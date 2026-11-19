"""Grounded RAG agent that retrieves evidence before answering."""

from __future__ import annotations

import re
from typing import Any

from src.config import get_settings
from src.rag.generator import describe_image_bytes, generate_answer
from src.rag.retriever import format_evidence, reciprocal_rank_fusion, search_by_image, search_documents
from src.agents.tools import build_search_tool


SYSTEM_PROMPT = """Tu es MAINTiDOC, un assistant documentaire local pour des techniciens de maintenance.
Utilise exclusivement les faits présents dans les passages des PDF indexés ci-dessous ou retournés par l'outil de recherche.
N'ajoute aucune connaissance générale, aucun livre, site web, auteur ou référence externe. N'invente jamais un document, une page, une figure ni une citation.
Cite chaque affirmation technique avec le format exact [nom du document, page N] correspondant à un passage fourni.
Si les preuves sont absentes ou ambiguës, dis clairement ce qui manque et demande la précision utile.
Pour les figures, sépare les éléments visibles des hypothèses. Distingue Scaled Dot-Product Attention (MatMul, Scale, SoftMax, MatMul) de Multi-Head Attention (projections linéaires, têtes parallèles, concaténation et projection finale).
Ne garantis jamais un diagnostic et ne recommande pas de piloter une machine.
Réponds en français, en 3 à 6 phrases courtes ou puces, sans répétition."""


def _text_content(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return str(content)


def _merge_evidence(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return reciprocal_rank_fusion(*groups, limit=8)


def strip_unsupported_citations(answer: str, sources: list[dict[str, Any]]) -> str:
    """Remove external scholarly citations and document/page claims absent from retrieved evidence."""
    allowed = {
        (str(item.get("source_file", "")).casefold(), int(item["page_number"]))
        for item in sources
        if item.get("source_file") and item.get("page_number") is not None
    }

    def keep_document_citation(match: re.Match[str]) -> str:
        citation = match.group(1).strip()
        source_match = re.fullmatch(
            r"(.+?),\s*page\s+(\d+)(?:,\s*(?:text|image|table))?", citation, flags=re.IGNORECASE
        )
        if not re.search(r"\bpage\s+\d+\b", citation, flags=re.IGNORECASE):
            return match.group(0)
        if source_match and (source_match.group(1).strip().casefold(), int(source_match.group(2))) in allowed:
            return match.group(0)
        return ""

    cleaned = re.sub(r"\[([^\[\]]+)\]", keep_document_citation, answer)
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    external_reference = re.compile(
        r"(?:\b[A-Z][A-Za-z'-]+(?:\s+et al\.)?\s*\(\s*(?:19|20)\d{2}\s*\)|"
        r"\([A-Z][A-Za-z'-]+(?:\s+et al\.)?,?\s*(?:19|20)\d{2}\)|https?://|www\.)",
        flags=re.IGNORECASE,
    )
    sentences = [sentence for sentence in sentences if not external_reference.search(sentence)]
    return " ".join(" ".join(sentences).split()).strip()


def has_supported_citation(answer: str, sources: list[dict[str, Any]]) -> bool:
    allowed = {
        (str(item.get("source_file", "")).casefold(), int(item["page_number"]))
        for item in sources
        if item.get("source_file") and item.get("page_number") is not None
    }
    for match in re.finditer(
        r"\[([^\[\]]+),\s*page\s+(\d+)(?:,\s*(?:text|image|table))?\]",
        answer,
        flags=re.IGNORECASE,
    ):
        if (match.group(1).strip().casefold(), int(match.group(2))) in allowed:
            return True
    return False


def answer_question(question: str, image_bytes: bytes | None = None) -> dict[str, Any]:
    """Retrieve textual/visual context, then ask the LangChain Ollama agent."""
    settings = get_settings()
    vision_description = ""
    if image_bytes:
        vision_description = describe_image_bytes(image_bytes)
    retrieval_query = question
    if vision_description:
        retrieval_query += "\nDescription visuelle fournie par Qwen2.5-VL : " + vision_description

    text_evidence = search_documents(retrieval_query)
    image_evidence = search_by_image(image_bytes) if image_bytes else []
    evidence = _merge_evidence(text_evidence, image_evidence)
    if not evidence:
        return {
            "answer": "Je n'ai pas trouvé de preuve suffisante dans les documents indexés pour répondre. Précise le document, l'équipement ou le passage concerné.",
            "sources": [],
            "vision_description": vision_description,
            "agent_warning": None,
        }
    context = format_evidence(evidence)
    user_message = (
        f"Question du technicien : {question}\n\n"
        f"Contexte documentaire pré-récupéré :\n{context}\n\n"
        + (f"Description de la photo utilisateur :\n{vision_description}\n\n" if vision_description else "")
        + "Réponds seulement avec les éléments vérifiables dans ces passages. N'ajoute aucune référence extérieure. "
        + "Utilise au maximum 6 phrases courtes et cite le nom exact du document et la page."
    )

    agent_warning = None
    try:
        from langchain.agents import create_agent
        from langchain_ollama import ChatOllama

        model = ChatOllama(
            model=settings.generation_model,
            base_url=settings.ollama_base_url,
            temperature=0,
        )
        agent = create_agent(model=model, tools=[build_search_tool()], system_prompt=SYSTEM_PROMPT)
        result = agent.invoke({"messages": [{"role": "user", "content": user_message}]})
        answer = _text_content(result["messages"][-1]).strip()
    except Exception as error:
        # Keep the working local Qwen3 path available when tool calling is unsupported.
        agent_warning = f"Agent LangChain indisponible pour cette requête; génération directe utilisée ({error})."
        answer = generate_answer(f"{SYSTEM_PROMPT}\n\n{user_message}")

    if not answer:
        answer = "Je n'ai pas pu produire de réponse à partir des éléments retrouvés."
    answer = strip_unsupported_citations(answer, evidence)
    if not answer or not has_supported_citation(answer, evidence):
        answer = "Les passages retrouvés ne permettent pas de produire une réponse vérifiable avec une citation documentaire valide."
    return {
        "answer": answer,
        "sources": evidence,
        "vision_description": vision_description,
        "agent_warning": agent_warning,
    }
