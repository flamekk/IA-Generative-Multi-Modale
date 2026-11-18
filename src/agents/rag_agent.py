"""Grounded RAG agent that retrieves evidence before answering."""

from __future__ import annotations

from typing import Any

from src.config import get_settings
from src.rag.generator import describe_image_bytes, generate_answer
from src.rag.retriever import format_evidence, search_by_image, search_documents
from src.agents.tools import build_search_tool


SYSTEM_PROMPT = """Tu es MAINTiDOC, un assistant documentaire pour des techniciens de maintenance.
Tu réponds uniquement à partir des passages retrouvés dans les PDF fournis.
Utilise l'outil de recherche avant de répondre. Cite chaque information technique sous la forme [nom du document, page N].
Si les sources ne répondent pas à la question, dis-le clairement et demande la précision utile.
Sépare les éléments observés dans une figure des hypothèses. Ne garantis jamais un diagnostic et ne recommande pas de piloter une machine.
Réponds en français, de façon concise et structurée."""


def _text_content(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return str(content)


def _merge_evidence(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for group in groups:
        for item in group:
            key = (item.get("source_file"), item.get("page_number"), item.get("content_type"), item.get("image_path"), item.get("text"))
            if key not in seen:
                seen.add(key)
                merged.append(item)
    return merged


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
    context = format_evidence(evidence)
    user_message = (
        f"Question du technicien : {question}\n\n"
        f"Contexte documentaire pré-récupéré :\n{context}\n\n"
        + (f"Description de la photo utilisateur :\n{vision_description}\n\n" if vision_description else "")
        + "Recherche les passages complémentaires si nécessaire, puis réponds avec des citations document/page."
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
    return {
        "answer": answer,
        "sources": evidence,
        "vision_description": vision_description,
        "agent_warning": agent_warning,
    }
