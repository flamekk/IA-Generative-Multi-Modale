"""Page-aware indexing and retrieval for text, tables, and figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core.vector_stores.types import VectorStoreQueryMode

from src.config import get_settings
from src.embeddings.image_embeddings import embed_image
from src.embeddings.text_embeddings import BgeM3TextEmbedding, ClipTextEmbedding
from src.rag.generator import describe_image
from src.vectorstore.pgvector_store import (
    get_image_index,
    get_image_vector_store,
    get_text_index,
    replace_document_nodes,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _table_text(table: dict[str, Any]) -> str:
    rows = table.get("rows", [])
    return "Tableau extrait :\n" + "\n".join(" | ".join(cell or "" for cell in row) for row in rows)


def index_pdf(pdf_path: str | Path) -> dict[str, Any]:
    """Run extraction, caption its figures, then replace this document in PGVector."""
    from src.ingestion.pdf_loader import ingest_pdf

    settings = get_settings()
    source = Path(pdf_path).expanduser().resolve()
    manifest = ingest_pdf(source, settings.project_root)
    document_id = manifest["document_id"]
    text_data = _read_json(settings.project_root / manifest["outputs"]["text_json"])
    table_data = _read_json(settings.project_root / manifest["outputs"]["tables_json"])
    # CLIP's text tower accepts at most 77 tokens; keep chunks comfortably below it.
    splitter = SentenceSplitter(chunk_size=60, chunk_overlap=8)
    text_nodes: list[TextNode] = []
    image_nodes: list[TextNode] = []
    captions: list[dict[str, Any]] = []

    base = {
        "document_id": document_id,
        "source_file": source.name,
        "source_path": str(source),
        "sha256": manifest["sha256"],
    }
    for page in text_data["pages"]:
        page_number = int(page["page_number"])
        text = page.get("text", "").strip()
        if text:
            for chunk_number, chunk in enumerate(splitter.split_text(text), start=1):
                text_nodes.append(
                    TextNode(
                        id_=f"{document_id}:p{page_number}:text:{chunk_number}",
                        text=chunk,
                        metadata={**base, "page_number": page_number, "content_type": "text"},
                    )
                )

    for table in table_data.get("tables", []):
        table_text = _table_text(table)
        for chunk_number, chunk in enumerate(splitter.split_text(table_text), start=1):
            text_nodes.append(
                TextNode(
                    id_=f"{document_id}:p{table['page_number']}:table{table['table_number']}:{chunk_number}",
                    text=chunk,
                    metadata={
                        **base,
                        "page_number": int(table["page_number"]),
                        "content_type": "table",
                        "table_number": int(table["table_number"]),
                    },
                )
            )

    image_records = manifest.get("images", [])
    image_root = settings.project_root / "data" / "extracted" / "images" / document_id
    captions_path = image_root / "captions.json"
    cached_captions = {}
    if captions_path.is_file():
        try:
            cached_captions = {
                item.get("filename"): item.get("caption", "")
                for item in json.loads(captions_path.read_text(encoding="utf-8"))
                if item.get("caption")
            }
        except (OSError, json.JSONDecodeError, AttributeError):
            cached_captions = {}
    for image in image_records:
        image_path = Path(image["path"])
        if not image_path.is_absolute():
            image_path = settings.project_root / image_path
        page_number = int(image["page_number"])
        caption = cached_captions.get(image["filename"], "")
        if not caption:
            try:
                caption = describe_image(image_path, page_number)
            except RuntimeError as error:
                caption = ""
                manifest.setdefault("warnings", []).append(f"Image {image['filename']} : {error}")
        caption_record = {
            "page_number": page_number,
            "filename": image["filename"],
            "caption": caption,
        }
        captions.append(caption_record)
        captions_path.write_text(json.dumps(captions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        searchable_text = caption or f"Figure extraite du document {source.name}, page {page_number}."
        vector = embed_image(image_path, settings.clip_model, settings.clip_device)
        image_nodes.append(
            TextNode(
                id_=f"{document_id}:p{page_number}:image:{image['image_number']}",
                text=f"Figure de {source.name}, page {page_number}.\nDescription : {searchable_text}",
                metadata={
                    **base,
                    "page_number": page_number,
                    "content_type": "image",
                    "image_filename": image["filename"],
                    "image_path": str(image_path),
                },
                embedding=vector,
            )
        )

    text_embedder = BgeM3TextEmbedding(model_id=settings.embedding_model)
    for start in range(0, len(text_nodes), 24):
        batch = text_nodes[start : start + 24]
        embeddings = text_embedder.get_text_embedding_batch(
            [node.get_content(metadata_mode="none") for node in batch]
        )
        for node, embedding in zip(batch, embeddings, strict=True):
            node.embedding = embedding

    replace_document_nodes(document_id, text_nodes, image_nodes)
    captions_path.write_text(json.dumps(captions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest["indexed_node_count"] = len(text_nodes) + len(image_nodes)
    manifest["text_node_count"] = len(text_nodes)
    manifest["image_node_count"] = len(image_nodes)
    manifest["image_descriptions"] = captions
    manifest["outputs"]["image_captions_json"] = captions_path.relative_to(settings.project_root).as_posix()
    (settings.project_root / "data" / "processed" / f"{document_id}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _as_evidence(node: Any, score: float | None) -> dict[str, Any]:
    return {
        "text": node.get_content(metadata_mode="none"),
        "source_file": node.metadata.get("source_file", "document inconnu"),
        "page_number": node.metadata.get("page_number"),
        "content_type": node.metadata.get("content_type", "text"),
        "image_path": node.metadata.get("image_path"),
        "score": float(score) if score is not None else None,
    }


def search_documents(query: str, top_k: int | None = None) -> list[dict[str, Any]]:
    settings = get_settings()
    limit = top_k or settings.retrieval_top_k
    text_index: VectorStoreIndex = get_text_index()
    image_index: VectorStoreIndex = get_image_index()
    text_retriever = text_index.as_retriever(similarity_top_k=limit)
    image_retriever = image_index.as_retriever(similarity_top_k=max(1, limit // 3))
    text_evidence = [_as_evidence(item.node, item.score) for item in text_retriever.retrieve(query)]
    image_evidence = [_as_evidence(item.node, item.score) for item in image_retriever.retrieve(query)]
    # Keep rankings separate: BGE text and CLIP figure scores are not comparable.
    return text_evidence + image_evidence


def search_by_image(image_bytes: bytes, top_k: int | None = None) -> list[dict[str, Any]]:
    settings = get_settings()
    vector = embed_image(image_bytes, settings.clip_model, settings.clip_device)
    result = get_image_vector_store().query(
        VectorStoreQuery(
            query_embedding=vector,
            similarity_top_k=top_k or settings.retrieval_top_k,
            mode=VectorStoreQueryMode.DEFAULT,
        )
    )
    scores = result.similarities or [None] * len(result.nodes or [])
    return [_as_evidence(node, scores[index] if index < len(scores) else None) for index, node in enumerate(result.nodes or [])]


def format_evidence(evidence: list[dict[str, Any]]) -> str:
    if not evidence:
        return "Aucun passage pertinent n'a été trouvé dans l'index."
    blocks = []
    for item in evidence:
        page = item.get("page_number")
        page_label = f"page {page}" if page else "page inconnue"
        blocks.append(f"[{item['source_file']}, {page_label}, {item['content_type']}]\n{item['text']}")
    return "\n\n---\n\n".join(blocks)
