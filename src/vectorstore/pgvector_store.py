"""LlamaIndex adapter for the persistent PostgreSQL/pgvector collection."""

from __future__ import annotations

from functools import lru_cache

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters
from llama_index.vector_stores.postgres import PGVectorStore

from src.config import get_settings
from src.embeddings.text_embeddings import BgeM3TextEmbedding, ClipTextEmbedding


def _create_store(table_name: str, embed_dim: int) -> PGVectorStore:
    settings = get_settings()
    try:
        return PGVectorStore.from_params(
            connection_string=settings.database_url,
            async_connection_string=settings.async_database_url,
            schema_name=settings.pg_schema,
            table_name=table_name,
            embed_dim=embed_dim,
            use_jsonb=True,
            perform_setup=True,
        )
    except Exception as error:
        raise RuntimeError(
            "Connexion à PGVector impossible. Démarrez PostgreSQL/pgvector avec "
            "`docker compose up -d pgvector`, puis vérifiez PG_HOST, PG_PORT et les identifiants."
        ) from error


@lru_cache(maxsize=1)
def get_text_vector_store() -> PGVectorStore:
    settings = get_settings()
    return _create_store(settings.pg_text_table, settings.text_vector_dimension)


@lru_cache(maxsize=1)
def get_image_vector_store() -> PGVectorStore:
    settings = get_settings()
    return _create_store(settings.pg_image_table, settings.image_vector_dimension)


def _delete_document_nodes(store: PGVectorStore, document_id: str) -> None:
    store.delete_nodes(
        filters=MetadataFilters(
            filters=[MetadataFilter(key="document_id", value=document_id, operator=FilterOperator.EQ)]
        )
    )


def replace_document_nodes(
    document_id: str,
    text_nodes: list[TextNode],
    image_nodes: list[TextNode],
) -> int:
    """Replace one document's nodes in both modality collections."""
    text_store = get_text_vector_store()
    image_store = get_image_vector_store()
    _delete_document_nodes(text_store, document_id)
    _delete_document_nodes(image_store, document_id)
    if text_nodes:
        text_store.add(text_nodes)
    if image_nodes:
        image_store.add(image_nodes)
    return len(text_nodes) + len(image_nodes)


def get_text_index() -> VectorStoreIndex:
    """Open the BGE-M3 text collection for LlamaIndex retrieval."""
    settings = get_settings()
    embed_model = BgeM3TextEmbedding(model_id=settings.embedding_model)
    return VectorStoreIndex.from_vector_store(get_text_vector_store(), embed_model=embed_model)


def get_image_index() -> VectorStoreIndex:
    """Open the CLIP image collection for text-to-figure retrieval."""
    settings = get_settings()
    embed_model = ClipTextEmbedding(model_id=settings.clip_model, device=settings.clip_device)
    return VectorStoreIndex.from_vector_store(get_image_vector_store(), embed_model=embed_model)
