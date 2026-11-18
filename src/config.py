"""Central application configuration for MAINTiDOC."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # Environment variables can still be set by the caller.
    pass


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    vision_model: str = os.getenv("OLLAMA_VISION_MODEL", "qwen2.5vl:3b")
    generation_model: str = os.getenv("OLLAMA_GENERATION_MODEL", "qwen3:4b")
    embedding_model: str = os.getenv("OLLAMA_EMBED_MODEL", "bge-m3")
    clip_model: str = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
    clip_device: str = os.getenv("CLIP_DEVICE", "cpu")
    pg_host: str = os.getenv("PG_HOST", "127.0.0.1")
    pg_port: int = int(os.getenv("PG_PORT", "5432"))
    pg_database: str = os.getenv("PG_DATABASE", "maintidoc")
    pg_user: str = os.getenv("PG_USER", "maintidoc")
    pg_password: str = os.getenv("PG_PASSWORD", "maintidoc-local-only")
    pg_schema: str = os.getenv("PG_SCHEMA", "public")
    pg_text_table: str = os.getenv("PG_TEXT_TABLE", "maintidoc_text")
    pg_image_table: str = os.getenv("PG_IMAGE_TABLE", "maintidoc_images")
    text_vector_dimension: int = 1024
    image_vector_dimension: int = 512
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "6"))

    @property
    def database_url(self) -> str:
        from urllib.parse import quote_plus

        user = quote_plus(self.pg_user)
        password = quote_plus(self.pg_password)
        return f"postgresql+psycopg2://{user}:{password}@{self.pg_host}:{self.pg_port}/{self.pg_database}"

    @property
    def async_database_url(self) -> str:
        from urllib.parse import quote_plus

        user = quote_plus(self.pg_user)
        password = quote_plus(self.pg_password)
        return f"postgresql+asyncpg://{user}:{password}@{self.pg_host}:{self.pg_port}/{self.pg_database}"


def get_settings() -> Settings:
    return Settings()
