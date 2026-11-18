"""CLIP text encoder exposed through LlamaIndex's embedding interface."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch
import torch.nn.functional as functional
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.embeddings import BaseEmbedding
from transformers import CLIPModel, CLIPProcessor

from src.config import get_settings


def _ollama_embed(texts: list[str], model_id: str) -> list[list[float]]:
    """Request dense BGE-M3 vectors from Ollama's local embedding endpoint."""
    import json
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen

    settings = get_settings()
    request = Request(
        f"{settings.ollama_base_url}/api/embed",
        data=json.dumps({"model": model_id, "input": texts, "truncate": True}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=300) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as error:
        raise RuntimeError(
            f"Impossible d'obtenir les embeddings {model_id!r} via Ollama. "
            f"Vérifiez le service et exécutez `ollama pull {model_id}`."
        ) from error
    vectors = payload.get("embeddings", [])
    if len(vectors) != len(texts):
        raise RuntimeError(f"Ollama a renvoyé {len(vectors)} vecteurs pour {len(texts)} passages.")
    expected = get_settings().text_vector_dimension
    if vectors and any(len(vector) != expected for vector in vectors):
        raise RuntimeError(f"Dimension d'embedding inattendue pour {model_id!r}; {expected} attendue.")
    return [[float(value) for value in vector] for vector in vectors]


class BgeM3TextEmbedding(BaseEmbedding):
    """Multilingual BGE-M3 text vectors served locally by Ollama."""

    model_id: str = "bge-m3"

    def __init__(self, model_id: str = "bge-m3", **kwargs: Any):
        super().__init__(model_name=model_id, model_id=model_id, **kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "maintidoc_bge_m3"

    def _get_query_embedding(self, query: str) -> list[float]:
        return _ollama_embed([query], self.model_id)[0]

    def _get_text_embedding(self, text: str) -> list[float]:
        return _ollama_embed([text], self.model_id)[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return _ollama_embed(texts, self.model_id)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embedding(text)

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._get_text_embeddings(texts)


@lru_cache(maxsize=2)
def _load_clip(model_id: str, device: str) -> tuple[CLIPModel, CLIPProcessor]:
    """Load a locally cached CLIP checkpoint; do not download weights at runtime."""
    try:
        processor = CLIPProcessor.from_pretrained(model_id, local_files_only=True)
        model = CLIPModel.from_pretrained(model_id, local_files_only=True).to(device).eval()
    except OSError as error:
        raise RuntimeError(
            f"Le modèle CLIP {model_id!r} n'est pas présent dans le cache local. "
            "Téléchargez-le une fois, puis relancez MAINTiDOC."
        ) from error
    return model, processor


def _projected_features(output: Any) -> torch.Tensor:
    """Handle the tensor and model-output return forms used by Transformers."""
    if hasattr(output, "pooler_output"):
        output = output.pooler_output
    elif isinstance(output, (tuple, list)):
        output = output[0]
    return functional.normalize(output.float(), p=2, dim=-1)


class ClipTextEmbedding(BaseEmbedding):
    """Cross-modal CLIP text embeddings; paired with CLIP image embeddings (512-D)."""

    model_id: str = "openai/clip-vit-base-patch32"
    device: str = "cpu"
    _model: Any = PrivateAttr()
    _processor: Any = PrivateAttr()

    def __init__(self, model_id: str = "openai/clip-vit-base-patch32", device: str = "cpu", **kwargs: Any):
        super().__init__(model_name=model_id, model_id=model_id, device=device, **kwargs)
        self._model, self._processor = _load_clip(model_id, device)

    @classmethod
    def class_name(cls) -> str:
        return "maintidoc_clip"

    def _encode_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        inputs = self._processor(
            text=texts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            features = _projected_features(self._model.get_text_features(**inputs))
        return features.cpu().tolist()

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._encode_texts([query])[0]

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._encode_texts([text])[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._encode_texts(texts)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embedding(text)

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._get_text_embeddings(texts)
