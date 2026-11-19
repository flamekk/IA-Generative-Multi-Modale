"""Small Ollama REST client for local vision descriptions and text generation."""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.config import get_settings


def normalize_model_text(text: str, *, max_chars: int = 1200, max_lines: int = 12) -> str:
    """Remove repeated model output and keep captions short enough for retrieval."""
    seen: set[str] = set()
    lines: list[str] = []
    for raw_line in str(text).replace("\r", "\n").splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", raw_line)
        line = re.sub(r"\*\*(.*?)\*\*|\*(.*?)\*", lambda match: match.group(1) or match.group(2), line)
        line = " ".join(line.split()).strip(" -*#\t")
        key = line.casefold()
        if not line or key in seen:
            continue
        seen.add(key)
        lines.append(line)
        if len(lines) >= max_lines:
            break
    normalized = "\n".join(lines)
    return normalized[:max_chars].rstrip()


def _post_chat(payload: dict, timeout: int = 300) -> dict:
    settings = get_settings()
    request = Request(
        f"{settings.ollama_base_url}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as error:
        raise RuntimeError(
            f"Ollama ne répond pas à {settings.ollama_base_url}. Vérifiez que le service local est démarré."
        ) from error


def describe_image(image_path: str | Path, page_number: int | None = None) -> str:
    settings = get_settings()
    image_bytes = Path(image_path).read_bytes()
    prompt = (
        "Décris cette figure technique en français en 3 à 6 puces courtes, sans répéter une idée. "
        "Transcris uniquement les libellés réellement lisibles et décris leur ordre et leurs flèches. "
        "Pour un schéma d'attention, distingue strictement Scaled Dot-Product Attention "
        "(MatMul → Scale → SoftMax → MatMul) de Multi-Head Attention "
        "(projections linéaires, têtes parallèles, concaténation, projection finale). "
        "Ne mélange pas leurs composants. Sépare observation et inférence; écris 'illisible' si nécessaire."
    )
    if page_number is not None:
        prompt += f" La figure vient de la page {page_number}."
    result = _post_chat(
        {
            "model": settings.vision_model,
            "messages": [{"role": "user", "content": prompt, "images": [base64.b64encode(image_bytes).decode("ascii")]}],
            "stream": False,
            "options": {"temperature": 0},
        }
    )
    return normalize_model_text(str(result.get("message", {}).get("content", "")))


def describe_image_bytes(image_bytes: bytes) -> str:
    settings = get_settings()
    result = _post_chat(
        {
            "model": settings.vision_model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Décris cette photo technique en français en 3 à 6 puces courtes, sans répétition. "
                        "Sépare les éléments visibles des hypothèses, indique les ambiguïtés et ne pose pas de diagnostic certain. "
                        "Si un schéma d'attention est visible, distingue les opérations MatMul/Scale/SoftMax "
                        "de projections linéaires, têtes parallèles et concaténation."
                    ),
                    "images": [base64.b64encode(image_bytes).decode("ascii")],
                }
            ],
            "stream": False,
            "options": {"temperature": 0},
        }
    )
    return normalize_model_text(str(result.get("message", {}).get("content", "")))


def generate_answer(prompt: str) -> str:
    settings = get_settings()
    result = _post_chat(
        {
            "model": settings.generation_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0, "num_predict": 512},
        }
    )
    return str(result.get("message", {}).get("content", "")).strip()
