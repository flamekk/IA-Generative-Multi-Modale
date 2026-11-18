"""Small Ollama REST client for local vision descriptions and text generation."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.config import get_settings


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
        "Décris en français cette figure technique pour permettre sa recherche documentaire. "
        "Identifie seulement les éléments lisibles, les relations et le sujet probable. "
        "Distingue ce qui est visible de ce qui est inféré; signale ce qui est illisible et n'invente rien."
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
    return str(result.get("message", {}).get("content", "")).strip()


def describe_image_bytes(image_bytes: bytes) -> str:
    settings = get_settings()
    result = _post_chat(
        {
            "model": settings.vision_model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Décris en français cette photo technique. Sépare les éléments visibles des hypothèses. "
                        "Ne pose pas de diagnostic certain et indique toute ambiguïté."
                    ),
                    "images": [base64.b64encode(image_bytes).decode("ascii")],
                }
            ],
            "stream": False,
            "options": {"temperature": 0},
        }
    )
    return str(result.get("message", {}).get("content", "")).strip()


def generate_answer(prompt: str) -> str:
    settings = get_settings()
    result = _post_chat(
        {
            "model": settings.generation_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.1},
        }
    )
    return str(result.get("message", {}).get("content", "")).strip()
