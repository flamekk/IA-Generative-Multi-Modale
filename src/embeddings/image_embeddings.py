"""CLIP image embeddings aligned with the project's text encoder."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import torch
from PIL import Image

from src.embeddings.text_embeddings import _load_clip, _projected_features


def embed_image(image: str | Path | bytes | BinaryIO, model_id: str, device: str = "cpu") -> list[float]:
    if isinstance(image, (str, Path)):
        opened = Image.open(image)
    elif isinstance(image, bytes):
        opened = Image.open(BytesIO(image))
    else:
        opened = Image.open(image)
    with opened.convert("RGB") as rgb_image:
        model, processor = _load_clip(model_id, device)
        inputs = processor(images=rgb_image, return_tensors="pt").to(device)
        with torch.inference_mode():
            features = _projected_features(model.get_image_features(**inputs))
        return features[0].cpu().tolist()
