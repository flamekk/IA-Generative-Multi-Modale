"""Extract embedded PDF images while retaining their source page and dimensions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pymupdf


def extract_page_images(
    document: pymupdf.Document,
    page: pymupdf.Page,
    page_number: int,
    image_dir: Path,
    document_id: str,
) -> list[dict[str, Any]]:
    """Write each distinct embedded image on a page and return its metadata."""
    records: list[dict[str, Any]] = []
    seen_xrefs: set[int] = set()
    for image_number, item in enumerate(page.get_images(full=True), start=1):
        xref = int(item[0])
        if xref <= 0 or xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)
        extracted = document.extract_image(xref)
        extension = extracted.get("ext", "bin").lower().lstrip(".")
        filename = f"{document_id}_page_{page_number:04d}_image_{image_number:03d}.{extension}"
        destination = image_dir / filename
        destination.write_bytes(extracted["image"])
        records.append(
            {
                "page_number": page_number,
                "image_number": image_number,
                "xref": xref,
                "filename": filename,
                "path": destination.as_posix(),
                "width": extracted.get("width"),
                "height": extracted.get("height"),
                "colorspace": extracted.get("colorspace"),
                "extension": extension,
            }
        )
    return records
