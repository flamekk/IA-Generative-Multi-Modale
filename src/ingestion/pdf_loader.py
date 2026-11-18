"""Ingest a PDF into page-aware text, extracted images, table data, and a manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pymupdf

try:  # Supports both ``python -m`` and direct script execution.
    from .image_extractor import extract_page_images
    from .table_extractor import extract_page_tables
    from .text_extractor import extract_page_text
except ImportError:  # pragma: no cover - exercised by direct invocation
    from image_extractor import extract_page_images
    from table_extractor import extract_page_tables
    from text_extractor import extract_page_text


def _document_id(stem: str) -> str:
    safe = re.sub(r"[^\w.-]+", "_", stem, flags=re.UNICODE).strip("._")
    return safe or "document"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def ingest_pdf(pdf_path: str | Path, project_root: str | Path | None = None) -> dict[str, Any]:
    """Extract a PDF and write outputs under the project's documented data folders.

    Existing inputs are never moved or deleted. Re-ingesting the same PDF replaces
    only the generated files bearing that PDF's document id.
    """
    source = Path(pdf_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"PDF introuvable : {source}")
    if source.suffix.lower() != ".pdf":
        raise ValueError(f"Le fichier doit être un PDF : {source.name}")
    root = Path(project_root).expanduser().resolve() if project_root else Path.cwd().resolve()
    document_id = _document_id(source.stem)

    text_dir = root / "data" / "extracted" / "text"
    image_dir = root / "data" / "extracted" / "images" / document_id
    table_dir = root / "data" / "extracted" / "tables"
    processed_dir = root / "data" / "processed"
    image_dir.mkdir(parents=True, exist_ok=True)

    try:
        document = pymupdf.open(source)
    except Exception as error:
        raise ValueError(f"Impossible d'ouvrir le PDF {source.name} : {error}") from error
    if not document.is_pdf:
        document.close()
        raise ValueError(f"Le fichier n'est pas un PDF valide : {source}")

    pages: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []
    warnings: list[str] = []
    try:
        for page_index, page in enumerate(document, start=1):
            page_text = extract_page_text(page)
            page_images = extract_page_images(document, page, page_index, image_dir, document_id)
            try:
                page_tables = extract_page_tables(page, page_index)
            except Exception as error:
                page_tables = []
                warnings.append(f"Page {page_index} : détection des tableaux impossible ({error})")
            pages.append(
                {
                    "page_number": page_index,
                    "text": page_text,
                    "character_count": len(page_text),
                    "image_count": len(page_images),
                    "table_count": len(page_tables),
                }
            )
            images.extend(page_images)
            tables.extend(page_tables)

        pdf_metadata = {
            str(key): value
            for key, value in (document.metadata or {}).items()
            if value not in (None, "")
        }
        page_count = len(document)
    finally:
        document.close()

    common_metadata = {
        "document_id": document_id,
        "source_filename": source.name,
        "source_path": str(source),
        "file_size_bytes": source.stat().st_size,
        "sha256": _sha256(source),
        "ingested_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "page_count": page_count,
        "pdf_metadata": pdf_metadata,
    }
    text_path = text_dir / f"{document_id}.json"
    tables_path = table_dir / f"{document_id}.json"
    manifest_path = processed_dir / f"{document_id}.json"
    _write_json(text_path, {**common_metadata, "pages": pages})
    _write_json(tables_path, {**common_metadata, "tables": tables})

    result = {
        **common_metadata,
        "pymupdf_version": pymupdf.__version__,
        "text_character_count": sum(page["character_count"] for page in pages),
        "image_count": len(images),
        "table_count": len(tables),
        "warnings": warnings,
        "outputs": {
            "text_json": text_path.relative_to(root).as_posix(),
            "images_directory": image_dir.relative_to(root).as_posix(),
            "tables_json": tables_path.relative_to(root).as_posix(),
        },
        "images": images,
    }
    _write_json(manifest_path, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extraire texte, images et tableaux d'un PDF MAINTiDOC.")
    parser.add_argument("pdf", type=Path, help="Chemin du PDF à ingérer")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Racine MAINTiDOC (par défaut : dossier courant)",
    )
    args = parser.parse_args(argv)
    try:
        result = ingest_pdf(args.pdf, args.project_root)
    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"Erreur : {error}", file=sys.stderr)
        return 2
    # Escaping non-ASCII keeps the CLI usable in legacy Windows consoles such
    # as the default cp1252 PowerShell host; JSON files remain UTF-8.
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
