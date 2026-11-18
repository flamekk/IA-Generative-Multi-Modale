"""Text extraction helpers for PDF pages."""

from __future__ import annotations

import pymupdf


def extract_page_text(page: pymupdf.Page) -> str:
    """Return page text in reading order, preserving the PDF's page granularity."""
    return page.get_text("text", sort=True).strip()
