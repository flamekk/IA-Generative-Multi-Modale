"""Best-effort extraction of text-based tables from PDF pages."""

from __future__ import annotations

from typing import Any

import pymupdf


def extract_page_tables(page: pymupdf.Page, page_number: int) -> list[dict[str, Any]]:
    """Extract detectable tables, with every record carrying a 1-based page number."""
    found = page.find_tables()
    records: list[dict[str, Any]] = []
    for table_number, table in enumerate(found.tables, start=1):
        rows = table.extract() or []
        normalized_rows = [[None if cell is None else str(cell).strip() for cell in row] for row in rows]
        records.append(
            {
                "page_number": page_number,
                "table_number": table_number,
                "bbox": [float(value) for value in table.bbox],
                "row_count": len(normalized_rows),
                "column_count": max((len(row) for row in normalized_rows), default=0),
                "rows": normalized_rows,
            }
        )
    return records
