"""LangChain retrieval tools used by the maintenance-document agent."""

from __future__ import annotations

import json

from langchain_core.tools import tool

from src.rag.retriever import search_documents


def build_search_tool():
    @tool("search_technical_documents")
    def search_technical_documents(query: str) -> str:
        """Search indexed technical PDF passages, tables, and figure descriptions. Use before answering."""
        results = search_documents(query)
        compact = [
            {
                "source_file": item["source_file"],
                "page_number": item["page_number"],
                "content_type": item["content_type"],
                "text": item["text"][:1600],
            }
            for item in results
        ]
        return json.dumps(compact, ensure_ascii=False)

    return search_technical_documents
