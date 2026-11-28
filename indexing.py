"""Helpers for describing knowledge-base documents.

The module only contains pure functions (no LLM client), so importing it is
cheap and never fails because an API key is missing. The LLM pipeline itself
lives in ``nodes.py``.
"""

from __future__ import annotations

from typing import Any

SUMMARY_FIELDS = ("short_summary", "main_topics", "main_topic", "summary")


def _get_doc_name(doc: Any) -> str:
    """Best-effort document name for dict or LangChain Document objects."""
    if isinstance(doc, dict):
        return (
            doc.get("document_name")
            or doc.get("file_name")
            or doc.get("source")
            or "Unknown"
        )

    metadata = getattr(doc, "metadata", None)
    if isinstance(metadata, dict):
        return (
            metadata.get("document_name")
            or metadata.get("file_name")
            or metadata.get("source")
            or "Unknown"
        )
    return "Unknown"


def _format_metadata(doc: Any) -> str:
    """Render a document's metadata as a single readable line."""
    metadata = doc.get("metadata") if isinstance(doc, dict) else getattr(doc, "metadata", None)

    if isinstance(metadata, dict):
        parts = [
            f"{key}: {value}"
            for key, value in metadata.items()
            if value not in (None, "", [], {})
        ]
        return " | ".join(parts) if parts else "No metadata"
    if isinstance(metadata, str) and metadata.strip():
        return metadata.strip()
    return "No metadata"


def build_metadata(file_name: str, content: str, page_count: int | None = None) -> dict:
    """Build the lightweight metadata stored next to each loaded document."""
    metadata = {
        "source": file_name,
        "file_name": file_name,
        "characters": len(content),
        "words": len(content.split()),
    }
    if page_count is not None:
        metadata["pages"] = page_count
    return metadata
