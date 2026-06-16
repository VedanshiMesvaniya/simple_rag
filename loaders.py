"""Document loaders for the PDF-only RAG pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_pdf(path: Path) -> str:
    """Extract text with PyMuPDF; append tables found by pdfplumber."""
    import pymupdf
    import pdfplumber

    with pymupdf.open(str(path)) as doc:
        pages = [page.get_text() for page in doc]

    tables: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables() or []:
                tables.append(
                    "\n".join("\t".join(c or "" for c in row) for row in table)
                )

    text = "\n\n".join(pages)
    if tables:
        text += "\n\n--- TABLES ---\n\n" + "\n\n".join(tables)
    return text


LOADERS: dict[str, callable] = {".pdf": load_pdf}


def load_document(path: Path) -> str:
    """Return extracted text for *path*, raising ValueError for unsupported types."""
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        raise ValueError(
            f"Unsupported file type: {path.suffix!r}. Supported: {list(LOADERS)}"
        )
    logger.info("Loading %s", path.name)
    text = loader(path)
    logger.info("  extracted %d characters", len(text))
    return text
