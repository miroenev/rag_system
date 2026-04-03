from __future__ import annotations

import logging
from pathlib import Path

from .base import BaseParser, ParsedDocument

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {".pdf"}


class PDFParser(BaseParser):
    """PDF text extraction using PyMuPDF (fitz)."""

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in _SUPPORTED_EXTENSIONS

    def parse(self, path: Path) -> ParsedDocument:
        import fitz

        logger.info("Parsing PDF: %s", path)
        doc = fitz.open(str(path))
        num_pages = len(doc)

        parts: list[str] = []
        for page in doc:
            text = page.get_text("text")
            if text.strip():
                parts.append(text)

        doc.close()

        content = "\n\n".join(parts)
        logger.info("  %s -> %d pages, %d chars extracted", path.name, num_pages, len(content))
        return ParsedDocument(
            source_path=path,
            content=content,
            metadata={
                "format": "pdf",
                "num_pages": num_pages,
            },
        )
