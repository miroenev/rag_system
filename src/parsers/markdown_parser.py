from __future__ import annotations

import logging
from pathlib import Path

from .base import BaseParser, ParsedDocument

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt"}


class MarkdownParser(BaseParser):
    """Native parser for Markdown and plain-text files."""

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in _SUPPORTED_EXTENSIONS

    def parse(self, path: Path) -> ParsedDocument:
        logger.info("Parsing Markdown/text: %s", path)
        content = path.read_text(encoding="utf-8")
        return ParsedDocument(
            source_path=path,
            content=content,
            metadata={"format": path.suffix.lower().lstrip(".")},
        )
