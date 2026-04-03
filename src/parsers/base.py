from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ParsedDocument:
    """Result of parsing a single document."""

    source_path: Path
    content: str
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)


class BaseParser(ABC):
    """Abstract interface for document parsers."""

    @abstractmethod
    def parse(self, path: Path) -> ParsedDocument:
        """Parse a single file and return structured content."""

    @abstractmethod
    def supports(self, path: Path) -> bool:
        """Return True if this parser can handle the given file."""

    def parse_many(self, paths: list[Path]) -> list[ParsedDocument]:
        return [self.parse(p) for p in paths if self.supports(p)]
