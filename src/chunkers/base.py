from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A single chunk of text with provenance information."""

    text: str
    chunk_index: int
    source_path: str
    start_char: int = 0
    end_char: int = 0
    section: str = ""
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)


class BaseChunker(ABC):
    """Abstract interface for text chunkers."""

    @abstractmethod
    def chunk(self, text: str, source_path: str, metadata: dict | None = None) -> list[Chunk]:
        """Split text into chunks, preserving source info."""
