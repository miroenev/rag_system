from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class RetrievalResult:
    """A single retrieved chunk with its score and metadata."""

    chunk_id: int
    text: str
    score: float
    source_path: str
    chunk_index: int
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)


class BaseRetriever(ABC):
    """Abstract interface for vector-store retrievers."""

    @abstractmethod
    def add(self, embeddings: np.ndarray, chunks: list[dict]) -> list[int]:
        """Add embeddings and associated chunk data. Returns assigned IDs."""

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int) -> list[RetrievalResult]:
        """Return the top-k closest chunks for a query embedding."""

    @abstractmethod
    def build_index(self) -> None:
        """Build/rebuild the search index from stored vectors."""

    @abstractmethod
    def save(self) -> None:
        """Persist index and metadata to disk."""

    @abstractmethod
    def load(self) -> None:
        """Load index and metadata from disk."""
