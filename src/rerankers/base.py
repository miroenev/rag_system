from __future__ import annotations

from abc import ABC, abstractmethod

from src.retrievers.base import RetrievalResult


class BaseReranker(ABC):
    """Abstract interface for rerankers."""

    @abstractmethod
    def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        """Rerank results by relevance to the query. Returns top_k results."""
