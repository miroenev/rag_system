from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseEmbedder(ABC):
    """Abstract interface for embedding providers."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts, returning an (N, D) float32 array."""

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query, returning a (D,) float32 array."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the embeddings produced."""
