from __future__ import annotations

import logging
import os

import numpy as np

from src.config import EmbedderConfig

from .base import BaseEmbedder

logger = logging.getLogger(__name__)


class APIEmbedder(BaseEmbedder):
    """OpenAI-compatible embeddings API client."""

    def __init__(self, config: EmbedderConfig) -> None:
        self._config = config
        self._dimension = config.dimensions
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_API_BASE")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is required for API embedder")

        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url

        self._client = OpenAI(**kwargs)
        return self._client

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        client = self._get_client()
        batch_size = self._config.batch_size
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = client.embeddings.create(
                model=self._config.model,
                input=batch,
                dimensions=self._config.dimensions,
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        arr = np.asarray(all_embeddings, dtype=np.float32)
        return arr

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([query])[0]
