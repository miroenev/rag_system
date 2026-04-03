from __future__ import annotations

import logging
import os
import warnings

import numpy as np

from src.config import EmbedderConfig

from .base import BaseEmbedder

logger = logging.getLogger(__name__)


class LocalEmbedder(BaseEmbedder):
    """GPU-accelerated embeddings via sentence-transformers."""

    def __init__(self, config: EmbedderConfig) -> None:
        self._config = config
        self._model = None
        self._dimension = config.dimensions

    def _get_model(self):
        if self._model is not None:
            return self._model

        from sentence_transformers import SentenceTransformer

        device = self._config.device
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Loading embedding model %s on %s", self._config.model, device)
        fd = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
        os.close(devnull)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._model = SentenceTransformer(
                    self._config.model,
                    device=device,
                    trust_remote_code=True,
                    local_files_only=True,
                    model_kwargs={"safe_serialization": True},
                )
        finally:
            os.dup2(fd, 2)
            os.close(fd)
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        model = self._get_model()
        embeddings = model.encode(
            texts,
            batch_size=self._config.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        arr = np.asarray(embeddings, dtype=np.float32)
        if self._config.dimensions < arr.shape[1]:
            arr = arr[:, : self._config.dimensions]
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            arr = arr / norms
        return arr

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([query])[0]
