from __future__ import annotations

import logging
import os
import warnings

import numpy as np

from src.config import EmbedderConfig

from .base import BaseEmbedder

logger = logging.getLogger(__name__)


class LocalEmbedder(BaseEmbedder):
    """CPU embeddings via sentence-transformers."""

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
            device = "cpu"

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

    def _has_prompts(self) -> bool:
        """Check if the loaded model has registered prompt templates."""
        model = self._get_model()
        return bool(getattr(model, "prompts", None))

    def _encode(self, texts: list[str], prompt_name: str | None = None) -> np.ndarray:
        model = self._get_model()
        kwargs: dict = {
            "batch_size": self._config.batch_size,
            "show_progress_bar": False,
            "normalize_embeddings": True,
        }
        if prompt_name and self._has_prompts():
            kwargs["prompt_name"] = prompt_name
        embeddings = model.encode(texts, **kwargs)
        arr = np.asarray(embeddings, dtype=np.float32)
        if self._config.dimensions < arr.shape[1]:
            arr = arr[:, : self._config.dimensions]
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            arr = arr / norms
        return arr

    def _resolve_prompt_name(self, role: str) -> str | None:
        """Map a semantic role to the model's registered prompt name.

        Models use different naming conventions (e.g. "search_document" vs
        "document"), so we try the most common variants in order.
        """
        model = self._get_model()
        prompts = getattr(model, "prompts", None)
        if not prompts:
            return None
        candidates = {
            "document": ["search_document", "document", "passage"],
            "query": ["search_query", "query"],
        }
        for name in candidates.get(role, []):
            if name in prompts:
                return name
        return None

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, prompt_name=self._resolve_prompt_name("document"))

    def embed_query(self, query: str) -> np.ndarray:
        return self._encode([query], prompt_name=self._resolve_prompt_name("query"))[0]
