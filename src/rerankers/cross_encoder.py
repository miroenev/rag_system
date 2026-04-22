from __future__ import annotations

import logging
import os
import warnings

from src.config import RerankerConfig
from src.retrievers.base import RetrievalResult

from .base import BaseReranker

logger = logging.getLogger(__name__)


class CrossEncoderReranker(BaseReranker):
    """Reranks retrieval candidates using a cross-encoder model.

    Cross-encoders jointly attend to (query, document) pairs, producing
    much more accurate relevance scores than bi-encoder cosine similarity.
    The tradeoff is latency: O(N) forward passes vs O(1) for vector search,
    so this is applied to a small candidate set (typically 20-40 items).
    """

    def __init__(self, config: RerankerConfig) -> None:
        self._config = config
        self._model = None

    def _get_model(self):
        if self._model is not None:
            return self._model

        from sentence_transformers import CrossEncoder

        device = self._config.device
        if device == "auto":
            device = "cpu"

        logger.info("Loading reranker model %s on %s", self._config.model, device)
        fd = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
        os.close(devnull)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._model = CrossEncoder(
                    self._config.model,
                    device=device,
                    trust_remote_code=True,
                )
        finally:
            os.dup2(fd, 2)
            os.close(fd)
        return self._model

    def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        if not results:
            return results

        model = self._get_model()
        pairs = [(query, r.text) for r in results]
        scores = model.predict(pairs, show_progress_bar=False)

        for result, score in zip(results, scores):
            result.score = float(score)

        results.sort(key=lambda r: r.score, reverse=True)
        reranked = results[:top_k]

        logger.debug(
            "Reranked %d candidates -> top %d (scores: %.4f to %.4f)",
            len(results), len(reranked),
            reranked[0].score if reranked else 0,
            reranked[-1].score if reranked else 0,
        )
        return reranked
