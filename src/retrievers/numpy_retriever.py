from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.config import RetrieverConfig

from .base import BaseRetriever, RetrievalResult
from .metadata_store import MetadataStore

logger = logging.getLogger(__name__)


class NumpyRetriever(BaseRetriever):
    """Brute-force vector search on CPU (NumPy) with SQLite metadata.

    Intended for small corpora or minimal deployments without FAISS.
    """

    def __init__(self, config: RetrieverConfig, dimension: int) -> None:
        self._config = config
        self._dimension = dimension
        self._index_dir = Path(config.index_path)
        self._index_dir.mkdir(parents=True, exist_ok=True)

        self._meta = MetadataStore(config.metadata_db)

        self._vectors: list[np.ndarray] = []
        self._id_map: list[int] = []
        self._np_dataset: np.ndarray | None = None

    def add(self, embeddings: np.ndarray, chunks: list[dict]) -> list[int]:
        ids = self._meta.insert_chunks(chunks)
        for idx, vec in zip(ids, embeddings):
            self._vectors.append(vec.astype(np.float32))
            self._id_map.append(idx)
        self._np_dataset = None
        return ids

    def build_index(self) -> None:
        if not self._vectors:
            logger.warning("No vectors to index")
            return

        n = len(self._vectors)
        if self._np_dataset is not None and self._np_dataset.shape[0] == n:
            logger.debug("NumPy dataset already matches vector count; skip rebuild")
            return

        self._np_dataset = np.vstack(self._vectors).astype(np.float32, copy=False)
        logger.info("Brute-force index ready (%d vectors, dim=%d)", n, self._dimension)

    def search(self, query_embedding: np.ndarray, top_k: int | None = None) -> list[RetrievalResult]:
        top_k = top_k or self._config.top_k
        min_len = self._config.min_chunk_length
        query = query_embedding.astype(np.float32)
        fetch_k = top_k * 3 if min_len > 0 else top_k

        if self._np_dataset is None or self._np_dataset.shape[0] == 0:
            return []

        k = min(int(fetch_k), len(self._id_map))
        indices, distances = self._search_numpy(query, k)

        chunk_ids = [int(self._id_map[i]) for i in indices if 0 <= i < len(self._id_map)]
        chunk_data = self._meta.get_chunks_by_ids(chunk_ids)
        id_to_data = {c["id"]: c for c in chunk_data}

        results: list[RetrievalResult] = []
        for idx, dist in zip(indices, distances):
            if idx >= len(self._id_map):
                continue
            chunk_id = self._id_map[idx]
            data = id_to_data.get(chunk_id)
            if data is None:
                continue
            if len(data["text"]) < min_len:
                continue

            if self._config.metric == "cosine":
                score = float(1.0 - dist)
            else:
                score = float(dist)

            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    text=data["text"],
                    score=score,
                    source_path=data["source_path"],
                    chunk_index=data["chunk_index"],
                    metadata=data["metadata"],
                )
            )
            if len(results) >= top_k:
                break
        return results

    def _search_numpy(self, query: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if self._np_dataset is None:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        if self._config.metric == "cosine":
            query_norm = query / (np.linalg.norm(query) + 1e-10)
            dataset_norms = self._np_dataset / (
                np.linalg.norm(self._np_dataset, axis=1, keepdims=True) + 1e-10
            )
            similarities = dataset_norms @ query_norm
            distances = 1.0 - similarities
        else:
            diff = self._np_dataset - query
            distances = np.sum(diff * diff, axis=1)

        top_k = min(top_k, len(distances))
        indices = np.argpartition(distances, max(top_k - 1, 0))[:top_k]
        indices = indices[np.argsort(distances[indices])]
        return indices, distances[indices]

    def save(self) -> None:
        vectors_path = self._index_dir / "vectors.npy"
        idmap_path = self._index_dir / "id_map.npy"

        if self._vectors:
            np.save(str(vectors_path), np.vstack(self._vectors))
            np.save(str(idmap_path), np.array(self._id_map, dtype=np.int64))

        logger.info("Index saved to %s", self._index_dir)

    def load(self) -> None:
        vectors_path = self._index_dir / "vectors.npy"
        idmap_path = self._index_dir / "id_map.npy"

        if not vectors_path.exists():
            logger.warning("No saved index found at %s", self._index_dir)
            self._np_dataset = None
            return

        dataset = np.load(str(vectors_path)).astype(np.float32, copy=False)
        self._id_map = np.load(str(idmap_path)).tolist()
        self._vectors = [dataset[i] for i in range(dataset.shape[0])]
        self._np_dataset = dataset
        logger.info("Loaded brute-force index (%d vectors)", dataset.shape[0])
