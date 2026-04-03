from __future__ import annotations

import contextlib
import io
import logging
import os
from pathlib import Path

import numpy as np

from src.config import RetrieverConfig

from .base import BaseRetriever, RetrievalResult
from .metadata_store import MetadataStore

logger = logging.getLogger(__name__)


class CuVSRetriever(BaseRetriever):
    """cuVS CAGRA/IVF-PQ vector retriever backed by SQLite metadata.

    On platforms without cuVS (e.g. CPU-only dev machines), falls back to
    brute-force NumPy search so the pipeline remains functional.
    """

    def __init__(self, config: RetrieverConfig, dimension: int) -> None:
        self._config = config
        self._dimension = dimension
        self._index_dir = Path(config.index_path)
        self._index_dir.mkdir(parents=True, exist_ok=True)

        self._meta = MetadataStore(config.metadata_db)

        self._vectors: list[np.ndarray] = []
        self._id_map: list[int] = []

        self._cuvs_index = None
        self._use_cuvs = False
        self._try_init_cuvs()

    def _try_init_cuvs(self) -> None:
        try:
            import cuvs  # noqa: F401
            self._use_cuvs = True
            logger.info("cuVS available -- using GPU-accelerated search")
        except ImportError:
            self._use_cuvs = False
            logger.warning(
                "cuVS not available -- falling back to brute-force NumPy search. "
                "Install cuvs-cu13 for GPU acceleration."
            )

    def add(self, embeddings: np.ndarray, chunks: list[dict]) -> list[int]:
        ids = self._meta.insert_chunks(chunks)

        for idx, vec in zip(ids, embeddings):
            self._vectors.append(vec.astype(np.float32))
            self._id_map.append(idx)

        return ids

    def build_index(self) -> None:
        if not self._vectors:
            logger.warning("No vectors to index")
            return

        dataset = np.vstack(self._vectors)
        n_vectors = dataset.shape[0]
        logger.info("Building index over %d vectors (dim=%d)", n_vectors, self._dimension)

        if self._use_cuvs:
            self._build_cuvs_index(dataset)
        else:
            self._np_dataset = dataset
            logger.info("Brute-force index ready (%d vectors)", n_vectors)

    def _build_cuvs_index(self, dataset: np.ndarray) -> None:
        from cuvs.neighbors import cagra
        import cupy as cp

        dataset_gpu = cp.asarray(dataset)
        n = dataset.shape[0]

        intermediate_graph_degree = min(128, n - 1)
        graph_degree = min(64, intermediate_graph_degree)

        index_params = cagra.IndexParams(
            metric="sqeuclidean" if self._config.metric == "l2" else "inner_product",
            graph_degree=graph_degree,
            intermediate_graph_degree=intermediate_graph_degree,
        )
        self._cuvs_index = cagra.build(index_params, dataset_gpu)
        logger.info("cuVS CAGRA index built (graph_degree=%d)", graph_degree)

    def search(self, query_embedding: np.ndarray, top_k: int | None = None) -> list[RetrievalResult]:
        top_k = top_k or self._config.top_k
        min_len = self._config.min_chunk_length
        query = query_embedding.astype(np.float32)

        fetch_k = top_k * 3 if min_len > 0 else top_k

        if self._use_cuvs and self._cuvs_index is not None:
            indices, distances = self._search_cuvs(query, min(fetch_k, len(self._id_map)))
        else:
            indices, distances = self._search_numpy(query, min(fetch_k, len(self._id_map)))

        query_norm = float(np.linalg.norm(query))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Query vector norm: %.6f", query_norm)
            logger.debug("Raw CAGRA distances (top 5): %s", distances[:5])
            logger.debug("Raw CAGRA indices   (top 5): %s", indices[:5])
            for i, (idx, dist) in enumerate(zip(indices[:5], distances[:5])):
                if idx < len(self._id_map) and idx < len(self._vectors):
                    vec = self._vectors[idx]
                    dot = float(np.dot(query, vec))
                    logger.debug(
                        "  [%d] idx=%d cagra=%.6f dot=%.6f",
                        i, int(idx), float(dist), dot,
                    )

        chunk_ids = [int(self._id_map[i]) for i in indices if i < len(self._id_map)]
        chunk_data = self._meta.get_chunks_by_ids(chunk_ids)
        id_to_data = {c["id"]: c for c in chunk_data}

        results: list[RetrievalResult] = []
        for rank, (idx, dist) in enumerate(zip(indices, distances)):
            if idx >= len(self._id_map):
                continue
            chunk_id = self._id_map[idx]
            data = id_to_data.get(chunk_id)
            if data is None:
                continue
            if len(data["text"]) < min_len:
                continue
            if self._config.metric == "cosine" and self._use_cuvs:
                actual_dot = float(np.dot(query, self._vectors[idx]))
                if abs(float(dist) - actual_dot) > 0.1:
                    logger.debug(
                        "Skipping idx=%d: CAGRA=%.4f vs actual_dot=%.4f (stale graph entry)",
                        int(idx), float(dist), actual_dot,
                    )
                    continue
                score = actual_dot
            elif self._config.metric == "cosine":
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

    def _search_cuvs(self, query: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        from cuvs.neighbors import cagra
        import cupy as cp

        query_gpu = cp.asarray(query.reshape(1, -1))
        search_params = cagra.SearchParams()
        distances, indices = cagra.search(search_params, self._cuvs_index, query_gpu, top_k)
        return cp.asnumpy(cp.asarray(indices))[0], cp.asnumpy(cp.asarray(distances))[0]

    def _search_numpy(self, query: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if not hasattr(self, "_np_dataset") or self._np_dataset is None:
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
        indices = np.argpartition(distances, top_k)[:top_k]
        indices = indices[np.argsort(distances[indices])]
        return indices, distances[indices]

    def save(self) -> None:
        vectors_path = self._index_dir / "vectors.npy"
        idmap_path = self._index_dir / "id_map.npy"

        if self._vectors:
            np.save(str(vectors_path), np.vstack(self._vectors))
            np.save(str(idmap_path), np.array(self._id_map, dtype=np.int64))

        if self._use_cuvs and self._cuvs_index is not None:
            from cuvs.neighbors import cagra
            cagra_path = self._index_dir / "cagra.index"
            cagra.save(str(cagra_path), self._cuvs_index)

        logger.info("Index saved to %s", self._index_dir)

    def load(self) -> None:
        vectors_path = self._index_dir / "vectors.npy"
        idmap_path = self._index_dir / "id_map.npy"

        if not vectors_path.exists():
            logger.warning("No saved index found at %s", self._index_dir)
            return

        dataset = np.load(str(vectors_path))
        self._id_map = np.load(str(idmap_path)).tolist()
        self._vectors = [dataset[i] for i in range(dataset.shape[0])]

        if self._use_cuvs:
            cagra_path = self._index_dir / "cagra.index"
            if cagra_path.exists():
                from cuvs.neighbors import cagra
                self._cuvs_index = cagra.load(str(cagra_path))
                logger.info("Loaded cuVS CAGRA index from %s", cagra_path)
                return

        self._np_dataset = dataset
        logger.info("Loaded brute-force index (%d vectors)", dataset.shape[0])
