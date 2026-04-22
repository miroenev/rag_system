from __future__ import annotations

import logging
from pathlib import Path

import faiss
import numpy as np

from src.config import RetrieverConfig

from .base import BaseRetriever, RetrievalResult
from .metadata_store import MetadataStore

logger = logging.getLogger(__name__)


class FaissRetriever(BaseRetriever):
    """FAISS-backed vector retriever (CPU) with SQLite metadata.

    Uses ``IndexFlatIP`` for cosine and inner-product metrics (FAISS returns
    raw inner product as distance) and ``IndexFlatL2`` for L2. Optional
    ``IndexHNSWFlat`` when ``algorithm`` is ``hnsw``.

    On disk: ``vectors.npy``, ``id_map.npy``, and ``faiss.index``. If
    ``faiss.index`` is missing but ``vectors.npy`` exists (e.g. migrated from
    another branch), the FAISS index is rebuilt from vectors on ``build_index``.
    """

    def __init__(self, config: RetrieverConfig, dimension: int) -> None:
        self._config = config
        self._dimension = dimension
        self._index_dir = Path(config.index_path)
        self._index_dir.mkdir(parents=True, exist_ok=True)

        self._meta = MetadataStore(config.metadata_db)

        self._vectors: list[np.ndarray] = []
        self._id_map: list[int] = []
        self._faiss_index: faiss.Index | None = None

    def add(self, embeddings: np.ndarray, chunks: list[dict]) -> list[int]:
        ids = self._meta.insert_chunks(chunks)
        for idx, vec in zip(ids, embeddings):
            self._vectors.append(vec.astype(np.float32))
            self._id_map.append(idx)
        self._faiss_index = None
        return ids

    def build_index(self) -> None:
        if not self._vectors:
            logger.warning("No vectors to index")
            return

        n = len(self._vectors)
        if self._faiss_index is not None and int(self._faiss_index.ntotal) == n:
            logger.debug("FAISS index already matches vector count; skip rebuild")
            return

        dataset = np.vstack(self._vectors).astype(np.float32, copy=False)
        d = dataset.shape[1]
        metric = self._config.metric
        algo = self._config.algorithm

        if algo == "hnsw":
            m = 32
            if metric == "l2":
                index = faiss.IndexHNSWFlat(d, m, faiss.METRIC_L2)
            else:
                index = faiss.IndexHNSWFlat(d, m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efSearch = 64
        elif metric == "l2":
            index = faiss.IndexFlatL2(d)
        else:
            index = faiss.IndexFlatIP(d)

        index.add(dataset)
        self._faiss_index = index
        logger.info("FAISS index built (%s, %s, n=%d, dim=%d)", algo, metric, n, d)

    def search(self, query_embedding: np.ndarray, top_k: int | None = None) -> list[RetrievalResult]:
        top_k = top_k or self._config.top_k
        min_len = self._config.min_chunk_length
        query = query_embedding.astype(np.float32)
        fetch_k = top_k * 3 if min_len > 0 else top_k

        if self._faiss_index is None or self._faiss_index.ntotal == 0:
            return []

        k = min(int(fetch_k), int(self._faiss_index.ntotal))
        q = query.reshape(1, -1).astype(np.float32)
        if self._config.metric == "cosine":
            norms = np.linalg.norm(q, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            q = (q / norms).astype(np.float32)

        distances, indices = self._faiss_index.search(q, k)
        dist_row = distances[0]
        idx_row = indices[0]

        chunk_ids = [int(self._id_map[i]) for i in idx_row if 0 <= i < len(self._id_map)]
        chunk_data = self._meta.get_chunks_by_ids(chunk_ids)
        id_to_data = {c["id"]: c for c in chunk_data}

        results: list[RetrievalResult] = []
        for idx, dist in zip(idx_row, dist_row):
            if idx < 0 or idx >= len(self._id_map):
                continue
            chunk_id = self._id_map[idx]
            data = id_to_data.get(chunk_id)
            if data is None:
                continue
            if len(data["text"]) < min_len:
                continue

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

    def save(self) -> None:
        vectors_path = self._index_dir / "vectors.npy"
        idmap_path = self._index_dir / "id_map.npy"

        if self._vectors:
            np.save(str(vectors_path), np.vstack(self._vectors))
            np.save(str(idmap_path), np.array(self._id_map, dtype=np.int64))

        if self._faiss_index is not None:
            faiss_path = self._index_dir / "faiss.index"
            faiss.write_index(self._faiss_index, str(faiss_path))

        logger.info("Index saved to %s", self._index_dir)

    def load(self) -> None:
        vectors_path = self._index_dir / "vectors.npy"
        idmap_path = self._index_dir / "id_map.npy"

        if not vectors_path.exists():
            logger.warning("No saved index found at %s", self._index_dir)
            self._faiss_index = None
            return

        dataset = np.load(str(vectors_path)).astype(np.float32, copy=False)
        self._id_map = np.load(str(idmap_path)).tolist()
        self._vectors = [dataset[i] for i in range(dataset.shape[0])]

        faiss_path = self._index_dir / "faiss.index"
        if faiss_path.exists():
            self._faiss_index = faiss.read_index(str(faiss_path))
            logger.info("Loaded FAISS index from %s (%d vectors)", faiss_path, dataset.shape[0])
        else:
            self._faiss_index = None
            logger.info(
                "No faiss.index at %s; FAISS index will be built from vectors on build_index()",
                faiss_path,
            )
