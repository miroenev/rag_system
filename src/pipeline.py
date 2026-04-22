from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from src.chunkers.base import Chunk
from src.config import EmbedderProvider, Settings
from src.parsers.base import ParsedDocument
from src.retrievers.base import RetrievalResult

logger = logging.getLogger(__name__)


@contextmanager
def _phase(name: str, timings: dict[str, float]):
    """Record wall-clock duration for a named phase into the timings dict."""
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[name] = time.perf_counter() - start


class RAGPipeline:
    """Orchestrates document ingestion and retrieval."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._parsers = self._build_parsers()
        self._chunker = self._build_chunker()
        self._quality_filter = self._build_quality_filter()
        self._embedder = self._build_embedder()
        self._reranker = self._build_reranker()
        self._retriever = self._build_retriever()

    def _build_parsers(self):
        from src.parsers.markdown_parser import MarkdownParser
        from src.parsers.pdf_parser import PDFParser

        return [
            PDFParser(),
            MarkdownParser(),
        ]

    def _build_chunker(self):
        from src.chunkers.recursive import RecursiveChunker

        return RecursiveChunker(self._settings.chunker)

    def _build_quality_filter(self):
        from src.chunkers.quality_filter import QualityFilter

        return QualityFilter(self._settings.quality_filter)

    def _build_embedder(self):
        if self._settings.embedder.provider == EmbedderProvider.OPENAI:
            from src.embedders.api import APIEmbedder

            return APIEmbedder(self._settings.embedder)
        else:
            from src.embedders.local import LocalEmbedder

            return LocalEmbedder(self._settings.embedder)

    def _build_reranker(self):
        if not self._settings.reranker.enabled:
            return None
        from src.rerankers.cross_encoder import CrossEncoderReranker

        return CrossEncoderReranker(self._settings.reranker)

    def _build_retriever(self):
        backend = self._settings.retriever.backend
        if backend == "numpy":
            from src.retrievers.numpy_retriever import NumpyRetriever

            return NumpyRetriever(self._settings.retriever, dimension=self._embedder.dimension)

        from src.retrievers.faiss_retriever import FaissRetriever

        return FaissRetriever(self._settings.retriever, dimension=self._embedder.dimension)

    def ingest(self, input_path: str | Path) -> int:
        """Ingest documents from a file or directory. Returns number of chunks indexed."""
        t0 = time.perf_counter()
        timings: dict[str, float] = {}

        input_path = Path(input_path)
        with _phase("collect", timings):
            files = self._collect_files(input_path)
        if not files:
            logger.warning("No supported files found at %s", input_path)
            return 0

        logger.info("Found %d files to ingest", len(files))
        all_chunks: list[Chunk] = []
        parse_s = 0.0
        chunk_s = 0.0
        filter_s = 0.0

        for file_path in files:
            t_parse = time.perf_counter()
            doc = self._parse_file(file_path)
            parse_s += time.perf_counter() - t_parse
            if doc is None:
                continue

            t_chunk = time.perf_counter()
            chunks = self._chunker.chunk(
                doc.content,
                source_path=str(doc.source_path),
                metadata=doc.metadata,
            )
            chunk_s += time.perf_counter() - t_chunk

            t_filter = time.perf_counter()
            chunks = self._quality_filter.filter(chunks)
            filter_s += time.perf_counter() - t_filter

            all_chunks.extend(chunks)
            logger.info("  %s -> %d chunks", file_path.name, len(chunks))

        timings["parse"] = parse_s
        timings["chunk"] = chunk_s
        timings["filter"] = filter_s

        if not all_chunks:
            logger.warning("No chunks produced from %d files", len(files))
            return 0

        texts = [c.text for c in all_chunks]
        logger.info("Embedding %d chunks...", len(texts))
        with _phase("embed", timings):
            embeddings = self._embed_batched(texts)

        chunk_dicts = [
            {
                "text": c.text,
                "source_path": c.source_path,
                "chunk_index": c.chunk_index,
                "metadata": c.metadata,
            }
            for c in all_chunks
        ]

        with _phase("index_add", timings):
            self._retriever.add(embeddings, chunk_dicts)
        with _phase("index_build", timings):
            self._retriever.build_index()
        with _phase("index_save", timings):
            self._retriever.save()

        total = time.perf_counter() - t0
        rate = len(all_chunks) / total if total > 0 else 0.0
        logger.info(
            "Ingested %d chunks from %d files in %.2fs (%.1f chunks/s)",
            len(all_chunks), len(files), total, rate,
        )
        self._log_timings("ingest", timings, total, extra=f"{len(all_chunks)} chunks")
        return len(all_chunks)

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Retrieve the most relevant chunks for a query."""
        t0 = time.perf_counter()
        timings: dict[str, float] = {}

        with _phase("load", timings):
            self._retriever.load()
        with _phase("index_build", timings):
            self._retriever.build_index()

        effective_k = top_k or self._settings.retriever.top_k
        with _phase("embed_query", timings):
            query_embedding = self._embedder.embed_query(query)

        if self._reranker is not None:
            fetch_k = effective_k * 4
            with _phase("search", timings):
                candidates = self._retriever.search(query_embedding, top_k=fetch_k)
            with _phase("rerank", timings):
                results = self._reranker.rerank(query, candidates, top_k=effective_k)
        else:
            with _phase("search", timings):
                results = self._retriever.search(query_embedding, top_k=effective_k)

        total = time.perf_counter() - t0
        self._log_timings("retrieve", timings, total, extra=f"{len(results)} results")
        return results

    @staticmethod
    def _log_timings(op: str, timings: dict[str, float], total: float, extra: str = "") -> None:
        """Emit one INFO line per phase and a summary."""
        tag = f" [{extra}]" if extra else ""
        logger.info("Timing breakdown for %s%s (total %.2fs):", op, tag, total)
        tracked = 0.0
        for name, dt in timings.items():
            pct = (dt / total * 100.0) if total > 0 else 0.0
            logger.info("  %-12s %7.3fs  (%4.1f%%)", name, dt, pct)
            tracked += dt
        other = max(total - tracked, 0.0)
        if other > 0.01:
            pct = (other / total * 100.0) if total > 0 else 0.0
            logger.info("  %-12s %7.3fs  (%4.1f%%)", "other", other, pct)

    def _collect_files(self, path: Path) -> list[Path]:
        if path.is_file():
            return [path]
        if path.is_dir():
            files: list[Path] = []
            for p in sorted(path.rglob("*")):
                if p.is_file() and any(parser.supports(p) for parser in self._parsers):
                    files.append(p)
            return files
        return []

    def _parse_file(self, path: Path) -> ParsedDocument | None:
        for parser in self._parsers:
            if parser.supports(path):
                try:
                    return parser.parse(path)
                except Exception:
                    logger.exception("Failed to parse %s", path)
                    return None
        logger.warning("No parser supports %s", path)
        return None

    def _embed_batched(self, texts: list[str]) -> np.ndarray:
        batch_size = self._settings.embedder.batch_size
        if len(texts) <= batch_size:
            return self._embedder.embed(texts)

        parts: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            parts.append(self._embedder.embed(batch))
        return np.vstack(parts)
