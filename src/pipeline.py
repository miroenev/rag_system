from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.chunkers.base import Chunk
from src.config import EmbedderProvider, Settings
from src.parsers.base import ParsedDocument
from src.retrievers.base import RetrievalResult

logger = logging.getLogger(__name__)


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
        from src.retrievers.cuvs_retriever import CuVSRetriever

        return CuVSRetriever(self._settings.retriever, dimension=self._embedder.dimension)

    def ingest(self, input_path: str | Path) -> int:
        """Ingest documents from a file or directory. Returns number of chunks indexed."""
        input_path = Path(input_path)
        files = self._collect_files(input_path)
        if not files:
            logger.warning("No supported files found at %s", input_path)
            return 0

        logger.info("Found %d files to ingest", len(files))
        all_chunks: list[Chunk] = []

        for file_path in files:
            doc = self._parse_file(file_path)
            if doc is None:
                continue
            chunks = self._chunker.chunk(
                doc.content,
                source_path=str(doc.source_path),
                metadata=doc.metadata,
            )
            chunks = self._quality_filter.filter(chunks)
            all_chunks.extend(chunks)
            logger.info("  %s -> %d chunks", file_path.name, len(chunks))

        if not all_chunks:
            logger.warning("No chunks produced from %d files", len(files))
            return 0

        texts = [c.text for c in all_chunks]
        logger.info("Embedding %d chunks...", len(texts))
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

        self._retriever.add(embeddings, chunk_dicts)
        self._retriever.build_index()
        self._retriever.save()

        logger.info("Ingested %d chunks from %d files", len(all_chunks), len(files))
        return len(all_chunks)

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Retrieve the most relevant chunks for a query."""
        self._retriever.load()
        self._retriever.build_index()

        effective_k = top_k or self._settings.retriever.top_k
        query_embedding = self._embedder.embed_query(query)

        if self._reranker is not None:
            fetch_k = effective_k * 4
            candidates = self._retriever.search(query_embedding, top_k=fetch_k)
            results = self._reranker.rerank(query, candidates, top_k=effective_k)
        else:
            results = self._retriever.search(query_embedding, top_k=effective_k)
        return results

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
