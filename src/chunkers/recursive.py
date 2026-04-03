from __future__ import annotations

import logging
import re

from src.config import ChunkerConfig

from .base import BaseChunker, Chunk

logger = logging.getLogger(__name__)

_DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


class RecursiveChunker(BaseChunker):
    """Recursive text splitter that tracks character offsets and section headings."""

    def __init__(self, config: ChunkerConfig) -> None:
        self._chunk_size = config.chunk_size
        self._chunk_overlap = config.chunk_overlap
        self._separators = list(_DEFAULT_SEPARATORS)

    def chunk(
        self, text: str, source_path: str, metadata: dict | None = None
    ) -> list[Chunk]:
        metadata = metadata or {}
        headings = self._extract_headings(text)
        raw_chunks = self._split_recursive(text, self._separators)
        positioned = self._locate_chunks(text, raw_chunks)
        merged = self._merge_with_overlap(positioned)

        results: list[Chunk] = []
        for i, (chunk_text, start, end) in enumerate(merged):
            if not chunk_text.strip():
                continue
            section = self._find_section(headings, start)
            chunk_meta = {**metadata, "start_char": start, "end_char": end}
            if section:
                chunk_meta["section"] = section
            results.append(
                Chunk(
                    text=chunk_text,
                    chunk_index=i,
                    source_path=source_path,
                    start_char=start,
                    end_char=end,
                    section=section,
                    metadata=chunk_meta,
                )
            )
        return results

    def _extract_headings(self, text: str) -> list[tuple[int, str]]:
        """Return (char_offset, heading_text) for each markdown heading."""
        return [(m.start(), m.group(2).strip()) for m in _HEADING_RE.finditer(text)]

    def _locate_chunks(self, text: str, chunks: list[str]) -> list[tuple[str, int, int]]:
        """Find each chunk's start/end character offset in the original text."""
        positioned: list[tuple[str, int, int]] = []
        search_from = 0
        for chunk in chunks:
            idx = text.find(chunk, search_from)
            if idx == -1:
                idx = text.find(chunk)
            if idx == -1:
                idx = search_from
            start = idx
            end = start + len(chunk)
            positioned.append((chunk, start, end))
            search_from = start + 1
        return positioned

    def _merge_with_overlap(
        self, positioned: list[tuple[str, int, int]]
    ) -> list[tuple[str, int, int]]:
        if not positioned or self._chunk_overlap == 0:
            return positioned

        merged: list[tuple[str, int, int]] = []
        for chunk_text, start, end in positioned:
            if merged and self._chunk_overlap > 0:
                prev_text, prev_start, prev_end = merged[-1]
                overlap_text = prev_text[-self._chunk_overlap :]
                new_text = overlap_text + chunk_text
                overlap_start = max(0, prev_end - self._chunk_overlap)
                merged.append((new_text, overlap_start, end))
            else:
                merged.append((chunk_text, start, end))
        return merged

    @staticmethod
    def _find_section(headings: list[tuple[int, str]], char_offset: int) -> str:
        """Find the nearest heading that precedes char_offset."""
        section = ""
        for heading_offset, heading_text in headings:
            if heading_offset <= char_offset:
                section = heading_text
            else:
                break
        return section

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        if not text:
            return []
        if len(text) <= self._chunk_size:
            return [text]
        if not separators:
            return [text[i : i + self._chunk_size] for i in range(0, len(text), self._chunk_size)]

        sep = separators[0]
        remaining_seps = separators[1:]

        if sep == "":
            return [text[i : i + self._chunk_size] for i in range(0, len(text), self._chunk_size)]

        parts = text.split(sep)
        result: list[str] = []
        current = ""

        for part in parts:
            candidate = f"{current}{sep}{part}" if current else part
            if len(candidate) <= self._chunk_size:
                current = candidate
            else:
                if current:
                    result.append(current)
                if len(part) > self._chunk_size:
                    result.extend(self._split_recursive(part, remaining_seps))
                else:
                    current = part
                    continue
                current = ""

        if current:
            result.append(current)
        return result
