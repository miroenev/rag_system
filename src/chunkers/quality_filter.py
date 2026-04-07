from __future__ import annotations

import logging
import re

from src.config import QualityFilterConfig

from .base import Chunk

logger = logging.getLogger(__name__)

_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_DOI_RE = re.compile(r"\b(doi|DOI)[:\s]+\S+")
_CITATION_RE = re.compile(r"^\s*\[\d+\]\s+", re.MULTILINE)
_REFERENCE_HEADING_RE = re.compile(
    r"(?:^|\n)\s*(?:#{1,6}\s+)?"
    r"(?:References|Bibliography|Works\s+Cited|Acknowledgments|Acknowledgements)"
    r"\s*$",
    re.IGNORECASE | re.MULTILINE,
)


class QualityFilter:
    """Scores and filters chunks by content quality before embedding."""

    def __init__(self, config: QualityFilterConfig) -> None:
        self._config = config

    def filter(self, chunks: list[Chunk]) -> list[Chunk]:
        if not self._config.enabled:
            return chunks

        kept: list[Chunk] = []
        dropped = 0
        for chunk in chunks:
            score = self.score(chunk.text, chunk.section)
            if score >= self._config.min_quality_score:
                kept.append(chunk)
            else:
                dropped += 1
                logger.debug(
                    "Dropped chunk (score=%.2f, %d chars, section=%r): %.60s...",
                    score, len(chunk.text), chunk.section, chunk.text,
                )
        if dropped:
            logger.info("Quality filter: kept %d, dropped %d chunks", len(kept), dropped)
        return kept

    def score(self, text: str, section: str = "") -> float:
        """Compute a 0-1 quality score for a chunk. Higher is better."""
        scores = [
            self._info_density(text),
            self._url_density(text),
            self._reference_score(text, section),
            self._title_page_score(text),
        ]
        return min(scores)

    def _info_density(self, text: str) -> float:
        """Low score for text with very low information density."""
        words = text.split()
        if len(words) < 5:
            return 0.1

        sentences = [s.strip() for s in re.split(r"[.!?]\s", text) if len(s.strip()) > 10]
        if len(sentences) < 1:
            return 0.2

        unique = set(w.lower() for w in words if len(w) > 2)
        ratio = len(unique) / len(words)
        if ratio < 0.15:
            return 0.1
        return min(1.0, ratio * 2)

    def _url_density(self, text: str) -> float:
        """Low score when URLs dominate the chunk."""
        if not text:
            return 1.0
        url_chars = sum(len(m.group()) for m in _URL_RE.finditer(text))
        ratio = url_chars / len(text)
        if ratio > self._config.max_url_ratio:
            return 0.1
        return 1.0 - ratio

    def _reference_score(self, text: str, section: str = "") -> float:
        """Low score for reference/bibliography sections."""
        if not self._config.skip_references:
            return 1.0

        lower_section = section.lower()
        if any(kw in lower_section for kw in ("references", "bibliography", "works cited")):
            return 0.0

        if _REFERENCE_HEADING_RE.search(text):
            return 0.1

        citation_matches = _CITATION_RE.findall(text)
        doi_matches = _DOI_RE.findall(text)
        lines = [l for l in text.splitlines() if l.strip()]
        if lines and (len(citation_matches) + len(doi_matches)) / max(len(lines), 1) > 0.4:
            return 0.15

        return 1.0

    def _title_page_score(self, text: str) -> float:
        """Low score for title/author pages (many emails, affiliations, few sentences)."""
        emails = len(_EMAIL_RE.findall(text))
        sentences = len(re.findall(r"[.!?]\s+[A-Z]", text))
        lines = [l for l in text.splitlines() if l.strip()]

        if emails >= 3 and sentences < 3:
            return 0.1

        affiliation_markers = sum(
            1 for l in lines
            if re.search(r"\b(?:University|Institute|Department|Laboratory|College|Inc\.|Corp\.)\b", l)
        )
        if affiliation_markers >= 3 and sentences < 3:
            return 0.15

        return 1.0
