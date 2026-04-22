from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


class MetadataStore:
    """SQLite-backed store for chunk text and metadata.

    Each chunk gets a monotonically increasing integer ID that serves as the
    foreign key into the vector index.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS chunks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                text        TEXT    NOT NULL,
                source_path TEXT    NOT NULL,
                chunk_index INTEGER NOT NULL,
                metadata    TEXT    NOT NULL DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_source
                ON chunks(source_path);
        """)
        self._conn.commit()

    def insert_chunks(self, chunks: list[dict]) -> list[int]:
        """Insert chunks and return their assigned integer IDs.

        Each dict must contain: text, source_path, chunk_index.
        Optional: metadata (dict).
        """
        ids: list[int] = []
        cursor = self._conn.cursor()
        for chunk in chunks:
            cursor.execute(
                "INSERT INTO chunks (text, source_path, chunk_index, metadata) VALUES (?, ?, ?, ?)",
                (
                    chunk["text"],
                    chunk["source_path"],
                    chunk["chunk_index"],
                    json.dumps(chunk.get("metadata", {})),
                ),
            )
            ids.append(cursor.lastrowid)  # type: ignore[arg-type]
        self._conn.commit()
        return ids

    def get_chunks_by_ids(self, ids: list[int]) -> list[dict]:
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self._conn.execute(
            f"SELECT id, text, source_path, chunk_index, metadata FROM chunks WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
        id_to_row = {row["id"]: row for row in rows}
        result = []
        for chunk_id in ids:
            row = id_to_row.get(chunk_id)
            if row is None:
                continue
            result.append({
                "id": row["id"],
                "text": row["text"],
                "source_path": row["source_path"],
                "chunk_index": row["chunk_index"],
                "metadata": json.loads(row["metadata"]),
            })
        return result

    def delete_by_source(self, source_path: str) -> list[int]:
        """Delete all chunks for a source and return deleted IDs."""
        rows = self._conn.execute(
            "SELECT id FROM chunks WHERE source_path = ?", (source_path,)
        ).fetchall()
        deleted_ids = [row["id"] for row in rows]
        if deleted_ids:
            self._conn.execute(
                "DELETE FROM chunks WHERE source_path = ?", (source_path,)
            )
            self._conn.commit()
        return deleted_ids

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) as cnt FROM chunks").fetchone()
        return row["cnt"]

    def close(self) -> None:
        self._conn.close()
