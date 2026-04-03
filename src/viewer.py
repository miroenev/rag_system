from __future__ import annotations

import json
import logging
import sqlite3
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)


class ViewerHandler(BaseHTTPRequestHandler):
    db_path: str = ""

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _json_response(self, data: object, status: int = 200) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html_response(self, html: str) -> None:
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/":
            self._html_response(_VIEWER_HTML)
        elif path == "/api/stats":
            self._handle_stats()
        elif path == "/api/sources":
            self._handle_sources()
        elif path == "/api/chunks":
            self._handle_chunks(params)
        elif path.startswith("/api/chunks/"):
            chunk_id = path.split("/")[-1]
            self._handle_chunk_detail(chunk_id)
        else:
            self.send_error(404)

    def _handle_stats(self) -> None:
        conn = self._connect()
        total = conn.execute("SELECT COUNT(*) as cnt FROM chunks").fetchone()["cnt"]
        sources = conn.execute("SELECT COUNT(DISTINCT source_path) as cnt FROM chunks").fetchone()["cnt"]
        avg_len = conn.execute("SELECT AVG(LENGTH(text)) as avg FROM chunks").fetchone()["avg"] or 0
        min_len = conn.execute("SELECT MIN(LENGTH(text)) as val FROM chunks").fetchone()["val"] or 0
        max_len = conn.execute("SELECT MAX(LENGTH(text)) as val FROM chunks").fetchone()["val"] or 0
        conn.close()
        self._json_response({
            "total_chunks": total,
            "total_sources": sources,
            "avg_chunk_length": round(avg_len, 1),
            "min_chunk_length": min_len,
            "max_chunk_length": max_len,
        })

    def _handle_sources(self) -> None:
        conn = self._connect()
        rows = conn.execute(
            "SELECT source_path, COUNT(*) as chunk_count, AVG(LENGTH(text)) as avg_len "
            "FROM chunks GROUP BY source_path ORDER BY source_path"
        ).fetchall()
        conn.close()
        self._json_response([
            {"source_path": r["source_path"], "chunk_count": r["chunk_count"], "avg_length": round(r["avg_len"], 1)}
            for r in rows
        ])

    def _handle_chunks(self, params: dict) -> None:
        conn = self._connect()
        source = params.get("source", [None])[0]
        search = params.get("search", [None])[0]
        limit = int(params.get("limit", [100])[0])
        offset = int(params.get("offset", [0])[0])

        where_clauses: list[str] = []
        bind: list[str] = []

        if source:
            where_clauses.append("source_path = ?")
            bind.append(source)
        if search:
            where_clauses.append("text LIKE ?")
            bind.append(f"%{search}%")

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        count = conn.execute(f"SELECT COUNT(*) as cnt FROM chunks {where_sql}", bind).fetchone()["cnt"]
        rows = conn.execute(
            f"SELECT id, chunk_index, source_path, metadata, LENGTH(text) as text_length, "
            f"SUBSTR(text, 1, 300) as preview FROM chunks {where_sql} "
            f"ORDER BY source_path, chunk_index LIMIT ? OFFSET ?",
            bind + [limit, offset],
        ).fetchall()
        conn.close()
        self._json_response({
            "total": count,
            "limit": limit,
            "offset": offset,
            "chunks": [
                {
                    "id": r["id"],
                    "chunk_index": r["chunk_index"],
                    "source_path": r["source_path"],
                    "text_length": r["text_length"],
                    "preview": r["preview"],
                    "metadata": json.loads(r["metadata"]),
                }
                for r in rows
            ],
        })

    def _handle_chunk_detail(self, chunk_id: str) -> None:
        conn = self._connect()
        row = conn.execute(
            "SELECT id, text, source_path, chunk_index, metadata FROM chunks WHERE id = ?",
            (chunk_id,),
        ).fetchone()
        conn.close()
        if row is None:
            self.send_error(404)
            return
        self._json_response({
            "id": row["id"],
            "text": row["text"],
            "source_path": row["source_path"],
            "chunk_index": row["chunk_index"],
            "metadata": json.loads(row["metadata"]),
            "text_length": len(row["text"]),
        })

    def log_message(self, format: str, *args: object) -> None:
        logger.debug(format, *args)


def serve(db_path: str, host: str = "0.0.0.0", port: int = 8501) -> None:
    ViewerHandler.db_path = db_path
    server = HTTPServer((host, port), ViewerHandler)
    logger.info("Chunk viewer running at http://%s:%d", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


_VIEWER_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RAG Chunk Viewer</title>
<style>
  :root {
    --bg: #0f1117;
    --surface: #1a1d27;
    --surface2: #242838;
    --border: #2e3347;
    --text: #e1e4ed;
    --text2: #8b90a5;
    --accent: #6c8aff;
    --accent2: #4a6aef;
    --green: #4ade80;
    --amber: #fbbf24;
    --red: #f87171;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    background: var(--bg); color: var(--text);
    line-height: 1.6;
  }
  .header {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 1rem 2rem;
    display: flex; align-items: center; gap: 1rem;
  }
  .header h1 { font-size: 1.25rem; font-weight: 600; }
  .header .tag {
    font-size: 0.75rem; background: var(--accent2); color: #fff;
    padding: 0.2rem 0.6rem; border-radius: 9999px;
  }
  .container { max-width: 1400px; margin: 0 auto; padding: 1.5rem 2rem; }

  .stats-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem; margin-bottom: 1.5rem;
  }
  .stat-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 0.75rem; padding: 1.25rem;
  }
  .stat-card .label { font-size: 0.8rem; color: var(--text2); margin-bottom: 0.25rem; }
  .stat-card .value { font-size: 1.75rem; font-weight: 700; }

  .controls {
    display: flex; gap: 0.75rem; margin-bottom: 1.5rem; flex-wrap: wrap;
  }
  .controls input, .controls select {
    background: var(--surface); border: 1px solid var(--border);
    color: var(--text); padding: 0.6rem 1rem; border-radius: 0.5rem;
    font-size: 0.9rem; outline: none; transition: border-color 0.2s;
  }
  .controls input:focus, .controls select:focus { border-color: var(--accent); }
  .controls input { flex: 1; min-width: 240px; }
  .controls select { min-width: 200px; }

  .chunk-list { display: flex; flex-direction: column; gap: 0.5rem; }
  .chunk-row {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 0.75rem; padding: 1rem 1.25rem;
    cursor: pointer; transition: border-color 0.2s, background 0.2s;
    display: grid; grid-template-columns: 60px 1fr auto; gap: 1rem; align-items: start;
  }
  .chunk-row:hover { border-color: var(--accent); background: var(--surface2); }
  .chunk-row.active { border-color: var(--accent); background: var(--surface2); }
  .chunk-id {
    font-size: 0.8rem; font-weight: 600; color: var(--accent);
    background: rgba(108,138,255,0.1); padding: 0.25rem 0.5rem;
    border-radius: 0.375rem; text-align: center;
  }
  .chunk-preview {
    font-size: 0.875rem; color: var(--text2);
    overflow: hidden; display: -webkit-box;
    -webkit-line-clamp: 2; -webkit-box-orient: vertical;
  }
  .chunk-meta {
    font-size: 0.75rem; color: var(--text2);
    text-align: right; min-width: 180px;
  }
  .chunk-meta .source {
    color: var(--green); font-weight: 500;
    max-width: 220px; overflow: hidden; text-overflow: ellipsis;
    white-space: nowrap;
  }
  .chunk-meta .section {
    color: var(--amber); font-weight: 500;
    max-width: 220px; overflow: hidden; text-overflow: ellipsis;
    white-space: nowrap; margin-top: 2px;
  }
  .chunk-meta .loc {
    color: var(--text2); margin-top: 2px;
  }

  .detail-panel {
    position: fixed; top: 0; right: -520px; width: 520px; height: 100vh;
    background: var(--surface); border-left: 1px solid var(--border);
    transition: right 0.3s ease; z-index: 100;
    display: flex; flex-direction: column; box-shadow: -4px 0 24px rgba(0,0,0,0.4);
  }
  .detail-panel.open { right: 0; }
  .detail-header {
    padding: 1.25rem; border-bottom: 1px solid var(--border);
    display: flex; justify-content: space-between; align-items: center;
  }
  .detail-header h2 { font-size: 1rem; }
  .detail-close {
    background: none; border: none; color: var(--text2); cursor: pointer;
    font-size: 1.5rem; line-height: 1;
  }
  .detail-close:hover { color: var(--text); }
  .detail-meta {
    padding: 1rem 1.25rem; border-bottom: 1px solid var(--border);
    display: flex; flex-direction: column; gap: 0.6rem;
    font-size: 0.8rem;
  }
  .detail-meta .dm-row {
    display: flex; justify-content: space-between; align-items: center;
  }
  .detail-meta .dm-label { color: var(--text2); }
  .detail-meta .dm-value { color: var(--text); font-weight: 500; text-align: right; }
  .detail-meta .dm-section {
    color: var(--amber); font-weight: 600; font-size: 0.9rem;
    padding: 0.4rem 0.75rem; background: rgba(251,191,36,0.08);
    border-radius: 0.375rem; border: 1px solid rgba(251,191,36,0.15);
  }
  .detail-meta .dm-loc {
    color: var(--accent); font-family: monospace; font-size: 0.8rem;
  }
  .detail-text {
    flex: 1; padding: 1.25rem; overflow-y: auto;
    font-size: 0.9rem; line-height: 1.8; white-space: pre-wrap;
    color: var(--text);
  }

  .pagination {
    display: flex; justify-content: center; align-items: center;
    gap: 1rem; margin-top: 1.5rem;
  }
  .pagination button {
    background: var(--surface); border: 1px solid var(--border);
    color: var(--text); padding: 0.5rem 1rem; border-radius: 0.5rem;
    cursor: pointer; font-size: 0.85rem;
  }
  .pagination button:hover { border-color: var(--accent); }
  .pagination button:disabled { opacity: 0.3; cursor: default; }
  .pagination .info { font-size: 0.85rem; color: var(--text2); }

  .overlay {
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(0,0,0,0.3); z-index: 99; display: none;
  }
  .overlay.open { display: block; }
  .empty { text-align: center; padding: 3rem; color: var(--text2); }
</style>
</head>
<body>

<div class="header">
  <h1>RAG Chunk Viewer</h1>
  <span class="tag" id="db-tag">loading...</span>
</div>

<div class="container">
  <div class="stats-grid" id="stats"></div>

  <div class="controls">
    <input type="text" id="search" placeholder="Search chunk text...">
    <select id="source-filter"><option value="">All sources</option></select>
  </div>

  <div class="chunk-list" id="chunk-list"></div>
  <div class="pagination" id="pagination"></div>
</div>

<div class="overlay" id="overlay" onclick="closeDetail()"></div>
<div class="detail-panel" id="detail">
  <div class="detail-header">
    <h2 id="detail-title">Chunk Detail</h2>
    <button class="detail-close" onclick="closeDetail()">&times;</button>
  </div>
  <div class="detail-meta" id="detail-meta"></div>
  <div class="detail-text" id="detail-text"></div>
</div>

<script>
const PAGE_SIZE = 50;
let currentOffset = 0;
let searchTimeout = null;

async function fetchJSON(url) {
  const r = await fetch(url);
  return r.json();
}

async function loadStats() {
  const s = await fetchJSON('/api/stats');
  document.getElementById('db-tag').textContent = s.total_chunks + ' chunks';
  document.getElementById('stats').innerHTML = `
    <div class="stat-card"><div class="label">Total Chunks</div><div class="value">${s.total_chunks}</div></div>
    <div class="stat-card"><div class="label">Sources</div><div class="value">${s.total_sources}</div></div>
    <div class="stat-card"><div class="label">Avg Length</div><div class="value">${s.avg_chunk_length}</div></div>
    <div class="stat-card"><div class="label">Min / Max</div><div class="value">${s.min_chunk_length} / ${s.max_chunk_length}</div></div>
  `;
}

async function loadSources() {
  const sources = await fetchJSON('/api/sources');
  const sel = document.getElementById('source-filter');
  sources.forEach(s => {
    const name = s.source_path.split('/').pop();
    const opt = document.createElement('option');
    opt.value = s.source_path;
    opt.textContent = name + ' (' + s.chunk_count + ')';
    sel.appendChild(opt);
  });
}

async function loadChunks() {
  const search = document.getElementById('search').value;
  const source = document.getElementById('source-filter').value;
  let url = '/api/chunks?limit=' + PAGE_SIZE + '&offset=' + currentOffset;
  if (source) url += '&source=' + encodeURIComponent(source);
  if (search) url += '&search=' + encodeURIComponent(search);
  const data = await fetchJSON(url);
  renderChunks(data);
}

function renderChunks(data) {
  const list = document.getElementById('chunk-list');
  if (data.chunks.length === 0) {
    list.innerHTML = '<div class="empty">No chunks found.</div>';
    document.getElementById('pagination').innerHTML = '';
    return;
  }
  list.innerHTML = data.chunks.map(c => {
    const section = c.metadata.section || '';
    const start = c.metadata.start_char;
    const end = c.metadata.end_char;
    const hasLoc = start !== undefined && end !== undefined;
    return `
    <div class="chunk-row" onclick="openDetail(${c.id})">
      <div class="chunk-id">#${c.id}</div>
      <div class="chunk-preview">${escHtml(c.preview)}</div>
      <div class="chunk-meta">
        <div class="source">${c.source_path.split('/').pop()}</div>
        ${section ? '<div class="section">&sect; ' + escHtml(section) + '</div>' : ''}
        <div class="loc">${hasLoc ? 'chars ' + start + '-' + end + ' &middot; ' : ''}${c.text_length} chars</div>
      </div>
    </div>`;
  }).join('');

  const pg = document.getElementById('pagination');
  const page = Math.floor(currentOffset / PAGE_SIZE) + 1;
  const pages = Math.ceil(data.total / PAGE_SIZE);
  pg.innerHTML = `
    <button onclick="prevPage()" ${currentOffset === 0 ? 'disabled' : ''}>Prev</button>
    <span class="info">Page ${page} of ${pages} (${data.total} chunks)</span>
    <button onclick="nextPage()" ${currentOffset + PAGE_SIZE >= data.total ? 'disabled' : ''}>Next</button>
  `;
}

async function openDetail(id) {
  const c = await fetchJSON('/api/chunks/' + id);
  document.getElementById('detail-title').textContent = 'Chunk #' + c.id;
  const m = c.metadata || {};
  const section = m.section || '';
  const start = m.start_char;
  const end = m.end_char;
  const fmt = m.format || '';
  let metaHtml = '';
  if (section) {
    metaHtml += `<div class="dm-section">&sect; ${escHtml(section)}</div>`;
  }
  metaHtml += `<div class="dm-row"><span class="dm-label">Source</span><span class="dm-value">${escHtml(c.source_path)}</span></div>`;
  if (start !== undefined && end !== undefined) {
    metaHtml += `<div class="dm-row"><span class="dm-label">Location</span><span class="dm-loc">chars ${start} &ndash; ${end}</span></div>`;
  }
  metaHtml += `<div class="dm-row"><span class="dm-label">Chunk Index</span><span class="dm-value">${c.chunk_index}</span></div>`;
  metaHtml += `<div class="dm-row"><span class="dm-label">Length</span><span class="dm-value">${c.text_length} chars</span></div>`;
  if (fmt) {
    metaHtml += `<div class="dm-row"><span class="dm-label">Format</span><span class="dm-value">${escHtml(fmt)}</span></div>`;
  }
  document.getElementById('detail-meta').innerHTML = metaHtml;
  document.getElementById('detail-text').textContent = c.text;
  document.getElementById('detail').classList.add('open');
  document.getElementById('overlay').classList.add('open');
}

function closeDetail() {
  document.getElementById('detail').classList.remove('open');
  document.getElementById('overlay').classList.remove('open');
}

function prevPage() { currentOffset = Math.max(0, currentOffset - PAGE_SIZE); loadChunks(); }
function nextPage() { currentOffset += PAGE_SIZE; loadChunks(); }

function escHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

document.getElementById('search').addEventListener('input', () => {
  clearTimeout(searchTimeout);
  searchTimeout = setTimeout(() => { currentOffset = 0; loadChunks(); }, 300);
});
document.getElementById('source-filter').addEventListener('change', () => { currentOffset = 0; loadChunks(); });
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeDetail(); });

loadStats();
loadSources();
loadChunks();
</script>
</body>
</html>
"""
