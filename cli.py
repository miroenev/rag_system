from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from src.config import Settings
from src.pipeline import RAGPipeline

app = typer.Typer(
    name="rag-system",
    help="GPU-accelerated document ingestion and retrieval for DGX Spark.",
    add_completion=False,
)
console = Console()

CONFIG_OPTION = typer.Option("config/default.yaml", "--config", "-c", help="Path to YAML config")
VERBOSE_OPTION = typer.Option(False, "--verbose", "-v", help="Enable debug logging")


_NOISY_LOGGERS = [
    "httpx",
    "httpcore",
    "urllib3",
    "huggingface_hub",
    "sentence_transformers",
    "transformers",
    "filelock",
]


class _QuietFilter(logging.Filter):
    _SUPPRESSED = [
        "unauthenticated requests",
        "All keys matched",
        "new version of the following files",
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(s in msg for s in self._SUPPRESSED)


def _setup_logging(verbose: bool) -> None:
    import os

    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"

    import warnings
    warnings.filterwarnings("ignore")

    level = logging.DEBUG if verbose else logging.INFO
    handler = RichHandler(console=console, rich_tracebacks=True)
    if not verbose:
        handler.addFilter(_QuietFilter())
    logging.basicConfig(level=level, format="%(message)s", handlers=[handler])

    third_party_level = logging.DEBUG if verbose else logging.ERROR
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(third_party_level)


def _load_pipeline(config_path: str) -> RAGPipeline:
    settings = Settings.from_yaml(config_path)
    return RAGPipeline(settings)


@app.command()
def ingest(
    path: str = typer.Argument(..., help="File or directory to ingest"),
    config: str = CONFIG_OPTION,
    verbose: bool = VERBOSE_OPTION,
) -> None:
    """Parse, chunk, embed, and index documents."""
    _setup_logging(verbose)
    pipeline = _load_pipeline(config)

    with console.status("[bold green]Ingesting documents..."):
        n_chunks = pipeline.ingest(path)

    console.print(f"\n[bold green]Done![/] Ingested [cyan]{n_chunks}[/] chunks.")


@app.command()
def retrieve(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(None, "--top-k", "-k", help="Number of results (overrides config)"),
    config: str = CONFIG_OPTION,
    verbose: bool = VERBOSE_OPTION,
) -> None:
    """Retrieve relevant chunks for a query."""
    _setup_logging(verbose)
    pipeline = _load_pipeline(config)

    with console.status("[bold green]Searching..."):
        results = pipeline.retrieve(query, top_k=top_k)

    if not results:
        console.print("[yellow]No results found.[/]")
        raise typer.Exit(1)

    table = Table(title=f"Results for: {query}", show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Score", justify="right", width=8)
    table.add_column("Location", style="cyan", max_width=50)
    table.add_column("Chunk", max_width=80)

    for i, r in enumerate(results, 1):
        source = Path(r.source_path).name
        section = r.metadata.get("section", "")
        start = r.metadata.get("start_char", "")
        end = r.metadata.get("end_char", "")
        loc_parts = [source]
        if section:
            loc_parts.append(f"[white]§ {section}[/]")
        if start != "":
            loc_parts.append(f"[dim]chars {start}-{end}[/]")
        location = "\n".join(loc_parts)
        preview = r.text[:200].replace("\n", " ")
        if len(r.text) > 200:
            preview += "..."
        table.add_row(str(i), f"{r.score:.4f}", location, preview)

    console.print(table)


@app.command()
def view(
    config: str = CONFIG_OPTION,
    port: int = typer.Option(8501, "--port", "-p", help="Port to serve on"),
    verbose: bool = VERBOSE_OPTION,
) -> None:
    """Launch the chunk viewer web UI with semantic search."""
    _setup_logging(verbose)
    pipeline = _load_pipeline(config)
    db_path = str(pipeline._settings.retriever.metadata_db)

    from src.viewer import serve

    console.print(f"[bold green]Chunk viewer[/] running at [cyan]http://localhost:{port}[/]")
    console.print("Press Ctrl+C to stop.")
    serve(db_path, port=port, pipeline=pipeline)


if __name__ == "__main__":
    app()
