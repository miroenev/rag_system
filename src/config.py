from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ParserConfig(BaseModel):
    pdf_backend: Literal["pymupdf"] = "pymupdf"


class ChunkerConfig(BaseModel):
    strategy: Literal["recursive"] = "recursive"
    chunk_size: int = Field(default=512, ge=64)
    chunk_overlap: int = Field(default=64, ge=0)


class EmbedderProvider(str, Enum):
    LOCAL = "local"
    OPENAI = "openai"


class EmbedderConfig(BaseModel):
    provider: EmbedderProvider = EmbedderProvider.LOCAL
    model: str = "nomic-ai/nomic-embed-text-v1.5"
    device: Literal["cuda", "cpu", "auto"] = "cuda"
    dimensions: int = Field(default=768, ge=64)
    batch_size: int = Field(default=64, ge=1)


class RetrieverAlgorithm(str, Enum):
    CAGRA = "cagra"
    IVF_PQ = "ivf_pq"
    IVF_FLAT = "ivf_flat"


class RetrieverConfig(BaseModel):
    backend: Literal["cuvs"] = "cuvs"
    algorithm: RetrieverAlgorithm = RetrieverAlgorithm.CAGRA
    metric: Literal["cosine", "l2", "inner_product"] = "cosine"
    top_k: int = Field(default=5, ge=1)
    min_chunk_length: int = Field(default=128, ge=0)
    index_path: Path = Path("/data/indexes")
    metadata_db: Path = Path("/data/metadata.db")


class Settings(BaseModel):
    parser: ParserConfig = ParserConfig()
    chunker: ChunkerConfig = ChunkerConfig()
    embedder: EmbedderConfig = EmbedderConfig()
    retriever: RetrieverConfig = RetrieverConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> Settings:
        path = Path(path)
        if not path.exists():
            return cls()
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)
