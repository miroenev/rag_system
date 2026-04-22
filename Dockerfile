FROM python:3.11-slim-bookworm

LABEL maintainer="rag-system"
LABEL description="CPU-only RAG ingestion and retrieval (cloud VM target)"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsqlite3-dev \
        build-essential \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch from the official CPU wheel index. Pinned before
# sentence-transformers so its torch dependency is already satisfied.
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch

# Python deps (duplicated from pyproject.toml for layer caching)
RUN pip install --no-cache-dir \
        "pydantic>=2.10" \
        "pyyaml>=6.0" \
        "typer>=0.15" \
        "rich>=13.0" \
        "sentence-transformers>=5.0" \
        "pymupdf>=1.25" \
        "numpy>=1.26" \
        "faiss-cpu>=1.9" \
        "einops>=0.8"

WORKDIR /app

# Copy application code
COPY pyproject.toml .
COPY cli.py .
COPY config/ config/
COPY src/ src/

# Install the project itself
RUN pip install --no-cache-dir --no-deps .

# Pre-download models so the container runs fully offline
RUN python -c "\
import warnings; warnings.filterwarnings('ignore'); \
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True, device='cpu'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', trust_remote_code=True, device='cpu')"

VOLUME ["/data"]

ENTRYPOINT ["python", "cli.py"]
CMD ["--help"]
