FROM nvcr.io/nvidia/pytorch:26.03-py3

LABEL maintainer="rag-system"
LABEL description="GPU-accelerated RAG ingestion and retrieval for DGX Spark"

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# cuVS + CuPy from NVIDIA's PyPI index
RUN pip install --no-cache-dir \
        cuvs-cu13 \
        cupy-cuda13x \
        --extra-index-url=https://pypi.nvidia.com

WORKDIR /app

# Python deps (duplicated from pyproject.toml for layer caching)
RUN pip install --no-cache-dir \
        "pydantic>=2.10" \
        "pyyaml>=6.0" \
        "typer>=0.15" \
        "rich>=13.0" \
        "sentence-transformers>=5.0" \
        "pymupdf>=1.25" \
        "numpy>=1.26"

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
SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', trust_remote_code=True)"

VOLUME ["/data"]

ENTRYPOINT ["python", "cli.py"]
CMD ["--help"]
