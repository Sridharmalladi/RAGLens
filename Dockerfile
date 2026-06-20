FROM python:3.11-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/app/.cache/huggingface

# Minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# ── 1. CPU-only torch (~200 MB vs 2-3 GB CUDA default) ──────────────────────
RUN pip install --no-cache-dir \
        torch==2.5.1+cpu \
        --index-url https://download.pytorch.org/whl/cpu

# ── 2. App dependencies ──────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── 3. Pre-download BGE models ───────────────────────────────────────────────
# BEFORE "COPY . ." so this heavy layer is cached across every code-only push.
# Only re-runs when requirements.txt or the torch version above changes.
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer, CrossEncoder
SentenceTransformer("BAAI/bge-small-en-v1.5")
CrossEncoder("BAAI/bge-reranker-base")
print("Models cached.")
PY

# ── 4. Source code ───────────────────────────────────────────────────────────
COPY . .

# ── 5. Pre-build FAISS index (~60 s; needs corpus/processed/chunks.json) ─────
RUN GROQ_API_KEY=build-placeholder python - <<'PY'
from src.corpus import get_chunks, get_index
chunks = get_chunks()
index  = get_index()
print(f"FAISS ready — {len(chunks)} chunks, dim={index.d}")
PY

# Non-root user (UID 1000 is what HF Spaces runs containers as)
RUN useradd -m -u 1000 user && chown -R user:user /app
USER user

EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
