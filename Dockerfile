FROM python:3.11-slim-bookworm

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/app/.cache/huggingface \
    HF_HUB_OFFLINE=0

RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch — must come before requirements.txt
RUN pip install --no-cache-dir torch \
        --index-url https://download.pytorch.org/whl/cpu

# The Docker image runs the in-process BGE + FAISS backend, so it installs the
# full dependency set (requirements.txt is the lighter hosted-retrieval one).
COPY requirements-full.txt .
RUN pip install --no-cache-dir -r requirements-full.txt

# Bake the retrieval models into the image so cold starts do not spend time or
# memory downloading ~1.2 GB from the HF CDN. Matters on scale-to-zero hosts
# (Cloud Run) where the filesystem is memory-backed.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('BAAI/bge-small-en-v1.5'); \
    CrossEncoder('BAAI/bge-reranker-base')"

# Pre-compile installed packages to .pyc so the first request is not blocked on
# Python parse time.
RUN python -m compileall -q /usr/local/lib/python3.11/site-packages 2>/dev/null || true

COPY . .

RUN python -m compileall -q . 2>/dev/null || true

RUN useradd -m -u 1000 user && chown -R user:user /app
USER user

# Listen on $PORT when the platform sets it (Cloud Run uses 8080), otherwise
# 7860 for Hugging Face Spaces, which reads app_port from README.md.
ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]

