FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download BGE models into the image so startup never waits for network.
# ~400 MB added to image size; model weights are cached in /root/.cache/huggingface.
RUN python - <<'PY'
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from sentence_transformers import SentenceTransformer, CrossEncoder
print("Downloading BGE-small-en-v1.5 …")
SentenceTransformer("BAAI/bge-small-en-v1.5")
print("Downloading BGE-reranker-base …")
CrossEncoder("BAAI/bge-reranker-base")
print("All models cached.")
PY

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
