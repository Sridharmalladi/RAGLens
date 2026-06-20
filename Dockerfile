FROM python:3.11-slim-bookworm

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/app/.cache/huggingface

RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch — must come before requirements.txt
RUN pip install --no-cache-dir torch \
        --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-compile all installed packages to .pyc bytecode.
# Eliminates the 15-19s Python parse time on cold start so HF's
# health check gets a response before it times out.
RUN python -m compileall -q /usr/local/lib/python3.11/site-packages 2>/dev/null || true

COPY . .

RUN python -m compileall -q . 2>/dev/null || true

RUN useradd -m -u 1000 user && chown -R user:user /app
USER user

EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
