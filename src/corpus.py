"""
Loads the FAISS index and chunk store.
If index.faiss is missing (e.g. fresh Docker container), it is built automatically
from chunks.json using BGE-small-en-v1.5. Takes ~60s on CPU; saved to disk so
subsequent startups are instant.
"""

import json
import logging
import os
import threading

import faiss
import numpy as np

logger = logging.getLogger(__name__)

_index: faiss.Index | None = None
_chunks: list[dict] | None = None
_load_lock = threading.Lock()


def _build_index(chunks: list[dict], index_path: str) -> faiss.Index:
    """Encode all chunks with BGE-small and build a flat L2 FAISS index.
    Uses the shared _get_embedder() singleton so there is never more than
    one SentenceTransformer load in flight at the same time."""
    from src.retrieval import _get_embedder  # thread-safe singleton with lock

    logger.info("FAISS index not found — building from %d chunks (one-time, ~60s)…", len(chunks))
    embedder = _get_embedder()
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(
        texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True
    )
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)
    logger.info("FAISS index built and saved to %s (dim=%d)", index_path, dim)
    return index


def _load() -> tuple[faiss.Index, list[dict]]:
    from config import FAISS_INDEX_PATH, CHUNKS_PATH

    if not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(
            f"Chunks file not found at {CHUNKS_PATH}. "
            "Ensure corpus/processed/chunks.json is committed to the repo."
        )

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if os.path.exists(FAISS_INDEX_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        logger.info("Corpus loaded: %d chunks, FAISS index dim=%d", len(chunks), index.d)
    else:
        index = _build_index(chunks, FAISS_INDEX_PATH)

    return index, chunks


def _ensure_loaded() -> None:
    global _index, _chunks
    if _index is None:
        with _load_lock:
            if _index is None:
                _index, _chunks = _load()


def get_index() -> faiss.Index:
    _ensure_loaded()
    return _index


def get_chunks() -> list[dict]:
    _ensure_loaded()
    return _chunks


def is_ready() -> bool:
    """True as long as chunks.json exists — index will be built if missing."""
    from config import CHUNKS_PATH
    return os.path.exists(CHUNKS_PATH)
