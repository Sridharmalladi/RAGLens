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

import numpy as np

# faiss is only imported where it is used, so the "hosted" retrieval backend
# (Vercel / Render) can run without it installed.

logger = logging.getLogger(__name__)

_index = None                       # faiss.Index, local backend only
_chunks: list[dict] | None = None
_doc_matrix: np.ndarray | None = None   # (N, dim) L2-normalised, hosted backend
_load_lock = threading.Lock()


def _load_precomputed_embeddings(embeddings_path: str) -> np.ndarray | None:
    """Load pre-computed embeddings from corpus/embeddings.json if present.
    Avoids running the encoder at startup — reduces cold-start from ~7 min to ~1 s."""
    if not os.path.exists(embeddings_path):
        return None
    import base64, json
    with open(embeddings_path) as f:
        payload = json.load(f)
    arr = np.frombuffer(base64.b64decode(payload["data"]), dtype=payload["dtype"])
    return arr.reshape(payload["shape"])


def _build_index(chunks: list[dict], index_path: str):
    """Build a flat L2 FAISS index from chunk embeddings (local backend only).
    Uses pre-computed embeddings from corpus/embeddings.json when available
    (instant). Falls back to encoding with BGE-small if the file is missing."""
    import faiss
    from config import EMBEDDINGS_PATH

    embeddings = _load_precomputed_embeddings(EMBEDDINGS_PATH)
    if embeddings is not None:
        logger.info("Loading pre-computed embeddings (%d chunks, dim=%d)…",
                    embeddings.shape[0], embeddings.shape[1])
    else:
        from src.retrieval import _get_embedder
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
    logger.info("FAISS index ready (dim=%d, %d chunks)", dim, len(chunks))
    return index


def _ensure_chunks() -> None:
    global _chunks
    if _chunks is not None:
        return
    from config import CHUNKS_PATH
    with _load_lock:
        if _chunks is not None:
            return
        if not os.path.exists(CHUNKS_PATH):
            raise FileNotFoundError(
                f"Chunks file not found at {CHUNKS_PATH}. "
                "Ensure corpus/processed/chunks.json is committed to the repo."
            )
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            _chunks = json.load(f)
        logger.info("Corpus loaded: %d chunks", len(_chunks))


def _ensure_index() -> None:
    global _index
    if _index is not None:
        return
    import faiss
    from config import FAISS_INDEX_PATH
    _ensure_chunks()
    with _load_lock:
        if _index is not None:
            return
        if os.path.exists(FAISS_INDEX_PATH):
            _index = faiss.read_index(FAISS_INDEX_PATH)
            logger.info("FAISS index loaded (dim=%d)", _index.d)
        else:
            _index = _build_index(_chunks, FAISS_INDEX_PATH)


def get_index():
    _ensure_index()
    return _index


def get_chunks() -> list[dict]:
    _ensure_chunks()
    return _chunks


def get_doc_matrix() -> np.ndarray:
    """(N, dim) L2-normalised corpus vectors for the hosted backend's cosine
    search. Reads the same corpus/embeddings.json the FAISS index is built from."""
    global _doc_matrix
    if _doc_matrix is not None:
        return _doc_matrix
    from config import EMBEDDINGS_PATH
    _ensure_chunks()
    with _load_lock:
        if _doc_matrix is not None:
            return _doc_matrix
        emb = _load_precomputed_embeddings(EMBEDDINGS_PATH)
        if emb is None:
            raise RuntimeError(
                "Hosted retrieval needs corpus/embeddings.json, which is missing."
            )
        emb = emb.astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        _doc_matrix = emb / np.clip(norms, 1e-9, None)
        logger.info("Corpus matrix ready for hosted retrieval (%d x %d)", *_doc_matrix.shape)
    return _doc_matrix


def is_ready() -> bool:
    """Whether a query can run without blocking on a cold load. Hosted: chunks +
    committed embeddings on disk. Local: that, plus the FAISS index and the BGE
    embedder/reranker actually loaded into memory — checking only file presence
    made the frontend show "ready" while the first query still paid for loading
    the models synchronously, which is the exact stall this flag exists to hide."""
    from config import CHUNKS_PATH, EMBEDDINGS_PATH, RETRIEVAL_BACKEND
    if not os.path.exists(CHUNKS_PATH):
        return False
    if RETRIEVAL_BACKEND == "hosted":
        return os.path.exists(EMBEDDINGS_PATH)
    if _index is None:
        return False
    from src.retrieval import models_loaded
    return models_loaded()
