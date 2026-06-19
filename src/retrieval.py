"""
Four retrieval strategies, each independently testable.
All functions return a list of chunk dicts: [{id, text, source, chunk_idx}].
"""

import logging
import threading

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

from config import (
    EMBEDDING_MODEL, RERANKER_MODEL,
    TOP_K, RERANK_TOP_N, HYBRID_ALPHA,
)

logger = logging.getLogger(__name__)

# Module-level singletons — guarded by locks so concurrent threads don't double-init
_embedder: SentenceTransformer | None = None
_reranker: CrossEncoder | None = None
_bm25: BM25Okapi | None = None
_bm25_chunks: list[dict] | None = None

_embedder_lock = threading.Lock()
_reranker_lock = threading.Lock()
_bm25_lock = threading.Lock()


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                logger.info("Loading embedding model %s", EMBEDDING_MODEL)
                _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                logger.info("Loading reranker model %s", RERANKER_MODEL)
                _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def _get_bm25(chunks: list[dict]) -> BM25Okapi:
    global _bm25, _bm25_chunks
    with _bm25_lock:
        if _bm25 is None or _bm25_chunks is not chunks:
            logger.info("Building BM25 index over %d chunks", len(chunks))
            tokenized = [c["text"].lower().split() for c in chunks]
            _bm25 = BM25Okapi(tokenized)
            _bm25_chunks = chunks
    return _bm25


def dense_retrieve(query: str, k: int = TOP_K) -> list[dict]:
    """Top-k dense retrieval using BGE embeddings + FAISS."""
    from src.corpus import get_index, get_chunks

    embedder = _get_embedder()
    index = get_index()
    chunks = get_chunks()

    q_vec = embedder.encode([query], normalize_embeddings=True).astype(np.float32)
    distances, indices = index.search(q_vec, k=min(k, len(chunks)))

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue
        results.append({**chunks[idx], "score": float(dist)})
    return results


def sparse_retrieve(query: str, k: int = TOP_K) -> list[dict]:
    """Top-k BM25 retrieval — pure Python, no external service."""
    from src.corpus import get_chunks

    chunks = get_chunks()
    bm25 = _get_bm25(chunks)

    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:k]

    return [{**chunks[i], "score": float(scores[i])} for i in top_indices if scores[i] > 0]


def hybrid_retrieve(query: str, k: int = TOP_K, alpha: float = HYBRID_ALPHA) -> list[dict]:
    """
    Combine dense and sparse scores via weighted sum.
    alpha=1.0 → pure dense, alpha=0.0 → pure sparse.
    Scores are min-max normalized before combining.
    """
    from src.corpus import get_chunks

    chunks = get_chunks()
    n = len(chunks)

    # Dense scores
    embedder = _get_embedder()
    index = __import__("src.corpus", fromlist=["get_index"]).get_index()
    q_vec = embedder.encode([query], normalize_embeddings=True).astype(np.float32)
    distances, indices = index.search(q_vec, k=n)

    dense_scores = np.zeros(n)
    for dist, idx in zip(distances[0], indices[0]):
        if 0 <= idx < n:
            dense_scores[idx] = 1.0 / (1.0 + dist)  # convert L2 distance to similarity

    # Sparse scores
    bm25 = _get_bm25(chunks)
    bm25_scores = bm25.get_scores(query.lower().split())

    # Min-max normalize each to [0, 1]
    def _normalize(arr: np.ndarray) -> np.ndarray:
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / (hi - lo + 1e-9)

    combined = alpha * _normalize(dense_scores) + (1 - alpha) * _normalize(bm25_scores)
    top_indices = np.argsort(combined)[::-1][:k]

    return [{**chunks[i], "score": float(combined[i])} for i in top_indices]


def rerank(query: str, chunks: list[dict], top_n: int = RERANK_TOP_N) -> list[dict]:
    """
    Cross-encoder reranking with BGE-reranker-base.
    Input chunks are already retrieved; this step re-scores and filters to top_n.
    """
    if not chunks:
        return []

    reranker = _get_reranker()
    pairs = [[query, c["text"]] for c in chunks]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [{**chunk, "rerank_score": float(score)} for score, chunk in ranked[:top_n]]
