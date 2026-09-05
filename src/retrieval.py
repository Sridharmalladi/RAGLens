"""
Four retrieval strategies, each independently testable.
All functions return a list of chunk dicts: [{id, text, source, chunk_idx}].
"""

import logging
import threading

import numpy as np
from rank_bm25 import BM25Okapi

# sentence-transformers is imported lazily inside _get_embedder / _get_reranker so
# the "hosted" backend (Vercel / Render) runs without torch installed.
from config import (
    EMBEDDING_MODEL, RERANKER_MODEL,
    TOP_K, RERANK_TOP_N, HYBRID_ALPHA,
    RETRIEVAL_BACKEND,
)

logger = logging.getLogger(__name__)

# Module-level singletons — guarded by locks so concurrent threads don't double-init
_embedder = None
_reranker = None
_bm25: BM25Okapi | None = None
_bm25_chunks: list[dict] | None = None

_embedder_lock = threading.Lock()
_reranker_lock = threading.Lock()
_bm25_lock = threading.Lock()


def _get_embedder():
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                from sentence_transformers import SentenceTransformer
                logger.info("Loading embedding model %s", EMBEDDING_MODEL)
                _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def models_loaded() -> bool:
    """True once the embedder and reranker are actually in memory (local backend)."""
    return _embedder is not None and _reranker is not None


def _get_reranker():
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                from sentence_transformers import CrossEncoder
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


def _hosted_query_vec(query: str) -> np.ndarray:
    from adapters import hf_inference
    return hf_inference.embed([query])[0].astype(np.float32)   # (dim,) L2-normalised


def _dense_hosted(query: str, k: int) -> list[dict]:
    """Dense retrieval as a numpy cosine over the committed corpus vectors."""
    from src.corpus import get_chunks, get_doc_matrix

    chunks = get_chunks()
    mat = get_doc_matrix()                      # (N, dim) L2-normalised
    sims = mat @ _hosted_query_vec(query)       # cosine similarity, higher is closer
    top = np.argsort(sims)[::-1][:min(k, len(chunks))]
    return [{**chunks[i], "score": float(sims[i])} for i in top]


def dense_retrieve(query: str, k: int = TOP_K) -> list[dict]:
    """Top-k dense retrieval. Local: BGE embeddings + FAISS. Hosted: HF Inference
    API query embedding + numpy cosine over the corpus vectors."""
    if RETRIEVAL_BACKEND == "hosted":
        return _dense_hosted(query, k)

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


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max to [0, 1]."""
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-9)


def hybrid_retrieve(query: str, k: int = TOP_K, alpha: float = HYBRID_ALPHA) -> list[dict]:
    """
    Combine dense and sparse scores via weighted sum.
    alpha=1.0 → pure dense, alpha=0.0 → pure sparse.
    Scores are min-max normalized before combining.
    """
    from src.corpus import get_chunks

    chunks = get_chunks()
    n = len(chunks)

    if RETRIEVAL_BACKEND == "hosted":
        from src.corpus import get_doc_matrix
        dense_scores = get_doc_matrix() @ _hosted_query_vec(query)   # cosine over all N
    else:
        from src.corpus import get_index
        embedder = _get_embedder()
        index = get_index()
        q_vec = embedder.encode([query], normalize_embeddings=True).astype(np.float32)
        distances, indices = index.search(q_vec, k=n)
        dense_scores = np.zeros(n)
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < n:
                dense_scores[idx] = 1.0 / (1.0 + dist)  # L2 distance → similarity

    # Sparse scores
    bm25 = _get_bm25(chunks)
    bm25_scores = np.asarray(bm25.get_scores(query.lower().split()))

    combined = alpha * _normalize(dense_scores) + (1 - alpha) * _normalize(bm25_scores)
    top_indices = np.argsort(combined)[::-1][:k]

    return [{**chunks[i], "score": float(combined[i])} for i in top_indices]


def rerank(query: str, chunks: list[dict], top_n: int = RERANK_TOP_N) -> list[dict]:
    """
    Cross-encoder reranking with BGE-reranker-base. Local: in-process CrossEncoder.
    Hosted: the HF Inference API text-ranking pipeline for the same model.
    Input chunks are already retrieved; this step re-scores and filters to top_n.
    """
    if not chunks:
        return []

    if RETRIEVAL_BACKEND == "hosted":
        from adapters import hf_inference
        try:
            scores = hf_inference.rerank(query, [c["text"] for c in chunks])
        except Exception as exc:
            logger.warning("Hosted rerank failed (%s); keeping retrieval order", exc)
            scores = [c.get("score", 0.0) for c in chunks]
    else:
        reranker = _get_reranker()
        scores = reranker.predict([[query, c["text"]] for c in chunks])

    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [{**chunk, "rerank_score": float(score)} for score, chunk in ranked[:top_n]]
