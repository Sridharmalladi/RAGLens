"""
Hugging Face Inference API adapter — query embedding and cross-encoder rerank.

Used only when RETRIEVAL_BACKEND == "hosted" (Vercel / Render / any host that
cannot carry torch + faiss). The corpus vectors were built with BGE-small and
this calls the same BGE-small model, so query and document vectors share a
space and nothing needs re-embedding.

Environment:
    HF_API_TOKEN  (or HF_TOKEN)   required in hosted mode
"""

import logging

import numpy as np

from config import HF_API_TOKEN, HF_INFERENCE_BASE, HF_EMBED_MODEL, HF_RERANK_MODEL

logger = logging.getLogger(__name__)

_TIMEOUT = 30.0


def available() -> bool:
    return bool(HF_API_TOKEN)


def _post(url: str, payload: dict):
    import httpx

    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"} if HF_API_TOKEN else {}
    resp = httpx.post(url, json=payload, headers=headers, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norms, 1e-9, None)


def embed(texts: list[str]) -> np.ndarray:
    """Return an L2-normalised (len(texts), dim) float32 array."""
    url = f"{HF_INFERENCE_BASE}/{HF_EMBED_MODEL}/pipeline/feature-extraction"
    out = _post(url, {"inputs": texts})
    arr = np.asarray(out, dtype=np.float32)
    if arr.ndim == 1:            # single input came back as a flat vector
        arr = arr[None, :]
    if arr.ndim == 3:            # token-level output, mean-pool
        arr = arr.mean(axis=1)
    return _l2_normalize(arr)


def rerank(query: str, texts: list[str]) -> list[float]:
    """Cross-encoder relevance score for each text, in the input order."""
    url = f"{HF_INFERENCE_BASE}/{HF_RERANK_MODEL}/pipeline/text-ranking"
    out = _post(url, {"inputs": {"query": query, "texts": texts}})
    scores = [0.0] * len(texts)
    for item in out:
        idx = item.get("index")
        if isinstance(idx, int) and 0 <= idx < len(texts):
            scores[idx] = float(item.get("score", 0.0))
    return scores
