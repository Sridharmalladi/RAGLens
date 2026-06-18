"""
Loads the pre-built FAISS index and chunk store from disk.
Building the corpus is done once in build_corpus.ipynb (Colab).
This module is read-only at runtime.
"""

import json
import logging
import os

import faiss
import numpy as np

logger = logging.getLogger(__name__)

_index: faiss.Index | None = None
_chunks: list[dict] | None = None  # [{id, text, source, chunk_idx}]


def _load() -> tuple[faiss.Index, list[dict]]:
    from config import FAISS_INDEX_PATH, CHUNKS_PATH

    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(
            f"FAISS index not found at {FAISS_INDEX_PATH}. "
            "Run build_corpus.ipynb in Google Colab to build it, "
            "then commit corpus/index.faiss and corpus/processed/chunks.json."
        )
    if not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(
            f"Chunks file not found at {CHUNKS_PATH}. "
            "Run build_corpus.ipynb to generate it."
        )

    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    logger.info("Corpus loaded: %d chunks, FAISS index dim=%d", len(chunks), index.d)
    return index, chunks


def get_index() -> faiss.Index:
    global _index, _chunks
    if _index is None:
        _index, _chunks = _load()
    return _index


def get_chunks() -> list[dict]:
    global _index, _chunks
    if _chunks is None:
        _index, _chunks = _load()
    return _chunks


def is_ready() -> bool:
    from config import FAISS_INDEX_PATH, CHUNKS_PATH
    return os.path.exists(FAISS_INDEX_PATH) and os.path.exists(CHUNKS_PATH)
