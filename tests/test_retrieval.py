"""
Tests for retrieval.py — run without a real corpus using a tiny mock index.
"""

import json
import os
import sys
import tempfile
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def mock_corpus(tmp_path, monkeypatch):
    """Create a minimal FAISS index + chunks.json for testing."""
    import faiss
    from sentence_transformers import SentenceTransformer

    chunks = [
        {"id": f"paper_{i}", "text": f"Sample text about RAG topic {i}. " * 10, "source": f"paper_{i}", "chunk_idx": 0}
        for i in range(20)
    ]
    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, normalize_embeddings=True).astype(np.float32)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(tmp_path / "index.faiss"))
    with open(tmp_path / "chunks.json", "w") as f:
        json.dump(chunks, f)

    monkeypatch.setenv("FAISS_INDEX_PATH", str(tmp_path / "index.faiss"))

    import src.corpus as corpus_mod
    corpus_mod._index = index
    corpus_mod._chunks = chunks

    return chunks


def test_dense_retrieve_returns_k_results(mock_corpus):
    from src.retrieval import dense_retrieve
    results = dense_retrieve("What is dense retrieval?", k=5)
    assert len(results) == 5
    assert all("text" in r for r in results)
    assert all("score" in r for r in results)


def test_dense_retrieve_k_capped_by_corpus(mock_corpus):
    from src.retrieval import dense_retrieve
    results = dense_retrieve("test query", k=100)
    assert len(results) == len(mock_corpus)  # capped at corpus size


def test_sparse_retrieve_returns_results(mock_corpus):
    from src.retrieval import sparse_retrieve
    results = sparse_retrieve("RAG topic retrieval", k=5)
    assert isinstance(results, list)
    assert len(results) > 0


def test_sparse_retrieve_zero_scores_excluded(mock_corpus):
    from src.retrieval import sparse_retrieve
    results = sparse_retrieve("xyzzy_notaword_atall", k=5)
    # BM25 returns empty for queries with no matching tokens
    assert all(r["score"] > 0 for r in results)


def test_hybrid_retrieve_returns_k_results(mock_corpus):
    from src.retrieval import hybrid_retrieve
    results = hybrid_retrieve("dense sparse retrieval", k=5)
    assert len(results) == 5


def test_rerank_reduces_to_top_n(mock_corpus):
    from src.retrieval import dense_retrieve, rerank
    candidates = dense_retrieve("retrieval augmented generation", k=5)
    reranked = rerank("retrieval augmented generation", candidates, top_n=3)
    assert len(reranked) == 3
    assert all("rerank_score" in r for r in reranked)


def test_rerank_empty_input(mock_corpus):
    from src.retrieval import rerank
    result = rerank("anything", [], top_n=3)
    assert result == []
