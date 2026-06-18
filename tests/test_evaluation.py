"""
Tests for evaluation.py and storage.py — no external API calls needed.
Scoring tests are skipped if GROQ_API_KEY is not set.
"""

import os
import sys
import pytest
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

HAS_GROQ = bool(os.environ.get("GROQ_API_KEY"))


# ---------------------------------------------------------------------------
# evaluation.py
# ---------------------------------------------------------------------------

def test_score_returns_null_without_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    # Reset the cached ready-state
    import src.evaluation as ev
    ev._ragas_ready = None

    result = ev.score("What is RAG?", "RAG is retrieval-augmented generation.", ["Context chunk."])
    assert result["faithfulness"] is None
    assert result["answer_relevancy"] is None
    assert result["context_precision"] is None


def test_score_returns_null_for_empty_answer():
    from src.evaluation import score
    result = score("query", "", ["some context"])
    assert all(v is None for v in result.values())


def test_score_returns_null_for_empty_contexts():
    from src.evaluation import score
    result = score("query", "some answer", [])
    assert all(v is None for v in result.values())


def test_scoring_available_false_without_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    import src.evaluation as ev
    ev._ragas_ready = None
    assert ev.scoring_available() is False


@pytest.mark.skipif(not HAS_GROQ, reason="GROQ_API_KEY not set")
def test_score_with_real_api():
    from src.evaluation import score
    result = score(
        "What is RAG?",
        "RAG stands for Retrieval-Augmented Generation. It combines a retrieval system with a language model.",
        ["Retrieval-Augmented Generation (RAG) is a technique that enhances LLMs with external knowledge."],
    )
    assert isinstance(result["faithfulness"], float) or result["faithfulness"] is None
    assert isinstance(result["answer_relevancy"], float) or result["answer_relevancy"] is None


# ---------------------------------------------------------------------------
# storage.py
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_PATH", db_path)
    import config
    config.DB_PATH = db_path
    import src.storage as storage
    storage.DB_PATH = db_path
    storage.init_db()
    return storage


def test_write_and_read_run(tmp_db):
    storage = tmp_db
    storage.write_run(
        model="test-model",
        config_id=2,
        config_name="Base + RAG",
        query="What is RAG?",
        scores={"faithfulness": 0.85, "answer_relevancy": 0.9, "context_precision": 0.75},
        latency_s=5.2,
    )
    rows = storage.read_recent(days=7)
    assert len(rows) == 1
    assert rows[0]["model"] == "test-model"
    assert rows[0]["faithfulness"] == pytest.approx(0.85)
    assert rows[0]["latency_s"] == pytest.approx(5.2)


def test_write_run_with_null_scores(tmp_db):
    storage = tmp_db
    storage.write_run(
        model="test-model",
        config_id=1,
        config_name="Baseline",
        query="test",
        scores={"faithfulness": None, "answer_relevancy": None, "context_precision": None},
        latency_s=2.0,
    )
    rows = storage.read_recent(days=7)
    assert rows[0]["faithfulness"] is None


def test_prune_old_removes_old_rows(tmp_db):
    from datetime import datetime, timedelta
    storage = tmp_db
    old_ts = (datetime.utcnow() - timedelta(days=35)).isoformat()
    storage.write_run("m", 1, "Baseline", "q", {}, 1.0, timestamp=old_ts)
    storage.write_run("m", 1, "Baseline", "q2", {}, 1.0)  # recent

    deleted = storage.prune_old(days=30)
    assert deleted == 1
    rows = storage.read_recent(days=7)
    assert len(rows) == 1


def test_detect_drift_flags_drop(tmp_db):
    from datetime import datetime, timedelta
    storage = tmp_db
    now = datetime.utcnow()

    # Prior window (24-48h ago): faithfulness 0.9
    prior_ts = (now - timedelta(hours=36)).isoformat()
    storage.write_run("m", 2, "Base + RAG", "q", {"faithfulness": 0.9}, 1.0, timestamp=prior_ts)

    # Recent window (last 24h): faithfulness 0.7 — a 0.2 drop
    recent_ts = (now - timedelta(hours=1)).isoformat()
    storage.write_run("m", 2, "Base + RAG", "q", {"faithfulness": 0.7}, 1.0, timestamp=recent_ts)

    alerts = storage.detect_drift(threshold=0.10)
    assert len(alerts) == 1
    assert alerts[0]["config_id"] == 2
    assert alerts[0]["drop"] == pytest.approx(0.2, abs=0.01)
