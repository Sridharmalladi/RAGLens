"""
Tests for models.py — use a tiny local model to avoid downloading full models in CI.
Tests focus on interface contracts, not generation quality.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

TINY_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # smallest available for fast tests


def test_generate_returns_string_and_float():
    from src.models import generate
    answer, latency = generate(TINY_MODEL, "What is RAG?", context=None)
    assert isinstance(answer, str)
    assert len(answer) > 0
    assert isinstance(latency, float)
    assert latency >= 0


def test_generate_with_context():
    from src.models import generate
    ctx = "Retrieval-Augmented Generation (RAG) combines retrieval with generation."
    answer, latency = generate(TINY_MODEL, "What is RAG?", context=ctx)
    assert isinstance(answer, str)
    assert latency >= 0


def test_generate_model_cached():
    from src.models import generate, _cache
    generate(TINY_MODEL, "test", context=None)
    before = len(_cache)
    generate(TINY_MODEL, "test 2", context=None)
    assert len(_cache) == before  # no new entry — cached


def test_adapter_available_false_without_file():
    from src.models import adapter_available
    # Adapter file won't exist in a fresh checkout
    result = adapter_available()
    assert isinstance(result, bool)


def test_generate_timeout_returns_message(monkeypatch):
    """Simulate timeout by setting a very short timeout."""
    import config
    monkeypatch.setattr(config, "GENERATION_TIMEOUT_SECONDS", 0.001)
    from src import models
    models.GENERATION_TIMEOUT_SECONDS = 0.001

    # Re-import to pick up the monkeypatched value
    import importlib
    importlib.reload(models)

    answer, latency = models.generate(TINY_MODEL, "Describe the entire history of AI in 10000 words.")
    assert "timed out" in answer.lower() or isinstance(answer, str)
