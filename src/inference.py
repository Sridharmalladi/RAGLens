"""
User-facing inference path.
Runs all 4 RAG configs in parallel for a given query + model.
NEVER writes to the monitoring database — this path is read-only.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Generator

from config import CONFIG_NAMES, CONFIG_DESCRIPTIONS, TOP_K, RERANK_TOP_N

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=4)


def _run_config(config_id: int, query: str, model_id: str) -> dict:
    """
    Execute one RAG config. Returns a result dict.
    config_id 1-4 maps to: baseline / dense RAG / FT+dense / FT+hybrid+rerank
    """
    from src.retrieval import dense_retrieve, hybrid_retrieve, rerank
    from src.models import generate

    try:
        context_chunks: list[dict] = []

        if config_id == 1:
            # Baseline: no retrieval
            context = None

        elif config_id == 2:
            # Dense retrieval only
            context_chunks = dense_retrieve(query, k=TOP_K)
            context = _chunks_to_context(context_chunks)

        elif config_id == 3:
            # Fine-tuned model + dense retrieval
            context_chunks = dense_retrieve(query, k=TOP_K)
            context = _chunks_to_context(context_chunks)

        elif config_id == 4:
            # Fine-tuned model + hybrid retrieval + reranking
            candidates = hybrid_retrieve(query, k=TOP_K * 2)
            context_chunks = rerank(query, candidates, top_n=RERANK_TOP_N)
            context = _chunks_to_context(context_chunks)

        else:
            return _error_result(config_id, "Unknown config ID")

        use_adapter = config_id in (3, 4)
        answer, latency = generate(model_id, query, context, use_adapter=use_adapter)

        return {
            "config_id": config_id,
            "config_name": CONFIG_NAMES[config_id],
            "description": CONFIG_DESCRIPTIONS[config_id],
            "answer": answer,
            "latency": latency,
            "context_chunks": [c["text"] for c in context_chunks],
            "sources": list({c["source"] for c in context_chunks}),
            "error": None,
        }

    except Exception as e:
        logger.error("Config %d failed for query %r: %s", config_id, query[:60], e)
        return _error_result(config_id, str(e))


def _chunks_to_context(chunks: list[dict]) -> str:
    """Format retrieved chunks as a numbered context block."""
    if not chunks:
        return ""
    parts = [f"[{i+1}] {c['text']}" for i, c in enumerate(chunks)]
    return "\n\n".join(parts)


def _error_result(config_id: int, message: str) -> dict:
    return {
        "config_id": config_id,
        "config_name": CONFIG_NAMES.get(config_id, f"Config {config_id}"),
        "description": CONFIG_DESCRIPTIONS.get(config_id, ""),
        "answer": None,
        "latency": 0.0,
        "context_chunks": [],
        "sources": [],
        "error": message,
    }


def run_all_configs(
    query: str,
    model_id: str,
) -> Generator[dict, None, None]:
    """
    Run all 4 configs in parallel. Yields each result as it completes.
    Caller receives results in completion order, not config order.
    This is a generator — iterate it to get progressive disclosure.
    """
    futures = {
        _executor.submit(_run_config, config_id, query, model_id): config_id
        for config_id in range(1, 5)
    }
    for future in as_completed(futures):
        try:
            yield future.result()
        except Exception as e:
            config_id = futures[future]
            logger.error("Unexpected error in config %d: %s", config_id, e)
            yield _error_result(config_id, str(e))


def run_all_configs_blocking(query: str, model_id: str) -> dict[int, dict]:
    """
    Blocking version — waits for all 4 configs and returns {config_id: result}.
    Used by monitoring.py which needs all results before writing to DB.
    """
    results = {}
    for result in run_all_configs(query, model_id):
        results[result["config_id"]] = result
    return results
