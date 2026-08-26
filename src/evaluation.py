"""
LLM-as-judge scoring via the OpenRouter API (openai/gpt-oss-20b).
Three metrics matching RAGAS axes: faithfulness, answer_relevancy, context_precision.
answer_relevancy runs for all configs; faithfulness + context_precision require context.
"""

import logging
import re

from adapters import openrouter

logger = logging.getLogger(__name__)


def _check_available() -> bool:
    return openrouter.api_key_present()


def _ask(prompt: str) -> float | None:
    """One judge call. Returns a float in [0,1], or None on failure or rate limit."""
    from config import JUDGE_MODEL

    try:
        text = openrouter.complete(
            [{"role": "user", "content": prompt}],
            model=JUDGE_MODEL,
            temperature=0,
            max_tokens=128,  # headroom for the model's reasoning before the number
            timeout=20.0,
            reasoning_effort="low",
        )
    except Exception as exc:
        logger.warning("OpenRouter scoring call failed: %s", exc)
        return None

    match = re.search(r"1\.0|0\.\d+|[01]", text)
    if match:
        return round(min(max(float(match.group()), 0.0), 1.0), 4)
    return None


def score(
    query: str,
    answer: str,
    contexts: list[str] | None = None,
    ground_truth: str | None = None,
) -> dict:
    """
    Score one RAG result.
    - answer_relevancy is computed for all configs (no context needed).
    - faithfulness and context_precision are only computed when contexts is non-empty.
    Returns dict with all three keys; values are floats in [0,1] or None.
    """
    null = {"faithfulness": None, "answer_relevancy": None, "context_precision": None}
    if not _check_available() or not answer:
        return null

    contexts = contexts or []

    relevancy = _ask(
        f"Question: {query}\n\nAnswer: {answer}\n\n"
        "Rate ANSWER RELEVANCY (0.0 to 1.0): how well does the answer address "
        "the question? 1.0 = perfectly on-topic, 0.0 = completely off-topic. "
        "Reply with a single decimal number only."
    )

    if not contexts:
        return {"faithfulness": None, "answer_relevancy": relevancy, "context_precision": None}

    from config import SCORING_CONTEXT_CHARS
    ctx = "\n\n".join(f"[{i+1}] {c[:SCORING_CONTEXT_CHARS]}" for i, c in enumerate(contexts[:3]))

    faithfulness = _ask(
        f"Context:\n{ctx}\n\n"
        f"Answer: {answer}\n\n"
        "Rate FAITHFULNESS (0.0 to 1.0): does the answer use ONLY information "
        "from the context, without adding facts not found there? "
        "1.0 = fully grounded, 0.0 = hallucinated. "
        "Reply with a single decimal number only."
    )

    precision = _ask(
        f"Question: {query}\n\n"
        f"Retrieved context:\n{ctx}\n\n"
        "Rate CONTEXT PRECISION (0.0 to 1.0): how useful is this retrieved "
        "context for answering the question? 1.0 = highly relevant, "
        "0.0 = useless noise. Reply with a single decimal number only."
    )

    return {
        "faithfulness": faithfulness,
        "answer_relevancy": relevancy,
        "context_precision": precision,
    }


def scoring_available() -> bool:
    return _check_available()
