"""
LLM-as-judge scoring via Groq API (llama-3.1-8b-instant).
Three structured prompts cover the same axes as RAGAS:
  faithfulness, answer_relevancy, context_precision.
No external eval frameworks — just the groq SDK already in requirements.
"""

import logging
import os
import re

logger = logging.getLogger(__name__)

_ragas_ready: bool | None = None  # None = unchecked


def _check_available() -> bool:
    global _ragas_ready
    if _ragas_ready is not None:
        return _ragas_ready
    _ragas_ready = bool(os.environ.get("GROQ_API_KEY"))
    if not _ragas_ready:
        logger.warning("GROQ_API_KEY not set — scoring disabled")
    return _ragas_ready


_RETRY_RE = re.compile(r"Please try again in (\d+\.?\d*)s")


def _ask(prompt: str) -> float | None:
    """One Groq call. Returns a float in [0, 1] or None on failure."""
    import time
    from groq import Groq, RateLimitError
    from config import JUDGE_MODEL

    client = Groq(api_key=os.environ["GROQ_API_KEY"], max_retries=0, timeout=15.0)
    for attempt in range(2):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10,
            )
            text = resp.choices[0].message.content.strip()
            m = re.search(r"1\.0|0\.\d+|[01]", text)
            if m:
                return round(min(max(float(m.group()), 0.0), 1.0), 4)
            return None
        except RateLimitError as e:
            m = _RETRY_RE.search(str(e))
            wait = float(m.group(1)) + 1.0 if m else 15.0
            if attempt == 0:
                logger.warning("Scoring rate limited — waiting %.1fs", wait)
                time.sleep(wait)
            else:
                logger.warning("Scoring rate limited after retry, skipping")
                return None
        except Exception as e:
            logger.warning("Groq scoring call failed: %s", e)
            return None
    return None


def score(
    query: str,
    answer: str,
    contexts: list[str],
    ground_truth: str | None = None,
) -> dict:
    """
    Score one RAG result.
    Returns {faithfulness, answer_relevancy, context_precision} in [0, 1],
    or all-None if scoring is unavailable or inputs are empty.
    """
    null = {"faithfulness": None, "answer_relevancy": None, "context_precision": None}

    if not _check_available() or not answer or not contexts:
        return null

    ctx = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts[:3]))

    faithfulness = _ask(
        f"Context:\n{ctx}\n\n"
        f"Answer: {answer}\n\n"
        "Rate FAITHFULNESS (0.0–1.0): does the answer use ONLY information "
        "from the context, without adding facts not found there? "
        "1.0 = fully grounded in context, 0.0 = hallucinated. "
        "Reply with a single decimal number only."
    )

    relevancy = _ask(
        f"Question: {query}\n\n"
        f"Answer: {answer}\n\n"
        "Rate ANSWER RELEVANCY (0.0–1.0): how well does the answer address "
        "the question? 1.0 = perfectly on-topic, 0.0 = completely off-topic. "
        "Reply with a single decimal number only."
    )

    precision = _ask(
        f"Question: {query}\n\n"
        f"Retrieved context:\n{ctx}\n\n"
        "Rate CONTEXT PRECISION (0.0–1.0): how useful is this retrieved "
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
