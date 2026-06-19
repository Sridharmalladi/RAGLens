"""
Generation via Groq API — no local model loading, sub-second responses.
Replaces the previous torch/transformers-based approach.
"""

import logging
import os
import re
import time

from config import GROQ_GENERATION_MODEL, MAX_NEW_TOKENS

logger = logging.getLogger(__name__)

_RETRY_AFTER_RE = re.compile(r"Please try again in (\d+\.?\d*)s")


def _retry_wait(exc) -> float | None:
    """Return seconds to wait from a Groq 429 error, or None if not applicable."""
    m = _RETRY_AFTER_RE.search(str(exc))
    return float(m.group(1)) + 1.0 if m else None


def generate(query: str, context: str | None = None, model: str | None = None) -> tuple[str, float]:
    """
    Generate an answer via Groq API.
    Returns (answer, latency_seconds).
    Retries once on rate limit, respecting the retry-after duration.
    """
    from groq import Groq, RateLimitError

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "[GROQ_API_KEY not set — generation unavailable]", 0.0

    chosen_model = model or GROQ_GENERATION_MODEL

    system = (
        "You are a helpful research assistant specialising in RAG systems and LLM evaluation. "
        "Answer concisely and accurately. When context is provided, base your answer on it."
    )
    user_msg = (
        f"Context:\n{context}\n\nQuestion: {query}" if context
        else f"Question: {query}\n\nAnswer from your training knowledge."
    )

    client = Groq(api_key=api_key, max_retries=0, timeout=30.0)

    for attempt in range(2):
        start = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=chosen_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.7,
                max_tokens=MAX_NEW_TOKENS,
            )
            return resp.choices[0].message.content.strip(), time.perf_counter() - start
        except RateLimitError as e:
            wait = _retry_wait(e) or 30.0
            if attempt == 0:
                logger.warning("Rate limited — waiting %.1fs before retry", wait)
                time.sleep(wait)
            else:
                logger.error("Groq generation failed after retry: %s", e)
                return "[Rate limited — please wait a moment and try again]", 0.0
        except Exception as e:
            logger.error("Groq generation failed: %s", e)
            return f"[Generation failed: {e}]", 0.0

    return "[Generation failed: exhausted retries]", 0.0
