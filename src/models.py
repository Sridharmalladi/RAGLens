"""Generation via the OpenRouter API. No local model weights, fast responses."""

import logging
import time

from adapters import openrouter
from config import OPENROUTER_GENERATION_MODEL, MAX_NEW_TOKENS

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a knowledgeable assistant. Answer the question directly and accurately "
    "in a few short paragraphs of plain prose. Do not use markdown tables, headings, "
    "or bullet lists. When context passages are provided, ground your answer in them "
    "and do not invent details they do not contain. If the context does not cover the "
    "question, say so briefly and answer from general knowledge."
)


def generate(
    query: str,
    context: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
) -> tuple[str, float, dict | None]:
    """
    Generate an answer via OpenRouter.
    Returns (answer, latency_seconds, usage) where usage is
    {prompt_tokens, completion_tokens, cost} or None on any failure.
    Retries once on a rate limit, respecting any retry-after hint.
    """
    if not openrouter.api_key_present():
        return "[OPENROUTER_API_KEY not set. Generation unavailable.]", 0.0, None

    chosen_model = model or OPENROUTER_GENERATION_MODEL
    user_msg = (
        f"Context:\n{context}\n\nQuestion: {query}" if context
        else f"Question: {query}\n\nAnswer from your training knowledge."
    )
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    start = time.perf_counter()
    try:
        answer, usage = openrouter.complete(
            messages,
            model=chosen_model,
            temperature=0.7,
            max_tokens=max_tokens or MAX_NEW_TOKENS,
            timeout=30.0,
            return_usage=True,
        )
        return answer, time.perf_counter() - start, usage
    except Exception as exc:
        from openai import RateLimitError

        if isinstance(exc, RateLimitError):
            logger.error("Generation rate limited after retry: %s", exc)
            return "[Rate limited. Please wait a moment and try again.]", 0.0, None
        logger.error("OpenRouter generation failed: %s", exc)
        return f"[Generation failed: {exc}]", 0.0, None
