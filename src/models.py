"""
Generation via Groq API — no local model loading, sub-second responses.
Replaces the previous torch/transformers-based approach.
"""

import logging
import os
import time

from config import GROQ_GENERATION_MODEL, MAX_NEW_TOKENS

logger = logging.getLogger(__name__)


def generate(query: str, context: str | None = None) -> tuple[str, float]:
    """
    Generate an answer via Groq API.
    Returns (answer, latency_seconds).
    """
    from groq import Groq

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "[GROQ_API_KEY not set — generation unavailable]", 0.0

    system = (
        "You are a helpful research assistant specialising in RAG systems and LLM evaluation. "
        "Answer concisely and accurately. When context is provided, base your answer on it."
    )

    if context:
        user_msg = f"Context:\n{context}\n\nQuestion: {query}"
    else:
        user_msg = f"Question: {query}\n\nAnswer from your training knowledge."

    client = Groq(api_key=api_key)
    start = time.perf_counter()

    try:
        resp = client.chat.completions.create(
            model=GROQ_GENERATION_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.7,
            max_tokens=MAX_NEW_TOKENS,
        )
        answer = resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error("Groq generation failed: %s", e)
        return f"[Generation failed: {e}]", 0.0

    return answer, time.perf_counter() - start
