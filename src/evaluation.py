"""
LLM-as-judge scoring via the OpenRouter API.

One judge call per answer. The context is sent once and the model returns every
metric in a single line, which keeps token use far below one call per metric.

Metrics (RAGAS axes):
  faithfulness       answer stays within the retrieved context   (needs context)
  answer_relevancy   answer addresses the question               (always)
  context_precision  retrieved context is useful for the question(needs context)

The scheduled monitoring job asks for faithfulness + relevancy only, since
context_precision is not charted anywhere. User-facing runs ask for all three.
"""

import logging
import re

from adapters import openrouter

logger = logging.getLogger(__name__)

_NULL = {"faithfulness": None, "answer_relevancy": None, "context_precision": None}
_NUM = r"(1(?:\.0+)?|0?\.\d+|[01])"


def _check_available() -> bool:
    return openrouter.api_key_present()


def _clamp(v: str) -> float:
    return round(min(max(float(v), 0.0), 1.0), 4)


def _judge(prompt: str, max_tokens: int = 200) -> str | None:
    # max_tokens is only a ceiling. Billed tokens are the model's actual
    # reasoning + output, which at "minimal" effort is a few dozen. The ceiling
    # just has to be high enough that the short answer is not clipped.
    from config import JUDGE_MODEL
    try:
        return openrouter.complete(
            [{"role": "user", "content": prompt}],
            model=JUDGE_MODEL,
            temperature=0,
            max_tokens=max_tokens,
            timeout=20.0,
            reasoning_effort="minimal",
        )
    except Exception as exc:
        logger.warning("Judge call failed: %s", exc)
        return None


def _parse(text: str, keys: tuple[str, ...]) -> dict:
    out = {}
    for key in keys:
        m = re.search(key + r"\s*[=:]\s*" + _NUM, text, re.I)
        if m:
            out[key] = _clamp(m.group(1))
    if not out:  # model ignored the format, fall back to numbers in order
        found = re.findall(_NUM, text)
        for key, val in zip(keys, found):
            out[key] = _clamp(val)
    return out


def score(
    query: str,
    answer: str,
    contexts: list[str] | None = None,
    full: bool = True,
) -> dict:
    """
    Score one answer with a single judge call.
    `full=False` skips context_precision (used by the scheduled job).
    Returns the three metric keys; values are floats in [0,1] or None.
    """
    if not _check_available() or not answer:
        return dict(_NULL)

    contexts = contexts or []

    # No retrieval: relevancy is the only meaningful axis.
    if not contexts:
        text = _judge(
            f"Question: {query}\n\nAnswer: {answer}\n\n"
            "Score from 0.00 to 1.00 how well the answer addresses the question. "
            "Reply with just the number.",
            max_tokens=60,
        )
        rel = None
        if text:
            m = re.search(_NUM, text)
            if m:
                rel = _clamp(m.group(1))
        return {"faithfulness": None, "answer_relevancy": rel, "context_precision": None}

    from config import SCORING_CONTEXT_CHARS
    ctx = "\n\n".join(
        f"[{i+1}] {c[:SCORING_CONTEXT_CHARS]}" for i, c in enumerate(contexts[:3])
    )

    if full:
        keys = ("faithfulness", "relevancy", "precision")
        rubric = (
            "faithfulness = the answer stays within what the passages say\n"
            "relevancy = the answer addresses the question\n"
            "precision = the passages are actually useful for this question\n"
        )
        fmt = "faithfulness=<n> relevancy=<n> precision=<n>"
    else:
        keys = ("faithfulness", "relevancy")
        rubric = (
            "faithfulness = the answer stays within what the passages say\n"
            "relevancy = the answer addresses the question\n"
        )
        fmt = "faithfulness=<n> relevancy=<n>"

    text = _judge(
        f"Question: {query}\n\nAnswer: {answer}\n\nContext passages:\n{ctx}\n\n"
        f"Score each from 0.00 to 1.00:\n{rubric}\n"
        f"Reply on one line, nothing else:\n{fmt}",
        max_tokens=200,
    )
    parsed = _parse(text or "", keys)
    return {
        "faithfulness": parsed.get("faithfulness"),
        "answer_relevancy": parsed.get("relevancy"),
        "context_precision": parsed.get("precision"),
    }


def scoring_available() -> bool:
    return _check_available()
