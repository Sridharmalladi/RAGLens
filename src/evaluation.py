"""
RAGAS scoring via Groq as the judge LLM.
Returns faithfulness, answer_relevancy, context_precision.
If GROQ_API_KEY is missing or the call fails, returns None scores — never fake zeros.
"""

import logging
import os

logger = logging.getLogger(__name__)

_ragas_ready: bool | None = None  # None = unchecked


def _check_ragas_available() -> bool:
    global _ragas_ready
    if _ragas_ready is not None:
        return _ragas_ready

    if not os.environ.get("GROQ_API_KEY"):
        logger.warning("GROQ_API_KEY not set — RAGAS scoring disabled")
        _ragas_ready = False
        return False

    try:
        import ragas  # noqa: F401
        from langchain_groq import ChatGroq  # noqa: F401
        _ragas_ready = True
    except ImportError as e:
        logger.warning("RAGAS dependencies not installed: %s — scoring disabled", e)
        _ragas_ready = False

    return _ragas_ready


def _build_ragas_llm():
    from langchain_groq import ChatGroq
    from ragas.llms import LangchainLLMWrapper
    from config import JUDGE_MODEL

    groq_llm = ChatGroq(
        model=JUDGE_MODEL,
        api_key=os.environ["GROQ_API_KEY"],
        temperature=0,
    )
    return LangchainLLMWrapper(groq_llm)


def _build_ragas_embeddings():
    from ragas.embeddings import HuggingfaceEmbeddings
    from config import EMBEDDING_MODEL
    return HuggingfaceEmbeddings(model_name=EMBEDDING_MODEL)


def score(
    query: str,
    answer: str,
    contexts: list[str],
    ground_truth: str | None = None,
) -> dict:
    """
    Score a single RAG result with RAGAS.
    Returns {faithfulness, answer_relevancy, context_precision} as floats in [0, 1],
    or {faithfulness: None, ...} if scoring is unavailable.
    """
    null_scores = {"faithfulness": None, "answer_relevancy": None, "context_precision": None}

    if not _check_ragas_available():
        return null_scores

    if not answer or not contexts:
        return null_scores

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision

        ragas_llm = _build_ragas_llm()
        ragas_emb = _build_ragas_embeddings()

        # Wire judge LLM and embeddings into each metric
        faithfulness.llm = ragas_llm
        answer_relevancy.llm = ragas_llm
        answer_relevancy.embeddings = ragas_emb
        context_precision.llm = ragas_llm

        data: dict = {
            "question": [query],
            "answer": [answer],
            "contexts": [contexts],
        }
        if ground_truth:
            data["ground_truth"] = [ground_truth]

        dataset = Dataset.from_dict(data)
        metrics = [faithfulness, answer_relevancy, context_precision]
        result = evaluate(dataset, metrics=metrics)

        return {
            "faithfulness": _safe_float(result.get("faithfulness")),
            "answer_relevancy": _safe_float(result.get("answer_relevancy")),
            "context_precision": _safe_float(result.get("context_precision")),
        }

    except Exception as e:
        logger.error("RAGAS scoring failed: %s", e)
        return null_scores


def _safe_float(value) -> float | None:
    try:
        v = float(value)
        return round(v, 4) if 0.0 <= v <= 1.0 else None
    except (TypeError, ValueError):
        return None


def scoring_available() -> bool:
    return _check_ragas_available()
