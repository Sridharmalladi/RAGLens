"""
OpenRouter chat-completion adapter.

OpenRouter exposes an OpenAI-compatible API, so this wraps the `openai`
SDK pointed at the OpenRouter base URL. No local model weights are loaded.
Both answer generation and LLM-as-judge scoring go through here.

Environment:
    OPENROUTER_API_KEY   required
    OPENROUTER_APP_NAME  optional, shown in OpenRouter dashboard rankings
    OPENROUTER_APP_URL   optional, referer sent with each request
"""

import logging
import os
import re
import time

logger = logging.getLogger(__name__)

BASE_URL = "https://openrouter.ai/api/v1"
API_KEY_ENV = "OPENROUTER_API_KEY"

_RETRY_AFTER_RE = re.compile(r"(?:try again in|retry after)\s+(\d+\.?\d*)\s*s", re.IGNORECASE)


def api_key_present() -> bool:
    return bool(os.environ.get(API_KEY_ENV))


def retry_wait_seconds(exc) -> float | None:
    """Parse a retry-after hint from a rate-limit error, or return None."""
    match = _RETRY_AFTER_RE.search(str(exc))
    return float(match.group(1)) + 1.0 if match else None


def _client(timeout: float):
    from openai import OpenAI

    key = os.environ.get(API_KEY_ENV)
    if not key:
        raise RuntimeError(f"{API_KEY_ENV} not set")

    headers = {"X-Title": os.environ.get("OPENROUTER_APP_NAME", "RAGLens")}
    referer = os.environ.get("OPENROUTER_APP_URL")
    if referer:
        headers["HTTP-Referer"] = referer

    return OpenAI(
        base_url=BASE_URL,
        api_key=key,
        max_retries=0,
        timeout=timeout,
        default_headers=headers,
    )


def complete(
    messages: list[dict],
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 600,
    timeout: float = 30.0,
    retries: int = 1,
    reasoning_effort: str | None = "low",
) -> str:
    """
    Send a chat completion and return the message text.

    Retries on a rate-limit error up to `retries` times, honouring any
    retry-after hint in the error. Other errors propagate to the caller
    so it can decide how to report them.

    `reasoning_effort` is passed through to OpenRouter for reasoning models
    such as gpt-oss. Keeping it low leaves room in the token budget for the
    visible answer and keeps latency down. Set it to None to omit the field.
    """
    from openai import RateLimitError

    client = _client(timeout)
    extra_body = {"reasoning": {"effort": reasoning_effort}} if reasoning_effort else {}

    attempt = 0
    while True:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body,
            )
            return (resp.choices[0].message.content or "").strip()
        except RateLimitError as exc:
            attempt += 1
            if attempt > retries:
                raise
            wait = retry_wait_seconds(exc) or 20.0
            logger.warning("OpenRouter rate limited, waiting %.1fs before retry", wait)
            time.sleep(wait)
