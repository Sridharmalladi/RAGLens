"""
Model loading and generation.
One model loaded at a time — swapped on dropdown change.
PEFT adapter applied on top of FT_BASE_MODEL for configs 3 & 4.
"""

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from config import (
    AVAILABLE_MODELS, FT_BASE_MODEL, ADAPTER_PATH,
    MAX_NEW_TOKENS, TEMPERATURE, GENERATION_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)

# Cache: {model_id: (model, tokenizer)}
_cache: dict[str, tuple] = {}
_cache_lock = threading.Lock()  # prevents double-load when preload + user query race
_ft_model_cache: tuple | None = None  # (peft_model, tokenizer) for FT configs
_executor = ThreadPoolExecutor(max_workers=1)


def _load_base(model_id: str) -> tuple:
    """Load a base model + tokenizer in bfloat16 (halves RAM vs float32)."""
    logger.info("Loading model %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()
    logger.info("Model %s loaded (%.1f GB)", model_id, _param_gb(model))
    return model, tokenizer


def _param_gb(model) -> float:
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9


def _load_ft_model() -> tuple | None:
    """Load FT_BASE_MODEL with QLoRA adapter. Returns None if adapter not found."""
    global _ft_model_cache
    if _ft_model_cache is not None:
        return _ft_model_cache

    if not os.path.exists(ADAPTER_PATH):
        logger.warning(
            "QLoRA adapter not found at %s — configs 3 & 4 will use base model. "
            "Train the adapter and commit it to enable fine-tuned configs.",
            ADAPTER_PATH,
        )
        return None

    try:
        from peft import PeftModel

        base_model, tokenizer = get_model(FT_BASE_MODEL)
        peft_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        peft_model.eval()
        _ft_model_cache = (peft_model, tokenizer)
        logger.info("QLoRA adapter loaded from %s", ADAPTER_PATH)
        return _ft_model_cache
    except Exception as e:
        logger.error("Failed to load QLoRA adapter: %s", e)
        return None


def get_model(model_id: str) -> tuple:
    """Return (model, tokenizer), loading and caching on first call."""
    if model_id in _cache:
        return _cache[model_id]
    with _cache_lock:
        if model_id not in _cache:  # re-check after acquiring lock
            if len(_cache) >= 2:
                oldest = next(iter(_cache))
                del _cache[oldest]
                logger.info("Evicted %s from model cache", oldest)
            _cache[model_id] = _load_base(model_id)
    return _cache[model_id]


def _build_prompt(query: str, context: str | None, model_id: str) -> str:
    """Build a chat-style prompt. Each model family has its own template."""
    system = "You are a helpful research assistant. Answer concisely based on the provided context."
    if context:
        user_msg = f"Context:\n{context}\n\nQuestion: {query}"
    else:
        user_msg = f"Question: {query}\n\nAnswer based on your training knowledge."

    # Gemma uses a different chat format
    if "gemma" in model_id.lower():
        return f"<start_of_turn>user\n{user_msg}<end_of_turn>\n<start_of_turn>model\n"

    # Most HF instruct models follow the standard chat template
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user_msg}]

    try:
        tokenizer = get_model(model_id)[1]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Fallback if model doesn't have a chat template
        return f"<|system|>{system}</s><|user|>{user_msg}</s><|assistant|>"


def _generate_inner(model, tokenizer, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode only the generated tokens (not the prompt)
    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def generate(
    model_id: str,
    query: str,
    context: str | None = None,
    use_adapter: bool = False,
) -> tuple[str, float]:
    """
    Generate an answer. Returns (answer, latency_seconds).
    use_adapter=True loads the QLoRA adapter for FT configs.
    Falls back to base model if adapter is unavailable.
    """
    prompt = _build_prompt(query, context, model_id if not use_adapter else FT_BASE_MODEL)

    if use_adapter:
        ft = _load_ft_model()
        if ft:
            model, tokenizer = ft
        else:
            # Adapter unavailable — fall back to base model, note it in the answer
            model, tokenizer = get_model(FT_BASE_MODEL)
    else:
        model, tokenizer = get_model(model_id)

    start = time.perf_counter()
    try:
        future = _executor.submit(_generate_inner, model, tokenizer, prompt)
        answer = future.result(timeout=GENERATION_TIMEOUT_SECONDS)
    except FuturesTimeout:
        return "[Generation timed out — try a shorter query or a smaller model]", GENERATION_TIMEOUT_SECONDS
    except Exception as e:
        logger.error("Generation error for %s: %s", model_id, e)
        return f"[Generation failed: {e}]", 0.0

    latency = time.perf_counter() - start
    return answer, latency


def adapter_available() -> bool:
    return os.path.exists(ADAPTER_PATH)
