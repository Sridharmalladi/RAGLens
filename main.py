"""
RAGLens FastAPI server.
Serves the static frontend and streams SSE results from the /api/compare endpoint.
"""

import asyncio
import json
import logging
import threading
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import (
    APP_TITLE,
    CORPUS_DESCRIPTION,
    AVAILABLE_MODELS,
    MODEL_IDS,
    MODEL_LABELS,
    DEFAULT_MODEL,
    MAX_COMPARE_MODELS,
    SUGGESTED_QUERIES,
    CONFIG_COLORS,
    CONFIG_NAMES,
    DRIFT_ALERT_THRESHOLD,
    CTX_PREVIEW_CHARS,
    CTX_MAX_CHUNKS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
# Silence noisy third-party loggers that flood the output with HTTP metadata pings
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub.utils._http").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

_ALLOWED_MODELS = set(MODEL_IDS)


def _warmup_models():
    """Build FAISS index + load BGE models into memory at startup.
    Runs in a daemon thread so the server is responsive immediately,
    but all heavy work is done before the first user query arrives.
    """
    try:
        # Trigger FAISS index build (or load from disk if already built)
        from src.corpus import get_index, get_chunks
        get_chunks()
        get_index()
        # Load embedding + reranker models into memory
        from src.retrieval import _get_embedder, _get_reranker
        _get_embedder()
        _get_reranker()
        logger.info("Warmup complete, corpus and models ready")
    except Exception as e:
        logger.warning("Warmup failed (will init on first request): %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    import os
    from src.storage import init_db
    from src.scheduler import start as start_scheduler

    init_db()

    # The background evaluation job only makes sense where one instance stays
    # alive between its runs. On a scale-to-zero host (Cloud Run) each cold start
    # would re-trigger a full cycle against a fresh empty DB and burn tokens, so
    # set ENABLE_SCHEDULER=0 there and rely on the /api/monitoring history that
    # a longer-lived deployment builds up.
    if os.environ.get("ENABLE_SCHEDULER", "1") != "0":
        start_scheduler()
    else:
        logger.info("Scheduler disabled (ENABLE_SCHEDULER=0)")

    threading.Thread(target=_warmup_models, daemon=True).start()
    logger.info("RAGLens started")
    yield
    logger.info("RAGLens shutting down")


app = FastAPI(title="RAGLens API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    models: list[str] | None = None    # one comparison per model, in this order
    model: str | None = None           # legacy single-model field, still accepted


def _resolve_models(request: QueryRequest) -> list[str]:
    """Pick the models to run: requested list, then legacy field, then default.
    Filtered to known slugs, de-duplicated in order, capped."""
    requested = list(request.models or [])
    if request.model:
        requested.append(request.model)

    seen: list[str] = []
    for m in requested:
        if m in _ALLOWED_MODELS and m not in seen:
            seen.append(m)
    if not seen:
        seen = [DEFAULT_MODEL]
    return seen[:MAX_COMPARE_MODELS]


async def _sse_error(msg: str):
    yield f'data: {json.dumps({"error": msg})}\n\n'


@app.get("/api/config")
async def app_config():
    """Everything the frontend needs to render its controls."""
    return {
        "title": APP_TITLE,
        "corpus": CORPUS_DESCRIPTION,
        "models": AVAILABLE_MODELS,
        "default_model": DEFAULT_MODEL,
        "max_models": MAX_COMPARE_MODELS,
        "suggestions": SUGGESTED_QUERIES,
    }


@app.post("/api/compare")
async def compare(request: QueryRequest):
    """Stream 4 RAG config results per selected model as Server-Sent Events,
    then stream a score event for each answer."""
    from src.corpus import is_ready

    query = request.query.strip()
    models = _resolve_models(request)

    if not query:
        return StreamingResponse(_sse_error("Empty query"), media_type="text/event-stream")

    if not is_ready():
        return StreamingResponse(_sse_error("Corpus not ready"), media_type="text/event-stream")

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _put(item):
        asyncio.run_coroutine_threadsafe(queue.put(item), loop)

    def _run_sync():
        from src.inference import run_all_configs
        from src.evaluation import score as eval_score, scoring_available

        can_score = scoring_available()

        def _trim(chunks):
            return [
                {**c, "text": (c.get("text") or "")[:CTX_PREVIEW_CHARS]}
                for c in (chunks or [])[:CTX_MAX_CHUNKS]
            ]

        def _worker(model: str):
            # One model at a time internally (rate-friendly), but models run
            # in parallel so a slow model does not hold up the others.
            pending = []  # (result, context_texts)
            for result in run_all_configs(query, model=model):
                chunks = result.get("chunks") or []
                context_texts = [c.get("text", "") for c in chunks]
                payload = {k: v for k, v in result.items() if k not in ("chunks", "rerank_pool")}
                payload["chunks"] = _trim(chunks)
                payload["rerank_pool"] = _trim(result.get("rerank_pool"))
                payload["model"] = model
                payload["scores"] = {}
                _put(payload)
                pending.append((result, context_texts))

            if not can_score:
                return
            for result, context_texts in pending:
                answer = result.get("answer") or ""
                if answer and not answer.startswith("["):
                    _put({
                        "type": "score",
                        "model": model,
                        "config_id": result["config_id"],
                        "scores": eval_score(query, answer, context_texts, explain=True),
                    })

        workers = [threading.Thread(target=_worker, args=(m,), daemon=True) for m in models]
        for w in workers:
            w.start()
        for w in workers:
            w.join()
        _put(None)

    threading.Thread(target=_run_sync, daemon=True).start()

    async def _stream():
        # Tell the client which models are running, and in what order
        yield f'data: {json.dumps({"type": "start", "models": models})}\n\n'
        while True:
            result = await asyncio.wait_for(queue.get(), timeout=300)
            if result is None:
                break
            result.pop("context_chunks", None)  # keep the payload small
            yield f"data: {json.dumps(result)}\n\n"

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/monitoring")
async def monitoring():
    from collections import defaultdict
    from src.storage import read_recent, detect_drift, read_last_run_time
    from src.scheduler import next_run_time

    rows = read_recent(days=7)
    alerts = detect_drift(threshold=DRIFT_ALERT_THRESHOLD)

    # One averaged point per (model, config, hour-slot)
    groups: dict[tuple, list] = defaultdict(list)
    for row in rows:
        hour_slot = row["timestamp"][:13]  # "YYYY-MM-DDTHH"
        groups[(row["model"], row["config_id"], hour_slot)].append(row)

    series_map: dict[tuple, dict] = {}
    for (model, config_id, hour_slot), grp in groups.items():
        key = (model, config_id)
        if key not in series_map:
            series_map[key] = {
                "model": model,
                "model_label": MODEL_LABELS.get(model, model),
                "config_id": config_id,
                "config_name": CONFIG_NAMES.get(config_id, f"Config {config_id}"),
                "color": CONFIG_COLORS.get(config_id, "#818CF8"),
                "points": [],
            }
        faiths = [r["faithfulness"] for r in grp if r.get("faithfulness") is not None]
        rels   = [r["answer_relevancy"] for r in grp if r.get("answer_relevancy") is not None]
        precs  = [r["context_precision"] for r in grp if r.get("context_precision") is not None]
        lats   = [r["latency_s"] for r in grp]
        series_map[key]["points"].append({
            "ts":                hour_slot.replace("T", " ") + ":00",
            "faithfulness":      round(sum(faiths) / len(faiths), 4) if faiths else None,
            "answer_relevancy":  round(sum(rels)   / len(rels),   4) if rels   else None,
            "context_precision": round(sum(precs)  / len(precs),  4) if precs  else None,
            "latency":           round(sum(lats)   / len(lats),   3) if lats   else None,
        })

    for s in series_map.values():
        s["points"].sort(key=lambda p: p["ts"])

    models_present = sorted(
        {(s["model"], s["model_label"]) for s in series_map.values()},
        key=lambda t: t[1],
    )

    return {
        "series": sorted(series_map.values(), key=lambda s: (s["model_label"], s["config_id"])),
        "models": [{"id": mid, "label": label} for mid, label in models_present],
        "alerts": alerts,
        "last_run": read_last_run_time(),
        "next_run": next_run_time(),
        "has_data": bool(rows),
    }


@app.get("/api/health")
async def health():
    from src.corpus import is_ready
    return {"status": "ok", "corpus_ready": is_ready()}


# ---------------------------------------------------------------------------
# Static frontend — must be last so API routes take priority
# ---------------------------------------------------------------------------
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
