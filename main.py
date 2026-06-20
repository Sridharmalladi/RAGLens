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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
# Silence noisy third-party loggers that flood the output with HTTP metadata pings
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub.utils._http").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


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
        logger.info("Warmup complete — corpus + models ready")
    except Exception as e:
        logger.warning("Warmup failed (will init on first request): %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from src.storage import init_db
    from src.scheduler import start as start_scheduler

    init_db()
    start_scheduler()
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

ALLOWED_MODELS = {
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "compound-beta",
}


class QueryRequest(BaseModel):
    query: str
    model: str | None = None  # falls back to GROQ_GENERATION_MODEL if omitted


async def _sse_error(msg: str):
    yield f'data: {json.dumps({"error": msg})}\n\n'


@app.post("/api/compare")
async def compare(request: QueryRequest):
    """Stream 4 RAG config results as Server-Sent Events, then score events."""
    from src.corpus import is_ready

    query = request.query.strip()
    model = request.model if request.model in ALLOWED_MODELS else None

    if not query:
        return StreamingResponse(_sse_error("Empty query"), media_type="text/event-stream")

    if not is_ready():
        return StreamingResponse(_sse_error("Corpus not ready"), media_type="text/event-stream")

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _run_sync():
        from src.inference import run_all_configs
        from src.evaluation import score as eval_score, scoring_available

        # Phase 1 — stream answers as they complete
        all_results = []
        for result in run_all_configs(query, model=model):
            contexts = result.get("context_chunks") or []
            payload = {k: v for k, v in result.items() if k != "context_chunks"}
            payload["scores"] = {}
            asyncio.run_coroutine_threadsafe(queue.put(payload), loop)
            all_results.append((result, contexts))

        # Phase 2 — score each config and stream score events
        if scoring_available():
            for result, contexts in all_results:
                answer = result.get("answer") or ""
                if answer and not answer.startswith("["):
                    s = eval_score(query, answer, contexts)
                    asyncio.run_coroutine_threadsafe(
                        queue.put({
                            "type": "score",
                            "config_id": result["config_id"],
                            "scores": s,
                        }),
                        loop,
                    )

        asyncio.run_coroutine_threadsafe(queue.put(None), loop)

    threading.Thread(target=_run_sync, daemon=True).start()

    async def _stream():
        while True:
            result = await asyncio.wait_for(queue.get(), timeout=300)
            if result is None:
                break
            # Don't send full chunk text to frontend — keep payload small
            result.pop("context_chunks", None)
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
    from config import DRIFT_ALERT_THRESHOLD, CONFIG_COLORS, CONFIG_NAMES

    rows = read_recent(days=7)
    alerts = detect_drift(threshold=DRIFT_ALERT_THRESHOLD)

    # Group rows by (config_id, hour-slot) so each monitoring run → one averaged point
    groups: dict[tuple, list] = defaultdict(list)
    for row in rows:
        hour_slot = row["timestamp"][:13]  # "YYYY-MM-DDTHH"
        groups[(row["config_id"], hour_slot)].append(row)

    # Build per-config series sorted by time
    config_map: dict[int, dict] = {}
    for (config_id, hour_slot), grp in groups.items():
        if config_id not in config_map:
            config_map[config_id] = {
                "config_id": config_id,
                "config_name": CONFIG_NAMES.get(config_id, f"Config {config_id}"),
                "color": CONFIG_COLORS.get(config_id, "#818CF8"),
                "points": [],
            }
        faiths = [r["faithfulness"] for r in grp if r.get("faithfulness") is not None]
        rels   = [r["answer_relevancy"] for r in grp if r.get("answer_relevancy") is not None]
        precs  = [r["context_precision"] for r in grp if r.get("context_precision") is not None]
        lats   = [r["latency_s"] for r in grp]
        config_map[config_id]["points"].append({
            "ts":                hour_slot.replace("T", " ") + ":00",
            "faithfulness":      round(sum(faiths) / len(faiths), 4) if faiths else None,
            "answer_relevancy":  round(sum(rels)   / len(rels),   4) if rels   else None,
            "context_precision": round(sum(precs)  / len(precs),  4) if precs  else None,
            "latency":           round(sum(lats)   / len(lats),   3) if lats   else None,
        })

    for s in config_map.values():
        s["points"].sort(key=lambda p: p["ts"])

    return {
        "series": sorted(config_map.values(), key=lambda s: s["config_id"]),
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
# Gradio — satisfies the Hugging Face Spaces `sdk: gradio` requirement.
# The actual UI is our custom static HTML; Gradio is mounted at /gradio and
# its launch() call starts the uvicorn server that serves everything below.
# ---------------------------------------------------------------------------
import gradio as gr

_demo = gr.Blocks(title="RAGLens")
with _demo:
    pass

gr.mount_gradio_app(app, _demo, path="/gradio")

# ---------------------------------------------------------------------------
# Static frontend — must be last so API routes take priority
# ---------------------------------------------------------------------------
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    _demo.launch(server_name="0.0.0.0", server_port=7860)
