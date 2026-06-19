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
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from src.storage import init_db
    from src.scheduler import start as start_scheduler

    init_db()
    start_scheduler()
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


@app.post("/api/compare")
async def compare(request: QueryRequest):
    """
    Stream 4 RAG config results as Server-Sent Events.
    Each event carries one completed config result (with scores).
    """
    from src.corpus import is_ready

    query = request.query.strip()

    if not query:
        async def _err():
            yield 'data: {"error": "Empty query"}\n\n'
        return StreamingResponse(_err(), media_type="text/event-stream")

    if not is_ready():
        async def _err():
            yield 'data: {"error": "Corpus not ready — upload corpus/index.faiss via HF Files tab"}\n\n'
        return StreamingResponse(_err(), media_type="text/event-stream")

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _run_sync():
        from src.inference import run_all_configs
        from src.evaluation import score as eval_score

        for result in run_all_configs(query):
            if not result.get("error") and result.get("answer"):
                try:
                    result["scores"] = eval_score(
                        query, result["answer"], result.get("context_chunks", [])
                    )
                except Exception as e:
                    logger.warning("Scoring failed: %s", e)
                    result["scores"] = {}
            else:
                result["scores"] = {}
            asyncio.run_coroutine_threadsafe(queue.put(result), loop)

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
    from src.storage import read_recent, detect_drift, read_last_run_time
    from src.scheduler import next_run_time
    from config import DRIFT_ALERT_THRESHOLD

    rows = read_recent(days=7)
    alerts = detect_drift(threshold=DRIFT_ALERT_THRESHOLD)

    return {
        "rows": rows,
        "alerts": alerts,
        "last_run": read_last_run_time(),
        "next_run": next_run_time(),
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
