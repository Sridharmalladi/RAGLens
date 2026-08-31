import os

# ── Generation via OpenRouter (OpenAI-compatible API, no local weights) ──────
# Models the UI offers and the monitoring job exercises. `id` is the OpenRouter
# slug; `label` is what the UI shows; `tag` is a short hint next to the label.
AVAILABLE_MODELS = [
    {"id": "openai/gpt-oss-20b",                        "label": "gpt-oss-20b",           "tag": "fast"},
    {"id": "openai/gpt-oss-120b",                       "label": "gpt-oss-120b",          "tag": "capable"},
    {"id": "google/gemini-2.5-flash-lite",              "label": "gemini-2.5-flash-lite", "tag": "fast"},
    {"id": "mistralai/mistral-small-24b-instruct-2501", "label": "mistral-small-24b",     "tag": "lean"},
    {"id": "meta-llama/llama-3.3-70b-instruct",         "label": "llama-3.3-70b",         "tag": "open"},
    {"id": "anthropic/claude-haiku-4.5",                "label": "claude-haiku-4.5",      "tag": "premium"},
]
DEFAULT_MODEL = "openai/gpt-oss-20b"
MAX_COMPARE_MODELS = 3  # cap on how many models one live comparison runs at once

OPENROUTER_GENERATION_MODEL = DEFAULT_MODEL  # fallback when a request names no model

# Token budgets. User-facing runs get room for a full answer; the scheduled job
# stays terse because it only needs a stable signal for the trend charts.
MAX_NEW_TOKENS = 500
CONTEXT_CHARS_PER_CHUNK = 1100  # trim each retrieved chunk before it goes to the model

# Retrieved-context inspector. How much of each chunk, and how many chunks, the
# /api/compare payload carries back to the browser so the UI can show what
# retrieval actually fed the model. Kept small so the SSE frames stay light.
CTX_PREVIEW_CHARS = 600
CTX_MAX_CHUNKS = 6

MODEL_IDS = [m["id"] for m in AVAILABLE_MODELS]
MODEL_LABELS = {m["id"]: m["label"] for m in AVAILABLE_MODELS}

# Embeddings & retrieval (local, small models ~400 MB total)
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K = 3
RERANK_TOP_N = 2
HYBRID_ALPHA = 0.5

# ── Retrieval backend ──────────────────────────────────────────────────────
# "local"  : sentence-transformers + FAISS in-process (needs requirements-full.txt,
#            ~4 GB with torch; used by the Docker image / Hugging Face / a dev box).
# "hosted" : query embedding + cross-encoder rerank via the Hugging Face Inference
#            API, dense search is a numpy cosine over the committed corpus vectors,
#            no torch / faiss (fits Vercel's 250 MB function and Render's 512 MB).
# The corpus vectors in corpus/embeddings.json are BGE-small, and the hosted path
# calls the same BGE-small model, so the two live in one space — no re-embedding.
def _default_retrieval_backend() -> str:
    override = os.environ.get("RETRIEVAL_BACKEND")
    if override:
        return override.strip().lower()
    try:
        import sentence_transformers  # noqa: F401
        return "local"
    except Exception:
        return "hosted"

RETRIEVAL_BACKEND = _default_retrieval_backend()

# Hugging Face Inference API (used only when RETRIEVAL_BACKEND == "hosted").
HF_API_TOKEN = os.environ.get("HF_API_TOKEN") or os.environ.get("HF_TOKEN")
HF_INFERENCE_BASE = os.environ.get(
    "HF_INFERENCE_BASE", "https://router.huggingface.co/hf-inference/models"
)
HF_EMBED_MODEL = os.environ.get("HF_EMBED_MODEL", EMBEDDING_MODEL)
HF_RERANK_MODEL = os.environ.get("HF_RERANK_MODEL", RERANKER_MODEL)

# Server-Sent Events stream the four config results as they land. Some hosts
# (Vercel's Python runtime) buffer the response body, which defeats the point;
# set STREAM_RESPONSES=0 there and /api/compare returns one JSON blob instead.
STREAM_RESPONSES = os.environ.get("STREAM_RESPONSES", "1").strip() not in ("0", "false", "no")

# Paths — anchored to this file so they resolve no matter the working directory
# (Vercel and some hosts do not run from the repo root).
_ROOT = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_PATH = os.path.join(_ROOT, "corpus", "index.faiss")
CHUNKS_PATH = os.path.join(_ROOT, "corpus", "processed", "chunks.json")
EMBEDDINGS_PATH = os.path.join(_ROOT, "corpus", "embeddings.json")
DB_PATH = os.environ.get("DB_PATH", os.path.join(_ROOT, "raglens.db"))

# Evaluation judge (same OpenRouter key, different role)
JUDGE_MODEL = "openai/gpt-oss-20b"

# ── Monitoring ─────────────────────────────────────────────────────────────
MONITORING_INTERVAL_HOURS = 8
RETENTION_DAYS = 30
DRIFT_ALERT_THRESHOLD = 0.10
MONITORING_MAX_TOKENS = 300   # terse answers during the scheduled job, trend only
SCORING_CONTEXT_CHARS = 350   # per-chunk cap for the judge's copy of the context;
                              # too low and every answer looks "unfaithful"

# Every scheduled cycle runs each probe query through all 4 configs for each of
# these models, so the charts stay comparable across models. Deliberately small:
# 2 models x 2 queries x 4 configs, once every 8 hours.
# Both are OpenAI's own open-weight models: no shared-pool rate limits, and they
# answer sensibly at a tight token cap. Third-party slugs on OpenRouter's free
# provider pools (mistral-small, gemini-flash-lite) 429 too often for an
# unattended job, so they stay in the picker only.
MONITORING_MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]

# Kept on-corpus so faithfulness is a meaningful signal for the trend.
MONITORING_QUERIES = [
    "What are the main differences between dense and sparse retrieval?",
    "How does reranking improve RAG performance?",
]

# Pool the UI samples from. It shuffles this list and shows a handful at a time,
# reshuffling on every visit and after every run so fresh prompts keep surfacing.
# A mix on purpose: some land squarely on the paper corpus, most do not, which
# is the point. You get to watch retrieval help on some questions and add noise
# on others.
SUGGESTED_QUERIES = [
    # everyday / general knowledge
    "Why is the sky blue during the day and red at sunset?",
    "How do noise-cancelling headphones actually cancel noise?",
    "What makes a sourdough starter rise, and why does it go flat?",
    "How does GPS on a phone figure out where you are?",
    "Why does coffee stop working for some people after a while?",
    "What happens in your body during the last mile of a marathon?",
    "How does a suspension bridge hold up under heavy traffic?",
    "Why does cutting an onion make your eyes water?",
    "How do vaccines teach the immune system without causing the disease?",
    "What causes the northern lights, and why are they green?",
    "How does a microwave heat food from the inside out?",
    "Why do we forget most of our dreams within minutes of waking?",
    "How does end-to-end encryption keep a message private?",
    "What makes some bridges hum or sway in the wind?",
    "Why does ice float instead of sinking like most solids?",
    "How do bees tell each other where the flowers are?",
    "What is compound interest, and why does starting early matter so much?",
    "How does a jet engine stay lit at cruising altitude?",
    "Why does the moon look bigger near the horizon?",
    "How do touchscreens know the difference between a finger and a knuckle?",
    # squarely on the paper corpus
    "Why do large language models hallucinate, and can retrieval fix it?",
    "When does hybrid retrieval beat plain vector search?",
    "What does a reranker do that embedding similarity cannot?",
    "How do you grade a RAG answer when you have no reference answer?",
    "Why can adding more retrieved context make an answer worse?",
    "When is reranking not worth the extra latency?",
]

# The 4 configs show progressively better retrieval strategies
CONFIG_NAMES = {
    1: "No RAG",
    2: "Dense RAG",
    3: "Hybrid RAG",
    4: "Hybrid + Rerank",
}

CONFIG_DESCRIPTIONS = {
    1: "Baseline. The model answers from training knowledge only.",
    2: "FAISS dense retrieval with BGE-small embeddings, top 3 chunks.",
    3: "Hybrid retrieval, dense plus BM25 sparse, top 3 chunks.",
    4: "Hybrid retrieval plus cross-encoder reranking, top 2 reranked.",
}

CONFIG_COLORS = {
    1: "#6B7280",
    2: "#60A5FA",
    3: "#FBBF24",
    4: "#34D399",
}

APP_TITLE = "RAGLens · Live RAG Benchmarking"
CORPUS_DESCRIPTION = "50 arXiv papers on RAG and LLM evaluation (1,665 chunks)"

# Demo backfill for the trend charts. On a host with no persistent disk the real
# scheduled job only ever lands one point before the DB is wiped, so with this
# set the runs table is seeded with ~8 days of plausible history when empty.
# Seeded rows are marked and the UI labels them "sample data".
SEED_DEMO_DATA = os.environ.get("SEED_DEMO_DATA", "0").strip().lower() not in ("0", "false", "no", "")
DEMO_MARKER = "[sample data]"
