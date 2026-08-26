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
MAX_NEW_TOKENS = 600

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

# Paths
FAISS_INDEX_PATH = "corpus/index.faiss"
CHUNKS_PATH = "corpus/processed/chunks.json"
EMBEDDINGS_PATH = "corpus/embeddings.json"
DB_PATH = os.environ.get("DB_PATH", "raglens.db")

# Evaluation judge (same OpenRouter key, different role)
JUDGE_MODEL = "openai/gpt-oss-20b"

# ── Monitoring ─────────────────────────────────────────────────────────────
MONITORING_INTERVAL_HOURS = 6
RETENTION_DAYS = 30
DRIFT_ALERT_THRESHOLD = 0.10
MONITORING_MAX_TOKENS = 500  # shorter answers during scheduled eval to conserve credit budget
SCORING_CONTEXT_CHARS = 300  # truncate each chunk before sending to judge to cut input tokens

# Every scheduled cycle runs each query through all 4 configs for each of these
# models, so the performance charts can be compared across models over time.
# Kept small and cheap since it runs unattended every few hours.
MONITORING_MODELS = ["openai/gpt-oss-20b", "google/gemini-2.5-flash-lite"]

MONITORING_QUERIES = [
    "What are the main differences between dense and sparse retrieval?",
    "How does reranking improve RAG performance?",
    "What is QLoRA and when should you use it?",
]

# Pool the UI samples from. It shuffles this list and shows a handful at a time,
# reshuffling after each run so fresh prompts keep surfacing.
SUGGESTED_QUERIES = [
    "Why do large language models hallucinate, and can retrieval fix it?",
    "What is the difference between fine-tuning and RAG for teaching a model new facts?",
    "How do you use an LLM to judge another LLM's answers?",
    "When does hybrid retrieval beat pure dense retrieval?",
    "What does a cross-encoder reranker do that embedding similarity cannot?",
    "How is chunk size chosen, and what breaks when it is too large or too small?",
    "What is faithfulness in RAG evaluation and how is it measured without ground truth?",
    "Why can adding more retrieved context make an answer worse?",
    "How does BM25 score a document against a query?",
    "What causes retrieval to miss a relevant passage even when it exists in the corpus?",
    "How do you evaluate a RAG system when you have no labelled answers?",
    "What is the role of embedding normalisation in dense retrieval?",
    "How does context precision differ from context recall?",
    "When is reranking not worth its added latency?",
    "What are common failure modes of LLM-as-judge scoring?",
    "How does query rewriting help multi-hop retrieval questions?",
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
