import os

# Models — all CPU-runnable, all ungated on HuggingFace
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
AVAILABLE_MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "google/gemma-2-2b-it",
]
# QLoRA adapter is trained on Phi-3.5-mini specifically
FT_BASE_MODEL = "microsoft/Phi-3.5-mini-instruct"
ADAPTER_PATH = "adapters/phi-3.5-mini-qlora-raglens"

# Embeddings & retrieval
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K = 5
RERANK_TOP_N = 3
HYBRID_ALPHA = 0.5

# Paths
CORPUS_RAW_DIR = "corpus/raw"
CORPUS_PROCESSED_DIR = "corpus/processed"
FAISS_INDEX_PATH = "corpus/index.faiss"
CHUNKS_PATH = "corpus/processed/chunks.json"
DB_PATH = os.environ.get("DB_PATH", "raglens.db")

# Generation — conservative for CPU inference
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
GENERATION_TIMEOUT_SECONDS = 600  # 10 min — covers cold-start model download on HF Spaces

# Evaluation — Groq free tier (14,400 req/day; monitoring uses ~288/day)
JUDGE_MODEL = "llama-3.1-8b-instant"
JUDGE_PROVIDER = "groq"  # requires GROQ_API_KEY env var

# Monitoring
MONITORING_INTERVAL_HOURS = 1
RETENTION_DAYS = 30
DRIFT_ALERT_THRESHOLD = 0.10

MONITORING_QUERIES = [
    "What are the main differences between dense and sparse retrieval?",
    "How does reranking improve RAG performance?",
    "What is QLoRA and when should you use it?",
]

SUGGESTED_QUERIES = [
    "Explain how hybrid retrieval combines dense and sparse search.",
    "What evaluation metrics matter most for production RAG?",
    "When does fine-tuning beat better retrieval?",
]

# Config metadata used in UI and monitoring
CONFIG_NAMES = {
    1: "Baseline",
    2: "Base + RAG",
    3: "FT + RAG",
    4: "FT + Optimal RAG",
}

CONFIG_DESCRIPTIONS = {
    1: "No retrieval — model answers from weights only",
    2: "Dense retrieval (BGE + FAISS) — top-5 chunks",
    3: "Fine-tuned model + dense retrieval",
    4: "Fine-tuned model + hybrid retrieval + reranking",
}

APP_TITLE = "RAGLens — Live RAG Benchmarking"
CORPUS_DESCRIPTION = "50 papers on RAG architectures and LLM evaluation"
