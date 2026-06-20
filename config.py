import os

# Generation — Groq API (no local model loading, sub-second responses)
GROQ_GENERATION_MODEL = "llama-3.1-8b-instant"
MAX_NEW_TOKENS = 300

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

# Evaluation judge (same Groq key, different role)
JUDGE_MODEL = "llama-3.1-8b-instant"
JUDGE_PROVIDER = "groq"

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
    "How does hybrid retrieval combine dense and sparse search?",
    "What evaluation metrics matter most for production RAG?",
    "When does fine-tuning beat better retrieval?",
]

# The 4 configs show progressively better retrieval strategies
CONFIG_NAMES = {
    1: "No RAG",
    2: "Dense RAG",
    3: "Hybrid RAG",
    4: "Hybrid + Rerank",
}

CONFIG_DESCRIPTIONS = {
    1: "Baseline — model answers from training knowledge only",
    2: "FAISS dense retrieval — BGE-small embeddings, top-3 chunks",
    3: "Hybrid retrieval — dense + BM25 sparse, top-3 chunks",
    4: "Hybrid + cross-encoder reranking — BGE-reranker, top-2 reranked",
}

CONFIG_COLORS = {
    1: "#6B7280",
    2: "#60A5FA",
    3: "#FBBF24",
    4: "#34D399",
}

APP_TITLE = "RAGLens — Live RAG Benchmarking"
CORPUS_DESCRIPTION = "50 arXiv papers on RAG & LLM evaluation (1,665 chunks)"
