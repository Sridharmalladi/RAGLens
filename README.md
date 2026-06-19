---
title: RAGLens
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# RAGLens — Live RAG Benchmarking

Compare 4 retrieval strategies side-by-side with live LLM-as-judge scoring and a 7-day monitoring dashboard.

## The 4 Configurations

| Config | Retrieval | What it shows |
|---|---|---|
| No RAG | None | Baseline — pure model knowledge |
| Dense RAG | BGE + FAISS | Dense retrieval contribution |
| Hybrid RAG | Dense + BM25 | Sparse signal on top |
| Hybrid + Rerank | Dense + BM25 + BGE cross-encoder | Best-of-all-worlds |

## Stack

- **Generation:** Groq llama-3.1-8b-instant (free tier, sub-second)
- **Embeddings:** BGE-small-en-v1.5 · **Reranker:** BGE-reranker-base
- **Retrieval:** FAISS flat L2 + BM25 (rank-bm25)
- **Evaluation:** LLM-as-judge via Groq (faithfulness, relevancy, precision)
- **Storage:** SQLite · **Scheduler:** APScheduler hourly cron
- **Hosting:** HF Spaces Docker, free tier

## Local Development

```bash
git clone <repo>
cd RAGLens
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add GROQ_API_KEY
uvicorn main:app --reload --port 7860
```

Open http://localhost:7860

> Built by **Sridhar Malladi** · Architecture over compute.
