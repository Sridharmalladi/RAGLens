---
title: RAGLens
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

<div align="center">

# RAGLens

**Run a query. Watch 4 RAG strategies answer it simultaneously. See exactly which one wins — and why.**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Groq](https://img.shields.io/badge/Groq-Llama_3.1-F55036?style=flat-square)](https://groq.com)
[![FAISS](https://img.shields.io/badge/FAISS-Meta_AI-0467DF?style=flat-square)](https://github.com/facebookresearch/faiss)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

</div>

---

## What It Does

Most RAG tutorials show one pipeline. RAGLens shows **four** — running live on your query, scored automatically, charted over time.

```
Your query ──► No RAG          ──► answer  score
           ──► Dense RAG       ──► answer  score
           ──► Hybrid RAG      ──► answer  score
           ──► Hybrid + Rerank ──► answer  score
```

Each result arrives as it finishes (SSE streaming). Each answer is scored by an LLM judge on faithfulness, relevancy, and context precision — no ground truth required.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Browser (SSE)                            │
│          Query ──────────────────────► Cards render live        │
└───────────────────────────┬─────────────────────────────────────┘
                            │ POST /api/compare
┌───────────────────────────▼─────────────────────────────────────┐
│                      FastAPI (main.py)                          │
│   Thread pool ──► run_all_configs() ──► StreamingResponse       │
└──────┬──────────────────────────────────────────────────────────┘
       │
       ├──► Config 1: No RAG ──────────────────────────────────────────────┐
       │                                                                   │
       ├──► Config 2: Dense RAG                                            │
       │      │                                                            │
       │      └──► BGE-small ──► FAISS (L2) ──► top-3 chunks              │
       │                                                                   ▼
       ├──► Config 3: Hybrid RAG                                    Groq API
       │      │                                                   (Llama 3.1)
       │      ├──► BGE-small ──► FAISS ──┐                               │
       │      └──► BM25 (keyword) ───────┴──► α-blend ──► top-3          │
       │                                                                   │
       └──► Config 4: Hybrid + Rerank                                     │
              │                                                            │
              ├──► Hybrid ──► top-6 candidates                            │
              └──► BGE-reranker (cross-encoder) ──► top-2 ───────────────►┘
                                                                          │
                                                                          ▼
                                                              answer + latency
                                                                          │
                                                    ┌─────────────────────▼──────┐
                                                    │   LLM-as-Judge (Groq)      │
                                                    │   faithfulness  [0–1]      │
                                                    │   answer_relevancy [0–1]   │
                                                    │   context_precision [0–1]  │
                                                    └─────────────────────┬──────┘
                                                                          │
                                               ┌──────────────────────────▼──────┐
                                               │   SQLite + APScheduler          │
                                               │   Hourly eval · 7-day charts    │
                                               │   Drift alerts (Δ > 10%)        │
                                               └─────────────────────────────────┘
```

---

## The 4 Configs

| # | Strategy | Retrieval | When It Wins |
|---|----------|-----------|--------------|
| 1 | **No RAG** | — | Baseline. Shows what the model already knows. |
| 2 | **Dense RAG** | BGE-small + FAISS | Semantic queries where words don't match exactly. |
| 3 | **Hybrid RAG** | Dense + BM25 (α=0.5) | Technical terms + semantic concepts together. |
| 4 | **Hybrid + Rerank** | Hybrid → BGE cross-encoder | Highest precision. Best for production. |

---

## Evaluation Metrics

| Metric | Question It Answers | Config |
|--------|---------------------|--------|
| **Faithfulness** | Did the answer stay grounded in the retrieved context? | 2, 3, 4 |
| **Answer Relevancy** | Does the answer actually address the question? | 1, 2, 3, 4 |
| **Context Precision** | Was the retrieved context useful noise-free? | 2, 3, 4 |

Scored automatically by Groq (no labelled data needed).

---

## Stack

```
Generation    Groq  ·  llama-3.1-8b-instant  ·  sub-second responses
Embeddings    BAAI/bge-small-en-v1.5  ·  384-dim  ·  ~130 MB
Reranker      BAAI/bge-reranker-base  ·  cross-encoder
Index         FAISS IndexFlatL2  ·  exact NN  ·  1,665 chunks
Keyword       rank-bm25  ·  pure Python  ·  no external service
Backend       FastAPI  ·  SSE streaming  ·  SQLite
Scheduler     APScheduler  ·  hourly cron  ·  drift detection
Corpus        50 arXiv papers  ·  RAG & LLM evaluation
```

---

## Quickstart

```bash
git clone https://github.com/your-username/raglens
cd raglens

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # paste your free Groq key
uvicorn main:app --reload --port 7860
```

→ Open **http://localhost:7860**

> Get a free Groq key at [console.groq.com](https://console.groq.com) — no credit card required.

---

## Data Flow at Request Time

```
User types query
      │
      ▼
POST /api/compare
      │
      ├─ Thread spawned (sync work off async event loop)
      │        │
      │        ├─ Config 1 → generate() → queue.put(result)  ──► SSE event 1
      │        ├─ Config 2 → retrieve() → generate() → queue.put()  ──► SSE event 2
      │        ├─ Config 3 → hybrid()   → generate() → queue.put()  ──► SSE event 3
      │        └─ Config 4 → rerank()   → generate() → queue.put()  ──► SSE event 4
      │                                                                       │
      │                                              score each answer ◄──────┘
      │                                                     │
      └──────────────────────────────────────────────── SSE score events (5–8)
```

The browser renders each answer card the moment it arrives — before the others finish.

---

<div align="center">

Architecture over compute.

</div>
