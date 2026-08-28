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

**Run a query. Watch 4 RAG strategies answer it at once. See exactly which one wins, and why.**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![OpenRouter](https://img.shields.io/badge/OpenRouter-gpt--oss-6467F2?style=flat-square)](https://openrouter.ai)
[![FAISS](https://img.shields.io/badge/FAISS-Meta_AI-0467DF?style=flat-square)](https://github.com/facebookresearch/faiss)
[![HF Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-FFD21E?style=flat-square)](https://huggingface.co/spaces/Malladi05/raglens)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

</div>

---

## What It Does

Most RAG tutorials show one pipeline. RAGLens shows **four**, running live on your query, scored automatically, charted over time.

```
Your query ──► No RAG          ──► answer  score
           ──► Dense RAG       ──► answer  score
           ──► Hybrid RAG      ──► answer  score
           ──► Hybrid + Rerank ──► answer  score
```

Each result arrives as it finishes (SSE streaming). Each answer is scored by an LLM judge on faithfulness, relevancy, and context precision, with no ground truth required, and the judge returns a one-line reason for every score plus any answer sentences it could not ground in the retrieved context.

Pick one model or several. Each selected model runs the full set of 4 configs, and the results stack in their own labelled group so you can read a model against a model, config for config. The suggested prompts are drawn from a pool and reshuffle after every run.

Every card is inspectable: expand it to see the exact chunks retrieval fed the model (config 4 shows the full rerank pool with which candidates survived), the token count and OpenRouter cost of the call, and the judge's reasoning. Once a run is scored a verdict strip names the winning config per model and its delta versus No RAG. Pin any two cards for a word-level answer diff, and export the whole run as Markdown + JSON.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Browser (SSE)                            │
│   Query ──► Cards render live ──► context inspector · cost ·    │
│             judge reasons · verdict strip · answer diff · export │
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
       ├──► Config 3: Hybrid RAG                                 OpenRouter API
       │      │                                                    (gpt-oss)
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
                                                    │   LLM-as-Judge (OpenRouter)│
                                                    │   faithfulness  [0–1] +why │
                                                    │   answer_relevancy [0–1]+why│
                                                    │   context_precision [0–1]  │
                                                    │   + ungrounded sentences   │
                                                    └─────────────────────┬──────┘
                                                                          │
                                               ┌──────────────────────────▼──────┐
                                               │   SQLite + APScheduler          │
                                               │   Scheduled eval · 7-day charts │
                                               │   Per-model drift (Δ > 10%)     │
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

The context inspector on each card lists the retrieved chunks with their retrieval scores; for config 4 it shows all six hybrid candidates ranked by the cross-encoder, marking the two that were kept and the four that were dropped.

---

## Evaluation Metrics

| Metric | Question It Answers | Config |
|--------|---------------------|--------|
| **Faithfulness** | Did the answer stay grounded in the retrieved context? | 2, 3, 4 |
| **Answer Relevancy** | Does the answer actually address the question? | 1, 2, 3, 4 |
| **Context Precision** | Was the retrieved context useful noise-free? | 2, 3, 4 |

Scored automatically through OpenRouter (no labelled data needed). One judge call per answer returns all three scores, a short reason for each, and a list of answer sentences not supported by the retrieved passages. The scheduled monitoring job asks only for faithfulness and relevancy to keep its token use minimal.

---

## Stack

```
Generation    OpenRouter  ·  openai/gpt-oss-20b  ·  fast hosted inference
Embeddings    BAAI/bge-small-en-v1.5  ·  384-dim  ·  ~130 MB
Reranker      BAAI/bge-reranker-base  ·  cross-encoder
Index         FAISS IndexFlatL2  ·  exact NN  ·  1,665 chunks
Keyword       rank-bm25  ·  pure Python  ·  no external service
Backend       FastAPI  ·  SSE streaming  ·  SQLite
Scheduler     APScheduler  ·  6-hour cycle  ·  per-model drift detection
Corpus        50 arXiv papers  ·  RAG & LLM evaluation
```

---

## Live Demo

**[huggingface.co/spaces/Malladi05/raglens](https://huggingface.co/spaces/Malladi05/raglens)**

---

## Quickstart

```bash
git clone https://github.com/Sridharmalladi/RAGLens
cd RAGLens

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # paste your OpenRouter key
uvicorn main:app --reload --port 7860
```

Then open **http://localhost:7860**

> Create an OpenRouter key at [openrouter.ai/keys](https://openrouter.ai/keys). Generation and scoring both draw from your OpenRouter credit balance.

---

## Deployment

Deployed on Hugging Face Spaces using Docker (`sdk: docker`).

- Base image: `python:3.11-slim-bookworm`
- CPU-only torch (no GPU needed)
- BGE embeddings pre-computed and committed (`corpus/embeddings.json`), so the FAISS index builds in about 1 s at startup instead of 7+ min
- All models (BGE-small, BGE-reranker) download once at container start via the warmup thread; a banner in the UI shows progress

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

The browser renders each answer card the moment it arrives, before the others finish.

---

<div align="center">

Architecture over compute.

</div>
