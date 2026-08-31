<div align="center">

# RAGLens

**Run a query. Watch 4 RAG strategies answer it at once. See exactly which one wins, and why.**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![OpenRouter](https://img.shields.io/badge/OpenRouter-gpt--oss-6467F2?style=flat-square)](https://openrouter.ai)
[![FAISS](https://img.shields.io/badge/FAISS-Meta_AI-0467DF?style=flat-square)](https://github.com/facebookresearch/faiss)
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
Embeddings    BAAI/bge-small-en-v1.5  ·  384-dim  ·  in-process or HF Inference API
Reranker      BAAI/bge-reranker-base  ·  cross-encoder  ·  in-process or HF Inference API
Index         FAISS IndexFlatL2 (local)  ·  numpy cosine (hosted)  ·  1,665 chunks
Keyword       rank-bm25  ·  pure Python  ·  no external service
Backend       FastAPI  ·  SSE streaming (JSON fallback)  ·  SQLite
Scheduler     APScheduler  ·  8-hour cycle  ·  per-model drift detection
Corpus        50 arXiv papers  ·  RAG & LLM evaluation
```

---

## Quickstart

```bash
git clone https://github.com/Sridharmalladi/RAGLens
cd RAGLens

python -m venv .venv && source .venv/bin/activate
pip install -r requirements-full.txt   # in-process BGE + FAISS
# or:  pip install -r requirements.txt  # lite: retrieval via HF Inference API

cp .env.example .env        # paste your OpenRouter key (+ HF_TOKEN for lite)
uvicorn main:app --reload --port 7860
```

Then open **http://localhost:7860**

> Create an OpenRouter key at [openrouter.ai/keys](https://openrouter.ai/keys). Generation and scoring both draw from your OpenRouter credit balance.

---

## Retrieval backends

`RETRIEVAL_BACKEND` picks how dense retrieval and reranking run. Left unset it
auto-detects: `local` when `sentence-transformers` is importable, else `hosted`.

| | `local` | `hosted` |
|---|---|---|
| Deps | `requirements-full.txt` (torch, faiss, ~4 GB) | `requirements.txt` (~30 MB) |
| Query embedding | in-process BGE-small | HF Inference API, BGE-small |
| Dense search | FAISS `IndexFlatL2` | numpy cosine over `corpus/embeddings.json` |
| Rerank | in-process cross-encoder | HF Inference API `text-ranking` |
| Needs | — | `HF_TOKEN` (read scope) |

Both call the **same** BGE-small model, so the committed corpus vectors work
either way — no re-embedding.

---

## Deployment

### Vercel (free)

Serverless. `vercel.json` routes everything to `api/index.py`, installs
`requirements.txt` (lite), and sets `RETRIEVAL_BACKEND=hosted`,
`STREAM_RESPONSES=0` (Vercel buffers streams — cards land together, not
progressively), `ENABLE_SCHEDULER=0`, `DB_PATH=/tmp/raglens.db`.

Import the repo at [vercel.com/new](https://vercel.com/new), then add two env
vars in the project settings:

- `OPENROUTER_API_KEY`
- `HF_TOKEN`

### Render (free)

`render.yaml` is a blueprint: web service, `requirements.txt` (lite), start
`uvicorn main:app --host 0.0.0.0 --port $PORT`. SSE streaming works here.

New + → Blueprint at [dashboard.render.com](https://dashboard.render.com), point
it at the repo, set `OPENROUTER_API_KEY` and `HF_TOKEN` when prompted. Free
instances sleep after 15 min idle (~40 s cold start).

### Docker (`local` backend, full models)

```bash
docker build -t raglens .
docker run -p 7860:7860 -e OPENROUTER_API_KEY=sk-or-... raglens
```

- Base image `python:3.11-slim-bookworm`, CPU-only torch, `requirements-full.txt`
- BGE-small + BGE-reranker baked into the image; FAISS index builds from the
  committed `corpus/embeddings.json` in ~1 s at startup
- Listens on `$PORT` (Cloud Run) or 7860. See `DEPLOY_CLOUD_RUN.md`.

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
