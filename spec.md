# RAGLens — Live RAG Benchmarking & Monitoring Platform

> **A production-aware tool that compares 4 RAG configurations across multiple LLMs, with continuous background monitoring and user-driven exploration.**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [System Architecture](#3-system-architecture)
4. [Design Principles](#4-design-principles)
5. [Component Specifications](#5-component-specifications)
6. [Data Flow](#6-data-flow)
7. [Tech Stack & Rationale](#7-tech-stack--rationale)
8. [Build Phases](#8-build-phases)
9. [Failure Modes & Fallbacks](#9-failure-modes--fallbacks)
10. [Self-Critique Protocol](#10-self-critique-protocol)
11. [Deployment](#11-deployment)
12. [Interview Talking Points](#12-interview-talking-points)
13. [Future Extensions](#13-future-extensions)

---

## 1. Project Overview

**RAGLens** is a live RAG benchmarking platform deployed on HuggingFace Spaces. It solves one specific problem: most RAG demos show models on their best day, with no way to verify they keep working over time.

RAGLens does two things in parallel:

| System | Purpose | Trigger |
|---|---|---|
| **Background Monitoring** | Continuously evaluates the same fixed queries on a schedule, logs RAGAS scores and latency, renders a 7-day trend dashboard. | Cron job (hourly) |
| **Interactive Comparison** | Lets a visitor pick a suggested query (or ask their own within the corpus) and see all 4 RAG configurations answer side-by-side with live RAGAS scoring. | User click |

The two systems share a UI but **never share writes** — monitoring data is controlled, never polluted by user input.

### Why This Matters

Every AI engineer can claim *"I built a RAG system."* Almost none can show:
- A live link that works in 10 seconds
- Comparative scoring across configurations
- Time-series monitoring proving continuous evaluation
- Honest failure modes with defined fallbacks

RAGLens demonstrates all four in one URL.

---

## 2. Problem Statement

### What RAG Demos Usually Miss

1. **One-shot demos.** A user asks one question, gets one answer. No way to compare strategies.
2. **No measurement.** Most demos show outputs without faithfulness, relevancy, or precision scores.
3. **No temporal awareness.** Models drift. Embeddings get stale. Demos never reflect this.
4. **No failure visibility.** When retrieval fails, demos either hide it or crash.

### What RAGLens Demonstrates

1. **Comparative architecture choice** — baseline → RAG → fine-tuning → optimal retrieval, scored side-by-side.
2. **Multi-model evaluation** — same query against different open-source LLMs to surface model-specific behavior.
3. **Continuous monitoring** — historical RAGAS scores rendered as a live dashboard.
4. **Honest fallbacks** — every component has a defined failure response. Nothing fails silently.

---

## 3. System Architecture

### High-Level View

```
┌──────────────────────────────────────────────────────────────┐
│                    HuggingFace Spaces                        │
│  ┌────────────────────────────────────────────────────────┐  │
│  │                  Gradio UI (app.py)                    │  │
│  │  ┌──────────────────┐    ┌─────────────────────────┐  │  │
│  │  │  Monitoring Tab  │    │  Comparison Tab         │  │  │
│  │  │  (read-only)     │    │  (live inference)       │  │  │
│  │  └────────┬─────────┘    └────────────┬────────────┘  │  │
│  └───────────┼───────────────────────────┼────────────────┘  │
│              │                           │                   │
│              ▼                           ▼                   │
│  ┌────────────────────┐    ┌────────────────────────────┐    │
│  │  monitoring.py     │    │  inference.py              │    │
│  │  (writes to DB)    │    │  (read-only, no DB write)  │    │
│  └────────┬───────────┘    └────────────┬───────────────┘    │
│           │                             │                    │
│           ▼                             ▼                    │
│  ┌────────────────────┐    ┌────────────────────────────┐    │
│  │  SQLite (history)  │    │  Retrieval + Generation     │    │
│  │  scores, latency   │    │  + RAGAS scoring (live)     │    │
│  └────────────────────┘    └─────────────────────────────┘    │
│           ▲                                                  │
│           │ writes only                                      │
│  ┌────────┴───────────┐                                      │
│  │  Scheduler         │                                      │
│  │  (APScheduler)     │                                      │
│  │  hourly cron       │                                      │
│  └────────────────────┘                                      │
└──────────────────────────────────────────────────────────────┘
```

### Critical Architectural Rule

> **The user-facing inference path NEVER writes to the monitoring database.**
> Monitoring writes only from the scheduled job. This separation keeps the historical trend honest. Violating it pollutes the entire project's story.

### Module Responsibilities

| Module | Owns | Never Does |
|---|---|---|
| `app.py` | UI rendering, routing user clicks | Direct DB writes, model loading |
| `inference.py` | Live query execution for user requests | Touches the monitoring DB |
| `monitoring.py` | Scheduled evaluations, DB writes | Serves user-facing requests |
| `retrieval.py` | All 4 retrieval strategies | Generation or scoring |
| `models.py` | Model + adapter loading, generation | Retrieval or evaluation |
| `evaluation.py` | RAGAS scoring | Storage or scheduling |
| `storage.py` | SQLite read/write abstraction | Business logic |
| `corpus.py` | Document loading, chunking, embedding | Runtime queries |
| `scheduler.py` | Cron timing, job triggering | Score computation |

---

## 4. Design Principles

These are the non-negotiables. Every line of code is judged against them.

### 4.1 Separation of Concerns
Each module owns exactly one responsibility. If a function needs to import from three different modules to do its job, the design is wrong.

### 4.2 Fail Loud, Fail Fast
Silent failures are the enemy of trust. Every external call defines:
- What success looks like
- What failure looks like
- What the user sees on failure

### 4.3 Boring Tech Wins
For every choice, pick the most well-documented, mature option. No experimental libraries. No bleeding-edge versions. Tools used in this project must have 1K+ GitHub stars and a commit within the last 6 months.

### 4.4 Reversibility
Every design decision should be undoable in under an hour. If a choice locks the project into a specific direction, write down the alternative explicitly.

### 4.5 Single Source of Truth
There is exactly one place where:
- Chunk size is defined (`config.py`)
- Model paths are listed (`config.py`)
- Query corpus is stored (`corpus/`)
- Historical data lives (`raglens.db`)

### 4.6 The Recruiter Test
At every step, ask: *"If a recruiter clicks this in 10 seconds, do they understand the value?"* If the answer is no, the feature is wrong-shaped, not the recruiter.

### 4.7 Cost-Aware
Every inference call has a cost. Document it. Cache aggressively. This project runs at **zero marginal cost**: Groq free tier covers all RAGAS evaluation calls; HuggingFace CPU free tier covers hosting; generation uses local in-process models with no API charges. The constraint is a feature — it proves the architecture matters more than the budget.

### 4.8 No Premature Abstraction
Don't build a "configurable framework." Hardcode values until duplication forces extraction. If you find yourself building a base class with one subclass, delete the base class.

### 4.9 Free-Stack by Design
RAGLens runs at zero marginal cost. All generation is in-process (no paid API). RAGAS evaluation uses the Groq free tier. Hosting is HuggingFace Spaces CPU Basic. This is not a compromise — it's a constraint that proves the architecture matters more than the budget. The same design scales identically to LLaMA 3.1 8B or any larger model; only compute changes. Put this in the About section so a recruiter reads it before they ask.

---

## 5. Component Specifications

### 5.1 The 4 RAG Configurations

| # | Config | Model | Retrieval | Purpose |
|---|---|---|---|---|
| 1 | **Baseline** | Phi-3.5 Mini (3.8B) Instruct | None | Establishes "no context" floor |
| 2 | **Base + RAG** | Phi-3.5 Mini (3.8B) Instruct | Top-k dense (BGE) | Shows retrieval alone's contribution |
| 3 | **Fine-tuned + RAG** | Phi-3.5 Mini + QLoRA | Top-k dense (BGE) | Shows fine-tuning's contribution |
| 4 | **FT + Optimal RAG** | Phi-3.5 Mini + QLoRA | Hybrid (dense + BM25) + rerank | Best-of-all-worlds config |

Reading the matrix left-to-right shows what RAG adds. Top-to-bottom shows what fine-tuning adds. Bottom-right is always the winner — and RAGAS scores must prove it numerically. If they don't, something is wrong with the fine-tuning or the corpus and that becomes the next debug task.

### 5.2 Multi-Model Comparison

Beyond the 4 configs, a model selector lets the user compare across:

| Model | Why Included |
|---|---|
| Phi-3.5 Mini (3.8B) | Default; includes fine-tuned QLoRA variant; CPU-runnable |
| Qwen 2.5 1.5B Instruct | Smallest option — fastest response, cost-conscious story |
| LLaMA 3.2 3B Instruct | Mid-size; different architecture from Phi family |
| Gemma-2 2B | Google's open model; recognizable name, competitive at 2B |

Default selection is Phi-3.5 Mini. User can switch via dropdown. All models run on CPU free tier — no GPU required.

### 5.3 Corpus

- **Source:** 50 papers on RAG architectures and LLM evaluation (publicly available, properly attributed)
- **Format:** Markdown after extraction from PDF
- **Chunking:** Fixed-size, 512 tokens, 50 overlap (hardcoded, single source of truth in `config.py`)
- **Embedding:** `BAAI/bge-small-en-v1.5` (768 dim, fast on CPU/T4)
- **Index:** FAISS flat L2 index, in-memory
- **Refresh:** On Space startup only — corpus is static for v1

### 5.4 Retrieval Layer

Four functions, each independently testable:

```python
def dense_retrieve(query: str, k: int = 5) -> list[Chunk]
def sparse_retrieve(query: str, k: int = 5) -> list[Chunk]
def hybrid_retrieve(query: str, k: int = 5, alpha: float = 0.5) -> list[Chunk]
def rerank(query: str, chunks: list[Chunk], top_n: int = 3) -> list[Chunk]
```

`alpha` controls the weight between dense and sparse in the hybrid case. Reranker uses `BAAI/bge-reranker-base` (not large — latency budget).

### 5.5 Generation Layer

```python
def generate(model_id: str, query: str, context: str | None = None) -> tuple[str, float]
```

Returns `(answer, latency_seconds)`. Context is optional — `None` for the baseline config.

Models run in standard float32/bfloat16 on CPU — no quantization needed at 1.5–3.8B params. Switching to the fine-tuned variant uses `peft` to swap in the QLoRA adapter without reloading the base model.

### 5.6 Evaluation Layer

```python
def score(query: str, answer: str, contexts: list[str], ground_truth: str | None = None) -> dict
```

Returns RAGAS scores:
- `faithfulness` — does the answer stay grounded in the context?
- `answer_relevancy` — does the answer address the query?
- `context_precision` — are the retrieved chunks actually relevant?

Judge model: `llama-3.1-8b-instant` via **Groq API** (free tier). Cost: $0. Groq free tier allows 14,400 requests/day; monitoring uses 288/day (3 queries × 4 configs × 24 hours) — well within limit. If `GROQ_API_KEY` is missing, the scoring layer disables and the UI shows "scoring unavailable" — not zeros.

### 5.7 Monitoring Layer

Scheduled job runs every hour:

1. Load the 3 fixed evaluation queries
2. Run each query through all 4 configs (for the default model — Phi-3.5 Mini)
3. Score every result via RAGAS
4. Write `(timestamp, model, config, query, score_dict, latency)` to SQLite
5. Retain only the last 30 days of data (older rows pruned)

The dashboard queries this DB on UI render and shows:
- Line chart of `faithfulness` over the last 7 days, one line per config
- Line chart of `latency` over the same window
- "Last run" and "Next run" timestamps
- Drift alert badge if any config's faithfulness dropped 10%+ in the last 24 hours

### 5.8 Interactive Comparison Layer

User flow:
1. Click a suggested query OR type a custom one (limited to 200 chars)
2. UI dispatches to `inference.run_all_configs(query, model_id)`
3. All 4 configs execute in parallel (asyncio or thread pool)
4. Each result rendered as it returns — progressive disclosure
5. RAGAS scores appear underneath each answer when ready
6. Total latency budget: under 90 seconds for all 4 configs

Important: this path **never writes to the monitoring DB.**

### 5.9 UI Specification

```
┌──────────────────────────────────────────────────────┐
│  RAGLens — Live RAG Benchmarking                     │
│  [Built by Sridhar Malladi · About]                  │
├──────────────────────────────────────────────────────┤
│                                                      │
│  📊 LIVE MONITORING (last 7 days)                    │
│  ┌────────────────────────────────────────────────┐ │
│  │  [Line chart: faithfulness × 4 configs]        │ │
│  │  [Line chart: latency × 4 configs]             │ │
│  └────────────────────────────────────────────────┘ │
│  Last run: 14 min ago · Next run: in 46 min          │
│  ⚠️ Drift alert: Base+RAG faithfulness down 12%      │
│  (alert shown only when triggered)                   │
│                                                      │
├──────────────────────────────────────────────────────┤
│  🧪 TRY IT YOURSELF                                  │
│                                                      │
│  Model: [Phi-3.5 Mini ▾]                              │
│                                                      │
│  Suggested:  [ Q1 ]  [ Q2 ]  [ Q3 ]                  │
│  Or ask your own:  [ ________________ ] [ Run ]      │
│                                                      │
│  📌 Corpus: 50 papers on RAG and LLM evaluation      │
│                                                      │
│  Results:                                            │
│  ┌──────────────┐  ┌──────────────┐                  │
│  │  Baseline    │  │  Base + RAG  │                  │
│  │  (answer)    │  │  (answer)    │                  │
│  │  F: 0.72     │  │  F: 0.84     │                  │
│  │  R: 0.81     │  │  R: 0.89     │                  │
│  │  P: —        │  │  P: 0.77     │                  │
│  └──────────────┘  └──────────────┘                  │
│  ┌──────────────┐  ┌──────────────┐                  │
│  │  FT + RAG    │  │  FT + Optimal│                  │
│  │  (answer)    │  │  (answer)    │                  │
│  │  F: 0.91     │  │  F: 0.94     │                  │
│  │  R: 0.92     │  │  R: 0.95     │                  │
│  │  P: 0.81     │  │  P: 0.88     │                  │
│  └──────────────┘  └──────────────┘                  │
│                                                      │
└──────────────────────────────────────────────────────┘
```

---

## 6. Data Flow

### 6.1 Scheduled Monitoring Flow

```
APScheduler (hourly trigger)
        │
        ▼
monitoring.run_evaluation_cycle()
        │
        ├─► For each of 3 fixed queries:
        │       │
        │       ├─► For each of 4 configs:
        │       │       │
        │       │       ├─► retrieval.{dense|hybrid+rerank|none}
        │       │       ├─► models.generate()
        │       │       └─► evaluation.score()
        │       │
        │       └─► storage.write_run(timestamp, query, config, scores, latency)
        │
        └─► storage.prune_old(days=30)
```

### 6.2 User Comparison Flow

```
User clicks "Run"
        │
        ▼
app.handle_run(query, model_id)
        │
        ▼
inference.run_all_configs(query, model_id)
        │
        ├─► (parallel) run config 1 ─► generate ─► score
        ├─► (parallel) run config 2 ─► retrieve ─► generate ─► score
        ├─► (parallel) run config 3 ─► retrieve ─► generate ─► score
        └─► (parallel) run config 4 ─► retrieve+rerank ─► generate ─► score
                                                                  │
                                                                  ▼
                                                            Stream results to UI
                                              (no DB write — strictly read-only path)
```

### 6.3 Dashboard Render Flow

```
User opens Space / refreshes monitoring tab
        │
        ▼
storage.read_recent(days=7)
        │
        ▼
app.render_charts(rows)
        │
        ▼
Plotly figures returned to Gradio
```

---

## 7. Tech Stack & Rationale

| Layer | Choice | Why This Over Alternatives |
|---|---|---|
| UI | Gradio 4.x | Native HF Spaces support, ships with auth + queueing, less code than Streamlit |
| Vector store | FAISS in-memory | Zero infrastructure, corpus is small and static — Pinecone is overkill |
| Embeddings | BGE-small-en-v1.5 | Strong quality at 33M params, CPU-fast, mature |
| Sparse retrieval | rank-bm25 | Pure Python, no Elasticsearch needed |
| Reranker | BGE-reranker-base | Same family as embeddings, latency budget allows base over large |
| Models | Phi-3.5 Mini (3.8B), Qwen2.5-1.5B, Llama-3.2-3B, Gemma-2-2B | All open, all ungated on HF, all CPU-runnable, all instruction-tuned |
| Fine-tuning | QLoRA via PEFT on Phi-3.5 Mini | Standard, mature, swappable adapter; trained free on Colab T4 |
| Evaluation | RAGAS | Industry standard, recruiter-recognizable |
| Judge model | llama-3.1-8b-instant via Groq | Free tier: 14,400 req/day; zero cost; fast inference |
| Scheduler | APScheduler | In-process, no external service needed |
| Storage | SQLite | Zero infra, perfect for time-series at this scale |
| Charts | Plotly | Interactive in Gradio, free, ubiquitous |
| Hosting | HuggingFace Spaces — CPU Basic (free) | Free tier, no GPU needed for sub-4B models, recruiters trust the URL |

### Explicit Non-Choices

| Rejected | Reason |
|---|---|
| LangChain | Hides too much, debugging is painful, no real benefit at this scale |
| LangGraph | Not orchestrating — this is parallel function calls |
| Pinecone / Weaviate | Persistent vector DB unnecessary for static corpus |
| Streamlit | More boilerplate for the dashboard, weaker HF integration |
| Modal / Replicate | Adds a hop, more failure modes, more cost |
| Custom React frontend | Wrong skill signal for an AI Engineer role |
| OpenAI API for generation | Costs money; same 4-config story achievable with local Phi-3.5 |
| bitsandbytes 4-bit quant | Needed only for 7B+ models; sub-4B runs fine on CPU without it |
| T4 GPU tier | Unnecessary for Phi-3.5 Mini (3.8B); CPU free tier is sufficient |
| gpt-4o-mini as judge | Replaced by Groq llama-3.1-8b-instant — free, fast, same quality |

---

## 8. Build Phases

Build in order. Do not skip ahead. Each phase ends with a self-critique gate.

### Phase 0 — Project Skeleton (1-2 hours)
- Repo structure
- `requirements.txt` with exact pins
- `config.py` with all constants
- Gradio launches a blank page locally
- README scaffold

### Phase 1 — Corpus Pipeline (2-3 hours)
- Drop 50 papers in `corpus/raw/`
- PDF → markdown extraction
- Chunking with the fixed strategy
- BGE embedding
- FAISS index built and saved to `corpus/index.faiss`
- Sanity check: retrieve 5 chunks for a known query, eyeball relevance

### Phase 2 — Retrieval Layer (2 hours)
- All 4 retrieval functions implemented
- Tested in isolation (no UI yet)
- Latency measured for each — must be under 2s

### Phase 3 — Generation Layer (3-4 hours)
- Base model loads in 4-bit
- Generation works on a hardcoded prompt
- QLoRA adapter loads on top
- Adapter swap works without full reload
- Latency measured — must be under 15s per call

### Phase 4 — Single Config End-to-End (2 hours)
- Wire retrieval + generation for config 2 (Base + RAG)
- Returns answer + context for one query
- No UI yet, run from CLI

### Phase 5 — All 4 Configs (3 hours)
- Parallel execution
- Total latency for one query across all 4 — must be under 90s
- Run from CLI, print results

### Phase 6 — RAGAS Integration (3 hours)
- Single config first
- Then all 4
- Verify scores look sensible
- Handle OpenAI API failures gracefully

### Phase 7 — SQLite Layer (1-2 hours)
- Schema: `runs(id, timestamp, model, config, query, faithfulness, relevancy, precision, latency)`
- Read/write functions
- 30-day pruning function

### Phase 8 — Scheduler (2 hours)
- APScheduler running hourly
- Runs the 3 fixed queries
- Writes to SQLite
- Test by triggering manually first

### Phase 9 — Gradio UI (4-5 hours)
- Monitoring tab — chart from SQLite
- Comparison tab — buttons + custom input + 2×2 grid
- Loading states
- Error messages

### Phase 10 — Polish & Deploy (3 hours)
- About section
- Disclaimer about corpus scope
- Deploy to HF Spaces
- Smoke test on the live URL

**Total estimated time: 25-30 hours of focused work.**

---

## 9. Failure Modes & Fallbacks

Every component has a defined failure response. This table is the contract.

| Component | Failure | Fallback | User Sees |
|---|---|---|---|
| FAISS index | Missing on startup | Rebuild from corpus | Spinner on startup, then normal |
| Embedding model | Fails to load | Hard fail at startup | Error banner: "Space initializing" |
| Base model | Fails to load | Log error, disable that model option | Dropdown option grayed out with message |
| QLoRA adapter | Missing/corrupt | Disable configs 3 & 4 | Those config cards grayed out with message |
| Retrieval | Returns 0 chunks | Pass empty context to generation | Config card shows "no retrieval result" |
| Generation | Timeout (>30s) | Cancel that config, return partial | Config card shows "generation timed out" |
| RAGAS | GROQ_API_KEY missing | Disable scoring entirely | Scores replaced with "scoring unavailable" |
| RAGAS | API call fails | Skip scoring for that run | Same — never show fake zeros |
| SQLite | Read fails | Empty dashboard with explainer | "Historical data unavailable" message |
| SQLite | Write fails | Log error, continue cycle | No user impact, monitored via logs |
| Scheduler | Job crashes | APScheduler restarts it on next tick | No user impact |

### The Honest Failure Principle

> A demo that hides failures is worse than one that doesn't. Recruiters who have shipped production systems can tell the difference. Show the failure, show the fallback, move on.

---

## 10. Self-Critique Protocol

This is run **after every build phase**. Do not skip.

```
SELF-CRITIQUE — PHASE [N]
=========================
1. What did I just build?
2. What's the simplest way this fails in production?
3. Did I add any code not strictly required by the brief?
4. Did I introduce a new dependency? If yes, was it absolutely necessary?
5. What's the latency profile? (measure, don't estimate)
6. If a recruiter clicked this right now, what's confusing?
7. What did I avoid building that I want to revisit later — but won't now?
8. Did I violate any of the 8 design principles?
```

**Rule:** If question 3 or 4 is "yes" without strong justification, remove the addition before continuing. If question 8 is "yes," fix the violation before moving on.

---

## 11. Deployment

### Local Development
```bash
git clone <repo>
cd raglens
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add GROQ_API_KEY (free key from console.groq.com)
python app.py
```

### HuggingFace Spaces Deployment
1. Create a Space, SDK = Gradio, Hardware = **CPU Basic (free)**
2. Push the repo
3. Add `GROQ_API_KEY` as a secret in Space settings (free key from console.groq.com)
4. No `HF_TOKEN` needed — all models are ungated
5. Space builds and starts the scheduler on boot

### Custom Domain (Optional)
- Buy `raglens.dev` or similar
- Add a CNAME record pointing to your HF Space URL
- Update Space settings to recognize the custom domain
- Resume link becomes `raglens.dev` instead of the longer HF URL

---

## 12. Interview Talking Points

When asked about this project, structure answers around these themes.

### "Walk me through the architecture."
Two parallel systems sharing a UI. Background monitoring runs scheduled evaluations and writes to SQLite — this powers the historical dashboard. The user-facing inference path runs comparisons live but never writes to the DB. This separation is the project's most important design decision: it keeps the monitoring data honest under any user load.

### "Why these 4 configurations?"
Each config isolates one variable. Baseline vs Base+RAG isolates retrieval's contribution. Base+RAG vs FT+RAG isolates fine-tuning's contribution. FT+RAG vs FT+Optimal isolates the retrieval strategy itself. Reading the 2×2 matrix, you can attribute exactly how much each component adds.

### "Why not LangChain?"
At this scale, LangChain hides more than it helps. The retrieval and generation paths are 4 functions called in parallel — adding a chain abstraction makes debugging harder without adding value. The cost of using LangChain is opacity; the benefit at this scale is zero.

### "How do you handle drift?"
The dashboard tracks faithfulness over time per config. A 10%+ drop in any config's 24-hour average triggers a visible alert badge. The same query is run hourly with the same model — so a drift signal points to either model behavior change or corpus relevance decay, both of which are real production failure modes.

### "What would you build next?"
Multi-judge evaluation — RAGAS uses a single judge model. Adding a second judge (a stronger or weaker model) and tracking score agreement would surface judge bias. After that, a robustness tab with adversarial queries (out-of-domain, contradictions, empty retrieval).

### "What's the biggest weakness?"
The corpus is static. In a real production RAG system, documents arrive continuously and the index must update incrementally. RAGLens rebuilds from scratch on Space restart. Solving that properly requires an external vector DB and ingestion pipeline — out of scope for v1, but the natural v2.

### "What did you learn building this?"
Two things. First — that monitoring is the differentiator. Anyone can show a RAG demo on its best day. Showing it across time, with measurable scores, is what separates a portfolio piece from production thinking. Second — that strict architectural boundaries (the user path never writes to monitoring) matter more than they seem. Without that rule, the entire project's value collapses within a day of public traffic.

---

## 13. Future Extensions

Tracked here for discipline — these are explicitly out of scope for v1.

| Extension | Why Later, Not Now |
|---|---|
| Document upload by users | Different product, different failure surface |
| Streaming responses | Complicates Gradio state — premature optimization |
| Multi-judge RAGAS | Doubles eval cost, marginal value for v1 |
| Custom corpus selection | UI complexity, low marginal value |
| Real-time corpus updates | Requires external vector DB, out of scope |
| A/B testing across users | No user volume to justify |
| User accounts / persistent sessions | Same — no volume |
| Cost dashboard | Nice-to-have, not differentiating |
| Robustness tab (adversarial queries) | Phase 2 — strong addition, not v1 critical |

---

## Appendix A — File Structure

```
raglens/
├── app.py                  # Gradio UI entry point
├── config.py               # All constants — single source of truth
├── corpus/
│   ├── raw/                # PDFs
│   ├── processed/          # Markdown chunks
│   └── index.faiss         # Built index
├── src/
│   ├── inference.py        # User-facing path
│   ├── monitoring.py       # Scheduled path
│   ├── retrieval.py        # 4 retrieval functions
│   ├── models.py           # Model loading + generation
│   ├── evaluation.py       # RAGAS wrapper
│   ├── storage.py          # SQLite read/write
│   ├── scheduler.py        # APScheduler setup
│   └── corpus.py           # Corpus loading + chunking
├── tests/
│   ├── test_retrieval.py
│   ├── test_generation.py
│   └── test_evaluation.py
├── requirements.txt
├── .env.example
├── README.md
└── raglens.db              # Created at runtime
```

## Appendix B — config.py (Single Source of Truth)

```python
# config.py — every constant in the project lives here

# Models — all CPU-runnable, all ungated on HuggingFace
DEFAULT_MODEL = "microsoft/Phi-3.5-mini-instruct"
AVAILABLE_MODELS = [
    "microsoft/Phi-3.5-mini-instruct",      # default; fine-tuned variant available
    "Qwen/Qwen2.5-1.5B-Instruct",           # smallest / fastest
    "meta-llama/Llama-3.2-3B-Instruct",     # mid-size, different architecture
    "google/gemma-2-2b-it",                 # Google open model
]
ADAPTER_PATH = "adapters/phi-3.5-mini-qlora-raglens"  # QLoRA adapter for configs 3 & 4

# Embeddings & retrieval
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K = 5
RERANK_TOP_N = 3
HYBRID_ALPHA = 0.5

# Generation
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
GENERATION_TIMEOUT_SECONDS = 30

# Evaluation — Groq free tier (14,400 req/day; monitoring uses ~288/day)
JUDGE_MODEL = "llama-3.1-8b-instant"
JUDGE_PROVIDER = "groq"  # env: GROQ_API_KEY
RAGAS_METRICS = ["faithfulness", "answer_relevancy", "context_precision"]

# Monitoring
MONITORING_INTERVAL_HOURS = 1
RETENTION_DAYS = 30
DRIFT_ALERT_THRESHOLD = 0.10

# Fixed monitoring queries (the same 3 every hour, forever)
MONITORING_QUERIES = [
    "What are the main differences between dense and sparse retrieval?",
    "How does reranking improve RAG performance?",
    "What is QLoRA and when should you use it?",
]

# Suggested user queries (the 3 buttons in the UI)
SUGGESTED_QUERIES = [
    "Explain how hybrid retrieval combines dense and sparse search.",
    "What evaluation metrics matter most for production RAG?",
    "When does fine-tuning beat better retrieval?",
]

# UI
APP_TITLE = "RAGLens — Live RAG Benchmarking"
CORPUS_DESCRIPTION = "50 papers on RAG architectures and LLM evaluation"
```

---

**End of specification.**

Built by Sridhar Malladi. Architectural patterns inspired by production ML systems and the principle that **the best portfolio project is one a recruiter can verify without your help.**
