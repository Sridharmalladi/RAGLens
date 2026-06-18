---
title: RAGLens
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
---

# RAGLens — Live RAG Benchmarking & Monitoring Platform

> Compare 4 RAG configurations side-by-side with live RAGAS scores and a 7-day monitoring dashboard.

## What It Does

**Interactive comparison** — pick a query, see all 4 RAG configs answer in parallel with faithfulness, relevancy, and precision scores.

**Background monitoring** — an hourly cron evaluates the same 3 fixed queries on all 4 configs, writes scores to SQLite, and renders a 7-day trend dashboard. User traffic never touches the monitoring data.

## The 4 Configurations

| Config | Model | Retrieval | What It Shows |
|---|---|---|---|
| Baseline | Qwen2.5-1.5B | None | No-context floor |
| Base + RAG | Qwen2.5-1.5B | Dense (BGE + FAISS) | Retrieval contribution |
| FT + RAG | Phi-3.5-mini + QLoRA | Dense | Fine-tuning contribution |
| FT + Optimal RAG | Phi-3.5-mini + QLoRA | Hybrid + reranking | Best-of-all-worlds |

## Stack

- **UI:** Gradio 4.x on HuggingFace Spaces CPU (free tier)
- **Models:** Qwen2.5-1.5B, Phi-3.5-mini, Llama-3.2-3B, Gemma-2-2B (all CPU-runnable)
- **Retrieval:** FAISS (dense) + BM25 (sparse) + BGE-reranker-base
- **Evaluation:** RAGAS with Groq llama-3.1-8b-instant as judge (free tier)
- **Storage:** SQLite · **Scheduler:** APScheduler

## Local Development

```bash
git clone <repo-url>
cd RAGLens
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your GROQ_API_KEY
python app.py
```

Note: the corpus must be built first. See [Building the Corpus](#building-the-corpus).

## Building the Corpus (one-time, in Google Colab)

1. Open `build_corpus.ipynb` in [Google Colab](https://colab.research.google.com)
2. Run all cells — downloads 50 arXiv papers, extracts text, embeds, and builds FAISS index
3. Download the two output files when prompted:
   - `corpus/index.faiss`
   - `corpus/processed/chunks.json`
4. Place them in your repo at those exact paths and commit

The HuggingFace Space loads these files at startup — no PDF processing at runtime.

## Deploying to HuggingFace Spaces

See [DEPLOY.md](DEPLOY.md) for step-by-step instructions.

## Built By

Sridhar Malladi — [GitHub](https://github.com/sridharmalladi)

> RAGLens runs at zero marginal cost. Architecture matters more than compute budget.
