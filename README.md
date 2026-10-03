<div align="center">

# RAGLens

**Compare retrieval strategies on the same question, with every result open to inspection.**

[Try RAGLens](https://huggingface.co/spaces/Malladi05/raglens)

</div>

RAGLens runs a query through four answer pipelines and scores the results with an LLM judge. Inspect the retrieved passages, judge notes, latency, token use, and cost for each answer.

## Strategies

| Pipeline | Retrieval |
| --- | --- |
| No RAG | Model knowledge only |
| Dense | Semantic search |
| Hybrid | Semantic + keyword search |
| Hybrid + reranker | Hybrid candidates reranked for precision |

Answers stream as they finish. Scores cover faithfulness, answer relevance, and context precision. Compare models, view answer differences, export a run as Markdown or JSON, and track scores over time. The bundled corpus contains 50 research papers.

## Run locally

Requires Python 3.11+, an OpenRouter API key, and an HF token for hosted retrieval.

```bash
git clone https://github.com/Sridharmalladi/RAGLens.git
cd RAGLens
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # set OPENROUTER_API_KEY and HF_TOKEN
uvicorn main:app --reload --port 7860
```

Open `http://localhost:7860`. For in-process embeddings and FAISS, install `requirements-full.txt` instead. Hosted deployment and scheduler settings are documented in `render.yaml` and `vercel.json`.
