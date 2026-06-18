"""
RAGLens — Gradio entry point.
Two tabs: Live Monitoring (read-only from SQLite) and Try It (live inference).
"""

import logging
import os
from datetime import datetime

import gradio as gr
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from config import (
    APP_TITLE, CORPUS_DESCRIPTION, SUGGESTED_QUERIES,
    AVAILABLE_MODELS, DEFAULT_MODEL, CONFIG_NAMES, DRIFT_ALERT_THRESHOLD,
)

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def _warmup_default_model():
    """Download + load the default model in the background so the first user query is fast."""
    try:
        from src.models import get_model
        from config import DEFAULT_MODEL
        logger.info("Warming up %s in background...", DEFAULT_MODEL)
        get_model(DEFAULT_MODEL)
        logger.info("Model warmup complete: %s", DEFAULT_MODEL)
    except Exception as e:
        logger.warning("Model warmup failed (non-fatal): %s", e)


def _startup():
    import threading
    from src.storage import init_db
    from src.scheduler import start as start_scheduler

    init_db()
    start_scheduler()
    threading.Thread(target=_warmup_default_model, daemon=True).start()
    logger.info("RAGLens started")

_startup()

# ---------------------------------------------------------------------------
# Monitoring tab helpers
# ---------------------------------------------------------------------------

CONFIG_COLORS = {
    "Baseline": "#6B7280",
    "Base + RAG": "#3B82F6",
    "FT + RAG": "#F59E0B",
    "FT + Optimal RAG": "#10B981",
}


def _build_monitoring_charts():
    from src.storage import read_recent, detect_drift, read_last_run_time
    from src.scheduler import next_run_time

    rows = read_recent(days=7)
    last_run = read_last_run_time()
    next_run = next_run_time()

    if not rows:
        empty_msg = (
            "No monitoring data yet.\n"
            "The first evaluation cycle runs automatically — "
            "check back in ~1 hour, or use the admin trigger below."
        )
        return (
            _empty_figure("Faithfulness (7 days)", empty_msg),
            _empty_figure("Latency in seconds (7 days)", empty_msg),
            "No runs yet",
            next_run or "Unknown",
            "",
        )

    # Group by config_name
    by_config: dict[str, dict] = {}
    for row in rows:
        name = row["config_name"]
        if name not in by_config:
            by_config[name] = {"ts": [], "faith": [], "latency": []}
        by_config[name]["ts"].append(row["timestamp"])
        by_config[name]["faith"].append(row["faithfulness"])
        by_config[name]["latency"].append(row["latency_s"])

    faith_fig = go.Figure()
    latency_fig = go.Figure()

    for name, data in by_config.items():
        color = CONFIG_COLORS.get(name, "#8B5CF6")
        faith_fig.add_trace(go.Scatter(
            x=data["ts"], y=data["faith"], mode="lines+markers",
            name=name, line=dict(color=color, width=2),
        ))
        latency_fig.add_trace(go.Scatter(
            x=data["ts"], y=data["latency"], mode="lines+markers",
            name=name, line=dict(color=color, width=2),
        ))

    _style_fig(faith_fig, "Faithfulness score (7 days)", y_range=[0, 1])
    _style_fig(latency_fig, "Generation latency — seconds (7 days)")

    last_run_str = _fmt_ago(last_run) if last_run else "Never"
    next_run_str = next_run or "Unknown"

    # Drift alerts
    alerts = detect_drift(threshold=DRIFT_ALERT_THRESHOLD)
    drift_text = ""
    if alerts:
        parts = [f"⚠️  {a['config_name']}: faithfulness down {a['drop']*100:.1f}% in 24h" for a in alerts]
        drift_text = "\n".join(parts)

    return faith_fig, latency_fig, last_run_str, next_run_str, drift_text


def _empty_figure(title: str, message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False, font=dict(size=14, color="#9CA3AF"))
    _style_fig(fig, title)
    return fig


def _style_fig(fig: go.Figure, title: str, y_range=None) -> None:
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#1F2937",
        font=dict(color="#F9FAFB"),
        legend=dict(bgcolor="#1F2937", bordercolor="#374151"),
        margin=dict(l=40, r=20, t=50, b=40),
        height=280,
    )
    if y_range:
        fig.update_yaxes(range=y_range)


def _fmt_ago(iso_ts: str) -> str:
    try:
        delta = datetime.utcnow() - datetime.fromisoformat(iso_ts)
        minutes = int(delta.total_seconds() / 60)
        if minutes < 60:
            return f"{minutes} min ago"
        return f"{minutes // 60}h {minutes % 60}m ago"
    except Exception:
        return iso_ts

# ---------------------------------------------------------------------------
# Comparison tab helpers
# ---------------------------------------------------------------------------

def _fmt_scores(scores: dict | None) -> str:
    if not scores:
        return "Scoring unavailable"
    parts = []
    labels = [("F", "faithfulness"), ("R", "answer_relevancy"), ("P", "context_precision")]
    for short, key in labels:
        v = scores.get(key)
        parts.append(f"{short}: {v:.2f}" if v is not None else f"{short}: —")
    return "  |  ".join(parts)


def _fmt_result_card(result: dict, scores: dict | None = None) -> str:
    if result.get("error"):
        return f"**{result['config_name']}**\n\n❌ {result['error']}"

    answer = result.get("answer", "").strip() or "(no answer)"
    latency = result.get("latency", 0)
    score_str = _fmt_scores(scores)
    sources = ", ".join(result.get("sources", [])) or "—"

    return (
        f"**{result['config_name']}**\n"
        f"_{result.get('description', '')}_\n\n"
        f"{answer}\n\n"
        f"---\n"
        f"⏱ {latency:.1f}s  |  {score_str}\n"
        f"📄 Sources: {sources}"
    )


def run_comparison(query: str, model_id: str):
    """
    Generator — yields (card1, card2, card3, card4) tuples as configs complete.
    Gradio streams these to the UI for progressive disclosure.
    """
    from src.corpus import is_ready
    from src.inference import run_all_configs
    from src.evaluation import score as ragas_score

    if not query or not query.strip():
        msg = "Please enter a query."
        yield msg, msg, msg, msg
        return

    if not is_ready():
        msg = "⚙️ Corpus not built yet. Run build_corpus.ipynb in Google Colab first."
        yield msg, msg, msg, msg
        return

    # Initialise all 4 placeholders as "running"
    placeholders = {i: f"**{CONFIG_NAMES[i]}**\n\n⏳ Running..." for i in range(1, 5)}
    yield tuple(placeholders[i] for i in range(1, 5))

    completed: dict[int, dict] = {}

    for result in run_all_configs(query.strip(), model_id):
        cid = result["config_id"]
        completed[cid] = result

        # Score this result (non-blocking from the UI perspective — runs inline)
        scores = None
        if not result.get("error") and result.get("answer"):
            try:
                scores = ragas_score(query, result["answer"], result.get("context_chunks", []))
            except Exception as e:
                logger.warning("Scoring failed for config %d: %s", cid, e)

        placeholders[cid] = _fmt_result_card(result, scores)
        yield tuple(placeholders[i] for i in range(1, 5))

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

CSS = """
.raglens-header { text-align: center; padding: 1rem 0; }
.config-card { background: #1F2937; border-radius: 8px; padding: 1rem; min-height: 220px; }
.drift-alert { color: #F87171; font-weight: bold; }
.score-row { font-family: monospace; font-size: 0.85rem; }
"""

with gr.Blocks(title=APP_TITLE, theme=gr.themes.Soft(), css=CSS) as demo:

    gr.HTML(f"""
    <div class="raglens-header">
        <h1>🔍 {APP_TITLE}</h1>
        <p style="color:#9CA3AF">
            4 RAG configurations · LLM-as-judge scoring · 7-day monitoring dashboard<br>
            Corpus: {CORPUS_DESCRIPTION} · Evaluation judge: Groq llama-3.1-8b-instant (free tier)<br>
            <small>Built by <strong>Sridhar Malladi</strong> ·
            RAGLens runs on free-tier infrastructure — architecture over compute.</small>
        </p>
        <p style="color:#6B7280;font-size:0.8rem">
            ⚡ First query downloads ~3 GB of model weights — expect 3–5 min on cold start.
            Subsequent queries are fast.
        </p>
    </div>
    """)

    # ── Tab 1: Monitoring ────────────────────────────────────────────────────
    with gr.Tab("📊 Live Monitoring"):
        with gr.Row():
            last_run_box = gr.Textbox(label="Last run", interactive=False, scale=1)
            next_run_box = gr.Textbox(label="Next run", interactive=False, scale=1)
            refresh_btn = gr.Button("🔄 Refresh", scale=0)

        drift_box = gr.Textbox(
            label="Drift alerts (10%+ faithfulness drop in 24h)",
            interactive=False,
            visible=True,
            elem_classes=["drift-alert"],
        )

        faith_chart = gr.Plot(label="Faithfulness score (7 days, 4 configs)")
        latency_chart = gr.Plot(label="Generation latency — seconds (7 days)")

        gr.Markdown(
            "Each line = one RAG config running the same 3 fixed queries every hour. "
            "Monitoring writes are isolated from user traffic — scores reflect only the scheduled job."
        )

        def _refresh_monitoring():
            faith, latency, last, nxt, drift = _build_monitoring_charts()
            return faith, latency, last, nxt, drift

        refresh_btn.click(
            _refresh_monitoring,
            outputs=[faith_chart, latency_chart, last_run_box, next_run_box, drift_box],
        )

        # Load charts on tab open
        demo.load(
            _refresh_monitoring,
            outputs=[faith_chart, latency_chart, last_run_box, next_run_box, drift_box],
        )

    # ── Tab 2: Try It ────────────────────────────────────────────────────────
    with gr.Tab("🧪 Try It Yourself"):
        gr.Markdown(f"**Corpus:** {CORPUS_DESCRIPTION}")

        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=AVAILABLE_MODELS,
                value=DEFAULT_MODEL,
                label="Model",
                scale=2,
            )

        gr.Markdown("**Suggested queries** (click to fill):")
        with gr.Row():
            btns = [gr.Button(q, size="sm") for q in SUGGESTED_QUERIES]

        query_input = gr.Textbox(
            label="Your query (max 200 chars)",
            placeholder="Ask anything about RAG, retrieval, or LLM evaluation...",
            max_lines=2,
        )
        run_btn = gr.Button("▶ Run all 4 configs", variant="primary")

        gr.Markdown("---")
        gr.Markdown("### Results")
        gr.Markdown(
            "_Results appear as each config completes. "
            "RAGAS scores (F=faithfulness, R=relevancy, P=precision) load after generation._"
        )

        with gr.Row():
            card1 = gr.Markdown(elem_classes=["config-card"])
            card2 = gr.Markdown(elem_classes=["config-card"])
        with gr.Row():
            card3 = gr.Markdown(elem_classes=["config-card"])
            card4 = gr.Markdown(elem_classes=["config-card"])

        outputs = [card1, card2, card3, card4]

        # Wire suggested query buttons
        for btn, q in zip(btns, SUGGESTED_QUERIES):
            btn.click(lambda _q=q: _q, outputs=query_input)

        run_btn.click(
            run_comparison,
            inputs=[query_input, model_dropdown],
            outputs=outputs,
        )
        query_input.submit(
            run_comparison,
            inputs=[query_input, model_dropdown],
            outputs=outputs,
        )

    # ── About ────────────────────────────────────────────────────────────────
    with gr.Tab("ℹ️ About"):
        gr.Markdown(f"""
## What is RAGLens?

RAGLens benchmarks **4 RAG configurations** on the same query — live, with RAGAS scores:

| # | Config | What it tests |
|---|---|---|
| 1 | Baseline | No retrieval — pure model knowledge |
| 2 | Base + RAG | Dense retrieval (BGE + FAISS) |
| 3 | FT + RAG | Fine-tuned model + dense retrieval |
| 4 | FT + Optimal RAG | Fine-tuned model + hybrid retrieval + reranking |

Reading left-to-right shows what retrieval adds. Top-to-bottom shows what fine-tuning adds.

## Free-Stack Design

RAGLens runs at **zero marginal cost**:
- **Hosting:** HuggingFace Spaces CPU Basic (free)
- **Generation:** Local in-process models (Qwen2.5-1.5B, Phi-3.5-mini, Llama-3.2-3B, Gemma-2-2B)
- **Evaluation:** Groq API free tier (llama-3.1-8b-instant as RAGAS judge, 14,400 req/day)
- **Storage:** SQLite

This is not a compromise — it proves that **production-aware RAG architecture matters more than compute budget.** The same design scales identically to LLaMA 3.1 8B or any larger model.

## Monitoring

The background job runs every hour, evaluating the same 3 fixed queries through all 4 configs.
User traffic never pollutes monitoring data — writes are strictly separated by design.

Built by **Sridhar Malladi** | [GitHub](https://github.com/sridharmalladi)
        """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
