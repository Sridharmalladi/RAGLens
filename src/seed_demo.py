"""
Demo backfill for the monitoring charts.

Free hosting has no persistent disk: the real scheduled job only ever lands one
point before the instance sleeps and /tmp/raglens.db is wiped, so the trend
charts render as a single dot. When SEED_DEMO_DATA is set and the runs table is
empty, this writes ~8 days of plausible points so the charts show movement.

Seeded rows carry DEMO_MARKER in the query column. /api/monitoring reports
demo=true whenever any are present and the UI shows a "sample data" badge. Real
scheduled runs append alongside these and are not marked.
"""

import logging
import math
import random
from datetime import datetime, timedelta

from config import MONITORING_MODELS, CONFIG_NAMES, DEMO_MARKER

logger = logging.getLogger(__name__)

# Per-config baselines: (faithfulness, answer_relevancy, latency_s). Config 1
# (No RAG) answers without retrieved context, so faithfulness stays None — the
# same shape the real job produces and the faithfulness chart already skips.
_BASE = {
    1: (None, 0.56, 1.4),
    2: (0.71, 0.77, 2.6),
    3: (0.80, 0.83, 3.3),
    4: (0.88, 0.87, 4.6),
}

# 120b scores a touch higher and runs slower than 20b.
_MODEL_ADJ = {
    "openai/gpt-oss-20b":  (0.00, 1.0),
    "openai/gpt-oss-120b": (0.03, 1.7),
}


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def seed_if_empty() -> int:
    """Backfill demo rows when the table has no runs yet. Returns rows written."""
    from src.storage import read_last_run_time, write_run

    if read_last_run_time() is not None:
        return 0

    rng = random.Random(20260831)
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    start = now - timedelta(days=8)
    step = timedelta(hours=8)

    written = 0
    t, tick = start, 0
    while t <= now:
        wave = math.sin(tick / 3.0) * 0.03      # slow shared ripple
        trend = tick * 0.0015                   # gentle improvement over the window
        for model in MONITORING_MODELS:
            f_adj, lat_mult = _MODEL_ADJ.get(model, (0.0, 1.0))
            for cid, (base_f, base_r, base_lat) in _BASE.items():
                faith = None
                if base_f is not None:
                    faith = round(_clamp(base_f + f_adj + wave + trend
                                         + rng.uniform(-0.025, 0.025)), 4)
                rel = round(_clamp(base_r + f_adj + wave * 0.7 + trend
                                   + rng.uniform(-0.03, 0.03)), 4)
                lat = round(max(0.3, base_lat * lat_mult + rng.uniform(-0.4, 0.6)), 3)
                write_run(
                    model=model,
                    config_id=cid,
                    config_name=CONFIG_NAMES.get(cid, f"Config {cid}"),
                    query=DEMO_MARKER,
                    scores={"faithfulness": faith,
                            "answer_relevancy": rel,
                            "context_precision": None},
                    latency_s=lat,
                    timestamp=t.isoformat(),
                )
                written += 1
        t += step
        tick += 1

    logger.info("Seeded %d demo monitoring rows (8d history)", written)
    return written
