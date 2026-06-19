"""
Scheduled evaluation cycle — the only writer to the monitoring DB.
Called by scheduler.py on an hourly cron. Never called from user-facing paths.
"""

import logging
from datetime import datetime

from config import GROQ_GENERATION_MODEL, MONITORING_QUERIES

logger = logging.getLogger(__name__)


def run_evaluation_cycle() -> None:
    from src.corpus import is_ready
    from src.inference import run_all_configs_blocking
    from src.evaluation import score
    from src.storage import write_run, prune_old

    if not is_ready():
        logger.warning("Corpus not ready — skipping monitoring cycle")
        return

    logger.info("Starting monitoring cycle at %s", datetime.utcnow().isoformat())
    total_written = 0

    for query in MONITORING_QUERIES:
        logger.info("Evaluating query: %s", query[:60])
        try:
            results = run_all_configs_blocking(query)
        except Exception as e:
            logger.error("run_all_configs_blocking failed: %s", e)
            continue

        for config_id, result in results.items():
            if result.get("error"):
                logger.warning("Config %d errored: %s", config_id, result["error"])
                continue

            try:
                scores = score(query, result.get("answer", ""), result.get("context_chunks", []))
            except Exception as e:
                logger.error("Scoring failed for config %d: %s", config_id, e)
                scores = {"faithfulness": None, "answer_relevancy": None, "context_precision": None}

            try:
                write_run(
                    model=GROQ_GENERATION_MODEL,
                    config_id=config_id,
                    config_name=result["config_name"],
                    query=query,
                    scores=scores,
                    latency_s=result.get("latency", 0.0),
                )
                total_written += 1
            except Exception as e:
                logger.error("DB write failed for config %d: %s", config_id, e)

    try:
        pruned = prune_old()
        logger.info("Pruned %d old rows", pruned)
    except Exception as e:
        logger.error("Prune failed: %s", e)

    logger.info("Monitoring cycle complete — wrote %d rows.", total_written)
