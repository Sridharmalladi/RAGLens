"""
APScheduler setup. Runs monitoring.run_evaluation_cycle() on a fixed interval.
Call start() once at app startup.
"""

import logging
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from config import MONITORING_INTERVAL_HOURS

logger = logging.getLogger(__name__)

_scheduler: BackgroundScheduler | None = None


def _job():
    try:
        from src.monitoring import run_evaluation_cycle
        run_evaluation_cycle()
    except Exception as e:
        logger.error("Monitoring job crashed: %s", e)
        # APScheduler retries on the next tick, no manual restart needed


def start() -> BackgroundScheduler:
    global _scheduler
    if _scheduler and _scheduler.running:
        return _scheduler

    _scheduler = BackgroundScheduler(timezone="UTC")
    _scheduler.add_job(
        _job,
        trigger=IntervalTrigger(hours=MONITORING_INTERVAL_HOURS),
        id="monitoring_cycle",
        name="RAGLens scheduled evaluation",
        replace_existing=True,
        misfire_grace_time=300,  # allow up to 5 min late start
    )
    _scheduler.start()
    logger.info("Scheduler started, monitoring runs every %dh", MONITORING_INTERVAL_HOURS)
    return _scheduler


def next_run_time() -> str | None:
    global _scheduler
    if not _scheduler:
        return None
    job = _scheduler.get_job("monitoring_cycle")
    if job and job.next_run_time:
        return job.next_run_time.strftime("%Y-%m-%d %H:%M UTC")
    return None
