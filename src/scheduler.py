"""
APScheduler setup. Runs monitoring.run_evaluation_cycle() on a fixed interval.
Call start() once at app startup.

On boot, if the DB has no recent run, one cycle is also scheduled a couple of
minutes out so the trend charts are not empty right after a deploy. The regular
interval then takes over.
"""

import logging
from datetime import datetime, timedelta

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.date import DateTrigger

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


def _data_is_stale() -> bool:
    """True when there is no monitoring run within the last interval."""
    try:
        from src.storage import read_last_run_time
        last = read_last_run_time()
    except Exception:
        return True
    if not last:
        return True
    try:
        ts = datetime.fromisoformat(last.replace("Z", ""))
        return datetime.utcnow() - ts > timedelta(hours=MONITORING_INTERVAL_HOURS)
    except Exception:
        return True


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

    if _data_is_stale():
        _scheduler.add_job(
            _job,
            trigger=DateTrigger(run_date=datetime.utcnow() + timedelta(minutes=2)),
            id="monitoring_kickoff",
            name="RAGLens startup evaluation",
            replace_existing=True,
            misfire_grace_time=600,
        )
        logger.info("Monitoring data stale, one kickoff cycle scheduled in 2 min")

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
