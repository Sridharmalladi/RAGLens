"""
SQLite read/write abstraction.
Schema: runs(id, timestamp, model, config_id, config_name, query,
             faithfulness, answer_relevancy, context_precision, latency_s)
Only monitoring.py writes here. inference.py never calls this module.
"""

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta

from config import DB_PATH, RETENTION_DAYS

logger = logging.getLogger(__name__)

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS runs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        TEXT    NOT NULL,
    model            TEXT    NOT NULL,
    config_id        INTEGER NOT NULL,
    config_name      TEXT    NOT NULL,
    query            TEXT    NOT NULL,
    faithfulness     REAL,
    answer_relevancy REAL,
    context_precision REAL,
    latency_s        REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_timestamp ON runs(timestamp);
CREATE INDEX IF NOT EXISTS idx_runs_config    ON runs(config_id);
"""


@contextmanager
def _conn():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def init_db() -> None:
    """Create tables if they don't exist. Safe to call multiple times."""
    with _conn() as con:
        con.executescript(CREATE_TABLE)
    logger.info("SQLite DB ready at %s", DB_PATH)


def write_run(
    model: str,
    config_id: int,
    config_name: str,
    query: str,
    scores: dict,
    latency_s: float,
    timestamp: str | None = None,
) -> None:
    ts = timestamp or datetime.utcnow().isoformat()
    with _conn() as con:
        con.execute(
            """
            INSERT INTO runs
                (timestamp, model, config_id, config_name, query,
                 faithfulness, answer_relevancy, context_precision, latency_s)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts, model, config_id, config_name, query,
                scores.get("faithfulness"),
                scores.get("answer_relevancy"),
                scores.get("context_precision"),
                latency_s,
            ),
        )


def read_recent(days: int = 7) -> list[dict]:
    """Return all runs from the last `days` days, newest first."""
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM runs WHERE timestamp >= ? ORDER BY timestamp DESC",
            (cutoff,),
        ).fetchall()
    return [dict(row) for row in rows]


def read_last_run_time() -> str | None:
    """Return the timestamp of the most recent monitoring run, or None."""
    with _conn() as con:
        row = con.execute(
            "SELECT timestamp FROM runs ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
    return row["timestamp"] if row else None


def prune_old(days: int = RETENTION_DAYS) -> int:
    """Delete rows older than `days`. Returns number of rows deleted."""
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
    with _conn() as con:
        cursor = con.execute("DELETE FROM runs WHERE timestamp < ?", (cutoff,))
    deleted = cursor.rowcount
    if deleted:
        logger.info("Pruned %d old rows (older than %d days)", deleted, days)
    return deleted


def detect_drift(threshold: float, hours: int = 24) -> list[dict]:
    """
    Return configs whose faithfulness dropped by `threshold` or more
    compared to the prior 24-hour window.
    """
    now = datetime.utcnow()
    recent_cutoff = (now - timedelta(hours=hours)).isoformat()
    prior_cutoff = (now - timedelta(hours=hours * 2)).isoformat()

    with _conn() as con:
        recent = con.execute(
            """
            SELECT config_id, config_name, AVG(faithfulness) as avg_f
            FROM runs WHERE timestamp >= ? AND faithfulness IS NOT NULL
            GROUP BY config_id
            """,
            (recent_cutoff,),
        ).fetchall()

        prior = con.execute(
            """
            SELECT config_id, AVG(faithfulness) as avg_f
            FROM runs WHERE timestamp >= ? AND timestamp < ? AND faithfulness IS NOT NULL
            GROUP BY config_id
            """,
            (prior_cutoff, recent_cutoff),
        ).fetchall()

    prior_map = {r["config_id"]: r["avg_f"] for r in prior}
    alerts = []
    for r in recent:
        cid = r["config_id"]
        if cid in prior_map and prior_map[cid] is not None and r["avg_f"] is not None:
            drop = prior_map[cid] - r["avg_f"]
            if drop >= threshold:
                alerts.append({
                    "config_id": cid,
                    "config_name": r["config_name"],
                    "drop": round(drop, 4),
                    "recent_avg": round(r["avg_f"], 4),
                    "prior_avg": round(prior_map[cid], 4),
                })
    return alerts
