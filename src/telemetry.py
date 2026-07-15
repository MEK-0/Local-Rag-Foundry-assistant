import json
import sqlite3
import time
from contextlib import contextmanager
from typing import Dict, Any, Optional

from src.db import get_db_connection

# --------------------------------------------------------------------------- #
# Structured request logging
#
# Previously, telemetry was just an in-memory dict built manually inside
# rag_pipeline.py and returned to the UI - useful for the current request,
# but nothing was persisted. This adds a query_log table so past requests
# (latency per stage, chunk counts, hop counts, success/failure) can be
# inspected after the fact, which is what "structured logging with request
# tracing" in the roadmap actually means.
# --------------------------------------------------------------------------- #

def init_telemetry_table() -> None:
    """Creates the query_log table if it doesn't exist. Safe to call on
    every startup alongside db.init_db()."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS query_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            query TEXT NOT NULL,
            advanced_mode INTEGER DEFAULT 1,
            hop_count INTEGER,
            chunk_count INTEGER,
            top_rerank_score REAL,
            generation_failed INTEGER DEFAULT 0,
            telemetry_json TEXT
        )
    """)
    conn.commit()
    conn.close()


@contextmanager
def timed_stage():
    """
    Context manager that measures elapsed time in milliseconds.
    Replaces the repeated start_time = time.perf_counter() / round(...)
    boilerplate that was duplicated across every phase in rag_pipeline.py.

    Usage:
        with timed_stage() as t:
            do_work()
        telemetry["stage_time_ms"] = t.elapsed_ms
    """
    class _Timer:
        elapsed_ms: float = 0.0

    timer = _Timer()
    start = time.perf_counter()
    try:
        yield timer
    finally:
        timer.elapsed_ms = round((time.perf_counter() - start) * 1000, 1)


def log_query_event(
    query: str,
    telemetry: Dict[str, Any],
    advanced_mode: bool = True,
    hop_count: Optional[int] = None,
    chunk_count: Optional[int] = None,
    top_rerank_score: Optional[float] = None,
    generation_failed: bool = False,
) -> None:
    """
    Persists one query's outcome to query_log. Called at the end of
    process_chat_query() so past requests can be inspected later
    (e.g. to spot queries that repeatedly fail generation, or stages
    that are consistently slow) without needing external log aggregation.

    Failures here are swallowed rather than raised - telemetry logging
    must never break the actual chat response.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO query_log
                (query, advanced_mode, hop_count, chunk_count, top_rerank_score, generation_failed, telemetry_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            query, int(advanced_mode), hop_count, chunk_count, top_rerank_score,
            int(generation_failed), json.dumps(telemetry),
        ))
        conn.commit()
        conn.close()
    except sqlite3.Error:
        pass  # telemetry logging must never break the chat response


def get_recent_queries(limit: int = 50) -> list:
    """Returns the most recent logged queries - useful for a future
    admin/debug view or for run_eval.py to compare historical runs."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM query_log ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]