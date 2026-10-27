"""
SQLite storage for structured bookkeeping — deliberately separate from
Qdrant. See docs/architecture.md §5 for why two stores exist.

This module owns exactly the three tables defined in the architecture doc.
No ORM for the MVP — the schema is small and stable enough that raw SQL is
more transparent than adding an ORM dependency.
"""

import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_DB_PATH = Path("data/sanad.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS ingested_documents (
    source_id TEXT PRIMARY KEY,
    ingested_at TIMESTAMP,
    n_chunks INTEGER,
    embedding_model TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    source_id TEXT REFERENCES ingested_documents(source_id),
    start_sentence INTEGER,
    end_sentence INTEGER
);

CREATE TABLE IF NOT EXISTS evaluation_runs (
    run_id TEXT PRIMARY KEY,
    config_name TEXT,
    started_at TIMESTAMP,
    n_examples INTEGER,
    precision_at_k REAL,
    recall_at_k REAL,
    p50_latency_seconds REAL,
    p95_latency_seconds REAL,
    embedding_model TEXT,
    llm_model TEXT,
    chunker_type TEXT,
    reranker_used TEXT,
    query_expansion_enabled INTEGER,
    is_dev_only_run INTEGER
);
"""

# Experimental axes added after the table already existed in the wild.
# SQLite has no ADD COLUMN IF NOT EXISTS, so migrate by inspection.
EVALUATION_RUN_AXES = (
    ("embedding_model", "TEXT"),
    ("llm_model", "TEXT"),
    ("chunker_type", "TEXT"),
    ("reranker_used", "TEXT"),
    ("query_expansion_enabled", "INTEGER"),
    ("is_dev_only_run", "INTEGER"),
)


def _ensure_evaluation_columns(conn) -> None:
    existing = {row[1] for row in conn.execute("PRAGMA table_info(evaluation_runs)")}
    for name, coltype in EVALUATION_RUN_AXES:
        if name not in existing:
            conn.execute(f"ALTER TABLE evaluation_runs ADD COLUMN {name} {coltype}")


@contextmanager
def get_connection(db_path: Path = DEFAULT_DB_PATH):
    """
    Bug found while testing Phase 6 standalone (outside the FastAPI app):
    previously, table creation only happened via the API's lifespan hook,
    so calling storage functions directly from a script (like the
    evaluation harness) crashed with "no such table". Fix: make the
    connection self-initializing — CREATE TABLE IF NOT EXISTS is cheap
    and idempotent, so running it on every connection removes the hidden
    dependency on the API process having started first.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    _ensure_evaluation_columns(conn)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA)


def record_ingested_document(
    source_id: str, n_chunks: int, embedding_model: str, db_path: Path = DEFAULT_DB_PATH
) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO ingested_documents "
            "(source_id, ingested_at, n_chunks, embedding_model) VALUES (?, ?, ?, ?)",
            (source_id, datetime.now(UTC).isoformat(), n_chunks, embedding_model),
        )


def delete_ingested_document(source_id: str, db_path: Path = DEFAULT_DB_PATH) -> None:
    """Remove bookkeeping rows when a source is quarantined from the index."""
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM chunks WHERE source_id = ?", (source_id,))
        conn.execute("DELETE FROM ingested_documents WHERE source_id = ?", (source_id,))


def _as_int(value) -> int | None:
    """SQLite has no BOOLEAN: persist flags as 0/1, keep missing as NULL."""
    if value is None:
        return None
    return 1 if bool(value) else 0


def record_evaluation_run(
    run_id: str,
    config_name: str,
    metrics: dict,
    db_path: Path = DEFAULT_DB_PATH,
) -> None:
    """Persist an evaluation run so benchmark_results.md is generated from
    real stored data, never hand-typed numbers."""
    with get_connection(db_path) as conn:
        conn.execute(
            """INSERT OR REPLACE INTO evaluation_runs
               (run_id, config_name, started_at, n_examples, precision_at_k,
                recall_at_k, p50_latency_seconds, p95_latency_seconds,
                embedding_model, llm_model, chunker_type, reranker_used,
                query_expansion_enabled, is_dev_only_run)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                config_name,
                datetime.now(UTC).isoformat(),
                metrics.get("n_examples"),
                metrics.get("precision_at_k"),
                metrics.get("recall_at_k"),
                metrics.get("p50_latency_seconds"),
                metrics.get("p95_latency_seconds"),
                metrics.get("embedding_model"),
                metrics.get("llm_model"),
                metrics.get("chunker_type"),
                metrics.get("reranker_used"),
                _as_int(metrics.get("query_expansion_enabled")),
                _as_int(metrics.get("is_dev_only_run")),
            ),
        )


def get_evaluation_run(run_id: str, db_path: Path = DEFAULT_DB_PATH) -> dict | None:
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM evaluation_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        return dict(row) if row else None
