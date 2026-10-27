import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from src.embeddings.base import HashEmbedding
from src.evaluation.harness import run_evaluation
from src.evaluation.storage import get_evaluation_run
from src.generation.llm_client import FakeLLM

GOLDEN_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "golden_dataset", "sample_golden.json"
)


@pytest.fixture
def isolated_env(monkeypatch, tmp_path):
    from src.retrieval.vector_store import reset_client_for_testing

    reset_client_for_testing()
    monkeypatch.setattr(
        "src.retrieval.vector_store.DEFAULT_LOCAL_PATH", str(tmp_path / "qdrant_local")
    )
    monkeypatch.setattr("src.evaluation.storage.DEFAULT_DB_PATH", tmp_path / "sanad.db")
    monkeypatch.delenv("QDRANT_URL", raising=False)
    yield
    reset_client_for_testing()


def test_run_evaluation_produces_real_metrics(isolated_env):
    metrics = run_evaluation(
        golden_dataset_path=GOLDEN_PATH,
        embedding_model=HashEmbedding(dims=64),
        llm=FakeLLM(),
    )

    assert metrics["n_examples"] == 4
    assert 0.0 <= metrics["precision_at_k"] <= 1.0
    assert 0.0 <= metrics["recall_at_k"] <= 1.0
    assert metrics["is_dev_only_run"] is True  # honest flag, not hidden


def test_evaluation_run_is_persisted_and_retrievable(isolated_env):
    metrics = run_evaluation(
        golden_dataset_path=GOLDEN_PATH,
        embedding_model=HashEmbedding(dims=64),
        llm=FakeLLM(),
        config_name="test-config",
    )

    stored = get_evaluation_run(metrics["run_id"])
    assert stored is not None
    assert stored["config_name"] == "test-config"
    assert stored["n_examples"] == 4


def test_recall_is_perfect_when_corpus_is_small_and_topk_generous(isolated_env):
    """Sanity check on the sample dataset itself: with only 3 tiny source
    documents and top_k=5, every relevant chunk should be retrievable —
    if this ever drops, the chunker/retriever broke, not the dataset."""
    metrics = run_evaluation(
        golden_dataset_path=GOLDEN_PATH,
        embedding_model=HashEmbedding(dims=64),
        llm=FakeLLM(),
        top_k=5,
    )
    assert metrics["recall_at_k"] == 1.0


def test_stored_run_keeps_experimental_axes(isolated_env):
    """Comparisons must be reproducible from the DB, not just Markdown:
    embedding/llm/chunker/reranker/expansion/dev-flag survive the roundtrip."""
    from src.evaluation.storage import record_evaluation_run

    run_id = "axes-probe"
    record_evaluation_run(
        run_id=run_id,
        config_name="axes",
        metrics={
            "n_examples": 4,
            "precision_at_k": 0.5,
            "recall_at_k": 0.75,
            "p50_latency_seconds": 0.01,
            "p95_latency_seconds": 0.02,
            "embedding_model": "hash-embedding-DEV-ONLY",
            "llm_model": "fake-llm-DEV-ONLY",
            "chunker_type": "arabic",
            "reranker_used": "BAAI/bge-reranker-base",
            "query_expansion_enabled": True,
            "is_dev_only_run": True,
        },
    )
    stored = get_evaluation_run(run_id)
    assert stored["embedding_model"] == "hash-embedding-DEV-ONLY"
    assert stored["llm_model"] == "fake-llm-DEV-ONLY"
    assert stored["chunker_type"] == "arabic"
    assert stored["reranker_used"] == "BAAI/bge-reranker-base"
    assert stored["query_expansion_enabled"] == 1
    assert stored["is_dev_only_run"] == 1


def test_legacy_db_without_axes_still_records(tmp_path):
    """Migration: databases created before the axes existed gain the
    columns on first connect instead of crashing the INSERT."""
    import sqlite3

    from src.evaluation import storage

    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """CREATE TABLE evaluation_runs (
            run_id TEXT PRIMARY KEY, config_name TEXT, started_at TIMESTAMP,
            n_examples INTEGER, precision_at_k REAL, recall_at_k REAL,
            p50_latency_seconds REAL, p95_latency_seconds REAL)"""
    )
    conn.execute(
        "INSERT INTO evaluation_runs (run_id, config_name) VALUES ('old', 'baseline')"
    )
    conn.commit()
    conn.close()

    storage.record_evaluation_run(
        run_id="new", config_name="x", metrics={"n_examples": 1},
        db_path=db,
    )
    stored = storage.get_evaluation_run("new", db_path=db)
    assert stored is not None
    assert stored["reranker_used"] is None  # absent axes read back as NULL
    assert storage.get_evaluation_run("old", db_path=db)["config_name"] == "baseline"
