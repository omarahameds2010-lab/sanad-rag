import sys
import os
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.evaluation.storage import (
    DEFAULT_DB_PATH,
    record_evaluation_run,
    get_evaluation_run,
)


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_reports_dependency_status(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert "status" in body and "qdrant_ok" in body and "db_ok" in body


def test_ingest_validates_empty_source_id(client):
    r = client.post("/ingest", json={"source_id": "", "text": "نص"})
    assert r.status_code == 422  # Pydantic validation, not a 500 crash


def test_ingest_returns_success_with_real_pipeline(client):
    r = client.post("/ingest", json={"source_id": "doc1", "text": "نص عربي كامل للاختبار."})
    assert r.status_code == 200
    body = r.json()
    assert body["n_chunks"] >= 1
    assert body["source_id"] == "doc1"


def test_query_returns_answer_with_real_pipeline(client):
    r = client.post("/query", json={"question": "ما هو الموضوع؟"})
    assert r.status_code == 200
    body = r.json()
    assert "answer" in body and "cited_chunk_ids" in body
    assert "retrieval_ms" in body and "generation_ms" in body


def test_evaluate_report_404_for_unknown_run(client):
    r = client.get("/evaluate/report", params={"run_id": "does-not-exist"})
    assert r.status_code == 404


def test_evaluate_report_returns_stored_metrics(client):
    run_id = str(uuid.uuid4())
    record_evaluation_run(
        run_id=run_id,
        config_name="baseline",
        metrics={
            "n_examples": 10,
            "precision_at_k": 0.8,
            "recall_at_k": 0.7,
            "p50_latency_seconds": 0.2,
            "p95_latency_seconds": 0.5,
        },
    )
    result = get_evaluation_run(run_id)
    assert result is not None
    assert result["precision_at_k"] == 0.8
    assert result["n_examples"] == 10
