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

# Test API key
TEST_API_KEY = "demo-key-123"


@pytest.fixture
def client():
    from src.security import rate_limiter

    # Rate limiting is per-process in-memory state: reset between tests so
    # each test gets a full bucket regardless of collection order/size.
    rate_limiter.buckets.clear()
    with TestClient(app) as c:
        # Add API key header to all requests
        c.headers.update({"X-API-Key": TEST_API_KEY})
        yield c


@pytest.fixture
def client_no_auth():
    """Client without API key for testing auth failures."""
    from src.security import rate_limiter

    rate_limiter.buckets.clear()
    with TestClient(app) as c:
        yield c


def test_health_reports_dependency_status(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert "status" in body and "qdrant_ok" in body and "db_ok" in body


def test_health_no_auth(client_no_auth):
    r = client_no_auth.get("/health")
    assert r.status_code == 200  # health endpoint doesn't require auth


def test_ingest_validates_empty_source_id(client):
    r = client.post("/ingest", json={"source_id": "", "text": "نص"})
    assert r.status_code == 422  # Pydantic validation, not a 500 crash


def test_ingest_returns_success_with_real_pipeline(client):
    r = client.post("/ingest", json={"source_id": "doc1", "text": "نص عربي كامل للاختبار."})
    assert r.status_code == 200
    body = r.json()
    assert body["n_chunks"] >= 1
    assert body["source_id"] == "doc1"


def test_ingest_requires_auth(client_no_auth):
    r = client_no_auth.post("/ingest", json={"source_id": "doc1", "text": "نص"})
    assert r.status_code == 401


def test_query_returns_answer_with_real_pipeline(client):
    r = client.post("/query", json={"question": "ما هو الموضوع؟"})
    assert r.status_code == 200
    body = r.json()
    assert "answer" in body and "cited_chunk_ids" in body
    assert "retrieval_ms" in body and "generation_ms" in body


def test_query_requires_auth(client_no_auth):
    r = client_no_auth.post("/query", json={"question": "ما هو الموضوع؟"})
    assert r.status_code == 401


def test_extractive_returns_passages_without_llm(client):
    # Ingest first so the extractive query has something to find.
    r = client.post("/ingest", json={"source_id": "ext-doc1", "text": "النيل أطول نهر في أفريقيا. القاهرة عاصمة مصر."})
    assert r.status_code == 200

    r = client.post("/query/extractive", json={"question": "ما هو أطول نهر؟", "top_k": 2})
    assert r.status_code == 200
    body = r.json()
    assert len(body["passages"]) >= 1
    assert body["cited_chunk_ids"] == [p["chunk_id"] for p in body["passages"]]
    assert all(p["text"] for p in body["passages"])
    assert "generation_ms" not in body
    assert body["retrieval_ms"] >= 0


def test_extractive_validates_empty_question(client):
    r = client.post("/query/extractive", json={"question": ""})
    assert r.status_code == 422


def test_extractive_requires_auth(client_no_auth):
    r = client_no_auth.post("/query/extractive", json={"question": "سؤال"})
    assert r.status_code == 401


def test_extractive_branch_filter_scopes_to_science(client):
    r = client.post("/ingest", json={
        "source_id": "branch-nahw-doc",
        "text": "الفاعل مرفوع دائما والمفعول منصوب.",
        "metadata": {"branch": "nahw"},
    })
    assert r.status_code == 200
    r = client.post("/ingest", json={
        "source_id": "branch-adab-doc",
        "text": "قال المتنبي الخيل والليل والبيداء تعرفني.",
        "metadata": {"branch": "adab"},
    })
    assert r.status_code == 200

    r = client.post("/query/extractive", json={"question": "المرفوع", "top_k": 5, "branch": "nahw"})
    assert r.status_code == 200
    body = r.json()
    assert body["branch"] == "nahw"
    assert len(body["passages"]) >= 1
    # Every returned passage carries the requested branch (the shared
    # index also holds gen_fixed_irab_v1, which is branch nahw too).
    for p in body["passages"]:
        assert p["branch"] == "nahw"

    r = client.post("/query/extractive", json={"question": "المرفوع", "top_k": 5, "branch": "no-such-branch"})
    assert r.status_code == 200
    assert r.json()["passages"] == []


def test_extractive_source_ids_filter(client):
    r = client.post("/query/extractive", json={
        "question": "المرفوع", "top_k": 5, "source_ids": ["branch-nahw-doc"],
    })
    assert r.status_code == 200
    body = r.json()
    assert body["source_ids"] == ["branch-nahw-doc"]
    assert len(body["passages"]) >= 1
    assert all(p["source_id"] == "branch-nahw-doc" for p in body["passages"])


def test_grounded_query_accepts_branch_filter(client):
    r = client.post("/query", json={"question": "المرفوع", "top_k": 2, "branch": "nahw"})
    assert r.status_code == 200
    body = r.json()
    assert body["branch"] == "nahw"
    assert "answer" in body and "retrieval_ms" in body


def test_evaluate_report_404_for_unknown_run(client):
    r = client.get("/evaluate/report", params={"run_id": "does-not-exist"})
    assert r.status_code == 404


def test_evaluate_report_404_requires_auth(client_no_auth):
    r = client_no_auth.get("/evaluate/report", params={"run_id": "does-not-exist"})
    assert r.status_code == 401


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


def test_evaluate_report_requires_auth(client_no_auth):
    run_id = str(uuid.uuid4())
    record_evaluation_run(
        run_id=run_id,
        config_name="baseline",
        metrics={"n_examples": 1, "precision_at_k": 0.5, "recall_at_k": 0.5},
    )
    r = client_no_auth.get("/evaluate/report", params={"run_id": run_id})
    assert r.status_code == 401