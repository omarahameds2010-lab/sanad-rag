import os
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from src.embeddings.base import HashEmbedding
from src.ingestion.pipeline import ingest_document
from src.retrieval.query_pipeline import answer_question
from src.generation.llm_client import FakeLLM


@pytest.fixture
def isolated_qdrant(monkeypatch, tmp_path):
    from src.retrieval.vector_store import reset_client_for_testing

    reset_client_for_testing()
    path = str(tmp_path / "qdrant_local")
    monkeypatch.setattr("src.retrieval.vector_store.DEFAULT_LOCAL_PATH", path)
    monkeypatch.delenv("QDRANT_URL", raising=False)
    yield
    reset_client_for_testing()


def test_answer_question_returns_full_contract(isolated_qdrant):
    embedding_model = HashEmbedding(dims=32)
    ingest_document(
        source_id="doc-" + str(uuid.uuid4()),
        raw_text="القاهرة هي عاصمة مصر. النيل هو أطول نهر في أفريقيا.",
        embedding_model=embedding_model,
    )

    result = answer_question(
        question="ما هي عاصمة مصر؟",
        embedding_model=embedding_model,
        primary_llm=FakeLLM(),
    )

    assert "answer" in result
    assert "cited_chunk_ids" in result
    assert result["retrieval_ms"] >= 0
    assert result["generation_ms"] >= 0
    assert len(result["cited_chunk_ids"]) >= 1


def test_answer_question_with_no_ingested_data_does_not_crash(isolated_qdrant):
    embedding_model = HashEmbedding(dims=32)
    from src.retrieval.vector_store import get_client, ensure_collection

    # ensure the collection exists (empty) so retrieval has something to
    # query against, mirroring what /health checks for
    ensure_collection(get_client(), dims=embedding_model.dimensions)

    result = answer_question(
        question="سؤال عن بيانات غير موجودة",
        embedding_model=embedding_model,
        primary_llm=FakeLLM(),
    )
    assert result["answer"]  # FakeLLM's "no info" fallback answer
    assert result["cited_chunk_ids"] == []


def test_primary_llm_failure_falls_back(isolated_qdrant):
    """Verifies the reliability strategy from architecture.md §6: if the
    primary LLM raises, the fallback LLM is used instead of crashing."""

    class BrokenLLM(FakeLLM):
        name = "broken-llm"

        def generate(self, question, context_chunks):
            raise RuntimeError("simulated primary LLM outage")

    embedding_model = HashEmbedding(dims=32)
    ingest_document(
        source_id="doc-" + str(uuid.uuid4()),
        raw_text="بيانات اختبار للـ fallback.",
        embedding_model=embedding_model,
    )

    result = answer_question(
        question="اختبار",
        embedding_model=embedding_model,
        primary_llm=BrokenLLM(),
        fallback_llm=FakeLLM(),
    )
    assert "fallback" in result["model_used"]
    assert result["answer"]  # fallback still produced an answer, no crash
