import os
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from src.embeddings.base import HashEmbedding
from src.ingestion.pipeline import ingest_document
from src.retrieval.query_pipeline import answer_question, retrieve_passages
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
        chunker_type="arabic",
    )

    assert "answer" in result
    assert "cited_chunk_ids" in result
    assert result["retrieval_ms"] >= 0
    assert result["generation_ms"] >= 0
    assert len(result["cited_chunk_ids"]) >= 1


def test_answer_question_with_no_ingested_data_does_not_crash(isolated_qdrant):
    embedding_model = HashEmbedding(dims=32)
    from src.retrieval.vector_store import get_client, ensure_collection, collection_for_chunker

    # ensure the collection exists (empty) so retrieval has something to
    # query against, mirroring what /health checks for
    ensure_collection(
        get_client(), dims=embedding_model.dimensions, collection=collection_for_chunker("arabic", embedding_model.dimensions)
    )

    result = answer_question(
        question="سؤال عن بيانات غير موجودة",
        embedding_model=embedding_model,
        primary_llm=FakeLLM(),
        chunker_type="arabic",
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
        chunker_type="arabic",
    )
    assert "fallback" in result["model_used"]
    assert result["answer"]  # fallback still produced an answer, no crash


def test_retrieve_passages_matches_answer_question_ranking(isolated_qdrant):
    """The extractive path must rank identically to the grounded path:
    same chunks, same order, no LLM involved."""
    embedding_model = HashEmbedding(dims=32)
    ingest_document(
        source_id="doc-" + str(uuid.uuid4()),
        raw_text="القاهرة هي عاصمة مصر. النيل هو أطول نهر في أفريقيا. الأزهر منارة العلم.",
        embedding_model=embedding_model,
    )

    chunks, retrieval_ms = retrieve_passages(
        question="ما هي عاصمة مصر؟",
        embedding_model=embedding_model,
        top_k=3,
        chunker_type="arabic",
    )
    grounded = answer_question(
        question="ما هي عاصمة مصر؟",
        embedding_model=embedding_model,
        primary_llm=FakeLLM(),
        top_k=3,
        chunker_type="arabic",
    )

    assert len(chunks) >= 1
    assert retrieval_ms >= 0
    assert [c["chunk_id"] for c in chunks] == grounded["retrieved_chunk_ids"]
    # FakeLLM cites the top retrieved chunk: extractive top == grounded citation
    assert chunks[0]["chunk_id"] in grounded["cited_chunk_ids"]


def test_retrieve_passages_empty_index_returns_empty(isolated_qdrant):
    embedding_model = HashEmbedding(dims=32)
    from src.retrieval.vector_store import get_client, ensure_collection, collection_for_chunker

    ensure_collection(
        get_client(), dims=embedding_model.dimensions, collection=collection_for_chunker("arabic", embedding_model.dimensions)
    )
    chunks, _ = retrieve_passages(
        question="سؤال بلا بيانات",
        embedding_model=embedding_model,
        chunker_type="arabic",
    )
    assert chunks == []


def test_infer_branch_taxonomy():
    from src.ingestion.branches import infer_branch

    assert infer_branch("gen_fixed_irab_v1") == "nahw"
    assert infer_branch("gen_sarf_tables_p1_v1") == "sarf"
    assert infer_branch("balagha_deep_v1") == "balagha"
    assert infer_branch("arood_qawafi_v1") == "arood"
    assert infer_branch("adab_3_v1") == "adab"
    assert infer_branch("gen_vocab_bank_v1") == "lugha"
    assert infer_branch("juzamma_tafsir_v1") == "quran"
    assert infer_branch("arbaeen_nabawiyya_v1") == "hadith"
    assert infer_branch("seera_nabaweya_v1") == "sira"
    assert infer_branch("reach_25k_v1") == "misc"
    assert infer_branch("") == "misc"


def test_branch_filter_scopes_to_science(isolated_qdrant):
    embedding_model = HashEmbedding(dims=32)
    ingest_document(
        source_id="sarf-book-" + str(uuid.uuid4()),
        raw_text="تصريف الفعل الماضي والمضارع والأمر والمصدر.",
        embedding_model=embedding_model,
        metadata={"branch": "sarf"},
    )
    ingest_document(
        source_id="adab-book-" + str(uuid.uuid4()),
        raw_text="ديوان الشعر العربي وبحور القصائد المعلقات.",
        embedding_model=embedding_model,
        metadata={"branch": "adab"},
    )

    sarf_chunks, _ = retrieve_passages(
        question="الفعل",
        embedding_model=embedding_model,
        top_k=5,
        chunker_type="arabic",
        branch="sarf",
    )
    assert len(sarf_chunks) >= 1
    assert all(c.get("branch") == "sarf" for c in sarf_chunks)

    unknown, _ = retrieve_passages(
        question="الفعل",
        embedding_model=embedding_model,
        top_k=5,
        chunker_type="arabic",
        branch="no-such-branch",
    )
    assert unknown == []


def test_source_ids_filter_scopes_to_documents(isolated_qdrant):
    embedding_model = HashEmbedding(dims=32)
    sid = "mydoc-" + str(uuid.uuid4())
    ingest_document(
        source_id=sid,
        raw_text="النحو فرع من فروع اللغة العربية.",
        embedding_model=embedding_model,
    )
    ingest_document(
        source_id="other-" + str(uuid.uuid4()),
        raw_text="البلاغة علم البيان والبديع والمعاني.",
        embedding_model=embedding_model,
    )

    chunks, _ = retrieve_passages(
        question="اللغة",
        embedding_model=embedding_model,
        top_k=5,
        chunker_type="arabic",
        source_ids=[sid],
    )
    assert len(chunks) >= 1
    assert all(c.get("source_id") == sid for c in chunks)


def test_prune_removes_source_points_and_records(isolated_qdrant, tmp_path):
    from src.retrieval.vector_store import delete_by_sources, get_client, collection_for_chunker
    from src.evaluation.storage import (
        delete_ingested_document,
        record_ingested_document,
        get_connection,
    )

    embedding_model = HashEmbedding(dims=32)
    sid = "filler-" + str(uuid.uuid4())
    ingest_document(source_id=sid, raw_text="حشو مكرر حشو مكرر.", embedding_model=embedding_model)

    client = get_client()
    collection = collection_for_chunker("arabic", embedding_model.dimensions)
    before, _ = client.scroll(collection_name=collection, limit=100, with_payload=False)
    assert any(True for _ in before)

    deleted = delete_by_sources([sid])
    assert deleted.get(collection, 0) >= 1
    from qdrant_client.models import FieldCondition, Filter, MatchAny
    pts, _ = client.scroll(
        collection_name=collection,
        scroll_filter=Filter(must=[FieldCondition(key="source_id", match=MatchAny(any=[sid]))]),
        limit=100,
        with_payload=False,
    )
    assert pts == []

    db = tmp_path / "t.db"
    record_ingested_document(sid, 1, "x", db_path=db)
    delete_ingested_document(sid, db_path=db)
    with get_connection(db) as conn:
        rows = conn.execute("SELECT * FROM ingested_documents WHERE source_id = ?", (sid,)).fetchall()
    assert rows == []