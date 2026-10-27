import os
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from src.embeddings.base import HashEmbedding
from src.generation.llm_client import FakeLLM
from src.ingestion.pipeline import ingest_document
from src.retrieval.query_pipeline import answer_question
from src.retrieval.vector_store import get_client, reset_client_for_testing


@pytest.fixture
def isolated_qdrant(monkeypatch, tmp_path):
    """Each test gets its own on-disk Qdrant path so tests don't interfere."""
    reset_client_for_testing()
    path = str(tmp_path / "qdrant_local")
    monkeypatch.setattr("src.retrieval.vector_store.DEFAULT_LOCAL_PATH", path)
    monkeypatch.delenv("QDRANT_URL", raising=False)
    yield
    reset_client_for_testing()


def test_hierarchical_ingest_creates_parent_and_child_collections(isolated_qdrant):
    """Hierarchical ingestion should create both parent and child collections."""
    embedding_model = HashEmbedding(dims=32)
    source_id = "test-hierarchical-" + str(uuid.uuid4())
    
    text = """النحو العربي علم يدرس أحكام الكلمة من حيث الإعراب والبناء.
    الإعراب هو تغير أواخر الكلم بتغير العوامل الداخلة عليها.
    البناء هو ثبات أواخر الكلم على حال واحدة.
    العوامل في الإعراب: الرفع، والنصب، والجر، والجزم.
    المرفوعات: المبتدأ، والخبر، والفاعل، ونائبه."""
    
    result = ingest_document(
        source_id=source_id,
        raw_text=text,
        embedding_model=embedding_model,
        chunker_type="hierarchical",
    )
    
    assert result["n_chunks"] > 0
    assert result["n_parents"] is not None
    assert result["n_children"] is not None
    assert result["n_parents"] > 0
    assert result["n_children"] > 0
    assert result["chunker_type"] == "hierarchical"
    
    # Verify both collections exist
    client = get_client()
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]
    
    assert any("hierarchical_children" in name for name in collection_names)
    assert any("hierarchical_parents" in name for name in collection_names)


def test_hierarchical_query_retrieves_parents(isolated_qdrant):
    """Querying hierarchical collection should return parent chunks."""
    embedding_model = HashEmbedding(dims=32)
    source_id = "test-hierarchical-query-" + str(uuid.uuid4())
    
    text = """النحو العربي علم يدرس أحكام الكلمة من حيث الإعراب والبناء.
    الإعراب هو تغير أواخر الكلم بتغير العوامل الداخلة عليها.
    البناء هو ثبات أواخر الكلم على حال واحدة.
    العوامل في الإعراب: الرفع، والنصب، والجر، والجزم.
    المرفوعات: المبتدأ، والخبر، والفاعل، ونائبه."""
    
    ingest_document(
        source_id=source_id,
        raw_text=text,
        embedding_model=embedding_model,
        chunker_type="hierarchical",
    )
    
    result = answer_question(
        question="ما هو الإعراب؟",
        embedding_model=embedding_model,
        primary_llm=FakeLLM(),
        chunker_type="hierarchical",
        top_k=3,
    )
    
    assert "answer" in result
    assert "retrieved_chunk_ids" in result
    assert len(result["retrieved_chunk_ids"]) > 0
    
    # Retrieved chunks should be parent chunks (not child chunks)
    for chunk_id in result["retrieved_chunk_ids"]:
        assert "parent_" in chunk_id  # Parent IDs contain "parent_"
        assert "child_" not in chunk_id  # Should not be child IDs


def test_hierarchical_query_with_reranking(isolated_qdrant):
    """Hierarchical query with reranking should work."""
    embedding_model = HashEmbedding(dims=32)
    source_id = "test-hierarchical-rerank-" + str(uuid.uuid4())
    
    text = """النحو العربي علم يدرس أحكام الكلمة من حيث الإعراب والبناء.
    الإعراب هو تغير أواخر الكلم بتغير العوامل الداخلة عليها.
    البناء هو ثبات أواخر الكلم على حال واحدة.
    العوامل في الإعراب: الرفع، والنصب، والجر، والجزم.
    المرفوعات: المبتدأ، والخبر، والفاعل، ونائبه."""
    
    ingest_document(
        source_id=source_id,
        raw_text=text,
        embedding_model=embedding_model,
        chunker_type="hierarchical",
    )
    
    from src.retrieval.reranker import CrossEncoderReranker
    
    result = answer_question(
        question="ما هو الإعراب؟",
        embedding_model=embedding_model,
        primary_llm=FakeLLM(),
        chunker_type="hierarchical",
        top_k=3,
        reranker=CrossEncoderReranker(),
    )
    
    assert "answer" in result
    assert result["reranker_used"] != "none"


def test_query_expansion_enabled_flag(isolated_qdrant):
    """Query with expansion enabled should return expansion flag in result."""
    embedding_model = HashEmbedding(dims=32)
    source_id = "test-expansion-" + str(uuid.uuid4())
    
    text = """النحو العربي علم يدرس أحكام الكلمة من حيث الإعراب والبناء.
    الإعراب هو تغير أواخر الكلم بتغير العوامل الداخلة عليها."""
    
    ingest_document(
        source_id=source_id,
        raw_text=text,
        embedding_model=embedding_model,
        chunker_type="arabic",
    )
    
    result = answer_question(
        question="ما هو الإعراب؟",
        embedding_model=embedding_model,
        primary_llm=FakeLLM(),
        chunker_type="arabic",
        top_k=3,
        enable_query_expansion=True,
    )
    
    assert "answer" in result
    assert result.get("query_expansion_enabled") is True


def test_query_expansion_disabled_flag(isolated_qdrant):
    """Query with expansion disabled should return expansion flag as False."""
    embedding_model = HashEmbedding(dims=32)
    source_id = "test-no-expansion-" + str(uuid.uuid4())
    
    text = """النحو العربي علم يدرس أحكام الكلمة من حيث الإعراب والبناء."""
    
    ingest_document(
        source_id=source_id,
        raw_text=text,
        embedding_model=embedding_model,
        chunker_type="arabic",
    )
    
    result = answer_question(
        question="ما هو الإعراب؟",
        embedding_model=embedding_model,
        primary_llm=FakeLLM(),
        chunker_type="arabic",
        top_k=3,
        enable_query_expansion=False,
    )
    
    assert "answer" in result
    assert result.get("query_expansion_enabled") is False


def test_hierarchical_with_query_expansion(isolated_qdrant):
    """Hierarchical + query expansion should work together."""
    embedding_model = HashEmbedding(dims=32)
    source_id = "test-both-" + str(uuid.uuid4())
    
    text = """النحو العربي علم يدرس أحكام الكلمة من حيث الإعراب والبناء.
    الإعراب هو تغير أواخر الكلم بتغير العوامل الداخلة عليها.
    البناء هو ثبات أواخر الكلم على حال واحدة.
    العوامل في الإعراب: الرفع، والنصب، والجر، والجزم."""
    
    ingest_document(
        source_id=source_id,
        raw_text=text,
        embedding_model=embedding_model,
        chunker_type="hierarchical",
    )
    
    result = answer_question(
        question="ما هو الإعراب؟",
        embedding_model=embedding_model,
        primary_llm=FakeLLM(),
        chunker_type="hierarchical",
        top_k=3,
        enable_query_expansion=True,
    )
    
    assert "answer" in result
    assert result.get("query_expansion_enabled") is True
    # Should retrieve parent chunks
    for chunk_id in result["retrieved_chunk_ids"]:
        assert "parent_" in chunk_id