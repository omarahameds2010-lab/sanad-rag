import os
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from src.embeddings.base import HashEmbedding
from src.ingestion.pipeline import ingest_document
from src.retrieval.vector_store import collection_for_chunker, get_client, search

ARABIC_SAMPLE = (
    "اللغة العربية لغة جميلة وغنية. تحتوي على تراث أدبي عريق. "
    "كثير من الطلاب يدرسون العربية اليوم."
)


@pytest.fixture
def isolated_qdrant_path(monkeypatch, tmp_path):
    """Each test gets its own on-disk Qdrant path so tests don't interfere."""
    from src.retrieval.vector_store import reset_client_for_testing

    reset_client_for_testing()
    path = str(tmp_path / "qdrant_local")
    monkeypatch.setattr("src.retrieval.vector_store.DEFAULT_LOCAL_PATH", path)
    monkeypatch.delenv("QDRANT_URL", raising=False)
    yield path
    reset_client_for_testing()


def test_ingest_document_produces_chunks_and_stores_vectors(isolated_qdrant_path):
    embedding_model = HashEmbedding(dims=32)
    result = ingest_document(
        source_id="test-doc-" + str(uuid.uuid4()),
        raw_text=ARABIC_SAMPLE,
        embedding_model=embedding_model,
    )
    assert result["n_chunks"] >= 1
    assert result["embedding_model"] == "hash-embedding-DEV-ONLY"


def test_ingested_chunks_are_retrievable_by_search(isolated_qdrant_path):
    embedding_model = HashEmbedding(dims=32)
    source_id = "test-doc-" + str(uuid.uuid4())
    ingest_document(source_id=source_id, raw_text=ARABIC_SAMPLE, embedding_model=embedding_model)

    client = get_client()
    query_vector = embedding_model.embed(["اللغة العربية"])[0]
    collection = collection_for_chunker("arabic", embedding_model.dimensions)
    results = search(client, query_vector, top_k=3, collection=collection)

    assert len(results) >= 1
    assert all(r["source_id"] == source_id for r in results)


def test_empty_text_ingests_zero_chunks(isolated_qdrant_path):
    embedding_model = HashEmbedding(dims=32)
    result = ingest_document(
        source_id="empty-doc", raw_text="   ", embedding_model=embedding_model
    )
    assert result["n_chunks"] == 0


def _count_source(collection: str, source_id: str) -> int:
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    client = get_client()
    return client.count(
        collection_name=collection,
        count_filter=Filter(
            must=[FieldCondition(key="source_id", match=MatchValue(value=source_id))]
        ),
        exact=True,
    ).count


def test_reingest_shorter_text_replaces_stale_chunks(isolated_qdrant_path):
    """A corrected/shorter source must not keep stale tail chunks.

    Regression: upserting chunk_ids 0..n over an older, longer ingestion used
    to leave the old tail behind, silently changing the corpus that eval
    numbers are measured on.
    """
    embedding_model = HashEmbedding(dims=32)
    source_id = "test-reingest-" + uuid.uuid4().hex
    collection = collection_for_chunker("arabic", embedding_model.dimensions)

    long_text = "\n".join([ARABIC_SAMPLE] * 20)
    first = ingest_document(
        source_id=source_id, raw_text=long_text, embedding_model=embedding_model
    )
    assert first["n_chunks"] > 1
    assert _count_source(collection, source_id) == first["n_chunks"]

    second = ingest_document(
        source_id=source_id, raw_text=ARABIC_SAMPLE, embedding_model=embedding_model
    )
    assert second["n_chunks"] == 1
    assert _count_source(collection, source_id) == 1


def test_hierarchical_reingest_replaces_children_and_parents(isolated_qdrant_path):
    embedding_model = HashEmbedding(dims=32)
    source_id = "test-h-reingest-" + uuid.uuid4().hex
    children_collection = collection_for_chunker(
        "hierarchical_children", embedding_model.dimensions
    )
    parents_collection = collection_for_chunker(
        "hierarchical_parents", embedding_model.dimensions
    )

    long_text = "\n".join([ARABIC_SAMPLE] * 30)
    first = ingest_document(
        source_id=source_id,
        raw_text=long_text,
        embedding_model=embedding_model,
        chunker_type="hierarchical",
    )
    assert first["n_children"] > 1
    assert _count_source(children_collection, source_id) == first["n_children"]
    assert _count_source(parents_collection, source_id) == first["n_parents"]

    second = ingest_document(
        source_id=source_id,
        raw_text=ARABIC_SAMPLE,
        embedding_model=embedding_model,
        chunker_type="hierarchical",
    )
    assert second["n_children"] == 1
    assert _count_source(children_collection, source_id) == 1
    assert _count_source(parents_collection, source_id) == second["n_parents"]
