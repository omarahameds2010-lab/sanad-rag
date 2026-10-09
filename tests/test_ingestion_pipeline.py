import os
import sys
import shutil
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from src.embeddings.base import HashEmbedding
from src.ingestion.pipeline import ingest_document
from src.retrieval.vector_store import get_client, search, collection_for_chunker


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
