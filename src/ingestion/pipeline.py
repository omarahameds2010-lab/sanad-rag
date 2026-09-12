"""
Ingestion pipeline — wires together the pieces built in Phase 1/3:
normalize -> chunk -> embed -> upsert to Qdrant -> record in SQLite.

This is the first piece of the architecture in docs/architecture.md §3
("Ingestion path") that is actually executable end-to-end, not just a
diagram.
"""

from src.ingestion.normalize import clean_for_display, normalize_for_retrieval
from src.chunking.arabic_chunker import chunk_text
from src.embeddings.base import EmbeddingModel
from src.retrieval.vector_store import get_client, ensure_collection, upsert_chunks
from src.evaluation.storage import record_ingested_document


def ingest_document(
    source_id: str,
    raw_text: str,
    embedding_model: EmbeddingModel,
    metadata: dict | None = None,
) -> dict:
    """
    Runs the full ingestion path for one document and returns a summary
    matching the /ingest response contract in docs/architecture.md §4.
    """
    display_text = clean_for_display(raw_text)
    chunks = chunk_text(display_text, metadata=metadata or {})

    if not chunks:
        return {"source_id": source_id, "n_chunks": 0, "embedding_model": embedding_model.name}

    # Embed against the retrieval-normalized form (diacritics stripped,
    # variants unified) — but store the original display text in the
    # payload, since that's what gets shown/cited to the user.
    normalized_texts = [normalize_for_retrieval(c.text) for c in chunks]
    vectors = embedding_model.embed(normalized_texts)

    chunk_ids = [f"{source_id}::{i}" for i in range(len(chunks))]
    payloads = [
        {
            "source_id": source_id,
            "raw_text": chunk.text,
            "start_sentence": chunk.start_sentence_idx,
            "end_sentence": chunk.end_sentence_idx,
            **chunk.metadata,
        }
        for chunk in chunks
    ]

    client = get_client()
    ensure_collection(client, dims=embedding_model.dimensions)
    upsert_chunks(client, chunk_ids, vectors, payloads)

    record_ingested_document(
        source_id=source_id, n_chunks=len(chunks), embedding_model=embedding_model.name
    )

    return {
        "source_id": source_id,
        "n_chunks": len(chunks),
        "embedding_model": embedding_model.name,
    }
