"""
Ingestion pipeline — wires together the pieces built in Phase 1/3:
normalize -> chunk -> embed -> upsert to Qdrant -> record in SQLite.

This is the first piece of the architecture in docs/architecture.md §3
("Ingestion path") that is actually executable end-to-end, not just a
diagram.

Phase 7 update: supports pluggable chunker selection (baseline vs Arabic-aware)
for optimization comparisons.

Phase 8 update: supports parent-child (hierarchical) chunking for decoupled
retrieval precision and generation context quality.
"""

from collections.abc import Callable

from src.chunking.arabic_chunker import chunk_text as arabic_chunk_text
from src.chunking.baseline_chunker import chunk_text as baseline_chunk_text
from src.chunking.hierarchical_chunker import hierarchical_chunk_text
from src.embeddings.base import EmbeddingModel
from src.ingestion.branches import infer_branch
from src.ingestion.normalize import clean_for_display, normalize_for_retrieval
from src.retrieval.vector_store import (
    collection_for_chunker,
    delete_by_sources,
    ensure_collection,
    get_client,
    upsert_chunks,
)

# NOTE: record_ingested_document is imported lazily inside functions
# (not at module top) to avoid a circular import: this module is imported
# by src.evaluation.harness, which is re-exported from src.evaluation/__init__,
# so a top-level "from src.evaluation.storage import ..." here would fail
# depending on import order (ImportError: partially initialized module).


def get_chunker(chunker_type: str = "arabic") -> Callable:
    """
    Returns the chunking function based on type.
    
    Args:
        chunker_type: "arabic" for Arabic-aware chunking, "baseline" for naive character splitting,
                     "hierarchical" for parent-child chunking
    
    Returns:
        Chunking function with signature (text, max_chunk_chars, overlap, metadata) -> list[Chunk]
    """
    if chunker_type == "baseline":
        return baseline_chunk_text
    if chunker_type == "hierarchical":
        # For hierarchical, we return a wrapper that provides parent chunks
        # The actual parent-child logic is handled in ingest_document
        from src.chunking.hierarchical_chunker import (
            chunk_text as hierarchical_chunk_text_compat,
        )
        return hierarchical_chunk_text_compat
    return arabic_chunk_text


def ingest_document(
    source_id: str,
    raw_text: str,
    embedding_model: EmbeddingModel,
    metadata: dict | None = None,
    chunker_type: str = "arabic",
) -> dict:
    """
    Runs the full ingestion path for one document and returns a summary
    matching the /ingest response contract in docs/architecture.md §4.
    
    Args:
        source_id: unique identifier for the document
        raw_text: raw document text
        embedding_model: embedding model to use
        metadata: optional metadata to attach to chunks
        chunker_type: "arabic", "baseline", or "hierarchical" chunking strategy
    """
    chunker = get_chunker(chunker_type)
    display_text = clean_for_display(raw_text)

    # Branch tagging for Explorer sub-filters: caller metadata wins,
    # otherwise infer from source_id. Additive only — never overrides.
    metadata = dict(metadata or {})
    metadata.setdefault("branch", infer_branch(source_id))

    # Hierarchical (parent-child) chunking: create parents for LLM context,
    # children for high-precision retrieval
    if chunker_type == "hierarchical":
        return _ingest_hierarchical(
            source_id, display_text, embedding_model, metadata, chunker_type
        )

    # Use appropriate chunker parameters based on type
    if chunker_type == "baseline":
        chunks = chunker(display_text, max_chunk_chars=800, overlap_chars=100, metadata=metadata)
    else:
        chunks = chunker(display_text, max_chunk_chars=800, overlap_sentences=1, metadata=metadata)

    if not chunks:
        return {
            "source_id": source_id,
            "n_chunks": 0,
            "embedding_model": embedding_model.name,
            "chunker_type": chunker_type,
        }

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
            "start_sentence": chunk.start_sentence_idx if hasattr(chunk, "start_sentence_idx") else chunk.start_char_idx,
            "end_sentence": chunk.end_sentence_idx if hasattr(chunk, "end_sentence_idx") else chunk.end_char_idx,
            "chunker_type": chunker_type,
            **chunk.metadata,
        }
        for chunk in chunks
    ]

    client = get_client()
    collection = collection_for_chunker(chunker_type, embedding_model.dimensions)
    ensure_collection(client, dims=embedding_model.dimensions, collection=collection)
    # Replace, don't merge. Upserting chunk_ids 0..n leaves a stale tail when
    # the new text is shorter than the previous ingestion (e.g. re-ingesting
    # a corrected source), which silently corrupts retrieval and evaluation.
    delete_by_sources([source_id], collections=[collection])
    upsert_chunks(client, chunk_ids, vectors, payloads, collection=collection)

    from src.evaluation.storage import record_ingested_document

    record_ingested_document(
        source_id=source_id, n_chunks=len(chunks), embedding_model=embedding_model.name
    )

    return {
        "source_id": source_id,
        "n_chunks": len(chunks),
        "embedding_model": embedding_model.name,
        "chunker_type": chunker_type,
    }


def _ingest_hierarchical(
    source_id: str,
    display_text: str,
    embedding_model: EmbeddingModel,
    metadata: dict | None,
    chunker_type: str,
) -> dict:
    """
    Ingest using parent-child (hierarchical) chunking:
    - Child chunks are indexed in Qdrant for retrieval
    - Parent chunks are stored in a separate collection for LLM context lookup
    """
    
    parents, children = hierarchical_chunk_text(
        display_text,
        max_parent_chars=1800,
        max_child_chars=400,
        parent_overlap_sentences=1,
        child_overlap_sentences=0,
        metadata=metadata,
    )

    if not parents or not children:
        return {
            "source_id": source_id,
            "n_chunks": 0,
            "n_parents": 0,
            "n_children": 0,
            "embedding_model": embedding_model.name,
            "chunker_type": chunker_type,
        }

    # ---- Index CHILD chunks in Qdrant (for retrieval) ----
    child_normalized = [normalize_for_retrieval(c.text) for c in children]
    child_vectors = embedding_model.embed(child_normalized)
    
    child_ids = [f"{source_id}::child::{c.child_id}" for c in children]
    child_payloads = [
        {
            "source_id": source_id,
            "raw_text": child.text,
            "parent_id": child.parent_id,
            "child_id": child.child_id,
            "start_sentence": child.start_sentence_idx,
            "end_sentence": child.end_sentence_idx,
            "chunker_type": chunker_type,
            "chunk_level": "child",
            # Explicit: build_child_chunks drops caller metadata, so the
            # branch would be lost here without this line.
            "branch": child.metadata.get("branch", metadata.get("branch", infer_branch(source_id))),
            **child.metadata,
        }
        for child in children
    ]

    client = get_client()
    child_collection = collection_for_chunker(f"{chunker_type}_children", embedding_model.dimensions)
    ensure_collection(client, dims=embedding_model.dimensions, collection=child_collection)
    delete_by_sources([source_id], collections=[child_collection])
    upsert_chunks(client, child_ids, child_vectors, child_payloads, collection=child_collection)

    # ---- Store PARENT chunks in a separate collection (for context lookup) ----
    # Parents are NOT embedded for retrieval; they are looked up by parent_id
    # when child chunks are retrieved. We store them with a simple key-value
    # approach using a dedicated "parents" collection.
    parent_ids = [f"{source_id}::parent::{p.parent_id}" for p in parents]
    # Create dummy vectors for parents (not used for search, but Qdrant requires vectors)
    # We'll use zero vectors since parents are never searched directly
    dims = embedding_model.dimensions
    zero_vector = [0.0] * dims
    parent_vectors = [zero_vector for _ in parents]
    
    parent_payloads = [
        {
            "source_id": source_id,
            "raw_text": parent.text,
            "parent_id": parent.parent_id,
            "child_ids": parent.child_ids,
            "start_sentence": parent.start_sentence_idx,
            "end_sentence": parent.end_sentence_idx,
            "chunker_type": chunker_type,
            "chunk_level": "parent",
            "branch": parent.metadata.get("branch", metadata.get("branch", infer_branch(source_id))),
            **parent.metadata,
        }
        for parent in parents
    ]
    
    parent_collection = collection_for_chunker(f"{chunker_type}_parents", embedding_model.dimensions)
    ensure_collection(client, dims=embedding_model.dimensions, collection=parent_collection)
    delete_by_sources([source_id], collections=[parent_collection])
    upsert_chunks(client, parent_ids, parent_vectors, parent_payloads, collection=parent_collection)

    from src.evaluation.storage import record_ingested_document

    record_ingested_document(
        source_id=source_id, n_chunks=len(children), embedding_model=embedding_model.name
    )

    return {
        "source_id": source_id,
        "n_chunks": len(children),  # children are the retrievable units
        "n_parents": len(parents),
        "n_children": len(children),
        "embedding_model": embedding_model.name,
        "chunker_type": chunker_type,
    }
