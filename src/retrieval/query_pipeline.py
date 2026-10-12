"""
Query pipeline — the "Query path" from docs/architecture.md §3, made real:
normalize -> hybrid retrieve -> rerank (optional) -> generate (with fallback) -> timed response.

Phase 7 update: supports optional reranking for optimization comparisons.
Phase 8 update: supports parent-child (hierarchical) retrieval and query expansion.
"""

from time import perf_counter
from typing import Optional

from src.ingestion.normalize import normalize_for_retrieval
from src.embeddings.base import EmbeddingModel
from src.retrieval.hybrid import hybrid_retrieve
from src.retrieval.reranker import Reranker, NoOpReranker
from src.retrieval.vector_store import collection_for_chunker, get_client
from src.generation.llm_client import LLMClient


def retrieve_passages(
    question: str,
    embedding_model: EmbeddingModel,
    top_k: int = 5,
    reranker: Optional[Reranker] = None,
    chunker_type: str = "arabic",
    enable_query_expansion: bool = False,
) -> tuple[list[dict], float]:
    """Retrieval-only half of the query path (no LLM involved).

    Returns (retrieved_chunks, retrieval_ms). Shared by answer_question
    and the extractive endpoint so both rank identically.
    """
    retrieval_start = perf_counter()
    normalized_question = normalize_for_retrieval(question)
    question_vector = embedding_model.embed([normalized_question])[0]

    # Determine collections based on chunker type
    is_hierarchical = chunker_type == "hierarchical"

    if is_hierarchical:
        # For hierarchical: retrieve from children collection, then map to parents
        if enable_query_expansion:
            retrieved_chunks = _hierarchical_retrieve_with_expansion(
                question,
                embedding_model,
                top_k=top_k * 2 if reranker else top_k,
                reranker=reranker,
            )
        else:
            retrieved_chunks = _hierarchical_retrieve(
                question_vector,
                normalized_question,
                embedding_model,
                top_k=top_k * 2 if reranker else top_k,
                reranker=reranker,
            )
    else:
        # Standard retrieval (with optional query expansion)
        if enable_query_expansion:
            retrieved_chunks = _retrieve_with_expansion(
                question,
                embedding_model,
                top_k=top_k * 2 if reranker else top_k,
                chunker_type=chunker_type,
                reranker=reranker,
            )
        else:
            retrieve_k = top_k * 2 if reranker else top_k
            collection = collection_for_chunker(chunker_type, embedding_model.dimensions)
            retrieved_chunks = hybrid_retrieve(
                question_vector, normalized_question, top_k=retrieve_k, collection=collection
            )

            # Apply reranking if provided
            if reranker:
                retrieved_chunks = reranker.rerank(normalized_question, retrieved_chunks, top_k=top_k)

    retrieval_ms = (perf_counter() - retrieval_start) * 1000
    return retrieved_chunks, retrieval_ms


def answer_question(
    question: str,
    embedding_model: EmbeddingModel,
    primary_llm: LLMClient,
    fallback_llm: Optional[LLMClient] = None,
    top_k: int = 5,
    reranker: Optional[Reranker] = None,
    chunker_type: str = "arabic",
    enable_query_expansion: bool = False,
) -> dict:
    """
    Returns a dict matching the /query response contract in
    docs/architecture.md §4: {answer, cited_chunk_ids, retrieval_ms,
    generation_ms}.
    
    Args:
        question: user's question
        embedding_model: embedding model to use
        primary_llm: primary LLM for generation
        fallback_llm: fallback LLM if primary fails
        top_k: number of chunks to retrieve
        reranker: optional reranker to re-rank retrieved chunks
        chunker_type: chunking strategy used at ingestion time
        enable_query_expansion: whether to expand the query with Arabic synonyms/variations
    """
    retrieved_chunks, retrieval_ms = retrieve_passages(
        question=question,
        embedding_model=embedding_model,
        top_k=top_k,
        reranker=reranker,
        chunker_type=chunker_type,
        enable_query_expansion=enable_query_expansion,
    )

    generation_start = perf_counter()
    try:
        result = primary_llm.generate(question, retrieved_chunks)
        used_model = primary_llm.name
    except Exception as primary_error:
        # Reliability strategy from docs/architecture.md §6: fall back
        # instead of surfacing a raw exception to the user.
        if fallback_llm is None:
            raise
        result = fallback_llm.generate(question, retrieved_chunks)
        used_model = f"{fallback_llm.name} (fallback, primary failed: {primary_error})"
    generation_ms = (perf_counter() - generation_start) * 1000

    return {
        "answer": result.answer,
        "cited_chunk_ids": result.cited_chunk_ids,
        "retrieved_chunk_ids": [c["chunk_id"] for c in retrieved_chunks],
        "retrieval_ms": round(retrieval_ms, 2),
        "generation_ms": round(generation_ms, 2),
        "model_used": used_model,
        "n_chunks_retrieved": len(retrieved_chunks),
        "reranker_used": reranker.name if reranker else "none",
        "chunker_type": chunker_type,
        "query_expansion_enabled": enable_query_expansion,
    }


def _hierarchical_retrieve(
    question_vector: list[float],
    normalized_question: str,
    embedding_model: EmbeddingModel,
    top_k: int = 5,
    reranker: Optional[Reranker] = None,
) -> list[dict]:
    """
    Hierarchical (parent-child) retrieval:
    1. Search child chunks in the children collection
    2. Map retrieved children to their parent chunks
    3. Deduplicate parents (keep highest-scoring child per parent)
    4. Return parent chunks for LLM context
    """
    from src.retrieval.vector_store import search as semantic_search
    from src.retrieval.bm25_index import bm25_search
    from src.retrieval.hybrid import RRF_K
    
    client = get_client()
    dims = embedding_model.dimensions
    
    # Search child collection
    child_collection = collection_for_chunker("hierarchical_children", dims)
    
    # Semantic search on children
    semantic_results = semantic_search(client, question_vector, top_k=top_k * 2, collection=child_collection)
    # BM25 search on children
    keyword_results = bm25_search(normalized_question, top_k=top_k * 2, collection=child_collection)
    
    # Fuse with RRF (same as hybrid_retrieve)
    fused_scores: dict[str, float] = {}
    payload_by_id: dict[str, dict] = {}
    
    for rank, result in enumerate(semantic_results):
        cid = result["chunk_id"]
        fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
        payload_by_id[cid] = result
    
    for rank, result in enumerate(keyword_results):
        cid = result["chunk_id"]
        fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
        payload_by_id.setdefault(cid, result)
    
    ranked_child_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)
    top_children = [
        {**payload_by_id[cid], "fused_score": fused_scores[cid]}
        for cid in ranked_child_ids[:top_k]
    ]
    
    # Map children to parents. Keyed by (source_id, parent_id) because
    # parent_ids restart at parent_0 inside every source document —
    # keying by parent_id alone returns a parent from the wrong document.
    parent_map: dict[tuple[str, str], dict] = {}

    for child in top_children:
        parent_id = child.get("parent_id")
        source_id = child.get("source_id", "")
        if not parent_id:
            continue

        key = (source_id, parent_id)
        score = child.get("fused_score", 0.0)
        child_id = child.get("chunk_id")

        if key not in parent_map or score > parent_map[key]["best_score"]:
            # Fetch parent chunk from parents collection
            parent_collection = collection_for_chunker("hierarchical_parents", dims)
            parent_payload = _fetch_parent_by_id(client, parent_collection, parent_id, source_id=source_id)

            if parent_payload:
                parent_map[key] = {
                    "payload": parent_payload,
                    "best_score": score,
                    "child_ids": [child_id],
                }
        else:
            parent_map[key]["child_ids"].append(child_id)

    # Convert parent map to list of chunks for LLM
    parent_chunks = []
    for (source_id, parent_id), data in parent_map.items():
        payload = data["payload"]
        payload = {
            **payload,
            "fused_score": data["best_score"],
            "child_ids": data["child_ids"],
            # Full stable ID (matches the Qdrant point ID scheme
            # "{source_id}::parent::{parent_id}") so citations are unambiguous
            # across documents.
            "chunk_id": f"{source_id}::parent::{parent_id}",
        }
        parent_chunks.append(payload)
    
    # Sort by best score
    parent_chunks.sort(key=lambda c: c.get("fused_score", 0.0), reverse=True)
    
    # Apply reranking if provided (rerank parents based on question)
    if reranker:
        parent_chunks = reranker.rerank(normalized_question, parent_chunks, top_k=top_k)
    else:
        parent_chunks = parent_chunks[:top_k]
    
    return parent_chunks


def _fetch_parent_by_id(
    client, collection: str, parent_id: str, source_id: str | None = None
) -> dict | None:
    """Fetch a parent chunk by its parent_id from the parents collection.

    source_id scoping is required because parent_ids restart at parent_0
    inside every source document.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    try:
        conditions = [FieldCondition(key="parent_id", match=MatchValue(value=parent_id))]
        if source_id:
            conditions.append(FieldCondition(key="source_id", match=MatchValue(value=source_id)))
        results = client.scroll(
            collection_name=collection,
            scroll_filter=Filter(must=conditions),
            limit=1,
            with_payload=True,
        )
        if results[0]:
            point = results[0][0]
            return {"chunk_id": parent_id, **point.payload}
    except Exception:
        pass
    return None


def _retrieve_with_expansion(
    question: str,
    embedding_model: EmbeddingModel,
    top_k: int = 5,
    chunker_type: str = "arabic",
    reranker: Optional[Reranker] = None,
) -> list[dict]:
    """
    Retrieve using multi-query expansion (Arabic synonyms/templates) + RRF fusion.
    """
    from src.retrieval.query_expansion import expand_and_retrieve
    
    return expand_and_retrieve(
        question,
        embedding_model,
        top_k=top_k,
        chunker_type=chunker_type,
        reranker=reranker,
        max_variations=3,
    )


def _hierarchical_retrieve_with_expansion(
    question: str,
    embedding_model: EmbeddingModel,
    top_k: int = 5,
    reranker: Optional[Reranker] = None,
) -> list[dict]:
    """
    Hierarchical retrieval with query expansion:
    1. Expand query into variations
    2. For each variation, run hierarchical retrieval
    3. Fuse parent results across variations via RRF
    """
    from src.retrieval.query_expansion import expand_query
    
    expanded_queries = expand_query(question, max_variations=3)
    
    all_parent_scores: dict[str, float] = {}
    parent_payloads: dict[str, dict] = {}
    
    for q_idx, expanded_q in enumerate(expanded_queries):
        # Get embedding for this variation
        from src.ingestion.normalize import normalize_for_retrieval
        normalized = normalize_for_retrieval(expanded_q)
        question_vector = embedding_model.embed([normalized])[0]
        
        # Run standard hierarchical retrieval for this variation
        parents = _hierarchical_retrieve(
            question_vector,
            normalized,
            embedding_model,
            top_k=top_k * 2,
            reranker=None,  # Don't rerank intermediate results
        )
        
        # Fuse across query variations
        for rank, parent in enumerate(parents):
            parent_id = parent.get("chunk_id")
            if not parent_id:
                continue
            score = 1.0 / (60 + rank + 1 + q_idx * 0.1)  # RRF with query damping
            all_parent_scores[parent_id] = all_parent_scores.get(parent_id, 0.0) + score
            parent_payloads[parent_id] = parent
    
    # Rank fused parents
    ranked_parent_ids = sorted(all_parent_scores, key=lambda pid: all_parent_scores[pid], reverse=True)
    fused_parents = [
        {**parent_payloads[pid], "fused_score": all_parent_scores[pid]}
        for pid in ranked_parent_ids[:top_k]
    ]
    
    # Apply reranking if provided
    if reranker:
        from src.ingestion.normalize import normalize_for_retrieval
        normalized_original = normalize_for_retrieval(question)
        fused_parents = reranker.rerank(normalized_original, fused_parents, top_k=top_k)
    else:
        fused_parents = fused_parents[:top_k]
    
    return fused_parents