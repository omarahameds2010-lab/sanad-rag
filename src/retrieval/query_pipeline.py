"""
Query pipeline — the "Query path" from docs/architecture.md §3, made real:
normalize -> hybrid retrieve -> generate (with fallback) -> timed response.
"""

from time import perf_counter

from src.ingestion.normalize import normalize_for_retrieval
from src.embeddings.base import EmbeddingModel
from src.retrieval.hybrid import hybrid_retrieve
from src.generation.llm_client import LLMClient


def answer_question(
    question: str,
    embedding_model: EmbeddingModel,
    primary_llm: LLMClient,
    fallback_llm: LLMClient | None = None,
    top_k: int = 5,
) -> dict:
    """
    Returns a dict matching the /query response contract in
    docs/architecture.md §4: {answer, cited_chunk_ids, retrieval_ms,
    generation_ms}.
    """
    retrieval_start = perf_counter()
    normalized_question = normalize_for_retrieval(question)
    question_vector = embedding_model.embed([normalized_question])[0]
    retrieved_chunks = hybrid_retrieve(question_vector, normalized_question, top_k=top_k)
    retrieval_ms = (perf_counter() - retrieval_start) * 1000

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
    }
