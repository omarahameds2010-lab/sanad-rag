"""
Hybrid retrieval: fuses semantic search (Qdrant) with keyword search
(BM25) using Reciprocal Rank Fusion (RRF).

Why RRF specifically: it combines rankings from two systems with
completely different, non-comparable score scales (cosine similarity vs
BM25 score) without needing to normalize either one — you only need the
*rank position* from each list, which sidesteps a whole class of "how do
I weight these two scores" bugs.
"""

from src.retrieval.vector_store import get_client, search as semantic_search, COLLECTION_NAME
from src.retrieval.bm25_index import bm25_search

RRF_K = 60  # standard RRF damping constant from the original paper


def hybrid_retrieve(
    question_vector: list[float],
    question_text: str,
    top_k: int = 5,
    collection: str = COLLECTION_NAME,
    query_filter=None,
    payload_filter=None,
) -> list[dict]:
    client = get_client()
    semantic_results = semantic_search(client, question_vector, top_k=top_k * 2, collection=collection, query_filter=query_filter)
    keyword_results = bm25_search(question_text, top_k=top_k * 2, collection=collection, payload_filter=payload_filter)

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

    ranked_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)
    return [
        {**payload_by_id[cid], "fused_score": fused_scores[cid]}
        for cid in ranked_ids[:top_k]
    ]
