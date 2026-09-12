"""
BM25 keyword search — the other half of the hybrid retrieval design in
docs/architecture.md.

Why keyword search alongside semantic search: embeddings are good at
paraphrase/meaning but weak on exact terms (names, numbers, rare technical
words) — especially with a dev-only HashEmbedding that has no real
semantic understanding at all. BM25 catches exact/near-exact term matches
that semantic search alone would miss.

MVP tradeoff (documented, not hidden): the index is rebuilt from Qdrant on
every query via `scroll()` rather than maintained as a persistent index.
This is correct for a student-scale corpus (hundreds/low-thousands of
chunks) and is exactly why docs/architecture_tiers.md lists "batched/
persistent index" as a Production-tier upgrade, not an MVP requirement.
"""

from rank_bm25 import BM25Okapi

from src.retrieval.vector_store import get_client, COLLECTION_NAME


def _tokenize(text: str) -> list[str]:
    # Simple whitespace tokenization. Arabic doesn't have the compounding
    # issues that make this tricky for e.g. German, but attached prefixes
    # (و، ف، ب) mean this is a coarse approximation — good enough for BM25
    # keyword matching, not used anywhere semantic.
    return text.split()


def build_bm25_index(collection: str = COLLECTION_NAME) -> tuple[BM25Okapi, list[dict]] | None:
    client = get_client()
    if not client.collection_exists(collection):
        return None

    points, _ = client.scroll(collection_name=collection, limit=10_000, with_payload=True)
    if not points:
        return None

    corpus_payloads = [p.payload for p in points]
    tokenized_corpus = [_tokenize(p.payload.get("raw_text", "")) for p in points]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, corpus_payloads


def bm25_search(question: str, top_k: int = 5, collection: str = COLLECTION_NAME) -> list[dict]:
    index = build_bm25_index(collection)
    if index is None:
        return []
    bm25, corpus_payloads = index

    scores = bm25.get_scores(_tokenize(question))
    ranked = sorted(zip(scores, corpus_payloads), key=lambda x: x[0], reverse=True)
    return [{"chunk_id": payload["chunk_id"], "score": float(score), **payload}
             for score, payload in ranked[:top_k] if score > 0]
