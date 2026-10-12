"""
Vector store wrapper around Qdrant.

Supports two modes, both using the exact same code path so behavior is
identical between dev and production:
- Local/embedded (`QDRANT_URL` unset or "local"): on-disk Qdrant with no
  server process — used for tests and offline development.
- Remote (`QDRANT_URL=http://localhost:6333` etc.): the Dockerized Qdrant
  service from docker-compose.yml — used for real runs.
"""

import os
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

DEFAULT_LOCAL_PATH = "data/qdrant_local"
COLLECTION_NAME = "sanad_chunks"


def collection_for_chunker(chunker_type: str, dims: int | None = None) -> str:
    """
    Phase 7: baseline vs Arabic-aware chunking comparisons must not
    clobber each other's vectors in one collection — both produce
    chunk_ids of the form "{source_id}::{i}" so upserting the second
    config would overwrite the first, corrupting the comparison.
    Including the embedding dimension also prevents a subtle crash when
    EMBEDDING_MODEL changes (a 128-dim HashEmbedding index cannot accept
    768-dim multilingual-e5 vectors): different models land in different
    collections instead of failing with a cryptic dim-mismatch error.
    """
    if dims is None:
        return f"{COLLECTION_NAME}_{chunker_type}"
    return f"{COLLECTION_NAME}_{chunker_type}_{dims}"


def build_scope_filter(
    source_ids: list[str] | None = None,
    branch: str | None = None,
):
    """Build a Qdrant Filter scoping retrieval to Explorer sub-filters.

    - source_ids: only chunks from these ingested sources (MatchAny).
    - branch: only chunks tagged with this library branch (MatchValue).
    Returns None when no scoping requested (search everything).
    """
    from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

    conditions = []
    if source_ids:
        conditions.append(FieldCondition(key="source_id", match=MatchAny(any=source_ids)))
    if branch:
        conditions.append(FieldCondition(key="branch", match=MatchValue(value=branch)))
    if not conditions:
        return None
    return Filter(must=conditions)


_client_singleton: QdrantClient | None = None


def get_client() -> QdrantClient:
    """
    Returns a cached singleton client within this process.

    Real bug found while testing Phase 5 end-to-end: the embedded/local
    Qdrant mode takes an on-disk file lock, so creating a *second*
    QdrantClient(path=...) instance in the same process — e.g. once from
    the ingestion call and again from a later query call — raises
    "Storage folder is already accessed by another instance of Qdrant
    client". Caching one client per process fixes this and matches how
    the remote/Docker mode behaves anyway (one client, reused).
    """
    global _client_singleton
    if _client_singleton is not None:
        return _client_singleton

    url = os.environ.get("QDRANT_URL")
    if url and url != "local":
        _client_singleton = QdrantClient(url=url)
    else:
        _client_singleton = QdrantClient(path=DEFAULT_LOCAL_PATH)
    return _client_singleton


def reset_client_for_testing() -> None:
    """Closes and clears the cached client — needed between isolated test
    runs that each want their own on-disk path."""
    global _client_singleton
    if _client_singleton is not None:
        _client_singleton.close()
        _client_singleton = None


def ensure_collection(client: QdrantClient, dims: int, collection: str = COLLECTION_NAME) -> None:
    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dims, distance=Distance.COSINE),
        )


def upsert_chunks(
    client: QdrantClient,
    chunk_ids: list[str],
    vectors: list[list[float]],
    payloads: list[dict],
    collection: str = COLLECTION_NAME,
) -> None:
    points = [
        PointStruct(id=_stable_point_id(cid), vector=vec, payload={**payload, "chunk_id": cid})
        for cid, vec, payload in zip(chunk_ids, vectors, payloads)
    ]
    client.upsert(collection_name=collection, points=points)


def search(
    client: QdrantClient,
    query_vector: list[float],
    top_k: int = 5,
    collection: str = COLLECTION_NAME,
    query_filter=None,
) -> list[dict]:
    from qdrant_client.models import Filter
    
    results = client.query_points(
        collection_name=collection, query=query_vector, limit=top_k, query_filter=query_filter
    ).points
    return [{"chunk_id": r.payload["chunk_id"], "score": r.score, **r.payload} for r in results]


def _stable_point_id(chunk_id: str) -> str:
    """Qdrant requires UUID or unsigned int point IDs — derive a stable
    UUID from our own chunk_id string so re-ingesting the same chunk
    upserts instead of duplicating."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))
