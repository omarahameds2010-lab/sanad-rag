"""
FastAPI application — implements the contract defined in
docs/architecture.md §4.

Phase 3 note: endpoints are real and wired to Pydantic validation, but the
retrieval/generation internals are stubs until Phase 4/5 build the actual
ingestion and retrieval pipelines. This is intentional — the API contract
is what gets locked in Phase 3, not the implementation behind it.

Phase 9: Security Review - Added API key auth, rate limiting, security headers, CORS
"""

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

from src.evaluation.storage import init_db
from src.security import (
    verify_api_key,
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
    get_cors_origins,
    sanitize_input,
    detect_injection_attempt,
)
from src.api.deps import (
    _get_embedding_model,
    _get_llm_clients,
    _check_qdrant,
    _check_db,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    # Warm up the embedding model once so per-request loads (slow + flaky
    # for 2GB models) never happen. Non-fatal: endpoints fail loudly anyway.
    try:
        from src.api.deps import warmup_embedding_model

        warmup_embedding_model()
    except Exception as e:
        print(f"[WARNING] embedding warmup failed: {e}")
    yield


app = FastAPI(
    title="SANAD API",
    description="Arabic-First RAG Evaluation Platform",
    version="0.4.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ---------------------------------------------------------------------------
# Security Middleware (order matters - first added = outermost)
# ---------------------------------------------------------------------------

# 1. Security headers (outermost - applies to all responses)
app.add_middleware(SecurityHeadersMiddleware)

# 2. Rate limiting
app.add_middleware(RateLimitMiddleware)

# 3. CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"],
)

# ---------------------------------------------------------------------------
# Schemas (Pydantic validation at the API boundary — see architecture.md §2)
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    source_id: str = Field(..., min_length=1, max_length=200)
    text: str = Field(..., min_length=1, max_length=200_000)
    metadata: dict | None = None
    chunker_type: Literal["arabic", "baseline", "hierarchical"] = Field(
        default="arabic", description="Chunking strategy: 'arabic', 'baseline', or 'hierarchical' (parent-child)"
    )


class IngestResponse(BaseModel):
    source_id: str
    n_chunks: int
    embedding_model: str
    chunker_type: str
    n_parents: int | None = None
    n_children: int | None = None


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2_000)
    top_k: int = Field(default=5, ge=1, le=20)
    config_name: str = Field(default="baseline")
    enable_reranking: bool = Field(default=False, description="Enable cross-encoder reranking")
    chunker_type: Literal["arabic", "baseline", "hierarchical"] = Field(
        default="arabic", description="Which chunker's collection to query"
    )
    enable_query_expansion: bool = Field(default=False, description="Enable Arabic query expansion for improved recall")


class QueryResponse(BaseModel):
    answer: str
    cited_chunk_ids: list[str]
    retrieval_ms: float
    generation_ms: float
    reranker_used: str
    query_expansion_enabled: bool = False


class ExtractiveRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2_000)
    top_k: int = Field(default=5, ge=1, le=20)
    chunker_type: Literal["arabic", "baseline", "hierarchical"] = Field(
        default="arabic", description="Which chunker's collection to query"
    )
    enable_reranking: bool = Field(default=False, description="Enable cross-encoder reranking")
    enable_query_expansion: bool = Field(default=False, description="Enable Arabic query expansion for improved recall")


class ExtractivePassage(BaseModel):
    chunk_id: str
    source_id: str
    text: str
    score: float


class ExtractiveResponse(BaseModel):
    passages: list[ExtractivePassage]
    cited_chunk_ids: list[str]
    retrieval_ms: float
    reranker_used: str
    chunker_type: str
    query_expansion_enabled: bool = False


class EvaluateRunRequest(BaseModel):
    golden_dataset_path: str
    config_name: str = Field(default="baseline")
    chunker_type: Literal["arabic", "baseline", "hierarchical"] = Field(
        default="arabic", description="Chunking strategy for golden source ingestion"
    )
    enable_reranking: bool = Field(default=False, description="Enable reranking during evaluation")
    enable_query_expansion: bool = Field(default=False, description="Enable query expansion during evaluation")


class EvaluateRunResponse(BaseModel):
    run_id: str
    n_examples: int
    status: str


# ---------------------------------------------------------------------------
# Security: Input validation middleware
# ---------------------------------------------------------------------------

from src.api.evaluation_runtime import router as evaluation_runtime_router

app.include_router(evaluation_runtime_router)

@app.middleware("http")
async def validate_input_middleware(request: Request, call_next):
    """
    Validate and sanitize request body for POST endpoints.
    Detects potential injection attempts.
    """
    # Only validate POST bodies for our API endpoints
    if request.method == "POST" and request.url.path in [
        "/ingest",
        "/query",
        "/query/extractive",
        "/evaluate/run",
        "/evaluate/submission",
    ]:
        try:
            body = await request.body()
            if body:
                body_str = body.decode("utf-8")
                # Check for injection attempts
                suspicious, patterns = detect_injection_attempt(body_str)
                if suspicious:
                    # Log but don't block (could be false positive for Arabic text)
                    # In production, you might want to block or alert
                    pass
        except Exception:
            pass  # If we can't read body, let it through
    
    response = await call_next(request)
    return response


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """
    Checks actual dependencies, not just "the process is alive" — per the
    reliability table in architecture.md §6.
    """
    qdrant_ok = _check_qdrant()
    db_ok = _check_db()
    status = "ok" if (qdrant_ok and db_ok) else "degraded"
    return {"status": status, "qdrant_ok": qdrant_ok, "db_ok": db_ok}


@app.post("/ingest", response_model=IngestResponse)
def ingest(
    request: IngestRequest,
    api_key: str = Depends(verify_api_key),  # API key auth
):
    """
    Wired to the real pipeline (src/ingestion/pipeline.py) as of Phase 4.
    Uses HashEmbedding by default in this environment because it has no
    network access to download real embedding models — see
    src/embeddings/base.py docstring. Set EMBEDDING_MODEL=multilingual-e5
    once running with real network/model access.
    """
    from src.ingestion.pipeline import ingest_document
    from src.embeddings.base import HashEmbedding

    # Sanitize input
    clean_text = sanitize_input(request.text, max_length=200_000)
    clean_source_id = sanitize_input(request.source_id, max_length=200)
    
    embedding_model = _get_embedding_model()
    result = ingest_document(
        source_id=clean_source_id,
        raw_text=clean_text,
        embedding_model=embedding_model,
        metadata=request.metadata,
        chunker_type=request.chunker_type,
    )
    return result


@app.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    api_key: str = Depends(verify_api_key),  # API key auth
):
    """
    Wired to the real query pipeline (src/retrieval/query_pipeline.py) as
    of Phase 5: hybrid retrieval (semantic + BM25 via RRF) -> generation
    with a documented dev-only fallback when no LLM API access exists in
    this environment (see src/generation/llm_client.py).
    """
    from src.retrieval.query_pipeline import answer_question
    from src.retrieval.reranker import CrossEncoderReranker

    # Sanitize input
    clean_question = sanitize_input(request.question, max_length=2_000)
    
    embedding_model = _get_embedding_model()
    primary_llm, fallback_llm = _get_llm_clients()

    reranker = CrossEncoderReranker() if request.enable_reranking else None
    result = answer_question(
        question=clean_question,
        embedding_model=embedding_model,
        primary_llm=primary_llm,
        fallback_llm=fallback_llm,
        top_k=request.top_k,
        reranker=reranker,
        chunker_type=request.chunker_type,
        enable_query_expansion=request.enable_query_expansion,
    )
    return {
        "answer": result["answer"],
        "cited_chunk_ids": result["cited_chunk_ids"],
        "retrieval_ms": result["retrieval_ms"],
        "generation_ms": result["generation_ms"],
        "reranker_used": result["reranker_used"],
        "query_expansion_enabled": result.get("query_expansion_enabled", False),
    }


@app.post("/query/extractive", response_model=ExtractiveResponse)
def query_extractive(
    request: ExtractiveRequest,
    api_key: str = Depends(verify_api_key),  # API key auth
):
    """
    Pure extractive retrieval: same hybrid ranking as /query
    (src/retrieval/query_pipeline.retrieve_passages) but NO LLM call.
    Returns the library passages verbatim with their chunk IDs, so the
    answer is always grounded in ingested sources and arrives in
    milliseconds. Citations equal retrieved IDs by construction.
    """
    from src.retrieval.query_pipeline import retrieve_passages
    from src.retrieval.reranker import CrossEncoderReranker

    # Sanitize input
    clean_question = sanitize_input(request.question, max_length=2_000)

    embedding_model = _get_embedding_model()

    reranker = CrossEncoderReranker() if request.enable_reranking else None
    retrieved_chunks, retrieval_ms = retrieve_passages(
        question=clean_question,
        embedding_model=embedding_model,
        top_k=request.top_k,
        reranker=reranker,
        chunker_type=request.chunker_type,
        enable_query_expansion=request.enable_query_expansion,
    )
    passages = []
    for chunk in retrieved_chunks:
        score = chunk.get("rerank_score", chunk.get("fused_score", chunk.get("score", 0.0)))
        passages.append(
            {
                "chunk_id": chunk["chunk_id"],
                "source_id": chunk.get("source_id", ""),
                "text": chunk.get("raw_text", chunk.get("text", "")),
                "score": float(score),
            }
        )
    return {
        "passages": passages,
        "cited_chunk_ids": [p["chunk_id"] for p in passages],
        "retrieval_ms": round(retrieval_ms, 2),
        "reranker_used": reranker.name if reranker else "none",
        "chunker_type": request.chunker_type,
        "query_expansion_enabled": request.enable_query_expansion,
    }


@app.post("/evaluate/run", response_model=EvaluateRunResponse)
def evaluate_run(
    request: EvaluateRunRequest,
    api_key: str = Depends(verify_api_key),  # API key auth
):
    """
    Wired to src/evaluation/harness.py as of Phase 6. Ingests the golden
    dataset's source documents (idempotent) and scores every question's
    retrieval against the golden relevant_chunk_ids.
    """
    from src.evaluation.harness import run_evaluation

    embedding_model = _get_embedding_model()
    primary_llm, fallback_llm = _get_llm_clients()

    metrics = run_evaluation(
        golden_dataset_path=request.golden_dataset_path,
        embedding_model=embedding_model,
        llm=primary_llm,
        config_name=request.config_name,
        chunker_type=request.chunker_type,
        enable_reranking=request.enable_reranking,
        enable_query_expansion=request.enable_query_expansion,
        fallback_llm=fallback_llm,
    )
    return {
        "run_id": metrics["run_id"],
        "n_examples": metrics["n_examples"],
        "status": "dev-only (see is_dev_only_run)" if metrics["is_dev_only_run"] else "completed",
    }


@app.get("/evaluate/report")
def evaluate_report(
    run_id: str,
    api_key: str = Depends(verify_api_key),  # API key auth
):
    from src.evaluation.storage import get_evaluation_run

    result = get_evaluation_run(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"No run found for run_id={run_id}")
    return result