"""
FastAPI application — implements the contract defined in
docs/architecture.md §4.

Phase 3 note: endpoints are real and wired to Pydantic validation, but the
retrieval/generation internals are stubs until Phase 4/5 build the actual
ingestion and retrieval pipelines. This is intentional — the API contract
is what gets locked in Phase 3, not the implementation behind it.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.evaluation.storage import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="SANAD API",
    description="Arabic-First RAG Evaluation Platform",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas (Pydantic validation at the API boundary — see architecture.md §2)
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    source_id: str = Field(..., min_length=1, max_length=200)
    text: str = Field(..., min_length=1, max_length=200_000)
    metadata: dict | None = None


class IngestResponse(BaseModel):
    source_id: str
    n_chunks: int
    embedding_model: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2_000)
    top_k: int = Field(default=5, ge=1, le=20)
    config_name: str = Field(default="baseline")


class QueryResponse(BaseModel):
    answer: str
    cited_chunk_ids: list[str]
    retrieval_ms: float
    generation_ms: float


class EvaluateRunRequest(BaseModel):
    golden_dataset_path: str
    config_name: str = Field(default="baseline")


class EvaluateRunResponse(BaseModel):
    run_id: str
    n_examples: int
    status: str


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
def ingest(request: IngestRequest):
    """
    Wired to the real pipeline (src/ingestion/pipeline.py) as of Phase 4.
    Uses HashEmbedding by default in this environment because it has no
    network access to download real embedding models — see
    src/embeddings/base.py docstring. Set EMBEDDING_MODEL=multilingual-e5
    once running with real network/model access.
    """
    from src.ingestion.pipeline import ingest_document
    from src.embeddings.base import HashEmbedding

    embedding_model = _get_embedding_model()
    result = ingest_document(
        source_id=request.source_id,
        raw_text=request.text,
        embedding_model=embedding_model,
        metadata=request.metadata,
    )
    return result


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Wired to the real query pipeline (src/retrieval/query_pipeline.py) as
    of Phase 5: hybrid retrieval (semantic + BM25 via RRF) -> generation
    with a documented dev-only fallback when no LLM API access exists in
    this environment (see src/generation/llm_client.py).
    """
    from src.retrieval.query_pipeline import answer_question
    from src.generation.llm_client import AnthropicLLM, FakeLLM

    embedding_model = _get_embedding_model()
    primary_llm, fallback_llm = _get_llm_clients()

    result = answer_question(
        question=request.question,
        embedding_model=embedding_model,
        primary_llm=primary_llm,
        fallback_llm=fallback_llm,
        top_k=request.top_k,
    )
    return {
        "answer": result["answer"],
        "cited_chunk_ids": result["cited_chunk_ids"],
        "retrieval_ms": result["retrieval_ms"],
        "generation_ms": result["generation_ms"],
    }


@app.post("/evaluate/run", response_model=EvaluateRunResponse)
def evaluate_run(request: EvaluateRunRequest):
    """
    Wired to src/evaluation/harness.py as of Phase 6. Ingests the golden
    dataset's source documents (idempotent) and scores every question's
    retrieval against the golden relevant_chunk_ids.
    """
    from src.evaluation.harness import run_evaluation

    embedding_model = _get_embedding_model()
    primary_llm, _ = _get_llm_clients()

    metrics = run_evaluation(
        golden_dataset_path=request.golden_dataset_path,
        embedding_model=embedding_model,
        llm=primary_llm,
        config_name=request.config_name,
    )
    return {
        "run_id": metrics["run_id"],
        "n_examples": metrics["n_examples"],
        "status": "dev-only (see is_dev_only_run)" if metrics["is_dev_only_run"] else "completed",
    }


@app.get("/evaluate/report")
def evaluate_report(run_id: str):
    from src.evaluation.storage import get_evaluation_run

    result = get_evaluation_run(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"No run found for run_id={run_id}")
    return result


# ---------------------------------------------------------------------------
# Dependency health checks
# ---------------------------------------------------------------------------

def _get_embedding_model():
    """
    Selects the embedding backend. Defaults to HashEmbedding because this
    sandboxed environment has no network access to huggingface.co or an
    OpenAI API key. On Omar's own machine (real network + model access),
    set EMBEDDING_MODEL=multilingual-e5 to use the real model documented
    in docs/research.md.
    """
    import os
    from src.embeddings.base import HashEmbedding, SentenceTransformerEmbedding

    choice = os.environ.get("EMBEDDING_MODEL", "hash-dev")
    if choice == "multilingual-e5":
        return SentenceTransformerEmbedding()
    return HashEmbedding()


def _get_llm_clients():
    """
    Selects primary + fallback LLM backends. Defaults to FakeLLM because
    this sandboxed environment has no ANTHROPIC_API_KEY / network access
    for real generation. On Omar's own machine, set ANTHROPIC_API_KEY to
    get real generation with the FakeLLM kept only as the fallback.
    """
    import os
    from src.generation.llm_client import AnthropicLLM, FakeLLM

    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicLLM(), FakeLLM()
    return FakeLLM(), None


def _check_qdrant() -> bool:
    try:
        from src.retrieval.vector_store import get_client

        client = get_client()
        client.get_collections()
        return True
    except Exception:
        return False


def _check_db() -> bool:
    try:
        from src.evaluation.storage import get_connection

        with get_connection() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False
