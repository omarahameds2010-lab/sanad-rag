# SANAD Integration Inventory
**Generated from actual sanad-ai-engine repository inspection**  
**Date:** 2026-10-08  
**SANAD Version:** 0.4.0 (Phase 9+10 complete)  
**Repository:** sanad-ai-engine (sanad-rag)

---

## 1. Actual Runtime / Entrypoint

| Property | Value |
|----------|-------|
| **Primary Entrypoint** | `uvicorn src.api.main:app --host 0.0.0.0 --port 8000` |
| **Framework** | FastAPI 0.115.0 (v0.142.2 in venv) |
| **Process Model** | Single-process HTTP server |
| **Workers** | None built-in (use gunicorn/uvicorn workers in production) |
| **CLI Scripts** | `scripts/run_evaluation.py` — evaluation runner only |
| **Background Jobs** | None (synchronous request/response) |

**No** separate worker process, message queue, or job scheduler exists in SANAD.

---

## 2. Public API Endpoints

All endpoints under `/` (no version prefix like `/v1`):

| Method | Path | Auth Required | Description |
|--------|------|---------------|-------------|
| `GET` | `/health` | No | Dependency status (Qdrant, SQLite) |
| `POST` | `/ingest` | **Yes** (API Key) | Ingest document → chunk → embed → store |
| `POST` | `/query` | **Yes** (API Key) | Query → hybrid retrieve → generate answer |
| `POST` | `/evaluate/run` | **Yes** (API Key) | Run evaluation on golden dataset |
| `GET` | `/evaluate/report` | **Yes** (API Key) | Get stored evaluation results |

---

## 3. Evaluation Request Schema

### POST `/evaluate/run`
```json
{
  "golden_dataset_path": "data/golden_dataset/sample_golden.json",
  "config_name": "baseline",
  "chunker_type": "arabic",
  "enable_reranking": false,
  "enable_query_expansion": false
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `golden_dataset_path` | string | Yes | — | Path to golden dataset JSON |
| `config_name` | string | No | `"baseline"` | Experiment label for comparison |
| `chunker_type` | enum | No | `"arabic"` | `"arabic"`, `"baseline"`, `"hierarchical"` |
| `enable_reranking` | boolean | No | `false` | Enable cross-encoder reranker |
| `enable_query_expansion` | boolean | No | `false` | Enable Arabic synonym expansion |

**Note:** SANAD's evaluation is **dataset-level**, not single-question. It ingests the golden dataset's source documents, runs all questions, and returns aggregate metrics (precision@k, recall@k, latency).

---

## 4. Evaluation Response Schema

### POST `/evaluate/run` → 200
```json
{
  "run_id": "uuid",
  "n_examples": 4,
  "status": "completed"
}
```

### GET `/evaluate/report?run_id=...` → 200
```json
{
  "run_id": "uuid",
  "config_name": "baseline",
  "started_at": "2026-10-08T...",
  "n_examples": 4,
  "precision_at_k": 0.75,
  "recall_at_k": 1.0,
  "p50_latency_seconds": 0.002,
  "p95_latency_seconds": 0.005
}
```

**Important:** SANAD does **NOT** have a single-question evaluation endpoint. The evaluation is batch-only against a golden dataset.

---

## 5. Query Request/Response (What LMS Would Actually Use)

### POST `/query`
```json
{
  "question": "ما هي عاصمة مصر؟",
  "top_k": 5,
  "config_name": "baseline",
  "enable_reranking": false,
  "chunker_type": "arabic",
  "enable_query_expansion": false
}
```

### Response
```json
{
  "answer": "القاهرة هي عاصمة مصر.",
  "cited_chunk_ids": ["geo_sample_1::0"],
  "retrieval_ms": 1.2,
  "generation_ms": 45.0,
  "reranker_used": "BAAI/bge-reranker-base",
  "query_expansion_enabled": false
}
```

---

## 6. Ingestion Request/Response

### POST `/ingest`
```json
{
  "source_id": "doc_123",
  "text": "النص العربي هنا...",
  "metadata": {"source": "textbook", "grade": 5},
  "chunker_type": "arabic"
}
```

### Response
```json
{
  "source_id": "doc_123",
  "n_chunks": 3,
  "embedding_model": "hash-embedding-DEV-ONLY",
  "chunker_type": "arabic",
  "n_parents": 1,
  "n_children": 3
}
```

**Chunking Strategies:**
| Type | Description | Parent/Child |
|------|-------------|--------------|
| `arabic` | Sentence-boundary aware | Flat |
| `baseline` | Naive character split | Flat |
| `hierarchical` | Parent (1800 chars) + Child (400 chars) | ✅ Yes |

---

## 7. Golden Dataset Format (Required for Evaluation)

```json
{
  "source_documents": [
    {"source_id": "geo_1", "text": "القاهرة هي عاصمة مصر..."},
    {"source_id": "geo_2", "text": "الرياض هي عاصمة السعودية..."}
  ],
  "questions": [
    {
      "id": "q1",
      "question": "ما هي عاصمة مصر؟",
      "relevant_chunk_ids": ["geo_1::0"],
      "reference_answer": "القاهرة هي عاصمة مصر."
    }
  ]
}
```

**Chunk IDs** follow format: `{source_id}::{chunk_index}` (flat) or `{source_id}::parent::{parent_id}` / `{source_id}::child::{child_id}` (hierarchical).

---

## 8. Authentication

| Mechanism | Implementation |
|-----------|----------------|
| **Type** | API Key (header-based) |
| **Header** | `X-API-Key` |
| **Config** | `SANAD_API_KEYS` env var (comma-separated) |
| **Dev Mode** | If `SANAD_API_KEYS` unset → no auth required |
| **Response on Fail** | `401` + `WWW-Authenticate: ApiKey` |

**No** JWT, OAuth2, HMAC, or service tokens implemented.

---

## 9. Async/Sync Behavior

| Endpoint | Behavior |
|----------|----------|
| `/health` | Sync (instant) |
| `/ingest` | **Sync** — blocks until chunked, embedded, stored |
| `/query` | **Sync** — blocks until retrieve + generate |
| `/evaluate/run` | **Sync** — blocks until all questions processed |

**No job IDs, no callbacks, no webhooks.** All endpoints are request/response.

---

## 10. Error Model

| Status | When |
|--------|------|
| `400` | Invalid JSON, missing required fields (Pydantic) |
| `401` | Missing/invalid API key (when auth enabled) |
| `404` | `/evaluate/report` with unknown `run_id` |
| `422` | Pydantic validation error (field constraints) |
| `429` | Rate limit exceeded (when `SANAD_RATE_LIMIT` set) |
| `500` | Unhandled exception (Qdrant down, model load fail, etc.) |

**Response Format:**
```json
{"detail": "Human-readable error message"}
```

---

## 11. Knowledge Ingestion Interface

| Aspect | Current State |
|--------|---------------|
| **File Upload** | ❌ Not implemented (text-only via `/ingest`) |
| **PDF Support** | ❌ Not implemented |
| **OCR** | ❌ Not implemented |
| **Text Extraction** | ❌ Not implemented |
| **Accepted Format** | Raw text string only (max 200KB) |
| **Chunking** | Built-in (3 strategies) |
| **Embedding** | Built-in (pluggable: HashEmbedding, multilingual-e5, bge-m3, OpenAI) |
| **Vector Store** | Qdrant (collections per chunker_type + dims) |
| **Tenant Isolation** | Via `source_id` prefix convention only |
| **Versioning** | None (re-ingest same `source_id` = overwrite) |
| **Status/Progress** | Sync response only |

---

## 12. Qdrant / Knowledge Identifiers

| Concept | Implementation |
|---------|----------------|
| **Collections** | `sanad_chunks_{chunker_type}_{dims}` (e.g., `sanad_chunks_arabic_128`) |
| **Hierarchical** | Separate: `*_children` (indexed) + `*_parents` (context lookup) |
| **Point ID** | UUID5 from `chunk_id` (stable, re-ingest = upsert) |
| **Payload** | `source_id`, `raw_text`, `chunk_id`, `chunker_type`, `chunk_level` (parent/child) |
| **Tenant Boundary** | Application-level via `source_id` prefix (e.g., `tenant_123_doc_1`) |

---

## 13. Embedding Models (Pluggable)

| Model | Env Var | Dim | Notes |
|-------|---------|-----|-------|
| `HashEmbedding` | `EMBEDDING_MODEL=hash-dev` | 128 | Dev-only, deterministic, no network |
| `multilingual-e5` | `EMBEDDING_MODEL=multilingual-e5` | 768 | Requires HF download |
| `bge-m3` | `EMBEDDING_MODEL=bge-m3` | 1024 | Requires HF download |
| `text-embedding-3-small` | `OPENAI_API_KEY` | 1536 | Requires OpenAI key |

---

## 14. LLM Generation (Pluggable)

| Provider | Class | Env Var | Notes |
|----------|-------|---------|-------|
| Anthropic | `AnthropicLLM` | `ANTHROPIC_API_KEY` | Primary (Claude Sonnet 4) |
| Fake | `FakeLLM` | None | Dev fallback, extracts top chunk |

---

## 15. Configuration / Environment Variables

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `QDRANT_URL` | `local` | No | `local` or `http://host:6333` |
| `EMBEDDING_MODEL` | `hash-dev` | No | `hash-dev`, `multilingual-e5`, `bge-m3` |
| `ANTHROPIC_API_KEY` | — | No | Required for real generation |
| `OPENAI_API_KEY` | — | No | Required for OpenAI embeddings |
| `SANAD_API_KEYS` | — | No | Comma-separated API keys (enables auth) |
| `SANAD_RATE_LIMIT` | `60` | No | Requests/minute per IP |
| `SANAD_RATE_BURST` | `10` | No | Token bucket burst |
| `SANAD_CORS_ORIGINS` | localhost | No | Comma-separated origins |
| `LOG_LEVEL` | `INFO` | No | Logging level |

---

## 16. Versioning / Provenance

| Item | Current State |
|------|---------------|
| **SANAD Version** | `0.4.0` (in `FastAPI` version) |
| **Model Version** | Embedded in `embedding_model` field (e.g., `hash-embedding-DEV-ONLY`) |
| **Prompt Version** | Not tracked (hardcoded Arabic grounding prompt) |
| **Knowledge Version** | Not tracked (re-ingest overwrites) |
| **Evaluation Run ID** | UUID4 per `/evaluate/run` |
| **Git Commit** | Not exposed via API |

---

## 17. Existing Tests / Fixtures

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `test_api.py` | 6 | Health, ingest, query, evaluate |
| `test_chunking.py` | 4 | Arabic sentence splitting, normalization |
| `test_ingestion_pipeline.py` | 3 | Ingest → store → retrieve |
| `test_query_pipeline.py` | 3 | Query → retrieve → generate |
| `test_evaluation_harness.py` | 3 | Batch evaluation metrics |
| `test_hierarchical_chunking.py` | 8 | Parent-child chunking |
| `test_query_expansion.py` | 9 | Arabic synonym/template expansion |
| `test_phase8_integration.py` | 6 | Hierarchical + rerank + expansion |
| `test_security.py` | 23 | Auth, rate limit, headers, sanitization |

**Total: 65 tests, all passing.**

---

## 18. Run/Deploy Instructions

### Local Dev
```bash
cd sanad-rag
cp .env.example .env
# Edit .env
uvicorn src.api.main:app --reload
```

### Docker (Production)
```bash
docker-compose -f docker-compose.prod.yml up -d
# Requires: ./ssl/fullchain.pem, ./ssl/privkey.pem
```

### Run Evaluation (Real Numbers)
```bash
EMBEDDING_MODEL=multilingual-e5 ANTHROPIC_API_KEY=xxx \
python scripts/run_evaluation.py --config baseline --chunker arabic
```

---

## 19. Missing Integration Capabilities (Gaps for LMS)

| Capability | Status | Notes |
|------------|--------|-------|
| Single-question evaluation | ❌ | Only batch golden-dataset evaluation |
| Async/job API | ❌ | All endpoints synchronous |
| Webhooks/callbacks | ❌ | No notification mechanism |
| PDF/document ingestion | ❌ | Text-only |
| OCR | ❌ | Not implemented |
| Multi-tenancy enforcement | ⚠️ | Via `source_id` convention only |
| Evaluation provenance/trace | ⚠️ | Only `run_id` + stored metrics |
| Prompt/config versioning | ❌ | Not versioned |
| Streaming/generator responses | ❌ | Full response only |

---

## 20. Summary: What LMS Can Actually Use Today

| LMS Need | SANAD Interface | Integration Pattern |
|----------|-----------------|---------------------|
| Grade student answer | `POST /query` with `question=student_answer + rubric` | Sync HTTP call |
| Ingest reference material | `POST /ingest` with `source_id` + `text` | Sync HTTP call |
| Run benchmark | `POST /evaluate/run` with golden dataset path | Sync HTTP call |
| Authenticate | `X-API-Key` header | Service-to-service |
| Rate limit handling | `429` + headers | Retry with backoff |

---

## 21. Recommended Integration Architecture for LMS

```
LMS (sanad-edtech)
├── modules/ai-evaluation/
│   ├── sanad-client/
│   │   ├── client.py          # Typed HTTP client
│   │   ├── models.py          # Request/response DTOs (from OpenAPI)
│   │   └── exceptions.py      # Mapped error types
│   ├── application/
│   │   └── evaluation_service.py  # Orchestrates submission → SANAD → result
│   └── domain/
│       └── evaluation.py      # LMS-owned evaluation entity
```

**Key Principle:** LMS calls SANAD as an external HTTP service. No shared DB, no code copying, independent deployments.

---

*This inventory reflects the actual sanad-ai-engine repository as of Phase 10 completion. No assumptions — only confirmed interfaces.*