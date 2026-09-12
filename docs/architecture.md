# Phase 3 — Architecture (Finalized)

This finalizes the architecture sketched in the README and validated by
`docs/research.md`. Every decision below states *why*, not just *what*,
because the goal is to be able to defend each choice in an interview.

## 1. System Architecture

```
                     ┌──────────────────────┐
                     │        Client          │  (dashboard, Phase 8)
                     └──────────┬───────────┘
                                │ HTTP (JSON)
                     ┌──────────▼───────────┐
                     │   FastAPI Application  │  src/api/main.py
                     │  ─────────────────────  │
                     │  /health                │
                     │  /ingest                │
                     │  /query                 │
                     │  /evaluate/run          │
                     │  /evaluate/report       │
                     └──┬────────────┬────────┘
                        │            │
        ┌───────────────▼──┐   ┌─────▼──────────────┐
        │  Ingestion +       │   │  Retrieval +        │
        │  Chunking Pipeline │   │  Generation Pipeline │
        │  (src/ingestion,   │   │  (src/retrieval)     │
        │   src/chunking)    │   │                     │
        └───────────────┬──┘   └─────┬──────────────┘
                        │            │
                ┌───────▼────────────▼───────┐
                │        Qdrant (vectors)      │
                │   + SQLite (run metadata,     │
                │     evaluation results)       │
                └───────────────────────────────┘
```

## 2. Why FastAPI (not Flask/Django)

- Native Pydantic request/response validation — matters because "what
  happens if the model returns invalid output" (a required failure-mode
  question) is partly solved at the API boundary for free.
- Async support matters for I/O-bound calls (embedding APIs, LLM calls,
  Qdrant queries) without adding a separate task queue for the MVP.
- Automatic OpenAPI docs — free, correct API documentation instead of a
  hand-maintained one that goes stale.

## 3. Data Flow (step by step)

**Ingestion path:**
1. Document arrives (`POST /ingest`) → saved to `data/raw/`.
2. `src/ingestion/normalize.py` produces (a) a display-clean version and
   (b) a retrieval-normalized version — kept separate on purpose (see the
   module docstring: normalization must never destroy what gets cited).
3. `src/chunking/arabic_chunker.py` splits into `Chunk` objects with
   sentence-index provenance (needed later for citation and for
   precision/recall scoring against the golden dataset).
4. Each chunk is embedded (pluggable model, `src/embeddings/base.py`) and
   upserted into Qdrant with payload: `{source_id, chunk_id, start_sentence,
   end_sentence, raw_text, normalized_text}`.
5. Chunk metadata is also written to SQLite (`ingested_documents`,
   `chunks` tables) — see §5 for why SQLite exists alongside Qdrant.

**Query path:**
1. `POST /query` receives a question.
2. Question is normalized the same way as ingested text (same function —
   this consistency is what makes retrieval actually work).
3. Hybrid retrieval: BM25 keyword search + Qdrant semantic search, results
   merged and reranked.
4. Top-k chunks become the LLM context; the LLM call requests a
   structured response `{answer, cited_chunk_ids}` — grounding is
   enforced at the schema level, not just hoped for.
5. If the primary LLM call fails or times out, a fallback model is used
   (see §6 — Reliability).
6. Response returned with latency breakdown (retrieval ms, generation ms)
   for cost/latency observability from day one.

**Evaluation path (the one that matters most for this project):**
1. `POST /evaluate/run` takes a golden dataset file + a named
   configuration (which chunker settings, which embedding model, reranking
   on/off).
2. Runs every golden question through the query path above, but logs
   every intermediate result instead of just the final answer.
3. `src/evaluation/metrics.py` computes precision@k, recall@k, latency
   percentiles against the logged results.
4. Results are written to SQLite (`evaluation_runs` table) so that
   baseline vs optimized comparisons in `docs/benchmark_results.md` are
   generated from real stored runs, never typed in by hand.

## 4. API Design (Phase 3 contract — implemented as stubs now, filled in Phase 4/5)

| Endpoint | Method | Request | Response | Notes |
|---|---|---|---|---|
| `/health` | GET | — | `{status, qdrant_ok, db_ok}` | checks dependencies, not just "the process is alive" |
| `/ingest` | POST | `{source_id, text, metadata?}` | `{source_id, n_chunks, embedding_model}` | synchronous for MVP; a queue is a Production-tier upgrade (see `docs/architecture_tiers.md`) |
| `/query` | POST | `{question, top_k?, config_name?}` | `{answer, cited_chunk_ids, retrieval_ms, generation_ms}` | `config_name` lets the same endpoint serve baseline vs optimized experiments |
| `/evaluate/run` | POST | `{golden_dataset_path, config_name}` | `{run_id, n_examples, status}` | kicks off a full evaluation run |
| `/evaluate/report` | GET | `?run_id=` | full metrics object from `src/evaluation/metrics.py` | this is what feeds `docs/benchmark_results.md` |

## 5. Database Design

Two stores, deliberately, not one — each justified for the MVP:

- **Qdrant** — vectors + payload needed at *query time* (raw/normalized
  text, sentence range, source id). Optimized for similarity search, not
  for relational queries.
- **SQLite** — structured records needed for *bookkeeping and evaluation*:
  which documents were ingested when, which evaluation runs happened with
  which config, and their aggregate metrics. SQLite (not Postgres) for the
  MVP because it's zero-infrastructure — a single file, no server to run —
  and this project's write volume is nowhere near what would need a real
  RDBMS server. Moving to Postgres is a one-line justified Production-tier
  change (see `docs/architecture_tiers.md`), not a redesign, because the
  schema below uses only standard SQL.

```sql
CREATE TABLE ingested_documents (
    source_id TEXT PRIMARY KEY,
    ingested_at TIMESTAMP,
    n_chunks INTEGER,
    embedding_model TEXT
);

CREATE TABLE chunks (
    chunk_id TEXT PRIMARY KEY,
    source_id TEXT REFERENCES ingested_documents(source_id),
    start_sentence INTEGER,
    end_sentence INTEGER
);

CREATE TABLE evaluation_runs (
    run_id TEXT PRIMARY KEY,
    config_name TEXT,
    started_at TIMESTAMP,
    n_examples INTEGER,
    precision_at_k REAL,
    recall_at_k REAL,
    p50_latency_seconds REAL,
    p95_latency_seconds REAL
);
```

## 6. Reliability (failure strategy per component)

| Component | What happens if it fails | Strategy |
|---|---|---|
| Embedding model call | Request would hang or error | Timeout + retry with exponential backoff (max 3 attempts) |
| Qdrant unavailable | Query path breaks entirely | `/health` reports it; `/query` returns a clear 503, not a silent wrong answer |
| LLM provider down/slow | Generation step fails | Fallback model configured in `src/embeddings/base.py`-style pluggable interface; documented, not yet implemented (Phase 5) |
| Malicious/oversized input | Resource exhaustion, injection into prompt | Pydantic request size limits + input sanitization before it ever reaches the LLM prompt (Phase 9 security review formalizes this) |
| Golden dataset malformed | Evaluation run silently produces garbage metrics | `evaluate_retrieval()` already asserts alignment between golden set and results (see `src/evaluation/metrics.py`) — fails loudly instead of reporting a wrong number |

## 7. What Phase 3 deliberately does NOT decide yet

- Exact reranker model — decided empirically in Phase 5/6 once the golden
  dataset exists to measure it against.
- Frontend framework details — Phase 8.
- Auth mechanism beyond "simple API key for MVP" — Phase 9.
