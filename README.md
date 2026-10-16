# SANAD (سند) — Arabic-First RAG Evaluation Platform

> **سند**: الأساس اللي بترتكز عليه — بالظبط زي ما المشروع ده بيبني أساس retrieval سليم للمحتوى العربي قبل ما نديه لأي LLM.

**Status:** Phase 11a complete — extractive retrieval, branch sub-filters, LMS integration surface, and filler quarantine live and tested (84 tests passing); real benchmark numbers pending a run with network/model access (see `docs/DEV_BENCHMARKS.md` for honest dev-only numbers)
[![Tests](https://github.com/omarahameds2010-lab/sanad-rag/actions/workflows/tests.yml/badge.svg)](https://github.com/omarahameds2010-lab/sanad-rag/actions/workflows/tests.yml)
**Author:** Omar Ahmed ([GitHub](https://github.com/omarahameds2010-lab) · [LinkedIn](https://linkedin.com/in/omar-ahmed-7b97613b1))

---

## 1. The Problem

Every mainstream embedding and retrieval stack (OpenAI `text-embedding-3`, Cohere, most open-source rerankers) is benchmarked and tuned primarily on **English** text. When you build a RAG system over Arabic content — educational material, legal documents, government publications — you run into three concrete failures:

1. **Chunking breaks words and sentence boundaries** because most chunkers assume Latin-script tokenization and don't account for Arabic diacritics (تشكيل) or right-to-left sentence structure.
2. **Retrieval precision drops** because embedding models trained mostly on English don't capture Arabic morphology (root-pattern system) well.
3. **Hallucination rate increases** because weak retrieval means the LLM gets poor context and starts filling gaps with invented facts.

This is a real, unsolved, and under-measured problem in the Arabic AI tooling space — most people building Arabic AI products right now are not measuring any of this, they're just shipping and hoping.

## 2. What This Project Actually Does

SANAD is **not** a "upload PDF → chatbot" demo. It is an evaluation-first RAG system that:

- Ingests Arabic (and mixed Arabic/English) documents through a purpose-built pipeline
- Compares multiple embedding strategies head-to-head on the same Arabic corpus
- Implements Arabic-aware chunking (sentence/paragraph boundary detection that respects Arabic punctuation and diacritics)
- Builds a **golden evaluation dataset** (manually verified Arabic question/answer pairs with known source passages)
- Measures retrieval precision, recall, groundedness, hallucination rate, latency, and cost — **before and after** each optimization
- Reports real numbers, not claims

## 3. Architecture (MVP)

```
┌─────────────────┐
│   Data Sources    │  (Arabic educational / public legal text)
└────────┬─────────┘
         ▼
┌─────────────────────────┐
│  Ingestion Layer          │  parsing → normalization (تشكيل handling)
│  src/ingestion/           │  → metadata extraction
└────────┬─────────────────┘
         ▼
┌─────────────────────────┐
│  Arabic-Aware Chunking    │  sentence-boundary aware, configurable
│  src/chunking/            │  overlap, size limits
└────────┬─────────────────┘
         ▼
┌─────────────────────────┐
│  Embedding Layer          │  pluggable: multilingual-e5, AraBERT-based,
│  src/embeddings/          │  OpenAI text-embedding-3 (comparison mode)
└────────┬─────────────────┘
         ▼
┌─────────────────────────┐
│  Vector Store             │  Qdrant (self-hosted, free) / pgvector
└────────┬─────────────────┘
         ▼
┌─────────────────────────┐
│  Retrieval + Reranking    │  hybrid search (BM25 + semantic) + reranker
│  src/retrieval/           │
└────────┬─────────────────┘
         ▼
┌─────────────────────────┐
│  Generation Layer         │  LLM call + structured grounding check +
│                            │  fallback model on failure
└────────┬─────────────────┘
         ▼
┌─────────────────────────┐
│  Evaluation Layer          │  golden dataset → precision/recall/
│  src/evaluation/           │  groundedness/hallucination/latency/cost
└─────────────────────────┘
```

## 4. Evaluation Methodology

No invented benchmark numbers. Every metric below is computed against `data/golden_dataset/` and logged to `docs/benchmark_results.md`:

| Metric | What it measures | Baseline | Optimized |
|---|---|---|---|
| Retrieval Precision@k | % of retrieved chunks that are actually relevant | TBD | TBD |
| Retrieval Recall@k | % of relevant chunks successfully retrieved | TBD | TBD |
| Groundedness | % of generated answers fully supported by retrieved context | TBD | TBD |
| Hallucination Rate | % of answers containing unsupported claims | TBD | TBD |
| p50 / p95 Latency | Response time distribution | TBD | TBD |
| Cost per query | Tokens + embedding + infra cost | TBD | TBD |

Experiments compared: **baseline chunking vs Arabic-aware chunking**, **single embedding model vs comparison ensemble**, **no reranking vs reranking enabled**.

## 5. Tech Stack

- **Backend:** Python (FastAPI)
- **Vector DB:** Qdrant (self-hosted, Docker)
- **Embeddings:** pluggable — starts with open-source multilingual models to keep cost at $0 during development
- **Frontend:** React dashboard (Phase 8) showing live comparisons and metrics
- **Deployment:** Docker Compose locally → Vercel/Render for the dashboard

## 6. Project Status / Roadmap

- [x] Phase 1 — Problem discovery & project scaffolding
- [x] Phase 2 — Research (see `docs/research.md`): confirmed the Arabic retrieval gap in public benchmarks, added `bge-m3` to the embedding shortlist, confirmed no existing system ships a reusable evaluation harness
- [x] Phase 3 — Architecture finalization (see `docs/architecture.md`): locked API contract (`/health`, `/ingest`, `/query`, `/evaluate/run`, `/evaluate/report`), SQLite + Qdrant dual-store design, per-component failure strategy — FastAPI skeleton implemented and tested (10/10 tests passing)
- [x] Phase 4 — Ingestion pipeline (see `src/ingestion/pipeline.py`): `/ingest` fully wired — normalize → Arabic-aware chunk → embed → Qdrant (local embedded mode for dev, Docker for production) → SQLite bookkeeping. Tested end-to-end with real Arabic text (13/13 tests passing). Uses a deterministic `HashEmbedding` in this dev environment (no network access to Hugging Face) — switches to real `multilingual-e5`/`bge-m3` via `EMBEDDING_MODEL` env var once real models are available
- [x] Phase 5 — Retrieval + generation (see `src/retrieval/hybrid.py`, `src/generation/llm_client.py`): `/query` fully wired — hybrid retrieval (semantic Qdrant + BM25 keyword search, fused via Reciprocal Rank Fusion) → LLM generation with a structured grounding prompt → automatic fallback on primary LLM failure (tested by simulating an outage). Uses `FakeLLM` in this dev environment (no `ANTHROPIC_API_KEY`/network access) — switches to real `AnthropicLLM` generation automatically once the key is set. 16/16 tests passing, including a real bug found and fixed: the embedded Qdrant client needs to be a per-process singleton or concurrent calls raise a file-lock error
- [x] Phase 6 — Golden dataset + evaluation harness (see `data/golden_dataset/sample_golden.json`, `src/evaluation/harness.py`, `scripts/run_evaluation.py`): `/evaluate/run` fully wired — ingests golden source documents idempotently, scores real retrieval (not generation citations) against golden `relevant_chunk_ids`, persists every run to SQLite, and a CLI script (`scripts/run_evaluation.py`) appends real rows to `docs/benchmark_results.md` — **run this on your own machine with real network access for trustworthy numbers.** A dev-only smoke test in this sandbox (documented, not counted as a real result) confirmed the harness itself works: 4/4 questions, 100% recall@5. Two real bugs found and fixed during this phase: (1) evaluation was originally scoring the LLM's *cited* chunks instead of the *retrieved* chunks — that would have measured generation behavior, not retrieval quality; (2) SQLite tables only existed if the FastAPI app had started first — fixed by making the storage layer self-initializing. 19/19 tests passing (since grown to 70/70 with Phases 7–9). Later extended with rubric-aware single-submission evaluation (`POST /evaluate/submission`, see `src/evaluation/runtime_evaluator.py`, `src/evaluation/schemas.py`, `src/api/evaluation_runtime.py`) — criterion-level scoring with quoted evidence, deterministic validation/aggregation, and provenance tracking
- [x] Phase 7 — Optimization pass (see `src/ingestion/pipeline.py`, `src/retrieval/query_pipeline.py`, `src/retrieval/reranker.py`, `src/retrieval/query_expansion.py`, `scripts/run_evaluation.py`): pluggable chunker selection (baseline vs Arabic-aware vs hierarchical parent-child), optional cross-encoder reranking, Arabic query expansion — every arm selectable per-request (`chunker_type`, `enable_reranking`, `enable_query_expansion`) and comparable via `config_name` runs
- [x] Phase 8 — User-facing frontend (see `frontend/`): Arabic RTL React app covering every API endpoint — material ingestion, RAG Q&A, rubric-aware submission evaluation with per-criterion evidence display, and benchmark lab; Dockerized and routed via nginx in `docker-compose.prod.yml`
- [x] Phase 9 — Security review (see `src/security/__init__.py`, `tests/test_security.py`): API-key auth (`X-API-Key`), per-IP rate limiting, security headers, CORS, input sanitization/injection detection
- [x] Phase 10 — Deployment (see `Dockerfile`, `docker-compose.prod.yml`, `nginx/`, `.github/workflows/ci-cd.yml`): multi-service prod compose (nginx + api + qdrant + frontend + redis), TLS reverse proxy routing UI vs API, CI pipeline (lint → security scan → tests → Docker build → deploy)
- [x] Phase 11a — LMS integration surface: pure extractive retrieval (`POST /query/extractive`, no LLM, ms responses, citations by construction), library branch taxonomy (`src/ingestion/branches.py`: nahw/sarf/balagha/arood/adab/lugha/quran/hadith/sira/misc) with `branch`/`source_ids` scoping on both query paths, graded-judgement fallback + forced teacher review on validation failure, filler quarantine tooling (`scripts/prune_index.py`), dev benchmarks log (`docs/DEV_BENCHMARKS.md`), and a working LMS client (`sanad-edtech: assignments/sanad.client.ts`)
- [ ] Phase 11b — Final documentation (remaining: real benchmark rows in `docs/benchmark_results.md` once a network/model run with `bge-m3` is available; LMS end-to-end proof waits on the platform DB connection)

## 7. Limitations (Honest, as of this stage)

- Golden dataset is currently small (student-scale) — statistical confidence is limited until it grows.
- No fine-tuning of embedding models is planned for MVP — out of scope for a student budget with no GPU cluster.
- This is not a production-scale system; see `docs/architecture_tiers.md` for MVP vs Production vs Hyperscale distinctions.

## 8. Setup

See `docs/setup.md` for full instructions (Docker, environment variables, running the pipeline).

---

*This project documents a real engineering journey: from full-stack development into AI engineering, with actual measurement, not just a demo.*
