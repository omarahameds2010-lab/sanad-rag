# SANAD (سند) — Arabic-First RAG Evaluation Platform

> **سند**: الأساس اللي بترتكز عليه — بالظبط زي ما المشروع ده بيبني أساس retrieval سليم للمحتوى العربي قبل ما نديه لأي LLM.

**Status:** 🚧 Phase 6 of 11 complete — evaluation harness live and tested; real benchmark numbers pending a run with network/model access
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
- [x] Phase 6 — Golden dataset + evaluation harness (see `data/golden_dataset/sample_golden.json`, `src/evaluation/harness.py`, `scripts/run_evaluation.py`): `/evaluate/run` fully wired — ingests golden source documents idempotently, scores real retrieval (not generation citations) against golden `relevant_chunk_ids`, persists every run to SQLite, and a CLI script (`scripts/run_evaluation.py`) appends real rows to `docs/benchmark_results.md` — **run this on your own machine with real network access for trustworthy numbers.** A dev-only smoke test in this sandbox (documented, not counted as a real result) confirmed the harness itself works: 4/4 questions, 100% recall@5. Two real bugs found and fixed during this phase: (1) evaluation was originally scoring the LLM's *cited* chunks instead of the *retrieved* chunks — that would have measured generation behavior, not retrieval quality; (2) SQLite tables only existed if the FastAPI app had started first — fixed by making the storage layer self-initializing. 19/19 tests passing
- [ ] Phase 7 — Optimization pass (measured before/after: baseline vs Arabic-aware chunking, single vs comparison-ensemble embeddings, reranking on/off — this is where multiple `config_name` runs get compared using the harness built in Phase 6)
- [ ] Phase 8 — Dashboard frontend
- [ ] Phase 9 — Security review
- [ ] Phase 10 — Deployment
- [ ] Phase 11 — Final documentation

## 7. Limitations (Honest, as of this stage)

- Golden dataset is currently small (student-scale) — statistical confidence is limited until it grows.
- No fine-tuning of embedding models is planned for MVP — out of scope for a student budget with no GPU cluster.
- This is not a production-scale system; see `docs/architecture_tiers.md` for MVP vs Production vs Hyperscale distinctions.

## 8. Setup

See `docs/setup.md` for full instructions (Docker, environment variables, running the pipeline).

---

*This project documents a real engineering journey: from full-stack development into AI engineering, with actual measurement, not just a demo.*
