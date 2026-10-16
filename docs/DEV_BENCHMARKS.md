# SANAD Dev Benchmarks Log (Honest Plumbing Proof — NOT Production Numbers)

> **Read this first.** Every number below was produced on dev-only backends
> (`HashEmbedding` + Gemini-with-`FakeLLM`-fallback, embedded Qdrant on one
> Windows machine). They prove the pipeline, the endpoints, and the harness
> work end-to-end. They do **NOT** measure Arabic retrieval quality and must
> **NOT** be copied into `docs/benchmark_results.md`, which is reserved for
> runs with real embeddings (`bge-m3`/`multilingual-e5`) on a real host.
> Each evaluation row traces to a stored SQLite run (`data/sanad.db`,
> `evaluation_runs`) — nothing here is hand-typed.

Environment (2026-10-16): Windows + Python 3.14 venv,
`EMBEDDING_MODEL=hash-dev` (128-dim trigram hash), LLM = Gemini 3.8 Flash
with `FakeLLM` fallback (fallback wired into `/evaluate/run` by commit
`456c988`; without it, Gemini 45s timeouts returned HTTP 500),
Qdrant embedded (`data/qdrant_local`), `bge-m3` unavailable on this host
(scipy DLL blocked by Application Control policy).

## 1. Test suite: 82/82 passing (~185s)

| File | Tests | Covers |
|---|---|---|
| `test_security.py` | 23 | sanitization, injection detection, rate limit, auth, headers |
| `test_api.py` | 17 | health, ingest, query, extractive + branch/source filters, evaluate report |
| `test_query_expansion.py` | 9 | Arabic synonyms/templates, variation caps |
| `test_query_pipeline.py` | 9 | full contract, empty index, fallback, extractive parity, branch/source scope, prune |
| `test_hierarchical_chunking.py` | 8 | parent-child linkage, size invariants |
| `test_phase8_integration.py` | 6 | hierarchical + rerank + expansion |
| `test_chunking.py` | 4 | Arabic sentence splitting, diacritics normalization |
| `test_ingestion_pipeline.py` | 3 | ingest → store → retrieve |
| `test_evaluation_harness.py` | 3 | batch metrics, persistence, recall tripwire |
| **Total** | **82** | **0 failures** |

Command: `$env:EMBEDDING_MODEL='hash-dev'; pytest tests/ -q`
(API server must be stopped first: embedded Qdrant takes an exclusive
file lock — a documented limitation, not a bug.)

## 2. Retrieval evaluation on the real library (`sanad_library_golden.json`)

Golden questions are written against chunk text read verbatim from the
banks (chunk ids reproduced by re-chunking with the same `arabic`
settings). Filler files (`adab_1..8`, `*touch*`, mojibake lists) are
deliberately excluded from the golden set.

| run_id (short) | config | n | Precision@5 | Recall@5 | p50 (s) | p95 (s) |
|---|---|---|---|---|---|---|
| `3610f320` | library-baseline | 8 | 0.125 | 0.625 | 10.69 | 161.73 |
| `98e0969a` | library-pruned | 8 | 0.125 | 0.625 | 5.56 | 40.89 |
| `d740fec8` | library-v2 | 10 | 0.140 | 0.700 | 1.86 | 8.00 |
| `42ed6da4` | library-v3 | 12 | 0.150 | 0.750 | 0.57 | 12.82 |
| `736268c2` | library-v4 | 14 | 0.143 | 0.714 | 3.90 | 14.73 |

Reading: recall climbs as genuine banks are added (0.625 → 0.75);
precision stays ~0.13–0.15, the expected ceiling of trigram-hash
retrieval. Latency is Gemini-dominated (timeouts + fallback), **not**
retrieval time — pure retrieval answers in ~30–55ms (see §4).

## 3. Live index snapshot (2026-10-16, `sanad_chunks_*`)

3,657 points total. Queryable library (`arabic_128`): 695 points across
27 sources — 17 genuine banks + golden samples + a few named test
residues (`branch-*-doc`, `doc1`, `test`). Legacy `*_1024` collections
(2,836 points) are stale remains from mixed-backend experiments, not
live data. 266 filler points were quarantined and deleted (commit
`e60b1c2`); the 10 filler sources are listed in
`scripts/prune_index.py::FILLER_SOURCES` with the audit rationale.

## 4. Live endpoint proofs (this host, hash-dev)

- `POST /query/extractive` (no LLM): 3 passages in **38.8ms**,
  `cited_chunk_ids == retrieved` by construction.
- `branch=nahw` → only `gen_fixed_irab` chunks; `branch=sarf` → only
  `gen_sarf_tables_p1`; `source_ids=[adab_2_v1, adab_3_v1]` → only those
  sources. All HTTP 200.
- `POST /evaluate/run` on 14 questions: HTTP 200 via fallback
  (previously HTTP 500 on Gemini timeout — fixed in `456c988`).

## 5. How to produce REAL numbers

1. Linux/Docker host without DLL restrictions.
2. `EMBEDDING_MODEL=bge-m3` (or `multilingual-e5`) + `QDRANT_URL`
   pointing at Qdrant server (`docker compose up -d qdrant` — no code
   change needed, `vector_store.get_client` already switches).
3. Re-ingest via `scripts/sanad_safe_injector.py`, then
   `POST /evaluate/run` on `data/golden_dataset/sanad_library_golden.json`.
4. Append the resulting row to `docs/benchmark_results.md` — that file,
   not this one, is the production record.

## 6. LMS integration verification (sanad-edtech, 2026-10-16)

Cross-repo proof that the Learning Platform actually consumes the
contracts above. SANAD commits: extractive endpoint (`a9f948c`),
branch filters (`117a508`, `7a002ee`), grading fallback (`456c988`),
forced review (`3ca918a`). LMS commits: `78b9d69` (SanadClient +
evaluate wiring + prisma secret fix), `3e4340d` (scope map, ask-sanad
proxy, AI grading panel).

| Check | Result |
|---|---|
| LMS api suite (`apps/api`, jest) | 187/187 passing (19 suites) |
| LMS api production build (`nest build` via pre-commit) | green, DI graph valid |
| LMS web suite (vitest) + `tsc --noEmit` | 51/51 passing, clean |
| Live `POST /query/extractive` + `branch`/`source_ids` filters | HTTP 200, scoped correctly |
| Live `POST /evaluate/submission` (2-criteria rubric) | `COMPLETED`, criteria scored, `needs_teacher_review` honored |
| Contract lessons (both enforced live) | `request_id`/`submission_id`/`question_id` must be UUIDs (else 422); rubric sum must equal `max_score` (else 422) |

Blocked live E2E (not a code gap): the LMS api needs its Railway
`DATABASE_URL`, which is truncated in the local `.env` — booting `:3002`
and running teacher evaluate → review → official grade waits on a
complete connection string pasted locally by the owner.
