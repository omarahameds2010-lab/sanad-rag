# SANAD Dev Benchmarks Log (Honest Plumbing Proof — NOT Production Numbers)

> **Read this first.** Every number below was produced on dev-only backends
> (`HashEmbedding` + Gemini-with-`FakeLLM`-fallback, embedded Qdrant on one
> Windows machine). They prove the pipeline, the endpoints, and the harness
> work end-to-end. They do **NOT** measure Arabic retrieval quality and must
> **NOT** be copied into `docs/benchmark_results.md`, which is reserved for
> runs with real embeddings (`bge-m3`/`multilingual-e5`) on a real host.
> Each evaluation row traces to a stored SQLite run (`data/sanad.db`,
> `evaluation_runs`) — nothing here is hand-typed.

> **2026-10-27 update:** `bge-m3` now loads and embeds on this host (the old
> scipy/Application Control blocker is gone). The first real retrieval-only
> run on the 23-source manifest corpus is recorded in
> `docs/benchmark_results.md`. The rows below remain dev-only and are kept
> as pipeline history, not as retrieval-quality claims.

Environment (2026-10-16): Windows + Python 3.14 venv,
`EMBEDDING_MODEL=hash-dev` (128-dim trigram hash), LLM = Gemini 3.8 Flash
with `FakeLLM` fallback (fallback wired into `/evaluate/run` by commit
`456c988`; without it, Gemini 45s timeouts returned HTTP 500),
Qdrant embedded (`data/qdrant_local`), `bge-m3` unavailable on this host
(scipy DLL blocked by Application Control policy).

## 1. Test suite: 100/100 passing (2026-10-27)

| File | Tests | Covers |
|---|---|---|
| `test_security.py` | 23 | sanitization, injection detection, rate limit, auth, headers |
| `test_api.py` | 17 | health, ingest, query, extractive + branch/source filters, evaluate report |
| `test_query_expansion.py` | 9 | Arabic synonyms/templates, variation caps |
| `test_query_pipeline.py` | 9 | full contract, empty index, fallback, extractive parity, branch/source scope, prune |
| `test_hierarchical_chunking.py` | 8 | parent-child linkage, size invariants |
| `test_evaluation_harness.py` | 8 | batch metrics, persistence, retrieval-only, no-ingest, axes roundtrip, legacy migration |
| `test_phase8_integration.py` | 6 | hierarchical + rerank + expansion |
| `test_corpus_manifest.py` | 6 | SHA-256 manifest, filler removal, strict audit |
| `test_ingestion_pipeline.py` | 5 | ingest → store → retrieve, stale-tail replacement |
| `test_chunking.py` | 4 | Arabic sentence splitting, diacritics normalization |
| `test_embeddings.py` | 3 | e5 query/passage prefixes, bge-m3 no-prefix |
| `test_runtime_validation.py` | 2 | forced teacher review on invalid grades |
| **Total** | **100** | **0 failures** |

Experimental axes (`embedding/llm/chunker/reranker/expansion/dev-flag`)
are now persisted per run — comparisons are DB-reproducible;
pre-existing rows read back with NULL axes.

Command: `$env:EMBEDDING_MODEL='hash-dev'; pytest tests/ -q`
(API server must be stopped first: embedded Qdrant takes an exclusive
file lock — a documented limitation, not a bug.)

## 1b. CI pipeline (2026-10-27)

The public badge was red for 21 consecutive runs. Root causes, all fixed:

| Symptom | Cause | Fix |
|---|---|---|
| Collection error in `tests.yml` | workflow installed a hand-written package subset instead of `requirements.txt` | installs `requirements-ci.txt` (same pins the suite verifies) |
| `ruff` failed | 175 violations, never linted | `pyproject.toml` selects E4/E7/E9/F/UP/I/RUF; **0 violations** on `src/ tests/ scripts/` |
| `pip-audit` failed | 14 CVEs in Starlette 0.46.0 | Starlette upgraded to 1.3.1 → **0 known vulnerabilities** |
| Qdrant service unhealthy | healthcheck used `wget`, absent from the image | runner-side probe of `/readyz` before tests |
| 5 auth tests failed without `.env` | app runs with auth **disabled** when `SANAD_API_KEYS` is unset (dev fail-open), while tests assert 401 | CI sets a test-only `SANAD_API_KEYS=demo-key-123` |

`ruff`/`mypy` also caught three real defects, fixed here: a duplicate
`"من"` dictionary key in `query_expansion.py` (the interrogative sense was
silently overwritten by the preposition sense), `c.max_score` read on a
dataclass that only defines `max` (`rubric.py`), and `temperature=` passed
to an Anthropic SDK version that no longer accepts it (`llm_client.py`).
The duck-typed `SimpleGradingConfig` was replaced by the real validated
`GradingConfig` model.

Verification, run in a clean venv with `requirements-ci.txt`, `.env` removed
and no `sentence-transformers`/`torch` installed (i.e. CI conditions):

| Mode | Result | Time |
|---|---|---|
| Embedded Qdrant (`tests.yml`) | **86 passed** | 3.91 s |
| Qdrant server 1.20.0 via Docker (`ci-cd.yml`) | **86 passed** | 4.71 s |
| `ruff check` / `mypy` | clean / clean (33 files) | — |
| `pip-audit -r requirements.txt` | 0 vulnerabilities | — |

CI deliberately omits the 2 GB torch install: every embedding import is
lazy. The cross-encoder test therefore exercises the documented keyword
fallback; real `bge-reranker-base` numbers stay in §2b (dev-host run).
Docker image build/push now happens only on releases, not on every commit.

Command: `python -m pytest tests/ -q` (CI simulation used the same command).

## 2. Retrieval evaluation on the real library (`sanad_library_golden.json`)

Golden questions are written against chunk text read verbatim from the
banks (chunk ids reproduced by re-chunking with the same `arabic`
settings). The shipped corpus is now governed by
`data/knowledge/corpus_manifest.json`: 23 sources with SHA-256 fingerprints,
enforced by `scripts/audit_corpus.py --strict` and
`tests/test_corpus_manifest.py`. Template filler (`adab_1..8`), the
English-padded `adab_al_talib` / `hadith_qudsiya_large` stubs, the
`seeds_*` generator inputs, and the 343 local-only `gen_supreme_core_*`
shards are excluded by policy and cannot be indexed by
`scripts/ingest_corpus.py`.

| run_id (short) | config | n | Precision@5 | Recall@5 | p50 (s) | p95 (s) |
|---|---|---|---|---|---|---|
| `3610f320` | library-baseline | 8 | 0.125 | 0.625 | 10.69 | 161.73 |
| `98e0969a` | library-pruned | 8 | 0.125 | 0.625 | 5.56 | 40.89 |
| `d740fec8` | library-v2 | 10 | 0.140 | 0.700 | 1.86 | 8.00 |
| `42ed6da4` | library-v3 | 12 | 0.150 | 0.750 | 0.57 | 12.82 |
| `736268c2` | library-v4 | 14 | 0.143 | 0.714 | 3.90 | 14.73 |
| `8257350b` | library-v5 | 16 | 0.150 | 0.750 | 12.91 | 152.17 |

## 2b. Phase 7 arm comparison (same golden v5, hash-dev)

First measured before/after of the optimization arms. Relative deltas
are meaningful even on dev embeddings; absolute values are not.

| run_id (short) | config | rerank | expansion | Precision@5 | Recall@5 |
|---|---|---|---|---|---|
| `8257350b` | library-v5 (baseline) | off | off | 0.150 | 0.750 |
| `fb791a1e` | library-exp | off | on | 0.138 | 0.688 |
| `97d3e429` | library-rerank | on (real `BAAI/bge-reranker-base`) | off | 0.150 | 0.750 |
| `f5e4bf9c` | library-both | on | on | 0.138 | 0.688 |

Reading: expansion slightly hurts here (11/16 vs 12/16) — synonym
variations add noise under coarse trigram retrieval; reranking changes
nothing on this set (same membership, reordered). Neither arm moves the
needle while retrieval itself is hash-based — the honest conclusion is
that **embeddings are the bottleneck**, not ranking. Caveats: the
reranker default is English-centric (`bge-reranker-base`, Arabic ideally
wants a multilingual reranker), and `evaluation_runs` storage drops the
`reranker_used` axis (comparison currently reproducible only from this
log + run_ids, not from the DB).

Reading: recall climbs as genuine banks are added (0.625 → 0.75);
precision stays ~0.13–0.15, the expected ceiling of trigram-hash
retrieval. Latency is Gemini-dominated (timeouts + fallback), **not**
retrieval time — pure retrieval answers in ~30–55ms (see §4).

## 2c. First real bge-m3 retrieval-only run (2026-10-27)

Same golden v5 (16 questions), same 23-source manifest corpus (722
chunks), Qdrant server 1.20.0 in Docker, CPU-only. No LLM was called;
`--no-ingest` scored the already-ingested collection.

| run_id (short) | config | embedding | n | Precision@5 | Recall@5 | p50 (s) | p95 (s) |
|---|---|---|---|---|---|---|---|
| `0b5110ff` | bge-m3-arabic-23 | BAAI/bge-m3 (1024d) | 16 | 0.1875 | 0.9375 | 0.3156 | 0.6182 |

Canonical row with host and reproduction details:
`docs/benchmark_results.md`. Against the hash-dev v5 baseline
(0.150/0.750), recall rose by 0.1875 absolute (15/16 vs 12/16 questions
hit) and precision by 0.0375. The latency is query embedding + Qdrant
search on CPU, so it is not comparable to the Gemini-dominated dev
latencies above.

## 3. Live index snapshot (2026-10-16, `sanad_chunks_*`)

3,657 points total. Queryable library (`arabic_128`): 695 points across
27 sources — 17 genuine banks + golden samples + a few named test
residues (`branch-*-doc`, `doc1`, `test`). Legacy `*_1024` collections
(2,836 points) are stale remains from mixed-backend experiments, not
live data. 266 filler points were quarantined and deleted (commit
`e60b1c2`); the 10 filler sources are listed in
`scripts/prune_index.py::FILLER_SOURCES` with the audit rationale.

> **2026-10-27:** the current `arabic_128` collection contains 733 points
> across 31 source ids (722 genuine points from the 23 manifest sources +
> 11 test residues). The new `arabic_1024` benchmark collection contains
> exactly the 722 genuine points, ingested with `BAAI/bge-m3`.

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
