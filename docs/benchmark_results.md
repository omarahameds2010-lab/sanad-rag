# Benchmark Results

Every row below comes from an actual run of `scripts/run_evaluation.py` — none are hand-typed. The `run_id` links back to the stored row in `data/sanad.db` (`evaluation_runs` table) for full traceability. Both canonical rows are real `BAAI/bge-m3` runs against Qdrant server 1.20.0 (Docker), retrieval-only, on the same 16-question golden set with `top-k=5`.

## Run 1 — audited 23-source corpus (722 chunks)

- **Corpus:** 23 SHA-256-pinned sources from `data/knowledge/corpus_manifest.json`, 722 chunks (the same chunk ids as the live library at the time).
- **Embedding:** `BAAI/bge-m3`, 1024 dimensions, normalized, no query/passage prefix.
- **Retrieval:** cosine, top-k=5, Arabic chunker (max 800 chars, 1-sentence overlap), no reranker, no query expansion.
- **LLM:** none — pure retrieval; latency is query embedding + vector search.
- **Host:** Windows 11 Pro (10.0.26300), AMD Ryzen 7 5700 (8C/16T), 15.8 GB RAM, CPU-only.
- **Golden set:** 16 manually verified questions (`data/golden_dataset/sanad_library_golden.json`).

## Run 2 — audited 35-source wave (891 chunks)

The 12 remaining audited files were promoted into the manifest, adding 169 genuine chunks (722 → 891). No synthetic or duplicated content was added; an exact/near-duplicate check against the 23-source corpus found no matches.

Two content/engineering corrections were made during this wave:

- The fully vocalized, punctuation-free poem `نظم الآجرومية` first produced 2 chunks (max 11,522 chars). The Arabic chunker now splits punctuation-free sentences at word boundaries, producing 18 chunks (max 800). The 23 earlier sources' chunk boundaries were verified byte-identical before/after that fix (722/722), so the golden labels remain valid.
- `duaa_quran_bank.txt` was mislabeled: it began with 394 proverb lines before its duaa section. Those lines were moved into `amthal_hikam.txt`, and the duaa source now contains 153 lines of supplications and Quranic lessons. An earlier 35-source pass (`b0877c18`) ran before this correction; it is superseded by the run below and retained in the database and `docs/DEV_BENCHMARKS.md` for audit history.

| Timestamp | Config | Chunker | Reranker | Embedding | LLM | N | Precision@k | Recall@k | p50 latency (s) | p95 latency (s) | Dev-only? | run_id |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-27T20:46:36.269005+00:00 | bge-m3-arabic-23 | arabic | none | BAAI/bge-m3 | none (retrieval-only) | 16 | 0.1875 | 0.9375 | 0.3156 | 0.6182 | False | 0b5110ff-bb90-4ca2-8a9e-752caffd2497 |
| 2026-10-27T21:20:48.094022+00:00 | bge-m3-arabic-35 | arabic | none | BAAI/bge-m3 | none (retrieval-only) | 16 | 0.1875 | 0.9375 | 0.3639 | 0.6194 | False | 64e2df16-eb01-4de7-947e-d4df65ce3a56 |

**Reading:** on both corpora, recall@5 is 0.9375 (15 of 16 questions retrieve a relevant chunk) and precision@5 is 0.1875 (15 relevant chunks across 80 slots). Precision@5 is bounded by the golden set's roughly one relevant chunk per question, so recall is the more informative metric here. Adding 169 genuine chunks did not displace the correct top-5 results. The dev HashEmbedding baseline on the same 16 questions was 0.150 / 0.750; bge-m3 raised recall by 0.1875 absolute.

**Latency:** p50 is 0.316–0.364 s and p95 0.612–0.619 s on CPU. This is sub-second, not millisecond-scale: bge-m3 query embedding dominates, while the Qdrant search is sub-millisecond at this corpus size. A strict millisecond SLO needs a GPU or a smaller embedding model.

**Reproduce the current 35-source run:**

```bash
EMBEDDING_MODEL=bge-m3 QDRANT_URL=http://localhost:6333 \
  python scripts/ingest_corpus.py --fresh
EMBEDDING_MODEL=bge-m3 QDRANT_URL=http://localhost:6333 \
  python scripts/run_evaluation.py \
  --golden data/golden_dataset/sanad_library_golden.json \
  --config bge-m3-arabic-35 --retrieval-only --no-ingest
```

The 23-source row used the manifest at commit `1588f34`; that corpus is a subset of the current 35-source manifest, and the `run_id` remains the canonical trace.
