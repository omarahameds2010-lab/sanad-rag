# Benchmark Results

Every row below comes from an actual run of `scripts/run_evaluation.py` — none are hand-typed. The `run_id` links back to the stored row in `data/sanad.db` (`evaluation_runs` table) for full traceability.

## First real run: bge-m3 on the audited 23-source corpus

- **Corpus:** 23 SHA-256-pinned sources from `data/knowledge/corpus_manifest.json`, 722 chunks (the same chunk ids as the live library).
- **Embedding:** `BAAI/bge-m3`, 1024 dimensions, normalized, no query/passage prefix.
- **Retrieval:** Qdrant server 1.20.0 (Docker), cosine, top-k=5, Arabic chunker (max 800 chars, 1-sentence overlap), no reranker, no query expansion.
- **LLM:** none — pure retrieval; latency is query embedding + vector search.
- **Host:** Windows 11 Pro (10.0.26300), AMD Ryzen 7 5700 (8C/16T), 15.8 GB RAM, CPU-only.
- **Golden set:** 16 manually verified questions (`data/golden_dataset/sanad_library_golden.json`).

| Timestamp | Config | Chunker | Reranker | Embedding | LLM | N | Precision@k | Recall@k | p50 latency (s) | p95 latency (s) | Dev-only? | run_id |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-27T20:46:36.269005+00:00 | bge-m3-arabic-23 | arabic | none | BAAI/bge-m3 | none (retrieval-only) | 16 | 0.1875 | 0.9375 | 0.3156 | 0.6182 | False | 0b5110ff-bb90-4ca2-8a9e-752caffd2497 |

**Reading:** recall@5 rose from 0.750 on the dev HashEmbedding baseline (same 23-source corpus, 16 questions) to 0.9375 with bge-m3; precision@5 rose from 0.150 to 0.1875. Precision@5 is bounded by the golden set's roughly one relevant chunk per question, so recall is the more informative metric here. The p50 is 0.316 s on CPU, not millisecond-scale: bge-m3 query embedding dominates. A production millisecond target needs a GPU or a smaller embedding model; the vector search itself is sub-millisecond at this corpus size.

**Reproduce:**

```bash
EMBEDDING_MODEL=bge-m3 QDRANT_URL=http://localhost:6333 \
  python scripts/ingest_corpus.py --fresh
EMBEDDING_MODEL=bge-m3 QDRANT_URL=http://localhost:6333 \
  python scripts/run_evaluation.py \
  --golden data/golden_dataset/sanad_library_golden.json \
  --config bge-m3-arabic-23 --retrieval-only --no-ingest
```
