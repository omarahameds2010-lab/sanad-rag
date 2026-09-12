# Benchmark Results

Every row below comes from an actual run of `scripts/run_evaluation.py` — none are hand-typed. The `run_id` links back to the stored row in `data/sanad.db` (`evaluation_runs` table) for full traceability.

**No rows yet.** Run `python scripts/run_evaluation.py` on a machine with real network/model access (see `.env.example` for `EMBEDDING_MODEL` and `ANTHROPIC_API_KEY`) to populate this table with real numbers. A dev-only smoke test (HashEmbedding + FakeLLM, run in the sandboxed build environment) confirmed the harness itself works end-to-end: 4/4 golden questions processed, 100% recall@5, ~0.0005s p50 latency — but those numbers are pipeline-plumbing proof, not retrieval-quality results, and are intentionally not included as a row here.

| Timestamp | Config | Embedding | LLM | N | Precision@k | Recall@k | p50 latency (s) | p95 latency (s) | Dev-only? | run_id |
|---|---|---|---|---|---|---|---|---|---|---|
