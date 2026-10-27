"""
Run this on your own machine (with real network/model access) to produce
real benchmark numbers — not in this sandboxed dev environment, where
HashEmbedding/FakeLLM are used only to prove the pipeline plumbing works.

Usage:
    python scripts/run_evaluation.py \\
        --golden data/golden_dataset/sample_golden.json \\
        --config baseline

Set EMBEDDING_MODEL=multilingual-e5 and ANTHROPIC_API_KEY in your .env
(see .env.example) before running this for real numbers.
"""

import argparse
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from _models import get_embedding_model

from src.evaluation.harness import run_evaluation


def get_real_llm():
    if os.environ.get("ANTHROPIC_API_KEY"):
        from src.generation.llm_client import AnthropicLLM

        return AnthropicLLM()
    from src.generation.llm_client import FakeLLM

    print(
        "⚠️  ANTHROPIC_API_KEY not set — using FakeLLM. Set it in .env "
        "for real generation."
    )
    return FakeLLM()


def append_benchmark_result(metrics: dict, benchmark_file: str = "docs/benchmark_results.md"):
    path = Path(benchmark_file)
    is_new = not path.exists()

    with open(path, "a", encoding="utf-8") as f:
        if is_new:
            f.write("# Benchmark Results\n\n")
            f.write(
                "Every row below comes from an actual run of "
                "`scripts/run_evaluation.py` — none are hand-typed. "
                "The `run_id` links back to the stored row in "
                "`data/sanad.db` (`evaluation_runs` table) for full "
                "traceability.\n\n"
            )
            f.write(
                "| Timestamp | Config | Chunker | Reranker | Embedding | LLM | N | "
                "Precision@k | Recall@k | p50 latency (s) | p95 latency (s) | "
                "Dev-only? | run_id |\n"
            )
            f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")

        f.write(
            f"| {datetime.now(UTC).isoformat()} | {metrics['config_name']} | "
            f"{metrics.get('chunker_type', 'arabic')} | {metrics.get('reranker_used', 'none')} | "
            f"{metrics['embedding_model']} | {metrics['llm_model']} | "
            f"{metrics['n_examples']} | {metrics['precision_at_k']} | "
            f"{metrics['recall_at_k']} | {metrics['p50_latency_seconds']} | "
            f"{metrics['p95_latency_seconds']} | {metrics['is_dev_only_run']} | "
            f"{metrics['run_id']} |\n"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default="data/golden_dataset/sample_golden.json")
    parser.add_argument("--config", default="baseline")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--chunker",
        default="arabic",
        choices=["arabic", "baseline"],
        help="Chunking strategy for golden source ingestion (Phase 7 comparison arm)",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable cross-encoder reranking (Phase 7 comparison arm)",
    )
    parser.add_argument(
        "--no-append",
        action="store_true",
        help="Do not append the result to docs/benchmark_results.md "
        "(use for dev-only smoke runs so they are never mistaken for real results)",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Score retrieval only (no LLM generation). Use this for honest "
        "embedding/retrieval benchmarks; latency is pure retrieval.",
    )
    parser.add_argument(
        "--no-ingest",
        action="store_true",
        help="Evaluate the existing collection; do not re-embed golden source "
        "documents (use after scripts/ingest_corpus.py).",
    )
    args = parser.parse_args()

    metrics = run_evaluation(
        golden_dataset_path=args.golden,
        embedding_model=get_embedding_model(),
        llm=None if args.retrieval_only else get_real_llm(),
        config_name=args.config,
        top_k=args.top_k,
        chunker_type=args.chunker,
        enable_reranking=args.rerank,
        ingest_sources=not args.no_ingest,
    )

    if not args.no_append:
        if metrics.get("is_dev_only_run"):
            print(
                "\nDev-only run: not appended to docs/benchmark_results.md. "
                "Only real embedding/LLM runs belong there."
            )
        else:
            append_benchmark_result(metrics)
            print("\nAppended to docs/benchmark_results.md")

    print("\n=== Evaluation Complete ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()