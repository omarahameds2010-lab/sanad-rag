"""
Evaluation harness.

This is the most important module in the project. Everything else exists
to produce good retrieval — this module is what proves whether it actually
is good, using data/golden_dataset/ (manually verified question -> correct
source chunk id -> reference answer triples).

No number produced here is ever hand-waved into the README; every metric in
docs/benchmark_results.md must trace back to a run of `evaluate_run()`.
"""

from dataclasses import dataclass
from time import perf_counter


@dataclass
class GoldenExample:
    question: str
    relevant_chunk_ids: set[str]
    reference_answer: str


@dataclass
class RetrievalResult:
    question: str
    retrieved_chunk_ids: list[str]
    latency_seconds: float


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for c in top_k if c in relevant)
    return hits / len(top_k)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    hits = len(top_k & relevant)
    return hits / len(relevant)


def evaluate_retrieval(
    golden_set: list[GoldenExample],
    results: list[RetrievalResult],
    k: int = 5,
) -> dict:
    """
    Aggregate precision/recall/latency across a full golden dataset run.
    `results` must be aligned 1:1 with `golden_set` by question order.
    """
    assert len(golden_set) == len(results), "golden set and results must align"

    precisions, recalls, latencies = [], [], []
    for golden, result in zip(golden_set, results):
        precisions.append(
            precision_at_k(result.retrieved_chunk_ids, golden.relevant_chunk_ids, k)
        )
        recalls.append(
            recall_at_k(result.retrieved_chunk_ids, golden.relevant_chunk_ids, k)
        )
        latencies.append(result.latency_seconds)

    latencies_sorted = sorted(latencies)
    p50_idx = int(len(latencies_sorted) * 0.5)
    p95_idx = min(int(len(latencies_sorted) * 0.95), len(latencies_sorted) - 1)

    return {
        "n_examples": len(golden_set),
        "precision_at_k": round(sum(precisions) / len(precisions), 4),
        "recall_at_k": round(sum(recalls) / len(recalls), 4),
        "p50_latency_seconds": round(latencies_sorted[p50_idx], 4) if latencies_sorted else None,
        "p95_latency_seconds": round(latencies_sorted[p95_idx], 4) if latencies_sorted else None,
    }


def check_groundedness(answer: str, retrieved_texts: list[str]) -> bool:
    """
    Placeholder groundedness check: verifies the answer does not introduce
    named entities or numbers absent from the retrieved context.
    This is a heuristic stand-in for Phase 6 — to be replaced with an
    LLM-as-judge grounding check once the golden dataset exists.
    """
    raise NotImplementedError(
        "Implement in Phase 6 once the golden dataset and generation "
        "pipeline are ready — do not fake this metric."
    )


def timed(fn, *args, **kwargs):
    """Utility: run fn and return (result, elapsed_seconds)."""
    start = perf_counter()
    result = fn(*args, **kwargs)
    elapsed = perf_counter() - start
    return result, elapsed
