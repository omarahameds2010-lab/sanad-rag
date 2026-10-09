"""
Evaluation harness — Phase 6/8. This is the module that turns the golden
dataset into real, stored numbers. It does NOT invent anything: every
number in the returned report traces back to an actual retrieval call
against the current index.

Two things this module deliberately keeps separate:
1. Loading/ingesting the golden dataset's source documents (idempotent —
   safe to call every run).
2. Running the questions and scoring retrieval — this is the part whose
   output is trustworthy only to the extent the embedding/generation
   backends used are real (see the loud warning this module emits when
   HashEmbedding/FakeLLM are detected).
"""

import json
import uuid
from pathlib import Path

from src.embeddings.base import EmbeddingModel
from src.generation.llm_client import LLMClient
from src.ingestion.pipeline import ingest_document
from src.retrieval.query_pipeline import answer_question
from src.evaluation.metrics import (
    GoldenExample,
    RetrievalResult,
    evaluate_retrieval,
)
from src.evaluation.storage import record_evaluation_run
from src.retrieval.reranker import CrossEncoderReranker


def load_golden_dataset(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def ingest_golden_sources(
    golden_data: dict, embedding_model: EmbeddingModel, chunker_type: str = "arabic"
) -> None:
    """Idempotent: re-running this just re-upserts the same chunk_ids."""
    for doc in golden_data.get("source_documents", []):
        ingest_document(
            source_id=doc["source_id"],
            raw_text=doc["text"],
            embedding_model=embedding_model,
            chunker_type=chunker_type,
        )


def run_evaluation(
    golden_dataset_path: str,
    embedding_model: EmbeddingModel,
    llm: LLMClient,
    config_name: str = "baseline",
    top_k: int = 5,
    chunker_type: str = "arabic",
    enable_reranking: bool = False,
    enable_query_expansion: bool = False,
) -> dict:
    """
    Runs every question in the golden dataset through the real query
    pipeline, scores retrieval against the golden relevant_chunk_ids, and
    persists the run. Returns the full metrics dict.
    """
    if embedding_model.name.endswith("DEV-ONLY") or llm.name.endswith("DEV-ONLY"):
        print(
            "[WARNING] running evaluation with a dev-only backend "
            f"(embedding={embedding_model.name}, llm={llm.name}). "
            "These numbers prove the harness works end-to-end — they are "
            "NOT real retrieval-quality or groundedness numbers. Do not "
            "put them in docs/benchmark_results.md as real results."
        )

    golden_data = load_golden_dataset(golden_dataset_path)
    ingest_golden_sources(golden_data, embedding_model, chunker_type=chunker_type)

    reranker = CrossEncoderReranker() if enable_reranking else None

    golden_examples = [
        GoldenExample(
            question=q["question"],
            relevant_chunk_ids=set(q["relevant_chunk_ids"]),
            reference_answer=q["reference_answer"],
        )
        for q in golden_data["questions"]
    ]

    retrieval_results = []
    for example in golden_examples:
        result = answer_question(
            question=example.question,
            embedding_model=embedding_model,
            primary_llm=llm,
            top_k=top_k,
            reranker=reranker,
            chunker_type=chunker_type,
            enable_query_expansion=enable_query_expansion,
        )
        retrieval_results.append(
            RetrievalResult(
                question=example.question,
                retrieved_chunk_ids=result["retrieved_chunk_ids"],
                latency_seconds=(result["retrieval_ms"] + result["generation_ms"]) / 1000,
            )
        )

    metrics = evaluate_retrieval(golden_examples, retrieval_results, k=top_k)
    metrics["config_name"] = config_name
    metrics["embedding_model"] = embedding_model.name
    metrics["llm_model"] = llm.name
    metrics["chunker_type"] = chunker_type
    metrics["reranker_used"] = reranker.name if reranker else "none"
    metrics["query_expansion_enabled"] = enable_query_expansion
    metrics["is_dev_only_run"] = embedding_model.name.endswith("DEV-ONLY") or llm.name.endswith(
        "DEV-ONLY"
    )

    run_id = str(uuid.uuid4())
    record_evaluation_run(run_id=run_id, config_name=config_name, metrics=metrics)
    metrics["run_id"] = run_id

    return metrics
