"""
Evaluation module exports.
"""

from src.evaluation.metrics import (
    GoldenExample,
    RetrievalResult,
    evaluate_retrieval,
    precision_at_k,
    recall_at_k,
    check_groundedness,
    timed,
)

from src.evaluation.harness import (
    run_evaluation,
    load_golden_dataset,
    ingest_golden_sources,
)

from src.evaluation.storage import (
    record_evaluation_run,
    get_evaluation_run,
    init_db,
    get_connection,
)

from src.evaluation.rubric import (
    Criterion,
    GradingConfig,
    CriterionResult,
    EvaluationResult,
    EvidenceType,
    ValidationResult,
)

from src.evaluation.validation import (
    validate_evaluation,
    aggregate_scores,
    to_criterion_responses,
)

from src.evaluation.prompts import (
    build_evaluation_prompt,
    build_context_from_chunks,
    parse_llm_evaluation_output,
)

from src.evaluation.runtime_evaluator import RuntimeEvaluator

from src.evaluation.schemas import (
    EvaluationRequest,
    EvaluationResponse,
    EvaluationConfigRequest,
    GradingConfigRequest,
    CriterionResponse,
    EvidenceType,
    QuestionType,
    ChunkerType,
    CriterionRequest,
    GradingConfigRequest,
    KnowledgeConfigRequest,
    EvaluationErrorResponse,
)

__all__ = [
    # Metrics
    "GoldenExample",
    "RetrievalResult",
    "evaluate_retrieval",
    "precision_at_k",
    "recall_at_k",
    "check_groundedness",
    "timed",
    # Harness
    "run_evaluation",
    "load_golden_dataset",
    "ingest_golden_sources",
    # Storage
    "record_evaluation_run",
    "get_evaluation_run",
    "init_db",
    "get_connection",
    # Rubric
    "Criterion",
    "GradingConfig",
    "CriterionResult",
    "EvaluationResult",
    "EvidenceType",
    "ValidationResult",
    # Validation
    "validate_evaluation",
    "aggregate_scores",
    "to_criterion_responses",
    # Prompts
    "build_evaluation_prompt",
    "build_context_from_chunks",
    "parse_llm_evaluation_output",
    # Runtime Evaluator
    "RuntimeEvaluator",
    # Schemas
    "EvaluationRequest",
    "EvaluationResponse",
    "EvaluationConfigRequest",
    "GradingConfigRequest",
    "CriterionResponse",
    "EvidenceType",
    "QuestionType",
    "ChunkerType",
    "CriterionRequest",
    "GradingConfigRequest",
    "KnowledgeConfigRequest",
    "EvaluationErrorResponse",
]