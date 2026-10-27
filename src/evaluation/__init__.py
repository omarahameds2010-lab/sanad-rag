"""
Evaluation module exports.
"""

from src.evaluation.harness import (
    ingest_golden_sources,
    load_golden_dataset,
    run_evaluation,
)
from src.evaluation.metrics import (
    GoldenExample,
    RetrievalResult,
    check_groundedness,
    evaluate_retrieval,
    precision_at_k,
    recall_at_k,
    timed,
)
from src.evaluation.prompts import (
    build_context_from_chunks,
    build_evaluation_prompt,
    parse_llm_evaluation_output,
)
from src.evaluation.rubric import (
    Criterion,
    CriterionResult,
    EvaluationResult,
    GradingConfig,
    ValidationResult,
)
from src.evaluation.runtime_evaluator import RuntimeEvaluator
from src.evaluation.schemas import (
    ChunkerType,
    CriterionRequest,
    CriterionResponse,
    EvaluationConfigRequest,
    EvaluationErrorResponse,
    EvaluationRequest,
    EvaluationResponse,
    EvidenceType,
    GradingConfigRequest,
    KnowledgeConfigRequest,
    QuestionType,
)
from src.evaluation.storage import (
    get_connection,
    get_evaluation_run,
    init_db,
    record_evaluation_run,
)
from src.evaluation.validation import (
    aggregate_scores,
    to_criterion_responses,
    validate_evaluation,
)

__all__ = [
    "ChunkerType",
    # Rubric
    "Criterion",
    "CriterionRequest",
    "CriterionResponse",
    "CriterionResult",
    "EvaluationConfigRequest",
    "EvaluationErrorResponse",
    # Schemas
    "EvaluationRequest",
    "EvaluationResponse",
    "EvaluationResult",
    "EvidenceType",
    # Metrics
    "GoldenExample",
    "GradingConfig",
    "GradingConfigRequest",
    "KnowledgeConfigRequest",
    "QuestionType",
    "RetrievalResult",
    # Runtime Evaluator
    "RuntimeEvaluator",
    "ValidationResult",
    "aggregate_scores",
    "build_context_from_chunks",
    # Prompts
    "build_evaluation_prompt",
    "check_groundedness",
    "evaluate_retrieval",
    "get_connection",
    "get_evaluation_run",
    "ingest_golden_sources",
    "init_db",
    "load_golden_dataset",
    "parse_llm_evaluation_output",
    "precision_at_k",
    "recall_at_k",
    # Storage
    "record_evaluation_run",
    # Harness
    "run_evaluation",
    "timed",
    "to_criterion_responses",
    # Validation
    "validate_evaluation",
]