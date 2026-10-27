"""
Score validation and aggregation logic for evaluation.
This module contains the deterministic validation layer that sits
between LLM output and the final API response.
"""

from src.evaluation.rubric import (
    EvaluationResult,
    EvidenceType,
    GradingConfig,
    ValidationResult,
)
from src.evaluation.schemas import CriterionResponse


def validate_evaluation(
    result: EvaluationResult,
    grading: GradingConfig,
) -> ValidationResult:
    """
    Validate complete evaluation against grading config.
    Returns ValidationResult with errors if any.
    """
    errors = []

    # 1. All criteria present
    result_ids = {c.criterion_id for c in result.criteria}
    rubric_ids = grading.get_criterion_ids()
    if result_ids != rubric_ids:
        return ValidationResult.failure([
            f"Criterion mismatch: expected {rubric_ids}, got {result_ids}"
        ])

    # 2. Per-criterion validation
    for c in result.criteria:
        criterion = grading.get_criterion(c.criterion_id)
        if criterion:
            criterion_errors = _validate_criterion(c, criterion)
            errors.extend(criterion_errors)
        else:
            errors.append(f"Unknown criterion: {c.criterion_id}")

    # 3. Sum constraint
    total = sum(c.earned for c in result.criteria)
    if total > result.max_score:
        errors.append(
            f"Sum of earned ({total}) exceeds max_score ({result.max_score})"
        )

    # 3b. Sum should equal suggested_score (deterministic aggregation)
    if total != result.suggested_score:
        errors.append(
            f"Sum of earned ({total}) does not match suggested_score ({result.suggested_score})"
        )

    # 4. Evidence type consistency
    for c in result.criteria:
        if c.earned == 0 and c.evidence_type != EvidenceType.ABSENT:
            errors.append(
                f"{c.criterion_id}: earned=0 requires evidence_type=ABSENT"
            )
        if c.earned == c.max and c.evidence_type != EvidenceType.SUPPORTED:
            errors.append(
                f"{c.criterion_id}: earned=max requires evidence_type=SUPPORTED"
            )

    # 5. Review signal bounds
    if not (0.0 <= result.review_signal <= 1.0):
        errors.append("review_signal must be in [0.0, 1.0]")

    # 6. Score bounds
    if not (0 <= result.suggested_score <= result.max_score):
        errors.append(f"suggested_score {result.suggested_score} not in [0, {result.max_score}]")

    if errors:
        return ValidationResult.failure(errors)
    return ValidationResult.success()


def _validate_criterion(
    criterion_result,
    criterion,
) -> list[str]:
    """Validate a single criterion result against rubric criterion."""
    errors = []

    # Bounds
    if not (0 <= criterion_result.earned <= criterion.max_score):
        errors.append(
            f"{criterion.id}: earned {criterion_result.earned} "
            f"not in [0, {criterion.max_score}]"
        )

    # Type
    if not isinstance(criterion_result.earned, int):
        errors.append(f"{criterion.id}: earned must be integer")

    # Evidence requirements
    if criterion_result.evidence_type == EvidenceType.SUPPORTED and not criterion_result.evidence:
        errors.append(f"{criterion.id}: evidence required for SUPPORTED")

    # Evidence type consistency with score
    if criterion_result.earned == 0 and criterion_result.evidence_type != EvidenceType.ABSENT:
        errors.append(
            f"{criterion.id}: earned=0 requires evidence_type=ABSENT"
        )
    if criterion_result.earned == criterion.max_score and criterion_result.evidence_type != EvidenceType.SUPPORTED:
        errors.append(
            f"{criterion.id}: earned=max requires evidence_type=SUPPORTED"
        )

    # Evidence must be string when present
    if criterion_result.evidence is not None and not isinstance(criterion_result.evidence, str):
        errors.append(f"{criterion.id}: evidence must be string or null")

    # Feedback must be non-empty
    if not criterion_result.feedback_ar or not criterion_result.feedback_ar.strip():
        errors.append(f"{criterion.id}: feedback_ar must be non-empty")

    return errors


def aggregate_scores(criterion_results: list) -> int:
    """
    Deterministically aggregate criterion scores.
    Returns sum of earned scores.
    """
    return sum(c.earned for c in criterion_results)


def to_criterion_responses(criterion_results: list) -> list[CriterionResponse]:
    """Convert internal criterion results to API response format."""
    return [
        CriterionResponse(
            criterion_id=c.criterion_id,
            earned=c.earned,
            max=c.max,
            evidence=c.evidence,
            evidence_type=c.evidence_type,
            feedback_ar=c.feedback_ar,
        )
        for c in criterion_results
    ]