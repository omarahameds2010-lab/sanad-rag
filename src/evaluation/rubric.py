"""
Rubric and criterion domain models for evaluation.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, Field, model_validator


class EvidenceType(StrEnum):
    SUPPORTED = "SUPPORTED"
    PARTIAL = "PARTIAL"
    ABSENT = "ABSENT"


class Criterion(BaseModel):
    """Single rubric criterion with scoring bounds."""
    id: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., min_length=1, max_length=500)
    max_score: int = Field(..., gt=0, le=20)


class GradingConfig(BaseModel):
    """Complete grading configuration for a question."""
    reference_answer: str = Field(..., min_length=1, max_length=10000)
    max_score: int = Field(..., gt=0, le=100)
    rubric_version: str = Field(..., min_length=1, max_length=50)
    criteria: list[Criterion] = Field(..., min_length=1, max_length=10)

    @model_validator(mode="after")
    def check_criteria_sum(self) -> Self:
        total = sum(c.max_score for c in self.criteria)
        if total != self.max_score:
            raise ValueError(
                f"Sum of criteria max_score ({total}) must equal "
                f"grading.max_score ({self.max_score})"
            )
        return self

    def get_criterion(self, criterion_id: str) -> Criterion | None:
        """Get criterion by ID."""
        for c in self.criteria:
            if c.id == criterion_id:
                return c
        return None

    def get_criterion_ids(self) -> set[str]:
        return {c.id for c in self.criteria}


@dataclass
class CriterionResult:
    """Result for a single criterion."""
    criterion_id: str
    earned: int
    max: int
    evidence: str | None = None
    evidence_type: str = "ABSENT"  # SUPPORTED, PARTIAL, ABSENT
    feedback_ar: str = ""

    def validate(self, criterion: Criterion) -> list[str]:
        """Validate this criterion result against the rubric criterion."""
        errors = []
        if not (0 <= self.earned <= criterion.max_score):
            errors.append(
                f"{criterion.id}: earned {self.earned} not in [0, {criterion.max_score}]"
            )
        if not isinstance(self.earned, int):
            errors.append(f"{criterion.id}: earned must be integer")
        if self.evidence_type == "SUPPORTED" and not self.evidence:
            errors.append(f"{criterion.id}: evidence required for SUPPORTED")
        return errors


@dataclass
class EvaluationResult:
    """Complete evaluation result before validation."""
    evaluation_id: str
    request_id: str
    suggested_score: int
    max_score: int
    criteria: list[CriterionResult]
    feedback_ar: str
    review_signal: float
    needs_teacher_review: bool
    knowledge_gaps: list[str] = field(default_factory=list)

    def validate(self, grading: GradingConfig) -> list[str]:
        """Validate complete evaluation against grading config."""
        errors = []

        # 1. All criteria present
        result_ids = {c.criterion_id for c in self.criteria}
        rubric_ids = grading.get_criterion_ids()
        if result_ids != rubric_ids:
            errors.append(
                f"Criterion mismatch: expected {rubric_ids}, got {result_ids}"
            )

        # 2. Per-criterion validation
        for c in self.criteria:
            criterion = grading.get_criterion(c.criterion_id)
            if criterion:
                errors.extend(c.validate(criterion))
            else:
                errors.append(f"Unknown criterion: {c.criterion_id}")

        # 3. Sum constraint
        total = sum(c.earned for c in self.criteria)
        if total > self.max_score:
            errors.append(
                f"Sum of earned ({total}) exceeds max_score ({self.max_score})"
            )

        # 4. Evidence type consistency
        for c in self.criteria:
            if c.earned == 0 and c.evidence_type != "ABSENT":
                errors.append(f"{c.criterion_id}: earned=0 requires evidence_type=ABSENT")
            if c.earned == c.max and c.evidence_type != "SUPPORTED":
                errors.append(f"{c.criterion_id}: earned=max requires evidence_type=SUPPORTED")

        # 5. Review signal bounds
        if not (0.0 <= self.review_signal <= 1.0):
            errors.append("review_signal must be in [0.0, 1.0]")

        # 6. Score bounds
        if not (0 <= self.suggested_score <= self.max_score):
            errors.append(f"suggested_score {self.suggested_score} not in [0, {self.max_score}]")

        return errors


class ValidationResult:
    def __init__(self, valid: bool, errors: list[str]):
        self.valid = valid
        self.errors = errors

    @classmethod
    def success(cls) -> "ValidationResult":
        return cls(valid=True, errors=[])

    @classmethod
    def failure(cls, errors: list[str]) -> "ValidationResult":
        return cls(valid=False, errors=errors)