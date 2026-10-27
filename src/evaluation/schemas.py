"""
Pydantic schemas for runtime evaluation API.
These define the contract between LMS and SANAD.
"""

from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator


class QuestionType(StrEnum):
    ESSAY = "ESSAY"
    SHORT_ANSWER = "SHORT_ANSWER"


class EvidenceType(StrEnum):
    SUPPORTED = "SUPPORTED"
    PARTIAL = "PARTIAL"
    ABSENT = "ABSENT"


class ChunkerType(StrEnum):
    ARABIC = "arabic"
    BASELINE = "baseline"
    HIERARCHICAL = "hierarchical"


class CriterionRequest(BaseModel):
    id: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., min_length=1, max_length=500)
    max_score: int = Field(..., gt=0, le=20)


class GradingConfigRequest(BaseModel):
    reference_answer: str = Field(..., min_length=1, max_length=10000)
    max_score: int = Field(..., gt=0, le=100)
    rubric_version: str = Field(..., min_length=1, max_length=50)
    criteria: list[CriterionRequest] = Field(..., min_length=1, max_length=10)

    @model_validator(mode="after")
    def check_criteria_sum(self):
        total = sum(c.max_score for c in self.criteria)
        if total != self.max_score:
            raise ValueError(
                f"Sum of criteria max_score ({total}) must equal "
                f"grading.max_score ({self.max_score})"
            )
        return self


class KnowledgeConfigRequest(BaseModel):
    source_ids: list[str] = Field(default_factory=list, max_length=50)
    version: str | None = Field(default=None, max_length=50)


class EvaluationConfigRequest(BaseModel):
    chunker_type: ChunkerType = Field(default=ChunkerType.HIERARCHICAL)
    enable_reranking: bool = True
    enable_query_expansion: bool = True
    top_k: int = Field(default=5, ge=1, le=10)


class EvaluationRequest(BaseModel):
    request_id: UUID
    submission_id: UUID
    tenant_id: str = Field(..., min_length=1, max_length=100)
    question_id: str = Field(..., min_length=1, max_length=100)

    question: dict = Field(...)  # {"text": "...", "type": "ESSAY"}
    student_answer: dict = Field(...)  # {"text": "..."}
    grading: GradingConfigRequest
    knowledge: KnowledgeConfigRequest = Field(default_factory=KnowledgeConfigRequest)
    config: EvaluationConfigRequest = Field(default_factory=EvaluationConfigRequest)


class CriterionResponse(BaseModel):
    criterion_id: str
    earned: int
    max: int
    evidence: str | None = None
    evidence_type: EvidenceType
    feedback_ar: str


class EvaluationResponse(BaseModel):
    evaluation_id: UUID
    request_id: UUID
    status: Literal["COMPLETED", "FAILED"]

    suggested_score: int
    max_score: int

    criteria: list[CriterionResponse]

    feedback_ar: str

    review_signal: float = Field(ge=0.0, le=1.0)
    needs_teacher_review: bool

    knowledge_gaps: list[str] = Field(default_factory=list)

    provenance: dict
    usage: dict


class EvaluationErrorResponse(BaseModel):
    error_code: str
    message: str
    details: dict | None = None