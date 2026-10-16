import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.embeddings.base import HashEmbedding
from src.generation.llm_client import FakeLLM
from src.evaluation.runtime_evaluator import RuntimeEvaluator


def make_grading():
    crits = [
        SimpleNamespace(id="c1", description="d1", max_score=5),
        SimpleNamespace(id="c2", description="d2", max_score=5),
    ]
    return SimpleNamespace(
        reference_answer="ref", max_score=10, rubric_version="v1", criteria=crits
    )


def make_evaluator():
    return RuntimeEvaluator(
        embedding_model=HashEmbedding(dims=32),
        primary_llm=FakeLLM(),
        fallback_llm=FakeLLM(),
    )


def test_invalid_grade_forces_teacher_review():
    ev = make_evaluator()
    # LLM dropped criterion c2 entirely: invalid per rubric validation.
    data = {
        "criteria": [
            {"criterion_id": "c1", "earned": 3, "evidence": "x",
             "evidence_type": "SUPPORTED", "feedback_ar": "good"},
        ],
        "suggested_score": 3,
        "overall_feedback_ar": "ok",
        "review_signal": 0.2,
        "needs_teacher_review": False,  # LLM claims no review needed
        "knowledge_gaps": [],
    }
    result = ev._parse_and_validate(data, make_grading(), 10)
    assert result.needs_teacher_review is True
    assert "تدقيق تلقائي" in result.feedback_ar


def test_valid_grade_preserves_llm_review_flag():
    ev = make_evaluator()
    data = {
        "criteria": [
            {"criterion_id": "c1", "earned": 3, "evidence": "x",
             "evidence_type": "SUPPORTED", "feedback_ar": "good"},
            {"criterion_id": "c2", "earned": 4, "evidence": "y",
             "evidence_type": "SUPPORTED", "feedback_ar": "good"},
        ],
        "suggested_score": 7,
        "overall_feedback_ar": "ok",
        "review_signal": 0.2,
        "needs_teacher_review": False,
        "knowledge_gaps": [],
    }
    result = ev._parse_and_validate(data, make_grading(), 10)
    assert result.needs_teacher_review is False
    assert "تدقيق تلقائي" not in result.feedback_ar
