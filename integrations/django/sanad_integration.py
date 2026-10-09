"""
Django integration example for SANAD (LMS side).

Install: pip install httpx
Settings (never expose to browser):
    SANAD_BASE_URL = "http://localhost:8000"
    SANAD_API_KEY = "demo-key-123"   # from env / secrets manager
    SANAD_TIMEOUT = 60

Flow:
    Teacher creates question + rubric + reference answer (LMS models)
    Teacher ingests material once  -> POST /ingest
    Student submits                -> save Submission (status=SUBMITTED)
    Celery task                    -> POST /evaluate/submission
    Save AIEvaluation              -> teacher reviews -> official grade
"""

import uuid
import httpx
from django.conf import settings
from django.db import models, transaction


# ---------------------------------------------------------------------------
# 1. Minimal models (adapt to your existing LMS models)
# ---------------------------------------------------------------------------

class AssignmentSubmission(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tenant_id = models.CharField(max_length=100)
    question_id = models.CharField(max_length=100)
    question_text = models.TextField()
    student_answer = models.TextField()
    status = models.CharField(max_length=20, default="SUBMITTED")  # SUBMITTED/AI_EVALUATED/GRADED
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = "assignments"


class AIEvaluation(models.Model):
    """LMS-owned AI evaluation record. Official grade lives elsewhere."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    request_id = models.UUIDField(unique=True)  # idempotency key
    submission = models.ForeignKey(AssignmentSubmission, on_delete=models.CASCADE, related_name="ai_evaluations")
    tenant_id = models.CharField(max_length=100)
    status = models.CharField(max_length=20, default="QUEUED")  # QUEUED/PROCESSING/COMPLETED/FAILED
    suggested_score = models.IntegerField(null=True, blank=True)
    max_score = models.IntegerField()
    criteria_json = models.JSONField(default=list)
    feedback_ar = models.TextField(blank=True, default="")
    review_signal = models.FloatField(null=True, blank=True)
    provenance_json = models.JSONField(default=dict)
    failure_code = models.CharField(max_length=50, null=True, blank=True)
    rubric_version = models.CharField(max_length=50, default="v1")
    knowledge_version = models.CharField(max_length=50, default="v1")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = "assignments"
        constraints = [
            models.UniqueConstraint(
                fields=["submission", "rubric_version", "knowledge_version"],
                name="uniq_eval_per_versions",
            )
        ]


class OfficialGrade(models.Model):
    """Teacher-approved final grade. Only LMS writes this."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    submission = models.OneToOneField(AssignmentSubmission, on_delete=models.CASCADE, related_name="official_grade")
    tenant_id = models.CharField(max_length=100)
    score = models.IntegerField()
    max_score = models.IntegerField()
    decided_by = models.CharField(max_length=100)  # teacher user id
    source = models.CharField(max_length=20)  # 'ai_approved' | 'teacher_override'
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = "assignments"


# ---------------------------------------------------------------------------
# 2. SANAD HTTP client (server-side only)
# ---------------------------------------------------------------------------

def _sanad_headers():
    return {"X-API-Key": settings.SANAD_API_KEY, "Content-Type": "application/json"}


def sanad_health():
    r = httpx.get(f"{settings.SANAD_BASE_URL}/health", timeout=10)
    r.raise_for_status()
    return r.json()


def sanad_ingest(source_id: str, text: str, metadata: dict | None = None, chunker_type: str = "hierarchical") -> dict:
    r = httpx.post(
        f"{settings.SANAD_BASE_URL}/ingest",
        headers=_sanad_headers(),
        json={"source_id": source_id, "text": text, "metadata": metadata or {}, "chunker_type": chunker_type},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def sanad_evaluate(payload: dict) -> dict:
    r = httpx.post(
        f"{settings.SANAD_BASE_URL}/evaluate/submission",
        headers=_sanad_headers(),
        json=payload,
        timeout=getattr(settings, "SANAD_TIMEOUT", 60),
    )
    r.raise_for_status()
    return r.json()


def build_evaluation_payload(*, submission: AssignmentSubmission, reference_answer: str,
                             rubric: list[dict], max_score: int, source_ids: list[str],
                             rubric_version: str = "v1", knowledge_version: str = "v1") -> dict:
    criteria_sum = sum(c["max_score"] for c in rubric)
    if criteria_sum != max_score:
        raise ValueError(f"rubric sum ({criteria_sum}) must equal max_score ({max_score})")
    return {
        "request_id": str(uuid.uuid4()),
        "submission_id": str(submission.id),
        "tenant_id": submission.tenant_id,
        "question_id": submission.question_id,
        "question": {"text": submission.question_text, "type": "ESSAY"},
        "student_answer": {"text": submission.student_answer},
        "grading": {
            "reference_answer": reference_answer,
            "max_score": max_score,
            "rubric_version": rubric_version,
            "criteria": [{"id": c["id"], "description": c["description"], "max_score": c["max_score"]} for c in rubric],
        },
        "knowledge": {"source_ids": source_ids, "version": knowledge_version},
        "config": {"chunker_type": "hierarchical", "enable_reranking": True, "enable_query_expansion": True, "top_k": 5},
    }


# ---------------------------------------------------------------------------
# 3. Views (student submit must NOT wait for SANAD)
# ---------------------------------------------------------------------------
# In views.py:
#
# from django.http import JsonResponse
# from .sanad_integration import AssignmentSubmission, AIEvaluation, build_evaluation_payload
# from .tasks import run_sanad_evaluation
#
# def submit_answer(request):
#     data = json.loads(request.body)
#     with transaction.atomic():
#         sub = AssignmentSubmission.objects.create(
#             tenant_id=request.tenant_id,  # from your auth middleware
#             question_id=data["question_id"],
#             question_text=data["question_text"],
#             student_answer=data["student_answer"],
#         )
#         payload = build_evaluation_payload(
#             submission=sub,
#             reference_answer=data["reference_answer"],  # normally fetched server-side, not from student!
#             rubric=data["rubric"],
#             max_score=data["max_score"],
#             source_ids=data["source_ids"],
#         )
#         AIEvaluation.objects.create(
#             request_id=payload["request_id"], submission=sub, tenant_id=sub.tenant_id,
#             status="QUEUED", max_score=payload["grading"]["max_score"],
#             rubric_version=payload["grading"]["rubric_version"],
#             knowledge_version=payload["knowledge"]["version"],
#         )
#         run_sanad_evaluation.delay(payload)  # Celery, async
#     return JsonResponse({"submission_id": str(sub.id), "status": "QUEUED"}, status=202)


# ---------------------------------------------------------------------------
# 4. Celery task (async worker -> SANAD sync call)
# ---------------------------------------------------------------------------
# In tasks.py:
#
# from celery import shared_task
# from .sanad_integration import sanad_evaluate, AIEvaluation
#
# @shared_task(bind=True, max_retries=3, autoretry_for=(httpx.TimeoutException, httpx.NetworkError),
#              retry_backoff=True, retry_backoff_max=60)
# def run_sanad_evaluation(self, payload: dict):
#     try:
#         result = sanad_evaluate(payload)
#     except httpx.HTTPStatusError as e:
#         status = e.response.status_code
#         if 400 <= status < 500 and status != 429:
#             AIEvaluation.objects.filter(request_id=payload["request_id"]).update(
#                 status="FAILED", failure_code=f"HTTP_{status}")
#             return  # do not retry client errors
#         raise self.retry(exc=e)
#     # Validate version match before persisting
#     ev = AIEvaluation.objects.get(request_id=payload["request_id"])
#     assert result["provenance"]["rubric_version"] == ev.rubric_version
#     ev.status = "COMPLETED"
#     ev.suggested_score = result["suggested_score"]
#     ev.criteria_json = result["criteria"]
#     ev.feedback_ar = result.get("feedback_ar", "")
#     ev.review_signal = result.get("review_signal")
#     ev.provenance_json = result.get("provenance", {})
#     ev.save()
#
#
# # Teacher review (official grade stays in LMS):
# #
# # def approve_evaluation(evaluation_id, teacher_id):
# #     ev = AIEvaluation.objects.select_related("submission").get(id=evaluation_id)
# #     OfficialGrade.objects.update_or_create(
# #         submission=ev.submission,
# #         defaults={"tenant_id": ev.tenant_id, "score": ev.suggested_score,
# #                   "max_score": ev.max_score, "decided_by": teacher_id, "source": "ai_approved"})
# #
# # def override_grade(submission_id, teacher_id, score, max_score):
# #     sub = AssignmentSubmission.objects.get(id=submission_id)
# #     OfficialGrade.objects.update_or_create(
# #         submission=sub,
# #         defaults={"tenant_id": sub.tenant_id, "score": score,
# #                   "max_score": max_score, "decided_by": teacher_id, "source": "teacher_override"})
