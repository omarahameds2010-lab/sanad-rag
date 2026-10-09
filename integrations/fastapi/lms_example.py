"""
FastAPI LMS-side integration example for SANAD.

pip install fastapi httpx uvicorn

ENV (server-side only):
    SANAD_BASE_URL=http://localhost:8000
    SANAD_API_KEY=demo-key-123
    SANAD_TENANT_ID=demo_tenant

Run: uvicorn lms_example:app --port 3000 --reload

Endpoints (LMS-owned):
    POST /lms/ingest      -> forwards reference material to SANAD /ingest
    POST /lms/submissions -> saves submission, returns 202 QUEUED (async via BackgroundTasks)
    GET  /lms/submissions/{id} -> evaluation status + official grade
    POST /lms/submissions/{id}/review -> teacher approve/override

NOTE: BackgroundTasks here is for demo only. In production use a real
queue (Celery/RQ/Arq + Redis) so retries survive restarts.
"""

import os
import uuid
from typing import Any, Optional

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

SANAD_BASE_URL = os.environ.get("SANAD_BASE_URL", "http://localhost:8000")
SANAD_API_KEY = os.environ.get("SANAD_API_KEY", "demo-key-123")
SANAD_TIMEOUT = float(os.environ.get("SANAD_TIMEOUT", "60"))
DEFAULT_TENANT = os.environ.get("SANAD_TENANT_ID", "demo_tenant")

app = FastAPI(title="LMS SANAD Integration Example")

# In-memory demo store (replace with Postgres in real LMS)
SUBMISSIONS: dict[str, dict] = {}


class RubricCriterion(BaseModel):
    id: str
    description: str
    max_score: int = Field(gt=0, le=20)


class SubmitBody(BaseModel):
    question_text: str
    student_answer: str
    reference_answer: str
    rubric: list[RubricCriterion]
    source_ids: list[str]
    max_score: int = Field(gt=0, le=100)
    question_id: Optional[str] = None


class IngestBody(BaseModel):
    source_id: str
    text: str
    metadata: dict[str, Any] = {}
    chunker_type: str = "hierarchical"


class ReviewBody(BaseModel):
    action: str  # approve | override
    override_score: Optional[int] = None


def _headers() -> dict:
    if not SANAD_API_KEY:
        raise RuntimeError("SANAD_API_KEY is required (server-side only)")
    return {"X-API-Key": SANAD_API_KEY, "Content-Type": "application/json"}


@app.get("/lms/health")
def lms_health():
    return {"status": "ok", "sanad_base_url": SANAD_BASE_URL}


@app.post("/lms/ingest")
def lms_ingest(body: IngestBody):
    """Store reference material in SANAD (call once per question/chapter)."""
    try:
        r = httpx.post(
            f"{SANAD_BASE_URL}/ingest",
            headers=_headers(),
            json={"source_id": body.source_id, "text": body.text,
                  "metadata": body.metadata, "chunker_type": body.chunker_type},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)


@app.post("/lms/submissions", status_code=202)
def create_submission(body: SubmitBody, background_tasks: BackgroundTasks):
    """Student submit. Persists submission, queues evaluation, returns immediately."""
    criteria_sum = sum(c.max_score for c in body.rubric)
    if criteria_sum != body.max_score:
        raise HTTPException(status_code=422, detail=f"rubric sum ({criteria_sum}) must equal max_score ({body.max_score})")
    submission_id = str(uuid.uuid4())
    request_id = str(uuid.uuid4())
    SUBMISSIONS[submission_id] = {
        "status": "SUBMITTED", "request_id": request_id,
        "body": body.model_dump(), "evaluation": None, "official_grade": None,
    }
    background_tasks.add_task(_run_evaluation, submission_id, request_id, body)
    return {"submission_id": submission_id, "request_id": request_id, "status": "QUEUED"}


def _run_evaluation(submission_id: str, request_id: str, body: SubmitBody, retries: int = 3):
    """Background worker: sync HTTP call to SANAD, then persist."""
    import time
    payload = {
        "request_id": request_id,
        "submission_id": submission_id,
        "tenant_id": DEFAULT_TENANT,
        "question_id": body.question_id or str(uuid.uuid4()),
        "question": {"text": body.question_text, "type": "ESSAY"},
        "student_answer": {"text": body.student_answer},
        "grading": {
            "reference_answer": body.reference_answer,
            "max_score": body.max_score,
            "rubric_version": "v1",
            "criteria": [{"id": c.id, "description": c.description, "max_score": c.max_score} for c in body.rubric],
        },
        "knowledge": {"source_ids": body.source_ids, "version": "v1"},
        "config": {"chunker_type": "hierarchical", "enable_reranking": True, "enable_query_expansion": True, "top_k": 5},
    }
    last_err = None
    for attempt in range(retries):
        try:
            r = httpx.post(f"{SANAD_BASE_URL}/evaluate/submission",
                           headers=_headers(), json=payload, timeout=SANAD_TIMEOUT)
            if 400 <= r.status_code < 500 and r.status_code != 429:
                SUBMISSIONS[submission_id]["status"] = "FAILED"
                SUBMISSIONS[submission_id]["error"] = r.text
                return
            r.raise_for_status()
            result = r.json()
            SUBMISSIONS[submission_id]["evaluation"] = result
            SUBMISSIONS[submission_id]["status"] = "AI_EVALUATED"
            return
        except (httpx.TimeoutException, httpx.NetworkError) as e:
            last_err = e
            time.sleep(2 ** attempt)
    SUBMISSIONS[submission_id]["status"] = "FAILED"
    SUBMISSIONS[submission_id]["error"] = str(last_err)


@app.get("/lms/submissions/{submission_id}")
def get_submission(submission_id: str):
    sub = SUBMISSIONS.get(submission_id)
    if not sub:
        raise HTTPException(status_code=404, detail="not found")
    return {"submission_id": submission_id, "status": sub["status"],
            "evaluation": sub.get("evaluation"), "official_grade": sub.get("official_grade")}


@app.post("/lms/submissions/{submission_id}/review")
def review_submission(submission_id: str, body: ReviewBody):
    """Teacher approve/override. Official grade lives in LMS only."""
    sub = SUBMISSIONS.get(submission_id)
    if not sub or not sub.get("evaluation"):
        raise HTTPException(status_code=404, detail="no AI evaluation yet")
    ev = sub["evaluation"]
    if body.action == "approve":
        sub["official_grade"] = {"score": ev["suggested_score"], "max_score": ev["max_score"], "source": "ai_approved"}
    elif body.action == "override":
        if body.override_score is None:
            raise HTTPException(status_code=400, detail="override_score required")
        sub["official_grade"] = {"score": body.override_score, "max_score": ev["max_score"], "source": "teacher_override"}
    else:
        raise HTTPException(status_code=400, detail="action must be approve or override")
    sub["status"] = "GRADED"
    return {"submission_id": submission_id, "official_grade": sub["official_grade"]}
