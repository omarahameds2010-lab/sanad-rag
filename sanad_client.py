"""
SANAD Integration Client for Python (FastAPI/Django/Flask)
Usage:
    client = SanadClient("http://localhost:8000", "your-api-key")
    await client.ingest_material("history_ch5_v1", "نص المادة...")
    result = await client.evaluate_submission(...)
"""

import httpx
import uuid
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import asyncio


@dataclass
class Criterion:
    id: str
    description: str
    max_score: int


@dataclass
class GradingConfig:
    reference_answer: str
    max_score: int
    rubric_version: str
    criteria: List[Criterion]


@dataclass
class KnowledgeConfig:
    source_ids: List[str]
    version: str


@dataclass
class EvaluationConfig:
    chunker_type: str = "hierarchical"
    enable_reranking: bool = True
    enable_query_expansion: bool = True
    top_k: int = 5


@dataclass
class EvaluationRequest:
    request_id: str
    submission_id: str
    tenant_id: str
    question_id: str
    question_text: str
    question_type: str = "ESSAY"
    student_answer: str
    grading: GradingConfig
    knowledge: KnowledgeConfig
    config: EvaluationConfig = None


@dataclass
class CriterionResult:
    criterion_id: str
    earned: int
    max: int
    evidence: Optional[str]
    evidence_type: str  # SUPPORTED, PARTIAL, ABSENT
    feedback_ar: str


@dataclass
class EvaluationResponse:
    evaluation_id: str
    request_id: str
    status: str
    suggested_score: int
    max_score: int
    criteria: List[CriterionResult]
    feedback_ar: str
    review_signal: float
    needs_teacher_review: bool
    knowledge_gaps: List[str]
    provenance: Dict[str, Any]
    usage: Dict[str, Any]


class SanadClient:
    """
    Async client for SANAD AI Evaluation Engine.
    
    Usage:
        client = SanadClient("http://localhost:8000", "your-api-key")
        await client.ingest_material("history_ch5_v1", "نص المادة...")
        result = await client.evaluate_submission(...)
    """
    
    def __init__(
        self, 
        base_url: str, 
        api_key: str,
        timeout: float = 60.0,
        max_retries: int = 3
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"X-API-Key": self.api_key},
            timeout=httpx.Timeout(self.timeout),
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()
    
    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={"X-API-Key": self.api_key},
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client
    
    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def _request_with_retry(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Make request with exponential backoff retry."""
        client = self._get_client()
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                response = await client.request(method, url, **kwargs)
                
                # Don't retry on client errors (4xx except 429)
                if 400 <= response.status_code < 500 and response.status_code != 429:
                    return response
                
                # Success or server error (5xx) or rate limit (429)
                if response.status_code < 400:
                    return response
                
            except (httpx.TimeoutException, httpx.NetworkError) as e:
                last_exception = e
            
            # Wait before retry (exponential backoff)
            if attempt < self.max_retries - 1:
                wait_time = (2 ** attempt) + (0.1 * attempt)
                await asyncio.sleep(wait_time)
        
        # All retries failed
        if last_exception:
            raise last_exception
        raise httpx.RequestError("Max retries exceeded")
    
    # ============================================================
    # Public API Methods
    # ============================================================
    
    async def health_check(self) -> dict:
        """Check SANAD service health."""
        r = await self._request_with_retry("GET", "/health")
        r.raise_for_status()
        return r.json()
    
    async def ingest_material(
        self,
        source_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunker_type: str = "hierarchical"
    ) -> dict:
        """
        Ingest reference material into SANAD.
        
        Args:
            source_id: Unique identifier for the document
            text: Raw text content (Arabic/mixed)
            metadata: Optional metadata dict
            chunker_type: "arabic" | "baseline" | "hierarchical"
        
        Returns:
            Dict with source_id, n_chunks, embedding_model, chunker_type
        """
        payload = {
            "source_id": source_id,
            "text": text,
            "metadata": metadata or {},
            "chunker_type": chunker_type
        }
        
        r = await self._request_with_retry("POST", "/ingest", json=payload)
        r.raise_for_status()
        return r.json()
    
    async def evaluate_submission(
        self,
        submission_id: str,
        question_text: str,
        student_answer: str,
        reference_answer: str,
        rubric: List[Dict[str, Any]],
        source_ids: List[str],
        max_score: int = 10,
        question_id: Optional[str] = None,
        question_type: str = "ESSAY",
        tenant_id: str = "default",
        rubric_version: str = "v1",
        knowledge_version: str = "v1",
        config: Optional[Dict[str, Any]] = None
    ) -> dict:
        """
        Evaluate a single student submission.
        
        Args:
            submission_id: LMS submission ID
            question_text: The question text
            student_answer: Student's answer text
            reference_answer: Teacher's reference answer
            rubric: List of {"id": "...", "description": "...", "max_score": int}
            source_ids: List of source_ids from /ingest
            max_score: Total max score
            question_id: Optional question ID
            question_type: "ESSAY" | "SHORT_ANSWER"
            tenant_id: Tenant identifier
            rubric_version: Rubric version string
            knowledge_version: Knowledge version string
            config: Optional config overrides
        
        Returns:
            Dict with evaluation_id, suggested_score, criteria, feedback_ar, etc.
        """
        criteria = [
            {"id": c["id"], "description": c["description"], "max_score": c["max_score"]}
            for c in rubric
        ]
        
        payload = {
            "request_id": str(uuid.uuid4()),
            "submission_id": submission_id,
            "tenant_id": tenant_id,
            "question_id": question_id or str(uuid.uuid4()),
            "question": {"text": question_text, "type": question_type},
            "student_answer": {"text": student_answer},
            "grading": {
                "reference_answer": reference_answer,
                "max_score": max_score,
                "rubric_version": rubric_version,
                "criteria": [
                    {"id": c["id"], "description": c["description"], "max_score": c["max_score"]}
                    for c in rubric
                ]
            },
            "knowledge": {
                "source_ids": source_ids,
                "version": knowledge_version
            },
            "config": config or {
                "chunker_type": "hierarchical",
                "enable_reranking": True,
                "enable_query_expansion": True,
                "top_k": 5
            }
        }
        
        r = await self._request_with_retry(
            "POST", 
            "/evaluate/submission", 
            json=payload
        )
        r.raise_for_status()
        return r.json()
    
    async def query(
        self,
        question: str,
        top_k: int = 5,
        chunker_type: str = "hierarchical",
        enable_reranking: bool = False,
        enable_query_expansion: bool = False
    ) -> dict:
        """
        Query SANAD for RAG Q&A (not evaluation).
        """
        payload = {
            "question": question,
            "top_k": top_k,
            "chunker_type": chunker_type,
            "enable_reranking": enable_reranking,
            "enable_query_expansion": enable_query_expansion
        }
        
        r = await self._request_with_retry("POST", "/query", json=payload)
        r.raise_for_status()
        return r.json()
    
    async def evaluate_batch(
        self,
        golden_dataset_path: str,
        config_name: str = "baseline",
        chunker_type: str = "arabic",
        enable_reranking: bool = False,
        enable_query_expansion: bool = False
    ) -> dict:
        """Run batch evaluation on golden dataset."""
        payload = {
            "golden_dataset_path": golden_dataset_path,
            "config_name": config_name,
            "chunker_type": chunker_type,
            "enable_reranking": enable_reranking,
            "enable_query_expansion": enable_query_expansion
        }
        
        r = await self._request_with_retry("POST", "/evaluate/run", json=payload)
        r.raise_for_status()
        return r.json()
    
    async def get_evaluation_report(self, run_id: str) -> dict:
        """Get evaluation report by run_id."""
        r = await self._request_with_retry("GET", "/evaluate/report", params={"run_id": run_id})
        r.raise_for_status()
        return r.json()


# ============================================================
# Convenience Functions
# ============================================================

async def evaluate_student_answer(
    base_url: str,
    api_key: str,
    submission_id: str,
    question_text: str,
    student_answer: str,
    reference_answer: str,
    rubric: List[Dict],
    source_ids: List[str],
    max_score: int = 10,
    tenant_id: str = "default"
) -> dict:
    """
    One-liner for evaluating a single submission.
    
    Usage:
        result = await evaluate_student_answer(
            "http://localhost:8000", "api-key",
            "sub_123", "سؤال...", "إجابة الطالب", "إجابة نموذجية",
            [{"id": "c1", "description": "...", "max_score": 5}],
            ["source_1"], 10
        )
    """
    async with SanadClient(base_url, api_key) as client:
        return await client.evaluate_submission(
            submission_id=submission_id,
            question_text=question_text,
            student_answer=student_answer,
            reference_answer=reference_answer,
            rubric=rubric,
            source_ids=source_ids,
            max_score=max_score,
            tenant_id=tenant_id,
        )


# Example usage
if __name__ == "__main__":
    async def main():
        async with SanadClient("http://localhost:8000", "demo-key-123") as client:
            # 1. Ingest reference material
            await client.ingest_material(
                source_id="history_ch5_v1",
                text="ثورة 1919 قامت بسبب الاستعمار البريطاني...",
                metadata={"subject": "history", "grade": 10}
            )
            
            # 2. Evaluate student submission
            result = await client.evaluate_submission(
                submission_id="sub_123",
                question_text="اشرح أسباب ثورة 1919.",
                student_answer="ثورة 1919 قامت بسبب الاستعمار البريطاني...",
                reference_answer="ثورة 1919 قامت بسبب الاستعمار البريطاني...",
                rubric=[
                    {"id": "c1", "description": "الأسباب السياسية", "max_score": 4},
                    {"id": "c2", "description": "الأسباب الاقتصادية", "max_score": 3},
                    {"id": "c3", "description": "الأسباب الاجتماعية", "max_score": 3},
                ],
                source_ids=["history_ch5_v1"],
                max_score=10
            )
            
            print(f"Score: {result['suggested_score']}/{result['max_score']}")
            print(f"Feedback: {result['feedback_ar']}")
            for c in result['criteria']:
                print(f"  {c['criterion_id']}: {c['earned']}/{c['max']} ({c['evidence_type']})")
    
    asyncio.run(main())