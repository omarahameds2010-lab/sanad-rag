"""
Runtime evaluation API endpoints.
POST /evaluate/submission — single-submission essay grading
"""

from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from src.api.deps import _get_embedding_model, _get_llm_clients, _check_qdrant, _check_db
from src.evaluation.schemas import (
    EvaluationRequest,
    EvaluationResponse,
)
from src.evaluation.runtime_evaluator import RuntimeEvaluator
from src.retrieval.reranker import CrossEncoderReranker
from src.security import verify_api_key


router = APIRouter(prefix="/evaluate", tags=["Evaluation"])


@router.post(
    "/submission",
    response_model=EvaluationResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate a single student submission",
    description="""
    Evaluate a single student's essay/short-answer submission using rubric-aware AI grading.
    
    This endpoint:
    - Retrieves relevant knowledge context from Qdrant
    - Runs rubric-aware evaluation via LLM
    - Validates scores against rubric constraints
    - Returns structured evaluation with criterion-level scores
    
    Requires API key authentication via X-API-Key header.
    """,
)
async def evaluate_submission(
    request: EvaluationRequest,
    api_key: str = Depends(verify_api_key),
) -> EvaluationResponse:
    """
    Evaluate a single student submission.
    
    Args:
        request: EvaluationRequest with question, student_answer, grading config, knowledge refs
        
    Returns:
        EvaluationResponse with structured evaluation result
        
    Raises:
        401: Invalid or missing API key
        422: Invalid request (validation errors)
        500: Evaluation failed
        503: SANAD dependencies unavailable
    """
    try:
        # Get dependencies
        embedding_model = _get_embedding_model()
        primary_llm, fallback_llm = _get_llm_clients()
        
        # Create evaluator
        evaluator = RuntimeEvaluator(
            embedding_model=embedding_model,
            primary_llm=primary_llm,
            fallback_llm=fallback_llm,
        )
        
        # Get reranker if requested
        reranker = CrossEncoderReranker() if request.config.enable_reranking else None
        
        # Run evaluation
        response = evaluator.evaluate(request, reranker=reranker)
        
        return response
        
    except HTTPException:
        raise
    except ValueError as e:
        # Validation errors from schemas or evaluation
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error_code": "VALIDATION_ERROR",
                "message": str(e),
            }
        )
    except Exception as e:
        # Internal errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "EVALUATION_FAILED",
                "message": "Evaluation failed due to internal error",
            }
        )


@router.get(
    "/health",
    summary="Evaluation service health check",
    description="Check if evaluation service dependencies are available.",
)
async def evaluation_health(
    api_key: str = Depends(verify_api_key),
) -> dict:
    """Check evaluation service dependencies."""
    from src.api.main import _check_qdrant, _check_db
    
    qdrant_ok = _check_qdrant()
    db_ok = _check_db()
    
    # Also check if LLM is available
    primary_llm, _ = _get_llm_clients()
    llm_ok = primary_llm.name != "fake-llm-DEV-ONLY"
    
    status_val = "ok" if (qdrant_ok and db_ok and llm_ok) else "degraded"
    
    return {
        "status": status_val,
        "qdrant_ok": qdrant_ok,
        "db_ok": db_ok,
        "llm_ok": llm_ok,
    }