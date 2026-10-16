"""
Runtime evaluation orchestration for single-submission essay grading.
This module coordinates retrieval, LLM evaluation, and validation.
"""

from time import perf_counter
from typing import Optional
from uuid import uuid4

from src.evaluation.schemas import (
    EvaluationRequest,
    EvaluationResponse,
    EvaluationConfigRequest,
    GradingConfigRequest,
    CriterionResponse,
    EvidenceType,
)
from src.evaluation.rubric import (
    GradingConfig,
    Criterion,
    CriterionResult,
    EvaluationResult,
)
from src.evaluation.validation import (
    validate_evaluation,
    aggregate_scores,
    to_criterion_responses,
)
from src.evaluation.prompts import build_evaluation_prompt, parse_llm_evaluation_output, build_context_from_chunks
from src.embeddings.base import EmbeddingModel
from src.generation.llm_client import LLMClient
from src.retrieval.query_pipeline import answer_question
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.vector_store import collection_for_chunker, get_client
from src.ingestion.normalize import normalize_for_retrieval


class RuntimeEvaluator:
    """
    Orchestrates single-submission essay evaluation.
    Reuses existing retrieval + LLM pipeline with evaluation-specific prompt.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        primary_llm: LLMClient,
        fallback_llm: Optional[LLMClient] = None,
    ):
        self.embedding_model = embedding_model
        self.primary_llm = primary_llm
        self.fallback_llm = fallback_llm

    def evaluate(
        self,
        request: EvaluationRequest,
        reranker: Optional[CrossEncoderReranker] = None,
    ) -> EvaluationResponse:
        """
        Evaluate a single student submission.
        
        Args:
            request: EvaluationRequest with question, student_answer, grading config
            reranker: Optional cross-encoder reranker
            
        Returns:
            EvaluationResponse with structured evaluation result
        """
        evaluation_id = uuid4()
        start_total = perf_counter()

        # Convert request grading to internal GradingConfig
        grading = self._to_grading_config(request.grading)

        # Build retrieval query: combine question + student answer for better recall
        retrieval_query = f"{request.question.get('text', '')} {request.student_answer.get('text', '')}"

        # Retrieve relevant context
        retrieval_start = perf_counter()
        context_chunks = self._retrieve_context(
            query=retrieval_query,
            request=request,
            top_k=request.config.top_k,
            reranker=reranker,
        )
        retrieval_ms = round((perf_counter() - retrieval_start) * 1000, 2)

        # Build context string for prompt
        context = build_context_from_chunks(context_chunks)

        # Build evaluation prompt
        prompt = build_evaluation_prompt(
            question_text=request.question.get("text", ""),
            student_answer=request.student_answer.get("text", ""),
            grading=self._to_grading_config(request.grading),
            context=build_context_from_chunks(context_chunks),
        )

        # Generate evaluation via LLM
        generation_start = perf_counter()
        try:
            # Use new evaluation-specific generation method
            evaluation_data = self._generate_evaluation(prompt, context_chunks)
            used_model = self.primary_llm.name
        except Exception as primary_error:
            if self.fallback_llm is None:
                raise
            evaluation_data = self._generate_evaluation_fallback(prompt, context_chunks)
            used_model = f"{self.fallback_llm.name} (fallback: {primary_error})"
        
        generation_ms = round((perf_counter() - generation_start) * 1000, 2)

        # Parse and validate LLM output
        evaluation_result = self._parse_and_validate(
            evaluation_data, 
            request.grading,
            request.grading.max_score,
        )

        # Build response
        total_ms = round((perf_counter() - start_total) * 1000, 2)

        return EvaluationResponse(
            evaluation_id=evaluation_id,
            request_id=request.request_id,
            status="COMPLETED",
            suggested_score=evaluation_result.suggested_score,
            max_score=evaluation_result.max_score,
            criteria=to_criterion_responses(evaluation_result.criteria),
            feedback_ar=evaluation_result.feedback_ar,
            review_signal=evaluation_result.review_signal,
            needs_teacher_review=evaluation_result.needs_teacher_review,
            knowledge_gaps=evaluation_result.knowledge_gaps,
            provenance={
                "sanad_version": "0.4.0",
                "model": used_model,
                "embedding_model": self.embedding_model.name,
                "chunker_type": request.config.chunker_type.value,
                "reranker": "BAAI/bge-reranker-base" if reranker else "none",
                "knowledge_version": request.knowledge.version,
                "rubric_version": request.grading.rubric_version,
                "prompt_version": "eval_v1",
            },
            usage={
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
                "total_ms": total_ms,
            },
        )

    def _to_grading_config(self, request_grading: GradingConfigRequest) -> GradingConfig:
        """Convert request schema to internal GradingConfig."""
        criteria = [
            type("Criterion", (), {"id": c.id, "description": c.description, "max_score": c.max_score})()
            for c in request_grading.criteria
        ]
        
        # Use a simple dict-like object that mimics GradingConfig interface
        class SimpleGradingConfig:
            def __init__(self, ref, max_s, version, criteria_list):
                self.reference_answer = ref
                self.max_score = max_s
                self.rubric_version = version
                self.criteria = criteria_list
                self._criteria = {c.id: c for c in criteria_list}
            
            def get_criterion_ids(self):
                return set(self._criteria.keys())
            
            def get_criterion(self, cid):
                return self._criteria.get(cid)
        
        return SimpleGradingConfig(
            request_grading.reference_answer,
            request_grading.max_score,
            request_grading.rubric_version,
            [type("C", (), {"id": c.id, "description": c.description, "max_score": c.max_score})() 
             for c in request_grading.criteria]
        )

    def _retrieve_context(
        self,
        query: str,
        request: EvaluationRequest,
        top_k: int,
        reranker,
    ) -> list[dict]:
        """
        Retrieve relevant context chunks for evaluation.
        Filters by tenant_id and authorized source_ids.
        """
        from src.retrieval.hybrid import hybrid_retrieve
        from src.retrieval.vector_store import search as semantic_search, collection_for_chunker
        from src.retrieval.bm25_index import bm25_search
        from src.retrieval.hybrid import RRF_K
        from src.retrieval.vector_store import get_client
        
        is_hierarchical = request.config.chunker_type.value == "hierarchical"
        
        normalized_query = normalize_for_retrieval(query)
        query_vector = self.embedding_model.embed([normalized_query])[0]
        
        if is_hierarchical:
            # Hierarchical retrieval with tenant filtering
            return self._hierarchical_retrieve_with_filter(
                query_vector, 
                query,
                request.tenant_id,
                request.knowledge.source_ids,
                top_k=top_k * 2,
                reranker=reranker,
            )
        else:
            # Standard retrieval with tenant filtering
            collection = collection_for_chunker(
                request.config.chunker_type.value, 
                self.embedding_model.dimensions
            )
            
            client = get_client()
            # Apply tenant + source filter
            from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
            tenant_filter = Filter(
                must=[
                    FieldCondition(key="tenant_id", match=MatchValue(value=request.tenant_id)),
                    FieldCondition(key="source_id", match=MatchAny(any=request.knowledge.source_ids)) 
                    if request.knowledge.source_ids else FieldCondition(key="source_id", match=MatchValue(value=""))
                ]
            )
            
            # Use semantic search with filter
            semantic_results = semantic_search(
                client, query_vector, top_k=top_k * 2, 
                collection=collection,
                query_filter=tenant_filter
            )
            
            keyword_results = bm25_search(
                query, top_k=top_k * 2, collection=collection
            )
            
            # Fuse with RRF
            fused_scores = {}
            payload_by_id = {}
            
            for rank, result in enumerate(semantic_results):
                cid = result["chunk_id"]
                fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (60 + rank + 1)
                payload_by_id[cid] = result
            
            for rank, result in enumerate(keyword_results):
                cid = result["chunk_id"]
                fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (60 + rank + 1)
                payload_by_id.setdefault(cid, result)
            
            ranked_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)
            return [
                {**payload_by_id[cid], "fused_score": fused_scores[cid]}
                for cid in ranked_ids[:top_k]
            ]

    def _hierarchical_retrieve_with_filter(
        self,
        query_vector: list[float],
        normalized_query: str,
        tenant_id: str,
        source_ids: list[str],
        top_k: int,
        reranker,
    ) -> list[dict]:
        """Hierarchical retrieval with tenant and source filtering."""
        from src.retrieval.vector_store import search as semantic_search, collection_for_chunker
        from src.retrieval.bm25_index import bm25_search
        from src.retrieval.hybrid import RRF_K
        from src.retrieval.vector_store import get_client
        from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
        
        client = get_client()
        dims = self.embedding_model.dimensions
        
        child_collection = collection_for_chunker("hierarchical_children", dims)
        
        # Build filter for tenant + source
        filter_conditions = [
            FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))
        ]
        if source_ids:
            filter_conditions.append(
                FieldCondition(key="source_id", match=MatchAny(any=source_ids))
            )
        
        tenant_filter = Filter(must=filter_conditions)
        
        # Search children with filter
        semantic_results = semantic_search(
            client, query_vector, top_k=top_k * 2, 
            collection=child_collection,
            query_filter=tenant_filter
        )
        keyword_results = bm25_search(
            normalized_query, top_k=top_k * 2, collection=child_collection
        )
        
        # RRF fusion
        fused_scores = {}
        payload_by_id = {}
        
        for rank, result in enumerate(semantic_results):
            cid = result["chunk_id"]
            fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
            payload_by_id[cid] = result
        
        for rank, result in enumerate(keyword_results):
            cid = result["chunk_id"]
            fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
            payload_by_id.setdefault(cid, result)
        
        ranked_child_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)
        top_children = [
            {**payload_by_id[cid], "fused_score": fused_scores[cid]}
            for cid in ranked_child_ids[:top_k]
        ]
        
        # Map to parents. Keyed by (source_id, parent_id) because
        # parent_ids restart at parent_0 inside every source document.
        parent_map = {}
        for child in top_children:
            parent_id = child.get("parent_id")
            child_source_id = child.get("source_id", "")
            if not parent_id:
                continue

            key = (child_source_id, parent_id)
            score = child.get("fused_score", 0.0)
            child_id = child.get("chunk_id")

            if key not in parent_map or score > parent_map[key]["best_score"]:
                parent_collection = collection_for_chunker("hierarchical_parents", dims)
                parent_results = client.scroll(
                    collection_name=parent_collection,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id)),
                            FieldCondition(key="source_id", match=MatchValue(value=child_source_id)),
                            FieldCondition(key="parent_id", match=MatchValue(value=parent_id))
                        ]
                    ),
                    limit=1,
                    with_payload=True,
                )
                if parent_results[0]:
                    parent_map[key] = {
                        "payload": parent_results[0][0].payload,
                        "best_score": score,
                        "child_ids": [child_id],
                    }
            else:
                parent_map[key]["child_ids"].append(child_id)

        # Convert to chunks
        parent_chunks = []
        for (child_source_id, parent_id), data in parent_map.items():
            payload = data["payload"]
            payload = {
                **payload,
                "fused_score": data["best_score"],
                "child_ids": data["child_ids"],
                "chunk_id": f"{child_source_id}::parent::{parent_id}",
            }
            parent_chunks.append(payload)
        
        parent_chunks.sort(key=lambda c: c.get("fused_score", 0.0), reverse=True)
        
        if reranker:
            parent_chunks = reranker.rerank(normalize_for_retrieval(""), parent_chunks, top_k=top_k)
        else:
            parent_chunks = parent_chunks[:top_k]
        
        return parent_chunks

    def _generate_evaluation(self, prompt: str, context_chunks: list[dict]) -> dict:
        """Generate evaluation using primary LLM with structured output."""
        from src.evaluation.prompts import parse_llm_evaluation_output
        
        # Use evaluation prompt directly (already contains full instruction)
        raw_output = self.primary_llm.generate_evaluation(prompt, None)
        
        # Parse structured output
        parsed = parse_llm_evaluation_output(raw_output)
        return parsed

    def _generate_evaluation_fallback(self, prompt: str, context_chunks: list[dict]) -> dict:
        """Fallback generation using fallback LLM."""
        from src.evaluation.prompts import parse_llm_evaluation_output
        raw_output = self.fallback_llm.generate_evaluation(prompt, None)
        return parse_llm_evaluation_output(raw_output)

    def _parse_and_validate(self, evaluation_data: dict, request_grading, max_score: int):
        """Parse LLM output and validate against rubric."""
        # Convert to internal format
        criteria_results = []
        for c in evaluation_data.get("criteria", []):
            criteria_results.append(CriterionResult(
                criterion_id=c["criterion_id"],
                earned=c["earned"],
                max=next((crit.max_score for crit in request_grading.criteria if crit.id == c["criterion_id"]), 0),
                evidence=c.get("evidence"),
                evidence_type=c.get("evidence_type", "ABSENT"),
                feedback_ar=c.get("feedback_ar", ""),
            ))
        
        result = EvaluationResult(
            evaluation_id="",  # Will be set by caller
            request_id="",
            suggested_score=evaluation_data.get("suggested_score", sum(c.earned for c in criteria_results)),
            max_score=max_score,
            criteria=criteria_results,
            feedback_ar=evaluation_data.get("overall_feedback_ar", ""),
            review_signal=evaluation_data.get("review_signal", 0.5),
            needs_teacher_review=evaluation_data.get("needs_teacher_review", True),
            knowledge_gaps=evaluation_data.get("knowledge_gaps", []),
        )
        
        # Validate
        from src.evaluation.validation import validate_evaluation
        from src.evaluation.rubric import GradingConfig
        
        # Create GradingConfig for validation
        grading = GradingConfig(
            reference_answer=request_grading.reference_answer,
            max_score=request_grading.max_score,
            rubric_version=request_grading.rubric_version,
            criteria=[Criterion(id=c.id, description=c.description, max_score=c.max_score) 
                     for c in request_grading.criteria],
        )
        
        validation = validate_evaluation(result, grading)
        needs_teacher_review = result.needs_teacher_review
        feedback_ar = result.feedback_ar
        if not validation.valid:
            # A grade missing criteria or breaking rubric arithmetic must
            # never look complete: force teacher review and surface why.
            needs_teacher_review = True
            reasons = "; ".join(str(e) for e in validation.errors)
            audit_note = f"[تدقيق تلقائي: الدرجة مخالفة للمعايير — {reasons}]"
            feedback_ar = f"{feedback_ar}\n{audit_note}".strip() if feedback_ar else audit_note

        return EvaluationResult(
            evaluation_id="",
            request_id="",
            suggested_score=result.suggested_score,
            max_score=result.max_score,
            criteria=result.criteria,
            feedback_ar=feedback_ar,
            review_signal=result.review_signal,
            needs_teacher_review=needs_teacher_review,
            knowledge_gaps=result.knowledge_gaps,
        )