export type ChunkerType = 'arabic' | 'baseline' | 'hierarchical';
export type QuestionType = 'ESSAY' | 'SHORT_ANSWER';
export type EvidenceType = 'SUPPORTED' | 'PARTIAL' | 'ABSENT';

export interface HealthResponse {
  status: string;
  qdrant_ok: boolean;
  db_ok: boolean;
}

export interface IngestResponse {
  source_id: string;
  n_chunks: number;
  embedding_model: string;
  chunker_type: string;
  n_parents?: number | null;
  n_children?: number | null;
}

export interface QueryResponse {
  answer: string;
  cited_chunk_ids: string[];
  retrieval_ms: number;
  generation_ms: number;
  reranker_used: string;
  query_expansion_enabled?: boolean;
}

export interface RubricCriterion {
  id: string;
  description: string;
  max_score: number;
}

export interface CriterionResult {
  criterion_id: string;
  earned: number;
  max: number;
  evidence: string | null;
  evidence_type: EvidenceType;
  feedback_ar: string;
}

export interface EvaluationResponse {
  evaluation_id: string;
  request_id: string;
  status: 'COMPLETED' | 'FAILED';
  suggested_score: number;
  max_score: number;
  criteria: CriterionResult[];
  feedback_ar: string;
  review_signal: number;
  needs_teacher_review: boolean;
  knowledge_gaps: string[];
  provenance: Record<string, unknown>;
  usage: Record<string, unknown>;
}

export interface EvaluateRunResponse {
  run_id: string;
  n_examples: number;
  status: string;
}

export interface EvaluateReport {
  run_id: string;
  config_name: string;
  started_at?: string;
  n_examples: number;
  precision_at_k: number | null;
  recall_at_k: number | null;
  p50_latency_seconds: number | null;
  p95_latency_seconds: number | null;
}

export interface ApiError {
  status: number;
  code: string;
  message: string;
  retryable: boolean;
}
