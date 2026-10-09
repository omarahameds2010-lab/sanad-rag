import type {
  ApiError,
  ChunkerType,
  EvaluateReport,
  EvaluateRunResponse,
  EvaluationResponse,
  HealthResponse,
  IngestResponse,
  QueryResponse,
  RubricCriterion,
} from './types';

export interface SanadSettings {
  baseUrl: string;
  apiKey: string;
  tenantId: string;
}

const SETTINGS_KEY = 'sanad-settings';

export function loadSettings(): SanadSettings {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as Partial<SanadSettings>;
      return {
        baseUrl: parsed.baseUrl || import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
        apiKey: parsed.apiKey ?? import.meta.env.VITE_SANAD_API_KEY ?? 'demo-key-123',
        tenantId: parsed.tenantId || 'demo_tenant',
      };
    }
  } catch {
    // ignore corrupt storage
  }
  return {
    baseUrl: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
    apiKey: import.meta.env.VITE_SANAD_API_KEY ?? 'demo-key-123',
    tenantId: 'demo_tenant',
  };
}

export function saveSettings(s: SanadSettings): void {
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(s));
}

function toApiError(status: number, body: unknown): ApiError {
  let message = `HTTP ${status}`;
  let code = 'HTTP_ERROR';
  if (body && typeof body === 'object') {
    const d = body as Record<string, unknown>;
    const detail = d.detail;
    if (typeof detail === 'string') {
      message = detail;
    } else if (detail && typeof detail === 'object') {
      const dd = detail as Record<string, unknown>;
      if (typeof dd.message === 'string') message = dd.message;
      if (typeof dd.error_code === 'string') code = dd.error_code;
    } else if (Array.isArray((d as { errors?: unknown }).errors)) {
      message = JSON.stringify((d as { errors: unknown }).errors);
      code = 'VALIDATION_ERROR';
    }
  }
  const retryable = status === 429 || status === 500 || status === 502 || status === 503 || status === 504 || status === 0;
  if (status === 401) code = 'UNAUTHENTICATED';
  else if (status === 404) code = 'NOT_FOUND';
  else if (status === 422) code = 'VALIDATION_ERROR';
  else if (status === 429) code = 'RATE_LIMITED';
  return { status, code, message, retryable };
}

async function request<T>(s: SanadSettings, path: string, init?: RequestInit): Promise<T> {
  const url = `${s.baseUrl.replace(/\/$/, '')}${path}`;
  let res: Response;
  try {
    res = await fetch(url, {
      ...init,
      headers: {
        'Content-Type': 'application/json',
        ...(s.apiKey ? { 'X-API-Key': s.apiKey } : {}),
        ...(init?.headers || {}),
      },
    });
  } catch (e) {
    throw { status: 0, code: 'NETWORK_ERROR', message: 'تعذر الاتصال بالخادم — تأكد أن الـ API يعمل', retryable: true } as ApiError;
  }
  if (!res.ok) {
    let body: unknown = null;
    try {
      body = await res.json();
    } catch {
      body = null;
    }
    throw toApiError(res.status, body);
  }
  return (await res.json()) as T;
}

export const api = {
  health: (s: SanadSettings) => request<HealthResponse>(s, '/health'),
  evalHealth: (s: SanadSettings) => request<{ status: string; qdrant_ok: boolean; db_ok: boolean; llm_ok: boolean }>(s, '/evaluate/health'),

  ingest: (s: SanadSettings, args: { source_id: string; text: string; metadata?: Record<string, unknown>; chunker_type: ChunkerType }) =>
    request<IngestResponse>(s, '/ingest', { method: 'POST', body: JSON.stringify(args) }),

  query: (
    s: SanadSettings,
    args: { question: string; top_k?: number; chunker_type?: ChunkerType; enable_reranking?: boolean; enable_query_expansion?: boolean },
  ) =>
    request<QueryResponse>(s, '/query', {
      method: 'POST',
      body: JSON.stringify({
        question: args.question,
        top_k: args.top_k ?? 5,
        chunker_type: args.chunker_type ?? 'hierarchical',
        enable_reranking: args.enable_reranking ?? false,
        enable_query_expansion: args.enable_query_expansion ?? false,
      }),
    }),

  evaluateSubmission: (
    s: SanadSettings,
    args: {
      submission_id: string;
      question_id: string;
      question_text: string;
      question_type?: 'ESSAY' | 'SHORT_ANSWER';
      student_answer: string;
      reference_answer: string;
      rubric: RubricCriterion[];
      max_score: number;
      rubric_version?: string;
      source_ids: string[];
      knowledge_version?: string;
      chunker_type?: ChunkerType;
      enable_reranking?: boolean;
      enable_query_expansion?: boolean;
      top_k?: number;
    },
  ) => {
    const criteriaSum = args.rubric.reduce((t, c) => t + c.max_score, 0);
    if (criteriaSum !== args.max_score) {
      throw { status: 422, code: 'INVALID_RUBRIC', message: `مجموع المعايير (${criteriaSum}) يجب أن يساوي الدرجة العظمى (${args.max_score})`, retryable: false } as ApiError;
    }
    return request<EvaluationResponse>(s, '/evaluate/submission', {
      method: 'POST',
      body: JSON.stringify({
        request_id: crypto.randomUUID(),
        submission_id: args.submission_id,
        tenant_id: s.tenantId,
        question_id: args.question_id,
        question: { text: args.question_text, type: args.question_type ?? 'ESSAY' },
        student_answer: { text: args.student_answer },
        grading: {
          reference_answer: args.reference_answer,
          max_score: args.max_score,
          rubric_version: args.rubric_version ?? 'v1',
          criteria: args.rubric,
        },
        knowledge: { source_ids: args.source_ids, version: args.knowledge_version ?? 'v1' },
        config: {
          chunker_type: args.chunker_type ?? 'hierarchical',
          enable_reranking: args.enable_reranking ?? true,
          enable_query_expansion: args.enable_query_expansion ?? true,
          top_k: args.top_k ?? 5,
        },
      }),
    });
  },

  evaluateRun: (
    s: SanadSettings,
    args: { golden_dataset_path: string; config_name?: string; chunker_type?: ChunkerType; enable_reranking?: boolean; enable_query_expansion?: boolean },
  ) =>
    request<EvaluateRunResponse>(s, '/evaluate/run', {
      method: 'POST',
      body: JSON.stringify({
        golden_dataset_path: args.golden_dataset_path,
        config_name: args.config_name ?? 'baseline',
        chunker_type: args.chunker_type ?? 'arabic',
        enable_reranking: args.enable_reranking ?? false,
        enable_query_expansion: args.enable_query_expansion ?? false,
      }),
    }),

  evaluateReport: (s: SanadSettings, runId: string) =>
    request<EvaluateReport>(s, `/evaluate/report?run_id=${encodeURIComponent(runId)}`),
};

export function formatApiError(e: unknown): string {
  if (e && typeof e === 'object' && 'message' in e) {
    const err = e as ApiError;
    const retry = err.retryable ? ' (قابلة لإعادة المحاولة)' : '';
    return `[${err.code || err.status}] ${err.message}${retry}`;
  }
  return 'خطأ غير متوقع';
}
