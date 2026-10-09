/**
 * SANAD Integration Client for Node.js / Express / NestJS (LMS side)
 *
 * npm i axios uuid
 *
 * Usage:
 *   const { SanadClient } = require('./sanadClient');
 *   const sanad = new SanadClient(process.env.SANAD_BASE_URL, process.env.SANAD_API_KEY);
 *   await sanad.ingestMaterial('history_ch5_v1', '...');
 *   const result = await sanad.evaluateSubmission({...});
 */

const axios = require('axios');
const { v4: uuidv4 } = require('uuid');

class SanadClient {
  /**
   * @param {string} baseUrl - e.g. http://localhost:8000
   * @param {string} apiKey - SANAD_API_KEYS value (server-side only!)
   * @param {object} [opts]
   * @param {number} [opts.timeoutMs=60000]
   * @param {number} [opts.maxRetries=3]
   */
  constructor(baseUrl, apiKey, opts = {}) {
    if (!baseUrl) throw new Error('SANAD baseUrl is required');
    if (!apiKey) throw new Error('SANAD apiKey is required (server-side only, never expose to browser)');
    this.client = axios.create({
      baseURL: baseUrl.replace(/\/$/, ''),
      headers: { 'X-API-Key': apiKey, 'Content-Type': 'application/json' },
      timeout: opts.timeoutMs || 60000,
    });
    this.maxRetries = opts.maxRetries ?? 3;
  }

  async _requestWithRetry(method, url, data = undefined, params = undefined) {
    let lastErr;
    for (let attempt = 0; attempt < this.maxRetries; attempt++) {
      try {
        const res = await this.client.request({ method, url, data, params });
        return res.data;
      } catch (err) {
        lastErr = err;
        const status = err?.response?.status;
        // Don't retry 4xx (except 429)
        if (status && status >= 400 && status < 500 && status !== 429) throw err;
        if (attempt < this.maxRetries - 1) {
          const wait = 2 ** attempt * 1000 + attempt * 100;
          await new Promise((r) => setTimeout(r, wait));
        }
      }
    }
    throw lastErr;
  }

  /** GET /health (no auth needed) */
  health() {
    return this._requestWithRetry('get', '/health');
  }

  /**
   * POST /ingest — store reference material once per question/chapter.
   * @param {string} sourceId e.g. 'history_ch5_v1'
   * @param {string} text Arabic reference text
   * @param {object} [metadata]
   * @param {string} [chunkerType='hierarchical']
   */
  ingestMaterial(sourceId, text, metadata = {}, chunkerType = 'hierarchical') {
    return this._requestWithRetry('post', '/ingest', {
      source_id: sourceId,
      text,
      metadata,
      chunker_type: chunkerType,
    });
  }

  /**
   * POST /evaluate/submission — rubric-aware essay grading.
   * @param {object} args
   * @param {string} args.submissionId
   * @param {string} args.questionText
   * @param {string} args.studentAnswer
   * @param {string} args.referenceAnswer
   * @param {Array<{id:string,description:string,max_score:number}>} args.rubric
   * @param {string[]} args.sourceIds
   * @param {number} [args.maxScore=10]
   * @param {string} [args.tenantId]
   * @param {string} [args.questionId]
   * @param {string} [args.questionType='ESSAY']
   * @param {string} [args.rubricVersion='v1']
   * @param {string} [args.knowledgeVersion='v1']
   */
  evaluateSubmission({
    submissionId,
    questionText,
    studentAnswer,
    referenceAnswer,
    rubric,
    sourceIds,
    maxScore = 10,
    tenantId,
    questionId,
    questionType = 'ESSAY',
    rubricVersion = 'v1',
    knowledgeVersion = 'v1',
    config,
  }) {
    if (!tenantId) throw new Error('tenantId is required for tenant isolation');
    if (!Array.isArray(rubric) || rubric.length === 0) throw new Error('rubric criteria required');
    const criteriaSum = rubric.reduce((s, c) => s + c.max_score, 0);
    if (criteriaSum !== maxScore) {
      throw new Error(`rubric criteria sum (${criteriaSum}) must equal maxScore (${maxScore})`);
    }
    if (!['ESSAY', 'SHORT_ANSWER'].includes(questionType)) {
      throw new Error(`unsupported questionType for SANAD: ${questionType}`);
    }
    return this._requestWithRetry('post', '/evaluate/submission', {
      request_id: uuidv4(),
      submission_id: submissionId,
      tenant_id: tenantId,
      question_id: questionId || uuidv4(),
      question: { text: questionText, type: questionType },
      student_answer: { text: studentAnswer },
      grading: {
        reference_answer: referenceAnswer,
        max_score: maxScore,
        rubric_version: rubricVersion,
        criteria: rubric.map((c) => ({ id: c.id, description: c.description, max_score: c.max_score })),
      },
      knowledge: { source_ids: sourceIds, version: knowledgeVersion },
      config: config || { chunker_type: 'hierarchical', enable_reranking: true, enable_query_expansion: true, top_k: 5 },
    });
  }

  /** POST /query — RAG Q&A only (NOT grading). */
  query(question, opts = {}) {
    return this._requestWithRetry('post', '/query', {
      question,
      top_k: opts.topK ?? 5,
      chunker_type: opts.chunkerType || 'hierarchical',
      enable_reranking: opts.enableReranking ?? false,
      enable_query_expansion: opts.enableQueryExpansion ?? false,
    });
  }
}

module.exports = { SanadClient };
