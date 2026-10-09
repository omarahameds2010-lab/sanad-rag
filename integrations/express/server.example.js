/**
 * Express integration example: submission endpoint + BullMQ worker calling SANAD.
 *
 * npm i express axios uuid bullmq ioredis dotenv
 *
 * ENV (server-side only):
 *   SANAD_BASE_URL=http://localhost:8000
 *   SANAD_API_KEY=demo-key-123
 *   SANAD_TENANT_ID=demo_tenant
 *   REDIS_URL=redis://localhost:6379
 *
 * NEVER expose SANAD_API_KEY to the browser.
 */

require('dotenv').config();
const express = require('express');
const { Queue, Worker } = require('bullmq');
const IORedis = require('ioredis');
const { SanadClient } = require('./sanadClient');

const app = express();
app.use(express.json({ limit: '1mb' }));

const sanad = new SanadClient(process.env.SANAD_BASE_URL, process.env.SANAD_API_KEY);
const connection = new IORedis(process.env.REDIS_URL || 'redis://localhost:6379', { maxRetriesPerRequest: null });

// In-memory store for demo (replace with Postgres/Prisma in real LMS)
const submissions = new Map(); // submissionId -> { status, evaluation, officialGrade }

// Queue: LMS owns async orchestration (SANAD stays synchronous)
const evalQueue = new Queue('ai-evaluation', {
  connection,
  defaultJobOptions: { attempts: 3, backoff: { type: 'exponential', delay: 1000 }, removeOnComplete: 100, removeOnFail: 50 },
});

// --- Student submit (MUST NOT wait for SANAD) ---
app.post('/api/submissions', async (req, res) => {
  const { questionText, studentAnswer, referenceAnswer, rubric, sourceIds, maxScore } = req.body;
  if (!questionText || !studentAnswer || !referenceAnswer || !rubric) {
    return res.status(400).json({ error: 'questionText, studentAnswer, referenceAnswer, rubric required' });
  }
  const submissionId = require('uuid').v4();
  submissions.set(submissionId, { status: 'SUBMITTED', evaluation: null, officialGrade: null });

  // Idempotency: BullMQ jobId = request_id
  const requestId = require('uuid').v4();
  await evalQueue.add('evaluate', {
    requestId,
    submissionId,
    tenantId: process.env.SANAD_TENANT_ID || 'demo_tenant',
    questionText, studentAnswer, referenceAnswer, rubric, sourceIds, maxScore: maxScore || 10,
  }, { jobId: requestId });

  return res.status(202).json({ submissionId, requestId, status: 'QUEUED' });
});

// --- Poll evaluation status (student/teacher UI) ---
app.get('/api/submissions/:id', (req, res) => {
  const sub = submissions.get(req.params.id);
  if (!sub) return res.status(404).json({ error: 'not found' });
  return res.json({ submissionId: req.params.id, ...sub });
});

// --- Teacher review: approve or override (official grade stays in LMS) ---
app.post('/api/submissions/:id/review', (req, res) => {
  const sub = submissions.get(req.params.id);
  if (!sub || !sub.evaluation) return res.status(404).json({ error: 'no AI evaluation yet' });
  const { action, overrideScore } = req.body; // action: 'approve' | 'override'
  if (action === 'approve') {
    sub.officialGrade = sub.evaluation.suggested_score;
  } else if (action === 'override') {
    if (typeof overrideScore !== 'number') return res.status(400).json({ error: 'overrideScore required' });
    sub.officialGrade = overrideScore;
    sub.teacherOverride = true;
  } else {
    return res.status(400).json({ error: "action must be 'approve' or 'override'" });
  }
  sub.status = 'GRADED';
  return res.json({ submissionId: req.params.id, officialGrade: sub.officialGrade });
});

// --- Worker: BullMQ -> SANAD (concurrency 3, NOT rate-limiter) ---
new Worker('ai-evaluation', async (job) => {
  const d = job.data;
  try {
    const result = await sanad.evaluateSubmission({
      submissionId: d.submissionId,
      questionText: d.questionText,
      studentAnswer: d.studentAnswer,
      referenceAnswer: d.referenceAnswer,
      rubric: d.rubric,
      sourceIds: d.sourceIds,
      maxScore: d.maxScore,
      tenantId: d.tenantId,
    });
    // Validate version match before persisting
    const sub = submissions.get(d.submissionId);
    if (sub) {
      sub.evaluation = result;
      sub.status = 'AI_EVALUATED';
    }
    return result;
  } catch (err) {
    const status = err?.response?.status;
    // 4xx (except 429) = do not retry; 5xx/timeout/429 = retry
    if (status && status >= 400 && status < 500 && status !== 429) throw err; // fail job -> DLQ path
    throw err; // retry with backoff
  }
}, { connection, concurrency: 3 });

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`LMS demo listening on :${PORT}`));
