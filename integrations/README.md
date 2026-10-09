# LMS Integration Guide — ربط المنصة التعليمية بـ SANAD

SANAD شغال كـ **خدمة مستقلة** (FastAPI على `:8000`).
المنصة (LMS) بتكلمه **server-to-server فقط** عبر HTTP + `X-API-Key`.
ممنوع المتصفح يكلم SANAD مباشرة، وممنوع مشاركة قواعد البيانات.

## 1. تشغيل SANAD (Docker — للإنتاج)

```bash
cd sanad-rag
cp .env.prod.example .env
# عدّل .env: SANAD_API_KEYS + GEMINI_API_KEY (أو ANTHROPIC_API_KEY)

docker compose -f docker-compose.prod.yml up -d --build
docker compose -f docker-compose.prod.yml ps

# تحقق:
curl http://localhost:8000/health
curl http://localhost:8000/docs   # Swagger UI
```

الخدمات: `nginx (:80/:443)` → `api (:8000)` → `qdrant (:6333)` + `redis (:6379)`.
ملفات الإعداد: `Dockerfile` + `docker-compose.prod.yml` + `nginx/` + `ssl/`.

## 2. المتغيرات الكاملة

| المتغير | أين | مثال |
|---|---|---|
| `SANAD_BASE_URL` | LMS فقط | `http://localhost:8000` |
| `SANAD_API_KEY` | LMS فقط | نفس قيمة `SANAD_API_KEYS` |
| `SANAD_API_KEYS` | SANAD فقط | `demo-key-123` |
| `GEMINI_API_KEY` | SANAD فقط | `AQ.…` أو `AIzaSy…` |
| `ANTHROPIC_API_KEY` | SANAD فقط (بديل) | `sk-ant-…` |
| `EMBEDDING_MODEL` | SANAD | `hash-dev` للديمو / `multilingual-e5` للإنتاج |
| `QDRANT_URL` | SANAD | `http://qdrant:6333` في Docker |
| `SANAD_TENANT_ID` | LMS | `demo_tenant` |
| `REDIS_URL` | LMS worker | `redis://localhost:6379` |

التفاصيل الكاملة: `.env.prod.example`.

## 3. أمثلة الربط (انسخ والصق)

| المنصة | الملف |
|---|---|
| Python (أي Framework) | `sanad_client.py` — `SanadClient` async مع retry |
| Django + Celery | `integrations/django/sanad_integration.py` — models + views + tasks |
| FastAPI (LMS تجريبي) | `integrations/fastapi/lms_example.py` — `uvicorn lms_example:app --port 3000` |
| Express + BullMQ | `integrations/express/sanadClient.js` + `server.example.js` — `npm i express axios uuid bullmq ioredis dotenv` |

كل مثال يطبق نفس الـ flow:

```
Student submit → LMS saves SUBMITTED → 202 QUEUED
  → queue/worker → POST /evaluate/submission (X-API-Key, timeout 60s, retry 3)
  → validate rubric_version → save AI evaluation
  → teacher approve/override → official grade (LMS فقط)
```

## 4. أهم 3 endpoints

```http
POST /ingest
X-API-Key: demo-key-123
{"source_id": "history_ch5_v1", "text": "...", "metadata": {}, "chunker_type": "hierarchical"}

POST /evaluate/submission
X-API-Key: demo-key-123
{
  "request_id": "evreq_…", "submission_id": "sub_…", "tenant_id": "demo_tenant",
  "question_id": "q_…",
  "question": {"text": "…", "type": "ESSAY"},
  "student_answer": {"text": "…"},
  "grading": {
    "reference_answer": "…", "max_score": 10, "rubric_version": "v1",
    "criteria": [{"id": "c1", "description": "…", "max_score": 4}, …]
  },
  "knowledge": {"source_ids": ["history_ch5_v1"], "version": "v1"},
  "config": {"chunker_type": "hierarchical", "enable_reranking": true, "enable_query_expansion": true, "top_k": 5}
}

GET /health   (بدون auth)
```

قاعدة: مجموع `criteria[].max_score` يجب أن يساوي `max_score`، وإلا `422`.

## 5. قواعد الإنتاج

- رسميًا: الدرجة النهائية في LMS فقط بعد مراجعة المعلم — لا تعتمد `suggested_score` مباشرة.
- `request_id` مختلف لكل تقييم (idempotency)؛ `source_id` يتضمن النسخة (`_v1`, `_v2`).
- أنواع الأسئلة المدعومة الآن: `ESSAY` و `SHORT_ANSWER` فقط.
- الأخطاء: `401` إعدادات auth، `429/5xx/timeout` قابلة لإعادة المحاولة، `422` خطأ عميل لا يُعاد.
