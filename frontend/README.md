# سند — الواجهة الأمامية (Frontend)

واجهة عربية (RTL) لمنصة سند: إدخال مواد، سؤال RAG، تقييم إجابات بالروبرك، ومختبر Benchmark.

## التشغيل المحلي

```bash
cd frontend
cp .env.example .env   # عدّل VITE_API_BASE_URL و VITE_SANAD_API_KEY
npm install
npm run dev            # http://localhost:5173
```

الـ backend يجب أن يعمل على `VITE_API_BASE_URL` (افتراضيًا `http://localhost:8000`):

```bash
cd ..
.venv\Scripts\python.exe -m uvicorn src.api.main:app --reload
```

## البناء

```bash
npm run build    # ينتج dist/
npm run preview  # معاينة على :3000
```

## Docker (إنتاج)

```bash
docker build -t sanad-frontend \
  --build-arg VITE_API_BASE_URL=https://sanad.example.com \
  --build-arg VITE_SANAD_API_KEY=demo-key-123 \
  ./frontend
```

أو عبر `docker-compose.prod.yml` (خدمة `frontend` على `:3000`، والـ nginx الرئيسي يوجّه `/` إليها).

## الصفحات

| تبويب | الـ API المستخدم |
|---|---|
| الرئيسية | `GET /health` + `GET /evaluate/health` |
| إدخال مادة | `POST /ingest` |
| اسأل | `POST /query` (RAG فقط) |
| تقييم إجابة | `POST /evaluate/submission` (روبرك + أدلة + provenance) |
| المختبر | `POST /evaluate/run` + `GET /evaluate/report` |

## ملاحظات أمنية

- مفتاح API في المتصفح **للتجربة المحلية فقط**.
- في الإنتاج: كل calls من backend الـ LMS (server-to-server)، والمتصفح لا يرى المفتاح.
- الدرجة المقترحة استرشادية — الاعتماد للمعلم، والدرجة الرسمية في الـ LMS.
