import { useState } from 'react';
import { api, formatApiError, type SanadSettings } from '../api';
import type { ChunkerType, EvaluationResponse, RubricCriterion } from '../types';

interface Props {
  settings: SanadSettings;
}

const DEFAULT_RUBRIC: RubricCriterion[] = [
  { id: 'c1', description: 'الأسباب السياسية', max_score: 4 },
  { id: 'c2', description: 'الأسباب الاقتصادية', max_score: 3 },
  { id: 'c3', description: 'الأسباب الاجتماعية', max_score: 3 },
];

export default function EvaluatePage({ settings }: Props) {
  const [questionText, setQuestionText] = useState('اشرح أسباب ثورة 1919.');
  const [studentAnswer, setStudentAnswer] = useState('ثورة 1919 قامت بسبب الاستعمار البريطاني وفرض الحماية على مصر...');
  const [referenceAnswer, setReferenceAnswer] = useState('ثورة 1919 قامت بسبب الاستعمار البريطاني والظروف الاقتصادية والاجتماعية...');
  const [maxScore, setMaxScore] = useState(10);
  const [rubric, setRubric] = useState<RubricCriterion[]>(DEFAULT_RUBRIC);
  const [sourceIds, setSourceIds] = useState('history_ch5_v1');
  const [knowledgeVersion, setKnowledgeVersion] = useState('v1');
  const [rubricVersion, setRubricVersion] = useState('v1');
  const [chunker, setChunker] = useState<ChunkerType>('hierarchical');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState<EvaluationResponse | null>(null);

  const criteriaSum = rubric.reduce((t, c) => t + (Number(c.max_score) || 0), 0);
  const rubricValid = criteriaSum === maxScore && rubric.length > 0;

  function updateCriterion(i: number, patch: Partial<RubricCriterion>) {
    setRubric(rubric.map((c, idx) => (idx === i ? { ...c, ...patch } : c)));
  }

  function addCriterion() {
    if (rubric.length >= 10) return;
    setRubric([...rubric, { id: `c${rubric.length + 1}`, description: '', max_score: 1 }]);
  }

  function removeCriterion(i: number) {
    setRubric(rubric.filter((_, idx) => idx !== i));
  }

  async function onEvaluate() {
    setLoading(true);
    setError('');
    setResult(null);
    try {
      const res = await api.evaluateSubmission(settings, {
        submission_id: crypto.randomUUID(),
        question_id: crypto.randomUUID().slice(0, 8),
        question_text: questionText,
        question_type: 'ESSAY',
        student_answer: studentAnswer,
        reference_answer: referenceAnswer,
        rubric,
        max_score: maxScore,
        rubric_version: rubricVersion,
        source_ids: sourceIds.split(',').map((s) => s.trim()).filter(Boolean),
        knowledge_version: knowledgeVersion,
        chunker_type: chunker,
        enable_reranking: true,
        enable_query_expansion: true,
        top_k: 5,
      });
      setResult(res);
    } catch (e) {
      setError(formatApiError(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <div className="card">
        <h2>تقييم إجابة طالب</h2>
        <p className="desc">
          تقييم واعٍ بالروبرك: كل معيار يُصحح مستقلًا مع دليل مقتبس من إجابة الطالب،
          والدرجة النهائية = مجموع درجات المعايير.
        </p>
        <div className="grid">
          <div>
            <label>السؤال</label>
            <textarea className="small" value={questionText} onChange={(e) => setQuestionText(e.target.value)} />
          </div>
          <div className="grid two">
            <div>
              <label>إجابة الطالب</label>
              <textarea value={studentAnswer} onChange={(e) => setStudentAnswer(e.target.value)} />
            </div>
            <div>
              <label>الإجابة النموذجية</label>
              <textarea value={referenceAnswer} onChange={(e) => setReferenceAnswer(e.target.value)} />
            </div>
          </div>
          <div className="grid two">
            <div>
              <label>الدرجة العظمى (مجموع المعايير يجب أن يساويها: حاليًا {criteriaSum})</label>
              <input type="number" min={1} max={100} value={maxScore} onChange={(e) => setMaxScore(Number(e.target.value))} />
              {!rubricValid && <div className="error">مجموع المعايير ({criteriaSum}) يجب أن يساوي الدرجة العظمى ({maxScore})</div>}
            </div>
            <div>
              <label>معرفات المصادر (مفصولة بفاصلة)</label>
              <input type="text" dir="ltr" value={sourceIds} onChange={(e) => setSourceIds(e.target.value)} />
            </div>
          </div>
          <div className="grid two">
            <div>
              <label>إصدار الروبرك</label>
              <input type="text" dir="ltr" value={rubricVersion} onChange={(e) => setRubricVersion(e.target.value)} />
            </div>
            <div>
              <label>إصدار المعرفة + التقطيع</label>
              <div className="row">
                <input type="text" dir="ltr" value={knowledgeVersion} onChange={(e) => setKnowledgeVersion(e.target.value)} style={{ width: 110 }} />
                <select value={chunker} onChange={(e) => setChunker(e.target.value as ChunkerType)}>
                  <option value="hierarchical">hierarchical</option>
                  <option value="arabic">arabic</option>
                  <option value="baseline">baseline</option>
                </select>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="card">
        <div className="row" style={{ justifyContent: 'space-between' }}>
          <h2 style={{ margin: 0 }}>الروبرك ({rubric.length} معايير)</h2>
          <button className="ghost" onClick={addCriterion} disabled={rubric.length >= 10}>+ معيار</button>
        </div>
        {rubric.map((c, i) => (
          <div className="criterion" key={i}>
            <div className="row">
              <div style={{ flex: 1 }}>
                <label>الوصف</label>
                <input type="text" value={c.description} onChange={(e) => updateCriterion(i, { description: e.target.value })} placeholder="ماذا يقيس هذا المعيار؟" />
              </div>
              <div style={{ width: 90 }}>
                <label>الدرجة</label>
                <input type="number" min={1} max={20} value={c.max_score} onChange={(e) => updateCriterion(i, { max_score: Number(e.target.value) })} />
              </div>
              <div style={{ width: 90 }}>
                <label>ID</label>
                <input type="text" dir="ltr" value={c.id} onChange={(e) => updateCriterion(i, { id: e.target.value })} />
              </div>
              <button className="ghost" onClick={() => removeCriterion(i)} disabled={rubric.length <= 1}>حذف</button>
            </div>
          </div>
        ))}
        <div className="row" style={{ marginTop: 12 }}>
          <button className="primary" disabled={loading || !rubricValid || !questionText.trim() || !studentAnswer.trim()} onClick={onEvaluate}>
            {loading ? 'جارٍ التقييم… (قد يستغرق ~60 ثانية)' : 'قيّم الإجابة'}
          </button>
        </div>
        {error && <div className="error">{error}</div>}
      </div>

      {result && (
        <div className="card">
          <h2>نتيجة التقييم</h2>
          <div className="score-hero">
            <div className="score-circle">{result.suggested_score}<small>/ {result.max_score}</small></div>
            <div>
              <div style={{ fontSize: 14 }}>{result.feedback_ar}</div>
              <div style={{ fontSize: 12, color: '#6b7280', marginTop: 4 }}>
                review_signal: <span className="mono">{result.review_signal}</span>
                {' · '}تحتاج مراجعة معلم: {result.needs_teacher_review ? 'نعم' : 'لا'}
              </div>
            </div>
          </div>
          {result.criteria.map((c) => (
            <div className="criterion" key={c.criterion_id}>
              <div className="head">
                <span>{c.criterion_id}</span>
                <span><span className={`badge ${c.evidence_type}`}>{c.evidence_type}</span> {c.earned}/{c.max}</span>
              </div>
              {c.evidence && <div className="evidence">«{c.evidence}»</div>}
              <div className="feedback">{c.feedback_ar}</div>
            </div>
          ))}
          {result.knowledge_gaps.length > 0 && (
            <div className="notice">فجوات معرفية: {result.knowledge_gaps.join('؛ ')}</div>
          )}
          <details style={{ marginTop: 10 }}>
            <summary style={{ cursor: 'pointer', fontSize: 13, color: '#6b7280' }}>Provenance + Usage (للتتبع)</summary>
            <pre className="json">{JSON.stringify({ provenance: result.provenance, usage: result.usage }, null, 2)}</pre>
          </details>
          <div className="notice">الدرجة المقترحة استرشادية — الاعتماد والمراجعة للمعلم، والدرجة الرسمية في الـ LMS.</div>
        </div>
      )}
    </div>
  );
}
