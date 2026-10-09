import { useState } from 'react';
import { api, formatApiError, type SanadSettings } from '../api';
import type { ChunkerType, IngestResponse } from '../types';

interface Props {
  settings: SanadSettings;
}

export default function IngestPage({ settings }: Props) {
  const [sourceId, setSourceId] = useState('history_ch5_v1');
  const [text, setText] = useState('ثورة 1919 قامت بسبب الاستعمار البريطاني وفرض الحماية على مصر...');
  const [chunker, setChunker] = useState<ChunkerType>('hierarchical');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState<IngestResponse | null>(null);

  async function onSubmit() {
    setLoading(true);
    setError('');
    setResult(null);
    try {
      const res = await api.ingest(settings, {
        source_id: sourceId.trim(),
        text,
        metadata: { via: 'frontend' },
        chunker_type: chunker,
      });
      setResult(res);
    } catch (e) {
      setError(formatApiError(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="card">
      <h2>إدخال مادة مرجعية</h2>
      <p className="desc">
        تُحفظ المادة مرة واحدة لكل فصل/مصدر، ثم تُستخدم في الاسترجاع والتقييم عبر <span className="mono">source_id</span>.
        أضف رقم النسخة في المعرف (مثال: <span className="mono">history_ch5_v1</span>).
      </p>
      <div className="grid">
        <div>
          <label>معرف المصدر (source_id)</label>
          <input type="text" dir="ltr" value={sourceId} onChange={(e) => setSourceId(e.target.value)} />
        </div>
        <div>
          <label>نص المادة المرجعية</label>
          <textarea value={text} onChange={(e) => setText(e.target.value)} />
        </div>
        <div className="row">
          <div>
            <label>استراتيجية التقطيع</label>
            <select value={chunker} onChange={(e) => setChunker(e.target.value as ChunkerType)}>
              <option value="hierarchical">hierarchical (أب/ابن — الأفضل للتقييم)</option>
              <option value="arabic">arabic (واعٍ بالجمل)</option>
              <option value="baseline">baseline (للمقارنة)</option>
            </select>
          </div>
          <button className="primary" disabled={loading || !sourceId.trim() || !text.trim()} onClick={onSubmit}>
            {loading ? 'جارٍ الإدخال…' : 'إدخال المادة'}
          </button>
        </div>
      </div>
      {error && <div className="error">{error}</div>}
      {result && (
        <div style={{ marginTop: 12 }}>
          <div className="kv"><span>source_id</span><span className="mono">{result.source_id}</span></div>
          <div className="kv"><span>عدد المقاطع</span><span>{result.n_chunks}</span></div>
          {result.n_parents != null && <div className="kv"><span>آباء / أبناء</span><span>{result.n_parents} / {result.n_children}</span></div>}
          <div className="kv"><span>نموذج التضمين</span><span className="mono">{result.embedding_model}</span></div>
          <div className="kv"><span>التقطيع</span><span className="mono">{result.chunker_type}</span></div>
          <div className="notice">استخدم نفس <span className="mono">source_id</span> في تبويب «تقييم إجابة» داخل حقل المصادر.</div>
        </div>
      )}
    </div>
  );
}
