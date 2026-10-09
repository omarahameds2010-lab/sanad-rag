import { useState } from 'react';
import { api, formatApiError, type SanadSettings } from '../api';
import type { ChunkerType, QueryResponse } from '../types';

interface Props {
  settings: SanadSettings;
}

export default function QueryPage({ settings }: Props) {
  const [question, setQuestion] = useState('ما هي عاصمة مصر؟');
  const [topK, setTopK] = useState(5);
  const [chunker, setChunker] = useState<ChunkerType>('hierarchical');
  const [rerank, setRerank] = useState(false);
  const [expand, setExpand] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState<QueryResponse | null>(null);

  async function onAsk() {
    setLoading(true);
    setError('');
    setResult(null);
    try {
      const res = await api.query(settings, {
        question,
        top_k: topK,
        chunker_type: chunker,
        enable_reranking: rerank,
        enable_query_expansion: expand,
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
      <h2>اسأل (RAG)</h2>
      <p className="desc">إجابة grounded من المقاطع المسترجعة — للاستكشاف ومساعد المعلم، وليست عقد التصحيح.</p>
      <div className="grid">
        <div>
          <label>السؤال</label>
          <textarea className="small" value={question} onChange={(e) => setQuestion(e.target.value)} />
        </div>
        <div className="row">
          <div>
            <label>top_k</label>
            <input type="number" min={1} max={20} value={topK} onChange={(e) => setTopK(Number(e.target.value))} style={{ width: 90 }} />
          </div>
          <div>
            <label>التقطيع</label>
            <select value={chunker} onChange={(e) => setChunker(e.target.value as ChunkerType)}>
              <option value="hierarchical">hierarchical</option>
              <option value="arabic">arabic</option>
              <option value="baseline">baseline</option>
            </select>
          </div>
          <label className="checkbox"><input type="checkbox" checked={rerank} onChange={(e) => setRerank(e.target.checked)} /> reranking</label>
          <label className="checkbox"><input type="checkbox" checked={expand} onChange={(e) => setExpand(e.target.checked)} /> query expansion</label>
          <button className="primary" disabled={loading || !question.trim()} onClick={onAsk}>
            {loading ? 'جارٍ البحث…' : 'اسأل'}
          </button>
        </div>
      </div>
      {error && <div className="error">{error}</div>}
      {result && (
        <div style={{ marginTop: 12 }}>
          <label>الإجابة</label>
          <div className="evidence">{result.answer}</div>
          <div className="kv"><span>المقاطع المستشهد بها</span><span className="mono">{result.cited_chunk_ids.join(', ') || '—'}</span></div>
          <div className="kv"><span>زمن الاسترجاع / التوليد</span><span className="mono">{result.retrieval_ms}ms / {result.generation_ms}ms</span></div>
          <div className="kv"><span>reranker</span><span className="mono">{result.reranker_used}</span></div>
        </div>
      )}
    </div>
  );
}
