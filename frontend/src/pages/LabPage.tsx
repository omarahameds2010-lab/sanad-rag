import { useState } from 'react';
import { api, formatApiError, type SanadSettings } from '../api';
import type { ChunkerType, EvaluateReport } from '../types';

interface Props {
  settings: SanadSettings;
}

export default function LabPage({ settings }: Props) {
  const [datasetPath, setDatasetPath] = useState('data/golden_dataset/sample_golden.json');
  const [configName, setConfigName] = useState('baseline');
  const [chunker, setChunker] = useState<ChunkerType>('arabic');
  const [rerank, setRerank] = useState(false);
  const [expand, setExpand] = useState(false);
  const [runId, setRunId] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [report, setReport] = useState<EvaluateReport | null>(null);

  async function onRun() {
    setLoading(true);
    setError('');
    setReport(null);
    try {
      const run = await api.evaluateRun(settings, {
        golden_dataset_path: datasetPath,
        config_name: configName,
        chunker_type: chunker,
        enable_reranking: rerank,
        enable_query_expansion: expand,
      });
      setRunId(run.run_id);
      const rep = await api.evaluateReport(settings, run.run_id);
      setReport(rep);
    } catch (e) {
      setError(formatApiError(e));
    } finally {
      setLoading(false);
    }
  }

  async function onFetchReport() {
    if (!runId.trim()) return;
    setLoading(true);
    setError('');
    try {
      const rep = await api.evaluateReport(settings, runId.trim());
      setReport(rep);
    } catch (e) {
      setError(formatApiError(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <div className="card">
        <h2>مختبر التقييم (Benchmark)</h2>
        <p className="desc">
          تشغيل مقاييس الاسترجاع على الـ golden dataset — للمقارنة بين الإعدادات،
          وليس لتصحيح الطلاب.
        </p>
        <div className="grid">
          <div>
            <label>مسار الـ golden dataset (على خادم SANAD)</label>
            <input type="text" dir="ltr" value={datasetPath} onChange={(e) => setDatasetPath(e.target.value)} />
          </div>
          <div className="row">
            <div>
              <label>config_name</label>
              <input type="text" dir="ltr" value={configName} onChange={(e) => setConfigName(e.target.value)} style={{ width: 160 }} />
            </div>
            <div>
              <label>التقطيع</label>
              <select value={chunker} onChange={(e) => setChunker(e.target.value as ChunkerType)}>
                <option value="arabic">arabic</option>
                <option value="baseline">baseline</option>
                <option value="hierarchical">hierarchical</option>
              </select>
            </div>
            <label className="checkbox"><input type="checkbox" checked={rerank} onChange={(e) => setRerank(e.target.checked)} /> reranking</label>
            <label className="checkbox"><input type="checkbox" checked={expand} onChange={(e) => setExpand(e.target.checked)} /> expansion</label>
            <button className="primary" disabled={loading} onClick={onRun}>
              {loading ? 'جارٍ التشغيل…' : 'تشغيل التقييم'}
            </button>
          </div>
        </div>
        {error && <div className="error">{error}</div>}
        {report && (
          <div style={{ marginTop: 12 }}>
            <div className="kv"><span>run_id</span><span className="mono">{report.run_id}</span></div>
            <div className="kv"><span>config</span><span className="mono">{report.config_name}</span></div>
            <div className="kv"><span>عدد الأمثلة</span><span>{report.n_examples}</span></div>
            <div className="kv"><span>Precision@k</span><span className="mono">{report.precision_at_k ?? '—'}</span></div>
            <div className="kv"><span>Recall@k</span><span className="mono">{report.recall_at_k ?? '—'}</span></div>
            <div className="kv"><span>p50 / p95 (ث)</span><span className="mono">{report.p50_latency_seconds ?? '—'} / {report.p95_latency_seconds ?? '—'}</span></div>
          </div>
        )}
      </div>

      <div className="card">
        <h2>استعلام عن run سابق</h2>
        <div className="row">
          <input type="text" dir="ltr" value={runId} onChange={(e) => setRunId(e.target.value)} placeholder="run_id" style={{ flex: 1 }} />
          <button className="ghost" disabled={loading || !runId.trim()} onClick={onFetchReport}>جلب التقرير</button>
        </div>
      </div>
    </div>
  );
}
