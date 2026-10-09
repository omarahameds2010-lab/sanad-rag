import { useCallback, useEffect, useState } from 'react';
import { api, loadSettings, saveSettings, type SanadSettings } from './api';
import type { HealthResponse } from './types';
import SettingsBar from './components/SettingsBar';
import HomePage from './pages/HomePage';
import IngestPage from './pages/IngestPage';
import QueryPage from './pages/QueryPage';
import EvaluatePage from './pages/EvaluatePage';
import LabPage from './pages/LabPage';

type Tab = 'home' | 'ingest' | 'query' | 'evaluate' | 'lab';

const TABS: { id: Tab; label: string }[] = [
  { id: 'home', label: 'الرئيسية' },
  { id: 'ingest', label: 'إدخال مادة' },
  { id: 'query', label: 'اسأل' },
  { id: 'evaluate', label: 'تقييم إجابة' },
  { id: 'lab', label: 'المختبر' },
];

export default function App() {
  const [tab, setTab] = useState<Tab>('home');
  const [settings, setSettings] = useState<SanadSettings>(() => loadSettings());
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [evalHealth, setEvalHealth] = useState<{ status: string; llm_ok?: boolean } | null>(null);

  const refreshHealth = useCallback(async (s: SanadSettings) => {
    try {
      const h = await api.health(s);
      setHealth(h);
    } catch {
      setHealth(null);
    }
    try {
      const eh = await api.evalHealth(s);
      setEvalHealth(eh);
    } catch {
      setEvalHealth(null);
    }
  }, []);

  useEffect(() => {
    refreshHealth(settings);
    const t = setInterval(() => refreshHealth(settings), 30000);
    return () => clearInterval(t);
  }, [refreshHealth, settings.baseUrl, settings.apiKey]);

  function onSave(s: SanadSettings) {
    setSettings(s);
    saveSettings(s);
    refreshHealth(s);
  }

  const healthText = health ? (health.status === 'ok' ? '● متصل' : '● متدهور') : '○ غير متصل';
  const healthClass = health ? (health.status === 'ok' ? 'ok' : 'warn') : 'bad';

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <div className="logo">س</div>
          <div>
            <h1>سند</h1>
            <small>منصة التقييم العربي</small>
          </div>
        </div>
        <nav className="nav">
          {TABS.map((t) => (
            <button key={t.id} className={tab === t.id ? 'active' : ''} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          ))}
        </nav>
        <div className="health">
          <span className={`dot ${healthClass}`} />
          <span>{healthText}</span>
        </div>
      </header>

      <main className="main">
        <SettingsBar settings={settings} onSave={onSave} healthText={`${settings.baseUrl}`} />
        {tab === 'home' && <HomePage health={health} evalHealth={evalHealth} />}
        {tab === 'ingest' && <IngestPage settings={settings} />}
        {tab === 'query' && <QueryPage settings={settings} />}
        {tab === 'evaluate' && <EvaluatePage settings={settings} />}
        {tab === 'lab' && <LabPage settings={settings} />}
      </main>

      <footer className="footer">
        سند — التقييم اقتراح استرشادي، والدرجة الرسمية في الـ LMS بعد مراجعة المعلم ·
        <a href={`${settings.baseUrl.replace(/\/$/, '')}/docs`} target="_blank" rel="noreferrer"> API docs</a>
      </footer>
    </div>
  );
}
