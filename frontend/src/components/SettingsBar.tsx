import { useState } from 'react';
import type { SanadSettings } from '../api';

interface Props {
  settings: SanadSettings;
  onSave: (s: SanadSettings) => void;
  healthText: string;
}

export default function SettingsBar({ settings, onSave, healthText }: Props) {
  const [open, setOpen] = useState(false);
  const [draft, setDraft] = useState(settings);

  return (
    <div className="card">
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <div style={{ fontSize: 13 }}>
          <strong>الاتصال:</strong> <span className="mono">{settings.baseUrl}</span>
          {' · '}
          <span>{healthText}</span>
        </div>
        <button className="ghost" onClick={() => { setDraft(settings); setOpen(!open); }}>
          {open ? 'إغلاق الإعدادات' : 'إعدادات الاتصال'}
        </button>
      </div>
      {open && (
        <div style={{ marginTop: 12 }}>
          <div className="grid two">
            <div>
              <label>عنوان الـ API</label>
              <input
                type="text"
                value={draft.baseUrl}
                dir="ltr"
                onChange={(e) => setDraft({ ...draft, baseUrl: e.target.value })}
                placeholder="http://localhost:8000"
              />
            </div>
            <div>
              <label>Tenant ID</label>
              <input
                type="text"
                value={draft.tenantId}
                dir="ltr"
                onChange={(e) => setDraft({ ...draft, tenantId: e.target.value })}
              />
            </div>
          </div>
          <div style={{ marginTop: 10 }}>
            <label>مفتاح API (X-API-Key) — يُحفظ في المتصفح للتجربة فقط</label>
            <input
              type="text"
              value={draft.apiKey}
              dir="ltr"
              onChange={(e) => setDraft({ ...draft, apiKey: e.target.value })}
              placeholder="demo-key-123"
            />
          </div>
          <div className="row" style={{ marginTop: 10 }}>
            <button className="primary" onClick={() => { onSave(draft); setOpen(false); }}>حفظ</button>
            <span style={{ fontSize: 12, color: '#6b7280' }}>في الإنتاج: المفتاح يبقى server-side فقط ولا يظهر في المتصفح.</span>
          </div>
        </div>
      )}
    </div>
  );
}
