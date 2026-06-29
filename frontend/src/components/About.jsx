import { useState, useEffect } from 'react'

const CLASS_COLORS = {
  Normal: '#10b981', Cyst: '#f59e0b', Tumor: '#ef4444', Stone: '#f97316',
}

const card = (extra = {}) => ({ background: '#0f172a', border: '1px solid rgba(51,65,85,.45)', borderRadius: 14, padding: '1.5rem', marginBottom: 16, ...extra })
const sec = { fontSize: '.72rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '.08em', color: '#64748b', marginBottom: 12 }

function LiveBadge({ ok }) {
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5, fontSize: '.72rem', fontWeight: 700,
      background: ok ? 'rgba(16,185,129,.12)' : 'rgba(239,68,68,.12)',
      border: `1px solid ${ok ? 'rgba(16,185,129,.3)' : 'rgba(239,68,68,.3)'}`,
      color: ok ? '#34d399' : '#fca5a5', borderRadius: 999, padding: '2px 10px' }}>
      <span style={{ width: 6, height: 6, borderRadius: '50%', background: ok ? '#10b981' : '#ef4444',
        animation: ok ? 'pulse 2s infinite' : 'none' }} />
      {ok ? 'Online' : 'Offline'}
    </span>
  )
}

export default function About() {
  const [health, setHealth] = useState(null)
  const [model, setModel] = useState(null)
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(true)
  const [lastRefresh, setLastRefresh] = useState(null)

  const load = async () => {
    setLoading(true)
    try {
      const [hRes, mRes, sRes] = await Promise.allSettled([
        fetch('/api/v1/health').then(r => r.json()),
        fetch('/api/v1/health/model-info').then(r => r.json()),
        fetch('/api/v1/records/history?limit=100&offset=0').then(r => r.json()),
      ])
      if (hRes.status === 'fulfilled') setHealth(hRes.value)
      if (mRes.status === 'fulfilled') setModel(mRes.value)
      if (sRes.status === 'fulfilled') {
        const items = sRes.value.items || []
        const counts = { Normal: 0, Cyst: 0, Tumor: 0, Stone: 0 }
        let totalConf = 0, uncertain = 0, totalMs = 0
        items.forEach(i => {
          counts[i.predicted_class] = (counts[i.predicted_class] || 0) + 1
          totalConf += i.confidence
          if (i.is_uncertain) uncertain++
          totalMs += i.inference_time_ms
        })
        setStats({ total: items.length, counts, avgConf: items.length ? totalConf/items.length : 0, uncertain, avgMs: items.length ? totalMs/items.length : 0 })
      }
      setLastRefresh(new Date())
    } finally { setLoading(false) }
  }

  useEffect(() => { load() }, [])

  const apiOk = health?.status === 'healthy'
  const modelOk = model?.is_loaded === true

  return (
    <div>
      <div style={{ marginBottom: '1.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', flexWrap: 'wrap', gap: 10 }}>
        <div>
          <h1 style={{ fontSize: '1.5rem', fontWeight: 700, marginBottom: 6 }}>About TumorVision</h1>
          <p style={{ color: '#64748b', fontSize: '.875rem' }}>
            Production-grade kidney CT scan classifier — EfficientNetB4 · 99.46% accuracy · Grad-CAM explainability.
          </p>
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          {lastRefresh && <span style={{ fontSize: '.72rem', color: '#475569' }}>Refreshed {lastRefresh.toLocaleTimeString()}</span>}
          <button onClick={load} disabled={loading}
            style={{ background: 'transparent', border: '1px solid rgba(51,65,85,.5)', borderRadius: 8, color: '#94a3b8', fontSize: '.8rem', padding: '7px 14px', cursor: 'pointer', fontFamily: 'inherit' }}>
            {loading ? '…' : '↻ Refresh'}
          </button>
        </div>
      </div>

      {/* Live status row */}
      <div style={card()}>
        <div style={sec}>Live System Status</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(180px,1fr))', gap: 12, marginBottom: 16 }}>
          {[
            { label: 'API', value: health?.status || '…', extra: <LiveBadge ok={apiOk} />, color: apiOk ? '#34d399' : '#ef4444' },
            { label: 'Model', value: modelOk ? 'Loaded' : (model ? 'Not loaded' : '…'), extra: <LiveBadge ok={modelOk} />, color: modelOk ? '#34d399' : '#ef4444' },
            { label: 'App Name', value: health?.app_name || '…', color: '#a5b4fc' },
            { label: 'Version', value: health?.app_version || '…', color: '#94a3b8' },
          ].map(k => (
            <div key={k.label} style={{ background: '#1e293b', border: '1px solid rgba(51,65,85,.4)', borderRadius: 10, padding: '12px 14px' }}>
              <div style={{ fontSize: '.68rem', color: '#64748b', textTransform: 'uppercase', letterSpacing: '.05em', marginBottom: 5 }}>{k.label}</div>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span style={{ fontWeight: 700, color: k.color, fontSize: '.92rem' }}>{k.value}</span>
                {k.extra}
              </div>
            </div>
          ))}
        </div>

        {model && (
          <div style={{ background: '#1e293b', borderRadius: 10, padding: '12px 14px', fontSize: '.8rem' }}>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px 24px' }}>
              <span style={{ color: '#64748b' }}>Repo: <span style={{ color: '#a5b4fc' }}>{model.model_repo}</span></span>
              <span style={{ color: '#64748b' }}>File: <span style={{ color: '#94a3b8' }}>{model.model_filename}</span></span>
              <span style={{ color: '#64748b' }}>Input: <span style={{ color: '#94a3b8' }}>{model.image_size}×{model.image_size}px</span></span>
              <span style={{ color: '#64748b' }}>Uncertainty threshold: <span style={{ color: '#94a3b8' }}>{model.uncertainty_threshold}</span></span>
              <span style={{ color: '#64748b' }}>Classes: {model.class_names?.map(c => (
                <span key={c} style={{ color: CLASS_COLORS[c], marginRight: 6, fontWeight: 600 }}>{c}</span>
              ))}</span>
            </div>
          </div>
        )}
      </div>

      {/* Session stats */}
      {stats && (
        <div style={card()}>
          <div style={sec}>Session Statistics</div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(150px,1fr))', gap: 12, marginBottom: 16 }}>
            {[
              { label: 'Total Predictions', value: stats.total, color: '#a5b4fc' },
              { label: 'Avg Confidence', value: `${Math.round(stats.avgConf*100)}%`, color: '#34d399' },
              { label: 'Avg Inference', value: `${Math.round(stats.avgMs)}ms`, color: '#22d3ee' },
              { label: 'Uncertain Flags', value: stats.uncertain, color: stats.uncertain > 0 ? '#ef4444' : '#34d399' },
            ].map(k => (
              <div key={k.label} style={{ background: '#1e293b', border: '1px solid rgba(51,65,85,.3)', borderRadius: 10, padding: '12px 14px' }}>
                <div style={{ fontSize: '.68rem', color: '#64748b', textTransform: 'uppercase', letterSpacing: '.05em', marginBottom: 5 }}>{k.label}</div>
                <div style={{ fontWeight: 700, color: k.color, fontSize: '1.6rem', lineHeight: 1 }}>{k.value}</div>
              </div>
            ))}
          </div>

          <div style={{ fontSize: '.78rem', fontWeight: 600, color: '#64748b', marginBottom: 8 }}>Predictions by class</div>
          {Object.entries(stats.counts).map(([cls, cnt]) => {
            const pct = stats.total > 0 ? Math.round((cnt/stats.total)*100) : 0
            return (
              <div key={cls} style={{ marginBottom: 8 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '.78rem', marginBottom: 3 }}>
                  <span style={{ color: CLASS_COLORS[cls] }}>{cls}</span>
                  <span style={{ color: '#475569' }}>{cnt} &nbsp;({pct}%)</span>
                </div>
                <div style={{ height: 6, background: '#1e293b', borderRadius: 999 }}>
                  <div style={{ height: '100%', width: `${pct}%`, background: CLASS_COLORS[cls], borderRadius: 999, transition: 'width 1s ease', opacity: .85 }} />
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* Model metrics */}
      <div style={card()}>
        <div style={sec}>Model Evaluation Metrics</div>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>{['Metric','Value','Notes'].map(h => (
                <th key={h} style={{ fontSize: '.68rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '.06em', color: '#64748b', padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid rgba(51,65,85,.4)' }}>{h}</th>
              ))}</tr>
            </thead>
            <tbody>
              {[
                ['Validation Accuracy', '99.46%', 'EfficientNetB4, 2-phase fine-tuning', '#a5b4fc'],
                ['Test AUC-ROC',        '98.38%', 'Macro-average across 4 classes', '#c4b5fd'],
                ['Test Sensitivity',    '85.36%', 'True positive rate (tumor recall)', '#34d399'],
                ['Test Specificity',    '95.00%', 'True negative rate', '#34d399'],
                ['Test F1 Score',       '85.25%', 'Harmonic mean precision/recall', '#fbbf24'],
                ['Training Dataset',    '12,446 scans', 'Normal · Cyst · Tumor · Stone', '#94a3b8'],
                ['Image Input Size',    '380 × 380 px', 'EfficientNetB4 native resolution', '#94a3b8'],
                ['Inference Time',      '~80ms', 'Per scan on CPU, includes MC Dropout', '#22d3ee'],
              ].map(([m,v,n,c]) => (
                <tr key={m} style={{ borderBottom: '1px solid rgba(51,65,85,.2)' }}>
                  <td style={{ padding: '10px 12px', fontSize: '.82rem' }}>{m}</td>
                  <td style={{ padding: '10px 12px', fontSize: '.82rem', fontWeight: 700, color: c }}>{v}</td>
                  <td style={{ padding: '10px 12px', fontSize: '.78rem', color: '#64748b' }}>{n}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* API Endpoints */}
      <div style={card()}>
        <div style={sec}>API Endpoints</div>
        {[
          { method: 'POST', path: '/api/v1/predict', desc: 'Multipart image upload → class + confidence + Grad-CAM + uncertainty', params: '?generate_report=true for async PDF' },
          { method: 'GET',  path: '/api/v1/predict/report/{task_id}', desc: 'Poll report status or download PDF when ready' },
          { method: 'GET',  path: '/api/v1/health', desc: 'API health, app name, version, environment' },
          { method: 'GET',  path: '/api/v1/health/model-info', desc: 'Model repo, classes, image size, uncertainty threshold, load status' },
          { method: 'GET',  path: '/api/v1/records/history', desc: 'Paginated prediction audit trail with feedback status', params: '?limit=20&offset=0' },
          { method: 'POST', path: '/api/v1/records/feedback', desc: 'Submit radiologist feedback: correct class, name, comment' },
        ].map(e => (
          <div key={e.path} style={{ display: 'flex', gap: 10, marginBottom: 10, padding: '10px 12px', background: '#1e293b', borderRadius: 8, alignItems: 'flex-start' }}>
            <span style={{ fontSize: '.7rem', fontWeight: 700, padding: '2px 8px', borderRadius: 4, flexShrink: 0, marginTop: 1,
              background: e.method === 'GET' ? 'rgba(16,185,129,.2)' : 'rgba(99,102,241,.2)',
              color: e.method === 'GET' ? '#34d399' : '#a5b4fc' }}>{e.method}</span>
            <div style={{ flex: 1, minWidth: 0 }}>
              <code style={{ fontSize: '.78rem', color: '#e2e8f0', display: 'block' }}>{e.path}</code>
              <div style={{ fontSize: '.75rem', color: '#64748b', marginTop: 2 }}>{e.desc}</div>
              {e.params && <div style={{ fontSize: '.72rem', color: '#475569', marginTop: 1 }}>{e.params}</div>}
            </div>
          </div>
        ))}
        <div style={{ marginTop: 12, fontSize: '.8rem', color: '#475569' }}>
          Full interactive docs: <a href="http://localhost:8110/docs" target="_blank" rel="noreferrer" style={{ color: '#6366f1' }}>localhost:8110/docs</a>
        </div>
      </div>

      {/* Tech stack */}
      <div style={card()}>
        <div style={sec}>Tech Stack</div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
          {[
            ['TensorFlow 2.21', '#FF6F00'], ['EfficientNetB4', '#6366f1'], ['Grad-CAM', '#8b5cf6'],
            ['MC Dropout', '#06b6d4'], ['FastAPI 0.136', '#009688'], ['Celery + Redis', '#dc2626'],
            ['React 18 + Vite', '#61dafb'], ['nginx', '#009639'], ['DVC Pipeline', '#10b981'],
            ['Docker Compose', '#2496ED'], ['ReportLab PDF', '#f59e0b'], ['SQLite', '#94a3b8'],
            ['GitHub Actions CI', '#a78bfa'],
          ].map(([t, c]) => (
            <span key={t} style={{ background: '#1e293b', border: '1px solid rgba(51,65,85,.4)', borderRadius: 8, padding: '6px 12px', fontSize: '.8rem', color: c || '#94a3b8' }}>{t}</span>
          ))}
        </div>
      </div>

      {/* Author */}
      <div style={{ ...card(), marginBottom: 0, display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12 }}>
        <div>
          <div style={{ fontWeight: 700 }}>Rayen Lassoued</div>
          <div style={{ fontSize: '.8rem', color: '#64748b', marginTop: 3 }}>Junior AI/ML Engineer · Bonn, Germany</div>
        </div>
        <div style={{ display: 'flex', gap: '1.5rem' }}>
          <a href="https://github.com/rayenx2/TumorVision" target="_blank" rel="noreferrer" style={{ color: '#64748b', fontSize: '.82rem', textDecoration: 'none' }}>github.com/rayenx2/TumorVision</a>
          <a href="https://linkedin.com/in/Rayen-Lassoued" target="_blank" rel="noreferrer" style={{ color: '#64748b', fontSize: '.82rem', textDecoration: 'none' }}>linkedin.com/in/Rayen-Lassoued</a>
        </div>
      </div>

      <style>{`@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.5;transform:scale(1.4)}}`}</style>
    </div>
  )
}
