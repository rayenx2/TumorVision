import { useState, useEffect, useCallback } from 'react'

const CLASS_COLORS = {
  Normal: { text: '#34d399', bg: 'rgba(16,185,129,.15)', border: 'rgba(16,185,129,.35)', bar: '#10b981' },
  Cyst:   { text: '#fde68a', bg: 'rgba(245,158,11,.15)', border: 'rgba(245,158,11,.35)', bar: '#f59e0b' },
  Tumor:  { text: '#fca5a5', bg: 'rgba(239,68,68,.15)',  border: 'rgba(239,68,68,.35)',  bar: '#ef4444' },
  Stone:  { text: '#fed7aa', bg: 'rgba(249,115,22,.15)', border: 'rgba(249,115,22,.35)', bar: '#f97316' },
}
const CLASSES = ['All', 'Normal', 'Cyst', 'Tumor', 'Stone']

const card = { background: '#0f172a', border: '1px solid rgba(51,65,85,.45)', borderRadius: 14, padding: '1.5rem' }

function StatCard({ label, value, color, sub }) {
  return (
    <div style={{ ...card, padding: '1rem 1.25rem' }}>
      <div style={{ fontSize: '.7rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '.06em', color: '#64748b', marginBottom: 6 }}>{label}</div>
      <div style={{ fontSize: '1.9rem', fontWeight: 700, color: color || '#f1f5f9', lineHeight: 1 }}>{value}</div>
      {sub && <div style={{ fontSize: '.72rem', color: '#475569', marginTop: 4 }}>{sub}</div>}
    </div>
  )
}

function MiniBar({ label, count, total, color }) {
  const pct = total > 0 ? Math.round((count / total) * 100) : 0
  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '.78rem', marginBottom: 3 }}>
        <span style={{ color: CLASS_COLORS[label]?.text || '#94a3b8' }}>{label}</span>
        <span style={{ color: '#64748b' }}>{count} ({pct}%)</span>
      </div>
      <div style={{ height: 6, background: '#1e293b', borderRadius: 999 }}>
        <div style={{ height: '100%', width: `${pct}%`, background: color, borderRadius: 999, transition: 'width 1s ease' }} />
      </div>
    </div>
  )
}

function FeedbackModal({ item, onClose, onSaved }) {
  const [cls, setCls] = useState(item.predicted_class)
  const [comment, setComment] = useState('')
  const [name, setName] = useState('')
  const [saving, setSaving] = useState(false)
  const [err, setErr] = useState(null)

  const submit = async () => {
    setSaving(true); setErr(null)
    try {
      const res = await fetch('/api/v1/records/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prediction_id: item.prediction_id,
          correct_class: cls,
          comment: comment || null,
          radiologist_name: name || null,
        }),
      })
      if (!res.ok) { const d = await res.json(); throw new Error(d.detail || `HTTP ${res.status}`) }
      onSaved()
      onClose()
    } catch (e) { setErr(e.message) }
    finally { setSaving(false) }
  }

  return (
    <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,.7)', zIndex: 999, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <div style={{ background: '#0f172a', border: '1px solid rgba(99,102,241,.4)', borderRadius: 16, padding: '1.75rem', width: 420, maxWidth: '90vw' }}>
        <div style={{ fontWeight: 700, fontSize: '1rem', marginBottom: 4 }}>Radiologist Feedback</div>
        <div style={{ fontSize: '.78rem', color: '#64748b', marginBottom: 18 }}>
          Prediction ID: <code style={{ color: '#94a3b8' }}>{item.prediction_id?.slice(0,16)}…</code>
          <br/>Model predicted: <span style={{ color: CLASS_COLORS[item.predicted_class]?.text }}>{item.predicted_class}</span>
        </div>

        <label style={{ fontSize: '.78rem', color: '#64748b', display: 'block', marginBottom: 5 }}>Correct Class</label>
        <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
          {['Normal','Cyst','Tumor','Stone'].map(c => (
            <button key={c} onClick={() => setCls(c)} style={{
              flex: 1, padding: '8px 4px', borderRadius: 8, fontSize: '.78rem', fontWeight: 600, cursor: 'pointer', fontFamily: 'inherit',
              background: cls === c ? CLASS_COLORS[c].bg : 'transparent',
              border: `1px solid ${cls === c ? CLASS_COLORS[c].border : 'rgba(51,65,85,.5)'}`,
              color: cls === c ? CLASS_COLORS[c].text : '#64748b',
            }}>{c}</button>
          ))}
        </div>

        <label style={{ fontSize: '.78rem', color: '#64748b', display: 'block', marginBottom: 5 }}>Radiologist Name (optional)</label>
        <input value={name} onChange={e => setName(e.target.value)} placeholder="Dr. Smith"
          style={{ width: '100%', background: '#1e293b', border: '1px solid rgba(51,65,85,.5)', borderRadius: 8, color: '#f1f5f9', fontSize: '.85rem', padding: '8px 12px', marginBottom: 14, fontFamily: 'inherit' }} />

        <label style={{ fontSize: '.78rem', color: '#64748b', display: 'block', marginBottom: 5 }}>Comment (optional)</label>
        <textarea value={comment} onChange={e => setComment(e.target.value)} placeholder="Additional observations…" rows={3}
          style={{ width: '100%', background: '#1e293b', border: '1px solid rgba(51,65,85,.5)', borderRadius: 8, color: '#f1f5f9', fontSize: '.85rem', padding: '8px 12px', marginBottom: 16, fontFamily: 'inherit', resize: 'vertical' }} />

        {err && <div style={{ background: 'rgba(239,68,68,.1)', border: '1px solid rgba(239,68,68,.3)', borderRadius: 8, padding: '8px 12px', fontSize: '.8rem', color: '#fca5a5', marginBottom: 12 }}>{err}</div>}

        <div style={{ display: 'flex', gap: 10 }}>
          <button onClick={submit} disabled={saving} style={{
            flex: 1, background: saving ? 'rgba(99,102,241,.3)' : 'linear-gradient(135deg,#6366f1,#8b5cf6)',
            border: 'none', borderRadius: 8, color: '#fff', fontSize: '.85rem', fontWeight: 700, padding: '10px', cursor: saving ? 'not-allowed' : 'pointer', fontFamily: 'inherit',
          }}>{saving ? 'Saving…' : 'Submit Feedback'}</button>
          <button onClick={onClose} style={{ background: 'transparent', border: '1px solid rgba(51,65,85,.5)', borderRadius: 8, color: '#64748b', fontSize: '.85rem', padding: '10px 16px', cursor: 'pointer', fontFamily: 'inherit' }}>Cancel</button>
        </div>
      </div>
    </div>
  )
}

export default function History() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [filter, setFilter] = useState('All')
  const [feedbackItem, setFeedbackItem] = useState(null)

  const load = useCallback(async () => {
    setLoading(true); setError(null)
    try {
      const res = await fetch('/api/v1/records/history?limit=50&offset=0')
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      setData(await res.json())
    } catch (e) { setError(e.message) }
    finally { setLoading(false) }
  }, [])

  useEffect(() => { load() }, [load])

  const exportCSV = () => {
    if (!data?.items?.length) return
    const rows = [
      ['prediction_id','predicted_class','confidence','uncertainty_score','is_uncertain','inference_time_ms','timestamp','feedback_received','correct_class'],
      ...data.items.map(i => [
        i.prediction_id, i.predicted_class, (i.confidence*100).toFixed(1)+'%',
        (i.uncertainty_score*100).toFixed(1)+'%', i.is_uncertain,
        Math.round(i.inference_time_ms), i.timestamp, i.feedback_received, i.correct_class || '',
      ])
    ]
    const csv = rows.map(r => r.join(',')).join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob)
    a.download = `tumorvision_history_${new Date().toISOString().slice(0,10)}.csv`; a.click()
  }

  const items = data?.items || []
  const filtered = filter === 'All' ? items : items.filter(i => i.predicted_class === filter)

  // Stats
  const total = items.length
  const classCounts = { Normal: 0, Cyst: 0, Tumor: 0, Stone: 0 }
  let totalConf = 0, uncertain = 0, withFeedback = 0
  items.forEach(i => {
    classCounts[i.predicted_class] = (classCounts[i.predicted_class] || 0) + 1
    totalConf += i.confidence
    if (i.is_uncertain) uncertain++
    if (i.feedback_received) withFeedback++
  })
  const avgConf = total > 0 ? Math.round((totalConf / total) * 100) : 0

  return (
    <div>
      <div style={{ marginBottom: '1.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', flexWrap: 'wrap', gap: 12 }}>
        <div>
          <h1 style={{ fontSize: '1.5rem', fontWeight: 700, marginBottom: 6 }}>Prediction History</h1>
          <p style={{ color: '#64748b', fontSize: '.875rem' }}>All scans analyzed — stored in SQLite audit trail. Submit radiologist feedback to improve model evaluation.</p>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button onClick={exportCSV} disabled={!total}
            style={{ background: 'transparent', border: '1px solid rgba(51,65,85,.5)', borderRadius: 8, color: '#94a3b8', fontSize: '.8rem', padding: '8px 14px', cursor: total ? 'pointer' : 'not-allowed', fontFamily: 'inherit' }}>
            ↓ Export CSV
          </button>
          <button onClick={load}
            style={{ background: 'transparent', border: '1px solid rgba(51,65,85,.5)', borderRadius: 8, color: '#94a3b8', fontSize: '.8rem', padding: '8px 14px', cursor: 'pointer', fontFamily: 'inherit' }}>
            ↻ Refresh
          </button>
        </div>
      </div>

      {/* Stats row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(150px,1fr))', gap: 12, marginBottom: 16 }}>
        <StatCard label="Total Scans" value={total} color="#a5b4fc" />
        <StatCard label="Avg Confidence" value={`${avgConf}%`} color="#34d399" />
        <StatCard label="Uncertain Flags" value={uncertain} color={uncertain > 0 ? '#ef4444' : '#34d399'} sub="flagged for review" />
        <StatCard label="Feedback Given" value={withFeedback} color="#f59e0b" sub={`of ${total} predictions`} />
      </div>

      {/* Class breakdown + filter */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(320px,1fr))', gap: 14, marginBottom: 16 }}>
        <div style={card}>
          <div style={{ fontSize: '.72rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '.06em', color: '#64748b', marginBottom: 12 }}>CLASS DISTRIBUTION</div>
          {Object.entries(classCounts).map(([cls, cnt]) => (
            <MiniBar key={cls} label={cls} count={cnt} total={total} color={CLASS_COLORS[cls].bar} />
          ))}
        </div>

        <div style={card}>
          <div style={{ fontSize: '.72rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '.06em', color: '#64748b', marginBottom: 12 }}>FILTER BY CLASS</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
            {CLASSES.map(c => (
              <button key={c} onClick={() => setFilter(c)} style={{
                padding: '7px 16px', borderRadius: 8, fontSize: '.8rem', fontWeight: 600, cursor: 'pointer', fontFamily: 'inherit', transition: 'all .15s',
                background: filter === c ? (c === 'All' ? 'rgba(99,102,241,.2)' : CLASS_COLORS[c]?.bg) : 'transparent',
                border: `1px solid ${filter === c ? (c === 'All' ? 'rgba(99,102,241,.5)' : CLASS_COLORS[c]?.border) : 'rgba(51,65,85,.5)'}`,
                color: filter === c ? (c === 'All' ? '#a5b4fc' : CLASS_COLORS[c]?.text) : '#64748b',
              }}>
                {c} {c !== 'All' && <span style={{ opacity: .7 }}>({classCounts[c] || 0})</span>}
              </button>
            ))}
          </div>
          <div style={{ marginTop: 14, fontSize: '.78rem', color: '#475569' }}>
            Showing <strong style={{ color: '#94a3b8' }}>{filtered.length}</strong> of {total} predictions
          </div>
        </div>
      </div>

      {/* Table */}
      <div style={card}>
        {loading && <div style={{ color: '#64748b', padding: '2rem', textAlign: 'center' }}>Loading history…</div>}
        {error && <div style={{ color: '#fca5a5', padding: '1rem', background: 'rgba(239,68,68,.08)', borderRadius: 8 }}>Error: {error}</div>}
        {!loading && !error && !filtered.length && (
          <div style={{ textAlign: 'center', color: '#475569', padding: '3rem 1rem' }}>
            <div style={{ fontSize: '2rem', marginBottom: 8, opacity: .3 }}>📋</div>
            {total === 0 ? 'No predictions yet — analyze a scan first.' : `No ${filter} predictions found.`}
          </div>
        )}
        {!loading && filtered.length > 0 && (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  {['ID','Class','Confidence','Uncertainty','Time (ms)','Feedback','Actions'].map(h => (
                    <th key={h} style={{ fontSize: '.68rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '.06em', color: '#64748b', padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid rgba(51,65,85,.4)', whiteSpace: 'nowrap' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {filtered.map(item => {
                  const cc = CLASS_COLORS[item.predicted_class] || CLASS_COLORS.Normal
                  return (
                    <tr key={item.prediction_id} style={{ borderBottom: '1px solid rgba(51,65,85,.2)' }}>
                      <td style={{ padding: '10px 12px', fontSize: '.74rem', fontFamily: 'monospace', color: '#475569' }}>
                        {item.prediction_id?.slice(0, 10)}…
                      </td>
                      <td style={{ padding: '10px 12px' }}>
                        <span style={{ fontSize: '.75rem', fontWeight: 700, padding: '3px 10px', borderRadius: 999, background: cc.bg, color: cc.text, border: `1px solid ${cc.border}` }}>
                          {item.predicted_class}
                        </span>
                      </td>
                      <td style={{ padding: '10px 12px', fontSize: '.82rem', fontWeight: 600, color: cc.bar }}>
                        {Math.round(item.confidence * 100)}%
                      </td>
                      <td style={{ padding: '10px 12px', fontSize: '.82rem', color: item.is_uncertain ? '#ef4444' : '#10b981' }}>
                        {Math.round(item.uncertainty_score * 100)}%
                        {item.is_uncertain && <span style={{ marginLeft: 4, fontSize: '.7rem' }}>⚠</span>}
                      </td>
                      <td style={{ padding: '10px 12px', fontSize: '.82rem', color: '#94a3b8' }}>
                        {Math.round(item.inference_time_ms)}
                      </td>
                      <td style={{ padding: '10px 12px', fontSize: '.78rem' }}>
                        {item.feedback_received ? (
                          <span style={{ color: '#34d399', display: 'flex', alignItems: 'center', gap: 4 }}>
                            ✓ {item.correct_class && item.correct_class !== item.predicted_class
                              ? <span style={{ color: CLASS_COLORS[item.correct_class]?.text }}>{item.correct_class}</span>
                              : 'Confirmed'}
                          </span>
                        ) : (
                          <span style={{ color: '#475569' }}>—</span>
                        )}
                      </td>
                      <td style={{ padding: '10px 12px' }}>
                        {!item.feedback_received && (
                          <button onClick={() => setFeedbackItem(item)} style={{
                            background: 'rgba(99,102,241,.1)', border: '1px solid rgba(99,102,241,.3)',
                            borderRadius: 6, color: '#a5b4fc', fontSize: '.72rem', fontWeight: 600,
                            padding: '4px 10px', cursor: 'pointer', fontFamily: 'inherit',
                          }}>
                            + Feedback
                          </button>
                        )}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {feedbackItem && (
        <FeedbackModal item={feedbackItem} onClose={() => setFeedbackItem(null)} onSaved={load} />
      )}
    </div>
  )
}
