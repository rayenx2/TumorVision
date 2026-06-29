import { useState, useRef, useCallback } from 'react'

const CLASS_COLORS = {
  Normal: { bg: 'rgba(16,185,129,.15)', border: 'rgba(16,185,129,.4)', text: '#34d399', bar: '#10b981' },
  Cyst:   { bg: 'rgba(245,158,11,.15)', border: 'rgba(245,158,11,.4)', text: '#fde68a', bar: '#f59e0b' },
  Tumor:  { bg: 'rgba(239,68,68,.15)',  border: 'rgba(239,68,68,.4)',  text: '#fca5a5', bar: '#ef4444' },
  Stone:  { bg: 'rgba(249,115,22,.15)', border: 'rgba(249,115,22,.4)', text: '#fed7aa', bar: '#f97316' },
}

const CLINICAL_NOTES = {
  Normal: 'No pathological findings detected. Kidney structure appears within normal anatomical parameters. Recommend routine follow-up.',
  Cyst: 'Simple renal cyst detected. Recommend Bosniak classification follow-up (likely Bosniak I–II). Generally benign — monitor for size changes.',
  Tumor: '⚠ Solid renal mass detected with high confidence malignancy pattern. Immediate radiologist review recommended. Consider contrast-enhanced CT or MRI for staging.',
  Stone: 'Calcification / nephrolithiasis detected. Recommend urology referral. Assess stone size, composition, and obstruction risk (KUB, ultrasound).',
}

function ProbBar({ name, probability }) {
  const c = CLASS_COLORS[name] || CLASS_COLORS.Normal
  const pct = Math.round(probability * 100)
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '.8rem', marginBottom: 4 }}>
        <span style={{ color: '#94a3b8' }}>{name}</span>
        <span style={{ fontWeight: 700, color: c.bar }}>{pct}%</span>
      </div>
      <div style={{ height: 8, background: '#1e293b', borderRadius: 999, overflow: 'hidden' }}>
        <div style={{ height: '100%', width: `${pct}%`, background: c.bar, borderRadius: 999, transition: 'width 1s cubic-bezier(.4,0,.2,1)' }} />
      </div>
    </div>
  )
}

function ReportButton({ predictionId, file }) {
  const [state, setState] = useState('idle') // idle | generating | polling | done | error
  const [taskId, setTaskId] = useState(null)
  const [errMsg, setErrMsg] = useState(null)
  const pollRef = useRef(null)

  const stopPolling = () => { if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null } }

  const pollReport = useCallback((tid) => {
    let attempts = 0
    pollRef.current = setInterval(async () => {
      attempts++
      if (attempts > 40) { stopPolling(); setState('error'); setErrMsg('Timed out waiting for report'); return }
      try {
        const res = await fetch(`/api/v1/predict/report/${tid}`)
        if (res.headers.get('content-type')?.includes('application/pdf')) {
          stopPolling()
          const blob = await res.blob()
          const url = URL.createObjectURL(blob)
          const a = document.createElement('a'); a.href = url; a.download = `tumorvision_${predictionId?.slice(0,8)}.pdf`; a.click()
          URL.revokeObjectURL(url); setState('done')
        } else {
          const data = await res.json()
          if (data.status === 'failed') { stopPolling(); setState('error'); setErrMsg(data.error || 'Report generation failed') }
        }
      } catch (e) { stopPolling(); setState('error'); setErrMsg(e.message) }
    }, 3000)
  }, [predictionId])

  const generate = async () => {
    setState('generating'); setErrMsg(null)
    const form = new FormData(); form.append('file', file)
    try {
      const res = await fetch(`/api/v1/predict/${predictionId}/report`, { method: 'POST', body: form })
      if (!res.ok) { const d = await res.json(); throw new Error(d.detail || `HTTP ${res.status}`) }
      const data = await res.json()
      setTaskId(data.task_id); setState('polling')
      pollReport(data.task_id)
    } catch (e) { setState('error'); setErrMsg(e.message) }
  }

  if (state === 'done') return <span style={{ fontSize: '.8rem', color: '#34d399' }}>✓ PDF downloaded</span>
  if (state === 'error') return <span style={{ fontSize: '.78rem', color: '#fca5a5' }}>✗ {errMsg}</span>

  return (
    <button onClick={generate} disabled={state !== 'idle'}
      style={{ background: 'rgba(16,185,129,.1)', border: '1px solid rgba(16,185,129,.3)', borderRadius: 8, color: '#34d399', fontSize: '.8rem', fontWeight: 600, padding: '7px 14px', cursor: state === 'idle' ? 'pointer' : 'not-allowed', fontFamily: 'inherit', display: 'flex', alignItems: 'center', gap: 6 }}>
      {state === 'idle' && '📄 Generate PDF Report'}
      {state === 'generating' && '⏳ Queuing…'}
      {state === 'polling' && '⏳ Generating report…'}
    </button>
  )
}

export default function Predict() {
  const [dragging, setDragging] = useState(false)
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const inputRef = useRef()

  const handleFile = useCallback((f) => {
    if (!f || !f.type.startsWith('image/')) { setError('Please upload a JPEG or PNG image.'); return }
    setFile(f); setPreview(URL.createObjectURL(f)); setResult(null); setError(null)
  }, [])

  const onDrop = useCallback((e) => { e.preventDefault(); setDragging(false); handleFile(e.dataTransfer.files[0]) }, [handleFile])
  const onDrag = useCallback((e) => { e.preventDefault(); setDragging(true) }, [])

  const analyze = async () => {
    if (!file) return
    setLoading(true); setError(null); setResult(null)
    const form = new FormData(); form.append('file', file)
    try {
      const res = await fetch('/api/v1/predict', { method: 'POST', body: form })
      if (!res.ok) { const d = await res.json(); throw new Error(d.detail || `HTTP ${res.status}`) }
      setResult(await res.json())
    } catch (e) { setError(e.message) }
    finally { setLoading(false) }
  }

  const reset = () => { setFile(null); setPreview(null); setResult(null); setError(null) }

  const cc = result ? (CLASS_COLORS[result.predicted_class] || CLASS_COLORS.Normal) : null

  return (
    <div>
      <div style={{ marginBottom: '1.5rem' }}>
        <h1 style={{ fontSize: '1.5rem', fontWeight: 700, marginBottom: 6 }}>CT Scan Classifier</h1>
        <p style={{ color: '#64748b', fontSize: '.875rem' }}>
          Upload a kidney CT scan — EfficientNetB4 classifies Normal / Cyst / Tumor / Stone
          with Grad-CAM heatmap, per-class probabilities, and MC Dropout uncertainty score.
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: 16 }}>

        {/* LEFT — Upload */}
        <div style={{ background: '#0f172a', border: '1px solid rgba(51,65,85,.45)', borderRadius: 14, padding: '1.5rem' }}>
          <div style={{ fontSize: '.72rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '.08em', color: '#64748b', marginBottom: 14 }}>CT SCAN IMAGE</div>

          {/* Drop zone */}
          <div
            onDrop={onDrop} onDragOver={onDrag} onDragLeave={() => setDragging(false)}
            onClick={() => !file && inputRef.current.click()}
            style={{
              border: `2px dashed ${dragging ? '#6366f1' : 'rgba(51,65,85,.6)'}`,
              borderRadius: 12, padding: '2rem 1rem', textAlign: 'center',
              cursor: file ? 'default' : 'pointer',
              background: dragging ? 'rgba(99,102,241,.06)' : 'rgba(15,23,42,.5)',
              transition: 'all .2s', minHeight: 200,
              display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
            }}
          >
            {preview ? (
              <>
                <img src={preview} alt="CT scan preview" style={{ maxHeight: 170, maxWidth: '100%', borderRadius: 8, objectFit: 'contain' }} />
                <div style={{ marginTop: 8, fontSize: '.78rem', color: '#64748b' }}>{file.name} · {(file.size/1024).toFixed(0)} KB</div>
              </>
            ) : (
              <>
                <svg width="40" height="40" viewBox="0 0 40 40" fill="none" style={{ marginBottom: 12, opacity: .35 }}>
                  <rect width="40" height="40" rx="10" fill="#334155"/>
                  <path d="M13 20h14M20 13v14" stroke="#94a3b8" strokeWidth="2.5" strokeLinecap="round"/>
                </svg>
                <div style={{ fontWeight: 600, marginBottom: 4 }}>Drop CT scan here</div>
                <div style={{ fontSize: '.78rem', color: '#64748b' }}>or click to browse — JPEG / PNG</div>
                <div style={{ fontSize: '.72rem', color: '#334155', marginTop: 6 }}>
                  Sample images in <code>tests/sample_images/</code>
                </div>
              </>
            )}
          </div>
          <input ref={inputRef} type="file" accept="image/*" style={{ display: 'none' }}
            onChange={e => handleFile(e.target.files[0])} />

          {error && (
            <div style={{ marginTop: 12, background: 'rgba(239,68,68,.1)', border: '1px solid rgba(239,68,68,.3)', borderRadius: 8, padding: '10px 14px', fontSize: '.82rem', color: '#fca5a5' }}>
              {error}
            </div>
          )}

          <div style={{ display: 'flex', gap: 10, marginTop: 14 }}>
            <button onClick={analyze} disabled={!file || loading}
              style={{
                flex: 1, background: (!file || loading) ? 'rgba(99,102,241,.3)' : 'linear-gradient(135deg,#6366f1,#8b5cf6)',
                border: 'none', borderRadius: 10, color: '#fff', fontSize: '.9rem', fontWeight: 700,
                padding: '11px', cursor: (!file || loading) ? 'not-allowed' : 'pointer', fontFamily: 'inherit',
              }}>
              {loading ? '⏳ Analyzing…' : '🔬 Analyze Scan'}
            </button>
            {file && (
              <button onClick={reset}
                style={{ background: 'transparent', border: '1px solid rgba(51,65,85,.5)', borderRadius: 10, color: '#64748b', fontSize: '.8rem', padding: '11px 16px', cursor: 'pointer', fontFamily: 'inherit' }}>
                Clear
              </button>
            )}
          </div>

          {/* How it works */}
          <div style={{ marginTop: 20, paddingTop: 16, borderTop: '1px solid rgba(51,65,85,.25)' }}>
            <div style={{ fontSize: '.72rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '.06em', color: '#64748b', marginBottom: 10 }}>HOW IT WORKS</div>
            {['Upload a kidney CT scan (JPEG/PNG)', 'EfficientNetB4 runs inference (~80ms)', 'Grad-CAM highlights decision regions', 'MC Dropout quantifies prediction uncertainty', 'Optional: generate PDF report for records'].map((s, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'flex-start', gap: 8, marginBottom: 7, fontSize: '.8rem' }}>
                <span style={{ background: 'rgba(99,102,241,.2)', color: '#a5b4fc', borderRadius: '50%', width: 20, height: 20, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '.68rem', fontWeight: 700, flexShrink: 0, marginTop: 1 }}>{i + 1}</span>
                <span style={{ color: '#94a3b8', lineHeight: 1.45 }}>{s}</span>
              </div>
            ))}
          </div>
        </div>

        {/* RIGHT — Result */}
        <div style={{ background: '#0f172a', border: '1px solid rgba(51,65,85,.45)', borderRadius: 14, padding: '1.5rem', display: 'flex', flexDirection: 'column', justifyContent: result ? 'flex-start' : 'center', alignItems: result ? 'stretch' : 'center', minHeight: 420 }}>

          {!result && !loading && (
            <div style={{ textAlign: 'center', color: '#475569' }}>
              <svg width="56" height="56" viewBox="0 0 56 56" fill="none" style={{ marginBottom: 14, opacity: .2 }}>
                <circle cx="28" cy="28" r="24" stroke="#64748b" strokeWidth="2"/>
                <circle cx="28" cy="28" r="12" stroke="#64748b" strokeWidth="1.5" fill="none" opacity=".6"/>
                <circle cx="34" cy="22" r="5" fill="#64748b" opacity=".4"/>
              </svg>
              <div style={{ fontSize: '.88rem' }}>Upload a scan to see results</div>
            </div>
          )}

          {loading && (
            <div style={{ textAlign: 'center', color: '#64748b' }}>
              <div style={{ fontSize: '2.5rem', marginBottom: 12, display: 'inline-block', animation: 'spin 1.5s linear infinite' }}>⏳</div>
              <div style={{ fontSize: '.9rem', fontWeight: 600, color: '#a5b4fc' }}>Running EfficientNetB4…</div>
              <div style={{ fontSize: '.78rem', marginTop: 6 }}>Grad-CAM + MC Dropout uncertainty</div>
            </div>
          )}

          {result && (
            <>
              {/* Header */}
              <div style={{ fontSize: '.72rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '.08em', color: '#64748b', marginBottom: 12 }}>PREDICTION RESULT</div>

              {/* Class badge + meta */}
              <div style={{ textAlign: 'center', marginBottom: 16 }}>
                <span style={{ display: 'inline-flex', alignItems: 'center', gap: 8, padding: '8px 20px', borderRadius: 999, fontSize: '1rem', fontWeight: 700, background: cc.bg, border: `1px solid ${cc.border}`, color: cc.text }}>
                  {result.predicted_class === 'Tumor' && '🚨'}
                  {result.predicted_class === 'Normal' && '✅'}
                  {result.predicted_class === 'Cyst' && '⚠️'}
                  {result.predicted_class === 'Stone' && '⚡'}
                  {result.predicted_class}
                </span>
                <div style={{ marginTop: 10, display: 'flex', justifyContent: 'center', gap: 20, fontSize: '.82rem', color: '#94a3b8', flexWrap: 'wrap' }}>
                  <span>Confidence: <strong style={{ color: '#f1f5f9' }}>{Math.round(result.confidence * 100)}%</strong></span>
                  <span>Uncertainty: <strong style={{ color: result.is_uncertain ? '#ef4444' : '#10b981' }}>{Math.round(result.uncertainty_score * 100)}%</strong>
                    {result.is_uncertain && <span style={{ marginLeft: 5, color: '#ef4444', fontSize: '.72rem' }}>⚠ Flag for review</span>}
                  </span>
                  <span style={{ color: '#475569', fontSize: '.78rem' }}>{Math.round(result.inference_time_ms)}ms · {result.model_version}</span>
                </div>
              </div>

              {/* Grad-CAM */}
              {result.gradcam_base64 && (
                <div style={{ marginBottom: 16 }}>
                  <div style={{ fontSize: '.72rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '.06em', color: '#64748b', marginBottom: 8 }}>GRAD-CAM HEATMAP</div>
                  <div style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
                    <div style={{ textAlign: 'center' }}>
                      <img src={preview} alt="original" style={{ width: 120, height: 120, objectFit: 'cover', borderRadius: 8, border: '1px solid rgba(51,65,85,.4)', display: 'block' }} />
                      <div style={{ fontSize: '.68rem', color: '#475569', marginTop: 4 }}>Original</div>
                    </div>
                    <div style={{ textAlign: 'center' }}>
                      <img src={result.gradcam_base64} alt="gradcam" style={{ width: 120, height: 120, objectFit: 'cover', borderRadius: 8, border: '1px solid rgba(51,65,85,.4)', display: 'block' }} />
                      <div style={{ fontSize: '.68rem', color: '#475569', marginTop: 4 }}>Grad-CAM</div>
                    </div>
                    <div style={{ flex: 1, fontSize: '.74rem', color: '#64748b', lineHeight: 1.5 }}>
                      <strong style={{ color: '#94a3b8', display: 'block', marginBottom: 4 }}>Reading the heatmap</strong>
                      Red/yellow regions indicate areas most influential to the model's decision.
                      Blue regions contributed less to the prediction.
                    </div>
                  </div>
                </div>
              )}

              {/* Probability bars */}
              <div style={{ marginBottom: 14 }}>
                <div style={{ fontSize: '.72rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '.06em', color: '#64748b', marginBottom: 10 }}>CLASS PROBABILITIES</div>
                {result.probabilities.map(p => <ProbBar key={p.class_name} name={p.class_name} probability={p.probability} />)}
              </div>

              {/* Clinical note */}
              <div style={{ background: result.predicted_class === 'Tumor' ? 'rgba(239,68,68,.06)' : 'rgba(15,23,42,.8)', border: `1px solid ${result.predicted_class === 'Tumor' ? 'rgba(239,68,68,.3)' : 'rgba(51,65,85,.3)'}`, borderRadius: 8, padding: '10px 14px', fontSize: '.78rem', color: '#94a3b8', lineHeight: 1.55, marginBottom: 14 }}>
                {CLINICAL_NOTES[result.predicted_class]}
              </div>

              {/* Actions */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 10 }}>
                <ReportButton predictionId={result.prediction_id} file={file} />
                <div style={{ fontSize: '.7rem', color: '#334155' }}>
                  ID: <code>{result.prediction_id?.slice(0, 16)}…</code>
                </div>
              </div>
            </>
          )}
        </div>
      </div>

      <style>{`@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }`}</style>
    </div>
  )
}
