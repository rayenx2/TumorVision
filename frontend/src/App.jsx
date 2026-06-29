import { BrowserRouter, Routes, Route, NavLink, Navigate } from 'react-router-dom'
import Predict from './components/Predict.jsx'
import History from './components/History.jsx'
import About from './components/About.jsx'

const TABS = [
  { path: '/predict',  label: 'Predict',  icon: '🔬' },
  { path: '/history',  label: 'History',  icon: '📋' },
  { path: '/about',    label: 'About',    icon: 'ℹ' },
]

const s = {
  root: { minHeight: '100vh', background: '#0a0f1a', color: '#f1f5f9', fontFamily: "'Inter', sans-serif" },
  topBar: { height: 2, background: 'linear-gradient(90deg,#6366f1,#8b5cf6,#06b6d4)' },
  header: {
    background: 'rgba(10,15,26,.97)', borderBottom: '1px solid rgba(51,65,85,.45)',
    padding: '0 2rem', display: 'flex', alignItems: 'center', gap: '2rem',
    height: 64, position: 'sticky', top: 0, zIndex: 100, backdropFilter: 'blur(12px)',
  },
  logo: { display: 'flex', alignItems: 'center', gap: 10, textDecoration: 'none' },
  logoText: { fontSize: '1.2rem', fontWeight: 700, color: '#f1f5f9', letterSpacing: '-.02em' },
  badge: {
    fontSize: '.6rem', fontWeight: 700, background: 'rgba(99,102,241,.15)',
    border: '1px solid rgba(99,102,241,.35)', color: '#a5b4fc', borderRadius: 4,
    padding: '2px 7px', letterSpacing: '.05em', textTransform: 'uppercase',
  },
  nav: { display: 'flex', gap: '.25rem', marginLeft: 'auto' },
  links: { display: 'flex', gap: '1rem', alignItems: 'center' },
  extLink: { color: '#475569', textDecoration: 'none', fontSize: '.8rem' },
  main: { maxWidth: 1200, margin: '0 auto', padding: '2rem 1.5rem' },
}

const tabStyle = (active) => ({
  background: active ? 'rgba(99,102,241,.15)' : 'transparent',
  border: active ? '1px solid rgba(99,102,241,.4)' : '1px solid transparent',
  color: active ? '#a5b4fc' : '#64748b',
  padding: '.4rem 1rem', borderRadius: 8, cursor: 'pointer',
  fontSize: '.875rem', fontWeight: 500, display: 'flex', alignItems: 'center',
  gap: '.4rem', textDecoration: 'none', transition: 'all .2s',
})

export default function App() {
  return (
    <BrowserRouter>
      <div style={s.root}>
        <div style={s.topBar} />
        <header style={s.header}>
          <NavLink to="/predict" style={s.logo}>
            <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
              <defs>
                <linearGradient id="lg" x1="0" y1="0" x2="32" y2="32" gradientUnits="userSpaceOnUse">
                  <stop offset="0%" stopColor="#4f46e5"/><stop offset="100%" stopColor="#7c3aed"/>
                </linearGradient>
              </defs>
              <rect width="32" height="32" rx="8" fill="url(#lg)"/>
              <circle cx="16" cy="16" r="8" stroke="white" strokeWidth="1.5" fill="none" opacity=".9"/>
              <line x1="8" y1="16" x2="24" y2="16" stroke="white" strokeWidth="1" opacity=".5"/>
              <line x1="16" y1="8" x2="16" y2="24" stroke="white" strokeWidth="1" opacity=".5"/>
              <circle cx="19" cy="13" r="3" fill="#06b6d4" opacity=".9"/>
              <circle cx="19" cy="13" r="1.5" fill="white"/>
            </svg>
            <span style={s.logoText}>Tumor<span style={{color:'#8b5cf6'}}>Vision</span></span>
            <span style={s.badge}>Deep Learning</span>
          </NavLink>

          <nav style={s.nav}>
            {TABS.map(t => (
              <NavLink key={t.path} to={t.path} style={({isActive}) => tabStyle(isActive)}>
                <span>{t.icon}</span>{t.label}
              </NavLink>
            ))}
          </nav>

          <div style={s.links}>
            <a href="http://localhost:8110/docs" target="_blank" rel="noreferrer" style={s.extLink}>API Docs</a>
            <a href="https://github.com/rayenx2/TumorVision" target="_blank" rel="noreferrer" style={s.extLink}>GitHub</a>
          </div>
        </header>

        <main style={s.main}>
          <Routes>
            <Route path="/" element={<Navigate to="/predict" replace />} />
            <Route path="/predict" element={<Predict />} />
            <Route path="/history" element={<History />} />
            <Route path="/about"   element={<About />} />
            <Route path="*"        element={<Navigate to="/predict" replace />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}
