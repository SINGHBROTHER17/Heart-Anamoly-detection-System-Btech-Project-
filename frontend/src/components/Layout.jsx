import { Link, NavLink, Outlet } from 'react-router-dom';
import { useDarkMode } from '../hooks/useDarkMode.js';

export default function Layout() {
  const [isDark, toggleDark] = useDarkMode();

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header — sticky so the nav stays accessible on small screens */}
      <header className="sticky top-0 z-40 bg-white/90 dark:bg-slate-900/90 backdrop-blur border-b border-slate-200 dark:border-slate-700">
        <div className="mx-auto max-w-4xl px-4 h-14 flex items-center justify-between">
          <Link to="/" className="flex items-center gap-2 font-semibold">
            <span className="text-xl" aria-hidden>❤️</span>
            <span>ECG&nbsp;Screen</span>
          </Link>
          <nav className="flex items-center gap-2 text-sm">
            <NavLink to="/" end className={navClass}>Upload</NavLink>
            <NavLink to="/history" className={navClass}>History</NavLink>
            <button
              type="button"
              onClick={toggleDark}
              aria-label="Toggle dark mode"
              className="btn-ghost px-2"
            >
              {isDark ? '☀️' : '🌙'}
            </button>
          </nav>
        </div>
      </header>

      <main className="flex-1 mx-auto w-full max-w-4xl px-4 py-6">
        <Outlet />
      </main>

      <footer className="mx-auto w-full max-w-4xl px-4 py-6 text-xs text-slate-500 dark:text-slate-400">
        Screening tool only — not a substitute for medical diagnosis.
      </footer>
    </div>
  );
}

function navClass({ isActive }) {
  return [
    'px-3 py-1.5 rounded-md transition-colors',
    isActive
      ? 'bg-brand-50 text-brand-700 dark:bg-slate-800 dark:text-brand-500'
      : 'text-slate-600 hover:text-slate-900 dark:text-slate-300 dark:hover:text-white',
  ].join(' ');
}
