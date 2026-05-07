import { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { listReports } from '../services/api.js';
import { fmtTimestamp } from '../utils/riskTiers.js';

const TIER_STYLES = {
  none:     { bg: 'bg-green-100',  text: 'text-green-700',  dot: 'bg-green-500',  label: 'Normal' },
  possible: { bg: 'bg-amber-100',  text: 'text-amber-700',  dot: 'bg-amber-400',  label: 'Possible' },
  likely:   { bg: 'bg-orange-100', text: 'text-orange-700', dot: 'bg-orange-500', label: 'Likely' },
  high:     { bg: 'bg-red-100',    text: 'text-red-700',    dot: 'bg-red-500',    label: 'High risk' },
};

export default function HistoryPage() {
  const navigate = useNavigate();
  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    listReports(50)
      .then((data) => setReports(data.reports || []))
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="flex flex-col px-4 pt-5 gap-4">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold text-slate-800">My Reports</h1>
        {!loading && reports.length > 0 && (
          <span className="text-xs text-slate-400">{reports.length} total</span>
        )}
      </div>

      {loading && (
        <div className="flex flex-col gap-3">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="skeleton h-20 rounded-2xl" />
          ))}
        </div>
      )}

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 rounded-2xl px-4 py-3 text-sm">
          {error}
        </div>
      )}

      {!loading && !error && reports.length === 0 && (
        <div className="card flex flex-col items-center text-center py-12 gap-4">
          <span className="text-5xl">📋</span>
          <div>
            <h2 className="font-bold text-slate-700">No reports yet</h2>
            <p className="text-sm text-slate-500 mt-1">
              Run your first ECG analysis to see it here.
            </p>
          </div>
          <button className="btn-primary" onClick={() => navigate('/analyze')}>
            Start Analysis
          </button>
        </div>
      )}

      {!loading && reports.length > 0 && (
        <ul className="flex flex-col gap-3">
          {reports.map((r) => {
            const top = [...r.results]
              .filter((c) => c.condition !== 'Normal Sinus Rhythm')
              .sort((a, b) => b.confidence - a.confidence)[0] || r.results[0];
            const style = TIER_STYLES[top.risk_tier] ?? TIER_STYLES.none;

            return (
              <li key={r.report_id}>
                <Link
                  to={`/report/${r.report_id}`}
                  className="card flex items-center gap-4 active:scale-[0.98] transition-transform"
                >
                  {/* Avatar with tier color */}
                  <div className={`w-12 h-12 rounded-2xl flex flex-col items-center justify-center flex-shrink-0 ${style.bg}`}>
                    <span className={`text-lg font-bold ${style.text}`}>
                      {Math.round(top.confidence * 100)}
                    </span>
                    <span className={`text-[9px] font-semibold ${style.text}`}>%</span>
                  </div>

                  {/* Info */}
                  <div className="flex-1 min-w-0">
                    <p className="font-semibold text-sm text-slate-800 truncate">{top.condition}</p>
                    <p className="text-xs text-slate-400 mt-0.5">{fmtTimestamp(r.timestamp)}</p>
                    <div className="flex items-center gap-1.5 mt-1">
                      <span className={`w-1.5 h-1.5 rounded-full ${style.dot}`} />
                      <span className={`text-[10px] font-semibold ${style.text}`}>{style.label}</span>
                      <span className="text-[10px] text-slate-300">·</span>
                      <span className="text-[10px] text-slate-400">
                        SQI {(r.signal_quality * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>

                  {/* Chevron */}
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
                       className="w-4 h-4 text-slate-300 flex-shrink-0">
                    <path d="M9 18l6-6-6-6" />
                  </svg>
                </Link>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
