import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { listReports } from '../services/api.js';
import ErrorBanner from '../components/ErrorBanner.jsx';
import { SkeletonCard } from '../components/Skeleton.jsx';
import { fmtTimestamp, TIERS } from '../utils/riskTiers.js';

export default function HistoryPage() {
  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    listReports(50)
      .then((data) => setReports(data.reports || []))
      .catch((e) => setError({ message: e.message }))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex flex-col gap-3">
        <h1 className="text-2xl font-bold">Report history</h1>
        {Array.from({ length: 4 }).map((_, i) => <SkeletonCard key={i} />)}
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col gap-4">
        <h1 className="text-2xl font-bold">Report history</h1>
        <ErrorBanner title="Could not load history" {...error} />
      </div>
    );
  }

  if (reports.length === 0) {
    return (
      <div className="flex flex-col gap-4">
        <h1 className="text-2xl font-bold">Report history</h1>
        <div className="card text-center py-10">
          <p className="text-slate-600 dark:text-slate-400 mb-4">
            No reports yet. Run your first analysis to see it here.
          </p>
          <Link to="/" className="btn-primary">New analysis</Link>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <h1 className="text-2xl font-bold">Report history</h1>

      <ul className="flex flex-col gap-2">
        {reports.map((r) => {
          const top = [...r.results]
            .filter((c) => c.condition !== 'Normal Sinus Rhythm')
            .sort((a, b) => b.confidence - a.confidence)[0] || r.results[0];
          const tier = TIERS[top.risk_tier] || TIERS.none;
          return (
            <li key={r.report_id}>
              <Link
                to={`/report/${r.report_id}`}
                className="card flex items-center gap-4 hover:shadow-md transition-shadow"
              >
                <div className={`h-10 w-10 rounded-full ${tier.color} flex items-center justify-center text-white font-bold`}>
                  {(top.confidence * 100).toFixed(0)}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="font-medium text-sm truncate">{top.condition}</p>
                  <p className="text-xs text-slate-500 dark:text-slate-400">
                    {fmtTimestamp(r.timestamp)} · SQI {(r.signal_quality * 100).toFixed(0)}%
                  </p>
                </div>
                <span className={`badge ${tier.color} ${tier.text} ring-transparent capitalize`}>
                  {top.risk_tier}
                </span>
              </Link>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
