import { useEffect, useState } from 'react';
import { useLocation, useNavigate, useParams } from 'react-router-dom';
import { getReport } from '../services/api.js';
import ConditionCard from '../components/ConditionCard.jsx';
import RiskMeter from '../components/RiskMeter.jsx';
import EcgPlot from '../components/EcgPlot.jsx';
import { SkeletonPlot, SkeletonResultList } from '../components/Skeleton.jsx';
import { useDarkMode } from '../hooks/useDarkMode.js';
import { useUserProfile } from '../hooks/useUserProfile.js';
import { fmtTimestamp } from '../utils/riskTiers.js';

export default function ReportPage() {
  const { id } = useParams();
  const location = useLocation();
  const navigate = useNavigate();
  const [isDark] = useDarkMode();
  const [profile] = useUserProfile();

  const [report, setReport] = useState(location.state?.report || null);
  const [loading, setLoading] = useState(!report);
  const [error, setError] = useState(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (report) return;
    setLoading(true);
    getReport(id)
      .then(setReport)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [id, report]);

  if (loading) return (
    <div className="px-4 pt-5 flex flex-col gap-5">
      <div className="skeleton h-7 w-48 rounded-xl" />
      <div className="skeleton h-32 rounded-2xl" />
      <SkeletonPlot height={300} />
      <SkeletonResultList count={4} />
    </div>
  );

  if (error) return (
    <div className="px-4 pt-5 flex flex-col gap-4">
      <div className="bg-red-50 border border-red-200 text-red-700 rounded-2xl p-4 text-sm">{error}</div>
      <button className="btn-secondary" onClick={() => navigate('/')}>Back to Home</button>
    </div>
  );

  if (!report) return null;

  const sorted = [...report.results].sort((a, b) => b.confidence - a.confidence);
  const topFinding = sorted.find((r) => r.condition !== 'Normal Sinus Rhythm') || sorted[0];
  const previewLeads = location.state?.leads || null;
  const name = profile.name || 'there';

  // Map top tier to overall risk display
  const isNormal = topFinding.risk_tier === 'none';

  const handleShare = async () => {
    try {
      if (navigator.share) {
        await navigator.share({ title: 'ECG Report', url: window.location.href });
      } else {
        await navigator.clipboard.writeText(window.location.href);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      }
    } catch { /* cancelled */ }
  };

  const handleSave = () => {
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ecg-report-${report.report_id.slice(0, 8)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-4 pt-4 pb-3">
        <button onClick={() => navigate(-1)} className="flex items-center gap-1 text-slate-500 text-sm">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-4 h-4">
            <path d="M15 18l-6-6 6-6" />
          </svg>
          Back
        </button>
        <h1 className="text-base font-bold text-slate-800">12-Lead ECG Report</h1>
        <div className="w-10 h-10 rounded-full bg-brand-500 flex items-center justify-center text-white text-xs font-bold">
          {name.split(' ').map((w) => w[0]).join('').slice(0, 2).toUpperCase() || '?'}
        </div>
      </div>

      {/* Notification banner */}
      <div className="mx-4 bg-green-50 border border-green-200 rounded-2xl px-4 py-3 flex items-start gap-3">
        <span className="text-green-500 text-lg mt-0.5">✅</span>
        <div className="flex-1">
          <p className="text-sm font-semibold text-slate-800">
            Hey {name}, your 12-Lead ECG Report is ready!
          </p>
          <p className="text-xs text-slate-400 mt-0.5">{fmtTimestamp(report.timestamp)}</p>
        </div>
      </div>

      {/* Risk meter */}
      <div className="mx-4 mt-4 card">
        <RiskMeter tier={topFinding.risk_tier} />
        <div className="mt-3 flex items-center gap-2">
          <span className={`w-5 h-5 rounded-full flex items-center justify-center text-white text-xs ${isNormal ? 'bg-green-500' : 'bg-brand-500'}`}>
            {isNormal ? '✓' : '!'}
          </span>
          <span className="font-semibold text-slate-700 text-sm">{topFinding.condition}</span>
        </div>
      </div>

      {/* Interpretation */}
      <div className="mx-4 mt-4 card">
        <h2 className="font-semibold text-slate-700 text-sm mb-2">Interpretation details</h2>
        <p className="text-sm text-slate-600 leading-relaxed">{report.overall_interpretation}</p>
      </div>

      {/* ECG Data metrics — top 4 findings as metric cards */}
      <div className="mx-4 mt-4">
        <h2 className="font-semibold text-slate-700 text-sm mb-3">ECG Data Analysis</h2>
        <div className="grid grid-cols-2 gap-2">
          {sorted.slice(0, 4).map((r) => {
            const pct = Math.round(r.confidence * 100);
            const tierColor = { none: '#22c55e', possible: '#f59e0b', likely: '#f97316', high: '#ef4444' }[r.risk_tier] ?? '#94a3b8';
            const shortName = r.condition.split(' ').slice(0, 2).join(' ');
            return (
              <div key={r.condition} className="card border-l-4 py-3" style={{ borderLeftColor: tierColor }}>
                <p className="text-[10px] text-slate-400 uppercase tracking-wide font-semibold">{shortName}</p>
                <p className="text-2xl font-bold mt-0.5" style={{ color: tierColor }}>{pct}%</p>
                <p className="text-[10px] text-slate-400">confidence</p>
              </div>
            );
          })}
        </div>
      </div>

      {/* ECG waveform */}
      {previewLeads && (
        <div className="mx-4 mt-4">
          <h2 className="font-semibold text-slate-700 text-sm mb-2">ECG characteristics</h2>
          <div className="card p-2 overflow-hidden">
            <EcgPlot
              signal={toSignalMap(previewLeads)}
              layout="full"
              height={400}
              isDark={isDark}
            />
            <p className="text-[10px] text-slate-400 text-center mt-1">Tap graph to enlarge view</p>
          </div>
        </div>
      )}

      {/* All conditions */}
      <div className="mx-4 mt-4">
        <h2 className="font-semibold text-slate-700 text-sm mb-3">All conditions screened</h2>
        <div className="flex flex-col gap-2">
          {sorted.map((r) => (
            <ConditionCard key={r.condition} result={r} />
          ))}
        </div>
      </div>

      {/* Signal quality */}
      <div className="mx-4 mt-4 card">
        <div className="flex items-center justify-between mb-1">
          <h2 className="font-semibold text-slate-700 text-sm">Signal Quality</h2>
          <span className="text-sm font-bold text-slate-600">
            {(report.signal_quality * 100).toFixed(0)}%
          </span>
        </div>
        <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
          <div
            className="h-full rounded-full bg-green-500"
            style={{ width: `${report.signal_quality * 100}%` }}
          />
        </div>
      </div>

      {/* Disclaimer */}
      <div className="mx-4 mt-4 bg-amber-50 border border-amber-100 rounded-2xl px-4 py-3">
        <p className="text-xs text-amber-700">
          <strong>Important:</strong> {report.disclaimer}
        </p>
      </div>

      {/* Action buttons */}
      <div className="mx-4 mt-5 mb-4 flex gap-3">
        <button onClick={handleSave} className="btn-primary flex-1 gap-2">
          <span>↓</span> Save Report
        </button>
        <button onClick={handleShare} className="btn-secondary flex-1 gap-2">
          <span>↗</span> {copied ? 'Link copied!' : 'Share Report'}
        </button>
      </div>
    </div>
  );
}

function toSignalMap(leads) {
  const map = {};
  for (const l of leads) map[l.lead_name] = l.samples;
  return map;
}
