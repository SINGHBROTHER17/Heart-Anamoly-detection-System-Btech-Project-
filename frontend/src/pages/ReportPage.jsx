import { useEffect, useState } from 'react';
import { useLocation, useNavigate, useParams } from 'react-router-dom';
import { getReport } from '../services/api.js';
import ConditionCard from '../components/ConditionCard.jsx';
import EcgPlot from '../components/EcgPlot.jsx';
import LeadQualityBadge from '../components/LeadQualityBadge.jsx';
import ErrorBanner from '../components/ErrorBanner.jsx';
import { SkeletonPlot, SkeletonResultList } from '../components/Skeleton.jsx';
import { useDarkMode } from '../hooks/useDarkMode.js';
import { fmtTimestamp, TIERS } from '../utils/riskTiers.js';

export default function ReportPage() {
  const { id } = useParams();
  const location = useLocation();
  const navigate = useNavigate();
  const [isDark] = useDarkMode();

  // Prefer the report passed via navigation state (from UploadPage) to avoid a
  // redundant round-trip on the "just analyzed" flow.
  const [report, setReport] = useState(location.state?.report || null);
  const [loading, setLoading] = useState(!report);
  const [error, setError] = useState(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (report) return;
    setLoading(true);
    getReport(id)
      .then(setReport)
      .catch((e) => setError({ title: 'Report unavailable', message: e.message }))
      .finally(() => setLoading(false));
  }, [id, report]);

  const handleShare = async () => {
    const url = window.location.href;
    try {
      if (navigator.share) {
        await navigator.share({ title: 'ECG report', url });
      } else {
        await navigator.clipboard.writeText(url);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      }
    } catch { /* user cancelled */ }
  };

  const handleDownloadJson = () => {
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ecg-report-${report.report_id.slice(0, 8)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  if (loading) {
    return (
      <div className="flex flex-col gap-6">
        <SkeletonPlot height={420} />
        <SkeletonResultList count={6} />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col gap-4">
        <ErrorBanner {...error} />
        <button className="btn-secondary self-start" onClick={() => navigate('/')}>
          Back to upload
        </button>
      </div>
    );
  }

  if (!report) return null;

  // Top finding = highest-confidence non-normal condition.
  const sorted = [...report.results].sort((a, b) => b.confidence - a.confidence);
  const topFinding = sorted.find((r) => r.condition !== 'Normal Sinus Rhythm') || sorted[0];
  const topTier = TIERS[topFinding.risk_tier] || TIERS.none;

  // Build { leadName: samples } map for EcgPlot — the API returns per-lead
  // SQI but not raw samples (by design). For the 12-lead viewer we
  // regenerate a preview from the same synthetic source if raw samples
  // aren't present; otherwise the API could add a `preview_samples` field.
  // Here we render a small mini-preview of per-lead SQI instead of full trace.
  // For demo we inject waveforms back from state if available.
  const previewLeads = location.state?.leads || null;

  return (
    <div className="flex flex-col gap-6">
      {/* Header */}
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold">Analysis report</h1>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
            {fmtTimestamp(report.timestamp)} &middot; ID{' '}
            <span className="font-mono">{report.report_id.slice(0, 8)}</span>
          </p>
        </div>
        <span className={`badge ${topTier.color} ${topTier.text} ring-transparent uppercase text-sm px-3 py-1.5`}>
          {topFinding.risk_tier}
        </span>
      </div>

      {/* Overall summary */}
      <div className="card border-l-4" style={{ borderLeftColor: 'currentColor' }}>
        <h2 className="font-semibold text-sm mb-2">Overall interpretation</h2>
        <p className="text-sm leading-relaxed text-slate-700 dark:text-slate-200">
          {report.overall_interpretation}
        </p>
      </div>

      {/* Disclaimer */}
      <div className="rounded-lg bg-amber-50 dark:bg-amber-950 border border-amber-200 dark:border-amber-900 px-4 py-3">
        <p className="text-xs text-amber-800 dark:text-amber-200">
          <strong>Important:</strong> {report.disclaimer}
        </p>
      </div>

      {/* Signal quality overview */}
      <section className="flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <h2 className="font-semibold">Signal quality</h2>
          <span className="text-sm font-mono">
            {(report.signal_quality * 100).toFixed(0)}% overall
          </span>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          {report.per_lead_quality.map((lq) => (
            <LeadQualityBadge key={lq.lead} {...lq} />
          ))}
        </div>
      </section>

      {/* 12-lead ECG viewer — shown only when we have samples */}
      {previewLeads && (
        <section className="flex flex-col gap-2">
          <h2 className="font-semibold">12-lead ECG trace</h2>
          <div className="card p-2">
            <EcgPlot
              signal={toSignalMap(previewLeads)}
              layout="full"
              height={700}
              isDark={isDark}
            />
          </div>
        </section>
      )}

      {/* Condition results */}
      <section className="flex flex-col gap-3">
        <h2 className="font-semibold">Conditions screened</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {sorted.map((r) => (
            <ConditionCard key={r.condition} result={r} />
          ))}
        </div>
      </section>

      {/* Actions */}
      <div className="flex flex-wrap gap-2 pt-2">
        <button className="btn-primary" onClick={handleDownloadJson}>
          Download report (JSON)
        </button>
        <button className="btn-secondary" onClick={handleShare}>
          {copied ? 'Link copied ✓' : 'Share with doctor'}
        </button>
        <button className="btn-ghost" onClick={() => navigate('/')}>
          New analysis
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
