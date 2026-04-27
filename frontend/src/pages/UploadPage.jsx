import { useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { analyzeCsv, analyzeJson } from '../services/api.js';
import { generateDemoSignal } from '../utils/synthEcg.js';
import ErrorBanner from '../components/ErrorBanner.jsx';

export default function UploadPage() {
  const navigate = useNavigate();
  const fileInputRef = useRef(null);

  const [file, setFile] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);

  const handleFile = (f) => {
    if (!f) return;
    if (!f.name.toLowerCase().endsWith('.csv')) {
      setError({ message: 'Only CSV files are supported.' });
      return;
    }
    setError(null);
    setFile(f);
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files?.[0];
    if (f) handleFile(f);
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setProgress(0);
    setError(null);
    const timer = setInterval(() => setProgress((p) => Math.min(p + 7, 92)), 150);
    try {
      const result = await analyzeCsv(file);
      setProgress(100);
      navigate(`/report/${result.report_id}`, { state: { report: result } });
    } catch (e) {
      setError({
        title: 'Analysis failed',
        message: e.message,
        perLeadSqi: e.perLeadSqi,
      });
    } finally {
      clearInterval(timer);
      setLoading(false);
    }
  };

  const handleDemo = async () => {
    setLoading(true);
    setProgress(0);
    setError(null);
    const timer = setInterval(() => setProgress((p) => Math.min(p + 5, 92)), 180);
    try {
      const leads = generateDemoSignal();
      const result = await analyzeJson(leads);
      setProgress(100);
      navigate(`/report/${result.report_id}`, { state: { report: result } });
    } catch (e) {
      setError({
        title: 'Demo failed',
        message: e.message,
        perLeadSqi: e.perLeadSqi,
      });
    } finally {
      clearInterval(timer);
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-bold">Upload an ECG recording</h1>
        <p className="mt-1 text-slate-600 dark:text-slate-400 text-sm">
          Drop a 12-lead CSV file or try a demo recording to see how the system works.
        </p>
      </div>

      {error && <ErrorBanner {...error} onRetry={() => setError(null)} />}

      {/* Dropzone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => fileInputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && fileInputRef.current?.click()}
        className={[
          'rounded-xl border-2 border-dashed p-8 text-center cursor-pointer transition-colors',
          dragging
            ? 'border-brand-500 bg-brand-50 dark:bg-slate-800'
            : 'border-slate-300 dark:border-slate-600 hover:border-brand-500 hover:bg-slate-50 dark:hover:bg-slate-800',
        ].join(' ')}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv"
          className="hidden"
          onChange={(e) => handleFile(e.target.files?.[0])}
        />
        <div className="text-4xl mb-2" aria-hidden>📄</div>
        <p className="font-medium">
          {file ? file.name : 'Drop a CSV file here or tap to browse'}
        </p>
        <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
          One column per lead (I, II, III, aVR, aVL, aVF, V1–V6). Up to 50 MB.
        </p>
      </div>

      {/* Progress bar during loading */}
      {loading && (
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span>Analyzing…</span>
            <span className="font-mono">{progress}%</span>
          </div>
          <div className="h-2 rounded-full bg-slate-200 dark:bg-slate-700 overflow-hidden">
            <div className="h-full bg-brand-500 transition-all" style={{ width: `${progress}%` }} />
          </div>
        </div>
      )}

      <div className="flex flex-wrap gap-2">
        <button
          className="btn-primary flex-1 sm:flex-none"
          disabled={!file || loading}
          onClick={handleUpload}
        >
          {loading ? 'Analyzing…' : 'Analyze uploaded file'}
        </button>
        <button
          className="btn-secondary flex-1 sm:flex-none"
          disabled={loading}
          onClick={handleDemo}
        >
          Simulate demo recording
        </button>
      </div>

      <div className="text-xs text-slate-500 dark:text-slate-400 border-t border-slate-200 dark:border-slate-700 pt-4">
        <strong>Manual lead entry:</strong> Paste per-lead sample data? That flow is
        available in the API via <code className="font-mono">POST /analyze/json</code>.
        The demo button above generates a complete synthetic 12-lead recording using
        that same endpoint.
      </div>
    </div>
  );
}
