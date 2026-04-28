import { useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { analyzeCsv, analyzeJson } from '../services/api.js';
import { generateDemoSignal } from '../utils/synthEcg.js';

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
      setError('Only CSV files are supported.');
      return;
    }
    setError(null);
    setFile(f);
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    handleFile(e.dataTransfer.files?.[0]);
  };

  const run = async (fn) => {
    setLoading(true);
    setProgress(0);
    setError(null);
    const timer = setInterval(() => setProgress((p) => Math.min(p + 6, 92)), 160);
    try {
      const result = await fn();
      setProgress(100);
      navigate(`/report/${result.report_id}`, { state: { report: result } });
    } catch (e) {
      setError(e.message || 'Analysis failed. Please try again.');
    } finally {
      clearInterval(timer);
      setLoading(false);
    }
  };

  const handleUpload = () => run(() => analyzeCsv(file));
  const handleDemo   = () => run(async () => analyzeJson(generateDemoSignal()));

  return (
    <div className="flex flex-col px-4 pt-5 gap-5">
      {/* Page header */}
      <div>
        <h1 className="text-xl font-bold text-slate-800">12-Lead ECG Analysis</h1>
        <p className="text-sm text-slate-500 mt-1">
          Upload a CSV file with 12-lead ECG data for AI-powered analysis.
        </p>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 text-sm rounded-xl px-4 py-3 flex items-start gap-2">
          <span className="flex-shrink-0">⚠️</span>
          <span>{error}</span>
        </div>
      )}

      {/* Dropzone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => !loading && fileInputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && fileInputRef.current?.click()}
        className={[
          'rounded-2xl border-2 border-dashed p-8 text-center cursor-pointer transition-all',
          dragging
            ? 'border-brand-500 bg-brand-50'
            : file
            ? 'border-green-400 bg-green-50'
            : 'border-slate-200 bg-white hover:border-brand-400 hover:bg-brand-50',
        ].join(' ')}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv"
          className="hidden"
          onChange={(e) => handleFile(e.target.files?.[0])}
        />
        <div className="text-4xl mb-3">
          {file ? '✅' : '📂'}
        </div>
        <p className="font-semibold text-slate-700 text-sm">
          {file ? file.name : 'Tap to select a CSV file'}
        </p>
        <p className="text-xs text-slate-400 mt-1">
          {file
            ? `${(file.size / 1024).toFixed(1)} KB — ready to analyze`
            : 'One column per lead (I, II, III, aVR, aVL, aVF, V1–V6). Up to 50 MB.'}
        </p>
        {file && (
          <button
            className="mt-3 text-xs text-brand-500 font-semibold"
            onClick={(e) => { e.stopPropagation(); setFile(null); setError(null); }}
          >
            Remove file
          </button>
        )}
      </div>

      {/* Progress bar */}
      {loading && (
        <div>
          <div className="flex justify-between text-xs text-slate-500 mb-1.5">
            <span>Analyzing ECG…</span>
            <span className="font-mono font-semibold">{progress}%</span>
          </div>
          <div className="h-2 rounded-full bg-slate-100 overflow-hidden">
            <div
              className="h-full bg-brand-500 rounded-full transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}

      {/* Actions */}
      <button
        className="btn-primary w-full py-4 text-base"
        disabled={!file || loading}
        onClick={handleUpload}
      >
        {loading ? 'Analyzing…' : 'Analyze ECG'}
      </button>

      <div className="flex items-center gap-3">
        <div className="flex-1 h-px bg-slate-200" />
        <span className="text-xs text-slate-400 font-medium">or</span>
        <div className="flex-1 h-px bg-slate-200" />
      </div>

      <button
        className="btn-secondary w-full py-3.5"
        disabled={loading}
        onClick={handleDemo}
      >
        Run demo recording
      </button>

      {/* Format guide */}
      <div className="card bg-slate-50 text-xs text-slate-500">
        <p className="font-semibold text-slate-600 mb-2">CSV format guide</p>
        <p>Each column should be a lead name (I, II, III, aVR, aVL, aVF, V1–V6) and each row a sample at 500 Hz.</p>
        <p className="mt-1">No header row required, but column order must match the 12-lead standard.</p>
      </div>
    </div>
  );
}
