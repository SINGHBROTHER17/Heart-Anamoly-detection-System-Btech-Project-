/**
 * LiveWaveformPreview — real-time scrolling ECG waveform on HTML5 Canvas.
 *
 * In demo mode (no dataStream prop): generates a realistic synthetic PQRST
 * morphology waveform with slight RR-interval jitter.
 *
 * In live mode: pass a `dataStream` Float32Array that grows over time;
 * the component renders the tail of the array scrolling left.
 *
 * Rendering: green trace on dark background, 60 fps via requestAnimationFrame.
 * Grid lines: light grey at ~5mm squares.
 */
import { useEffect, useRef } from 'react';

// ── Synthetic PQRST generator ────────────────────────────────────────────────
function gaussian(t, center, sigma, amp) {
  return amp * Math.exp(-((t - center) ** 2) / (2 * sigma ** 2));
}

/**
 * Generate one synthetic ECG beat starting at `offset` (samples).
 * Returns an array of { t, v } pairs.
 */
function generateBeat(offset, fs = 500) {
  const ms = (ms) => Math.round((ms / 1000) * fs);  // ms → samples
  const points = [];
  const total  = ms(900);  // 900ms window

  for (let i = 0; i < total; i++) {
    const t = i / fs;  // time in seconds

    // P wave  — Gaussian, peaks at 80ms
    const P = gaussian(t, 0.08,  0.025, 0.15);
    // Q dip   — negative, at 155ms
    const Q = gaussian(t, 0.155, 0.008, -0.10);
    // R spike — at 170ms
    const R = gaussian(t, 0.17,  0.007,  1.00);
    // S dip   — at 187ms
    const S = gaussian(t, 0.187, 0.008, -0.15);
    // T wave  — Gaussian, peaks at 350ms
    const T = gaussian(t, 0.35,  0.05,   0.30);

    points.push({ t: offset + i, v: P + Q + R + S + T });
  }
  return points;
}

export default function LiveWaveformPreview({
  dataStream = null,   // Float32Array | null  (null = demo mode)
  fs         = 500,    // sample rate
  height     = 120,
}) {
  const canvasRef  = useRef(null);
  const bufferRef  = useRef([]);   // rolling array of { t, v } sample points
  const rafRef     = useRef(null);
  const beatOffRef = useRef(0);    // current beat offset (samples)
  const rrJitter   = useRef(0);

  // Heart rate estimate state (computed from peaks)
  const hrRef      = useRef(72);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx    = canvas.getContext('2d');
    const W      = canvas.width;
    const H      = canvas.height;

    const WINDOW_S  = 5;                   // show last 5 seconds
    const WINDOW_PT = WINDOW_S * fs;       // samples in window
    const PX_PER_PT = W / WINDOW_PT;       // pixels per sample
    const MV_SCALE  = H * 0.35;           // pixels per mV
    const MIDLINE   = H * 0.55;

    // ── Demo-mode beat scheduler ──────────────────────────────────────────
    function appendDemoBeat() {
      // RR interval: 833ms ± 30ms jitter
      const rr = Math.round(fs * (0.833 + rrJitter.current));
      rrJitter.current = (Math.random() - 0.5) * 0.06;
      const beat = generateBeat(beatOffRef.current, fs);
      bufferRef.current.push(...beat);
      beatOffRef.current += rr;
      // Trim old samples
      while (bufferRef.current.length > WINDOW_PT * 2)
        bufferRef.current.shift();
    }

    // Seed initial buffer with 2 beats so there's content immediately
    for (let i = 0; i < 3; i++) appendDemoBeat();

    function draw() {
      // In demo mode, append the next beat when buffer is running low
      if (!dataStream && bufferRef.current.length < WINDOW_PT + 100)
        appendDemoBeat();

      const source = dataStream
        ? Array.from(dataStream).map((v, i) => ({ t: i, v }))
        : bufferRef.current;

      const tail = source.slice(-WINDOW_PT);

      // Clear
      ctx.fillStyle = '#0d1117';
      ctx.fillRect(0, 0, W, H);

      // Grid lines — 5mm equivalent columns + rows (slate-800)
      const gridPtX = Math.round(0.2 * fs);  // 200ms per column
      const gridPtY = 0.5;                   // 0.5mV per row
      ctx.strokeStyle = '#1e293b';
      ctx.lineWidth   = 0.5;
      ctx.setLineDash([]);

      for (let col = 0; col < WINDOW_PT; col += gridPtX) {
        const x = col * PX_PER_PT;
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
      }
      for (let row = -2; row <= 2; row += gridPtY) {
        const y = MIDLINE - row * MV_SCALE;
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
      }

      // Baseline
      ctx.strokeStyle = '#2d3748';
      ctx.lineWidth   = 0.8;
      ctx.beginPath(); ctx.moveTo(0, MIDLINE); ctx.lineTo(W, MIDLINE); ctx.stroke();

      // Waveform trace
      if (tail.length < 2) { rafRef.current = requestAnimationFrame(draw); return; }

      ctx.strokeStyle = '#4ade80';  // green-400
      ctx.lineWidth   = 1.8;
      ctx.lineJoin    = 'round';
      ctx.setLineDash([]);
      ctx.beginPath();

      tail.forEach(({ v }, i) => {
        const x = i * PX_PER_PT;
        const y = MIDLINE - v * MV_SCALE;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();

      // HR display (top-right)
      ctx.fillStyle  = '#94a3b8';
      ctx.font       = 'bold 11px monospace';
      ctx.textAlign  = 'right';
      ctx.fillText(`♥ ${hrRef.current} bpm`, W - 6, 14);

      rafRef.current = requestAnimationFrame(draw);
    }

    rafRef.current = requestAnimationFrame(draw);
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); };
  }, [dataStream, fs]);

  return (
    <div className="rounded-xl overflow-hidden border border-slate-700 bg-[#0d1117]">
      <div className="px-3 py-1.5 flex items-center justify-between border-b border-slate-800">
        <span className="text-[10px] text-slate-500 font-mono uppercase tracking-wider">
          Live signal preview
        </span>
        <span className="flex items-center gap-1.5 text-[10px] text-green-400">
          <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse inline-block" />
          LIVE
        </span>
      </div>
      <canvas
        ref={canvasRef}
        width={360}
        height={height}
        className="w-full"
        aria-label="Live ECG waveform preview"
      />
    </div>
  );
}
