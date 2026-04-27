/**
 * Client-side synthetic ECG generator — powers the "Simulate Demo Recording"
 * button so users can try the app without uploading real data.
 *
 * Builds 10 s of 12-lead ECG at 500 Hz using Gaussian pulses for PQRST.
 */

const LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                    'V1', 'V2', 'V3', 'V4', 'V5', 'V6'];

export function generateDemoSignal({
  durationSec = 10,
  fs = 500,
  hrBpm = 72,
  noiseSigma = 0.02,
  seed = 42,
} = {}) {
  const N = Math.floor(durationSec * fs);
  const rng = mulberry32(seed);

  const t = new Float32Array(N);
  for (let i = 0; i < N; i++) t[i] = i / fs;

  // Single beat template, then replicate at the target heart rate.
  const base = new Float32Array(N);
  const beatInterval = 60 / hrBpm;
  for (let bStart = 0.3; bStart < durationSec - 0.3; bStart += beatInterval) {
    for (let i = 0; i < N; i++) {
      const dt = t[i] - bStart;
      // P wave
      base[i] += 0.15 * Math.exp(-(dt * dt) / (0.04 * 0.04));
      // Q
      const dQ = t[i] - (bStart + 0.15);
      base[i] -= 0.10 * Math.exp(-(dQ * dQ) / (0.01 * 0.01));
      // R
      const dR = t[i] - (bStart + 0.17);
      base[i] += 1.00 * Math.exp(-(dR * dR) / (0.012 * 0.012));
      // S
      const dS = t[i] - (bStart + 0.20);
      base[i] -= 0.20 * Math.exp(-(dS * dS) / (0.015 * 0.015));
      // T
      const dT = t[i] - (bStart + 0.35);
      base[i] += 0.30 * Math.exp(-(dT * dT) / (0.04 * 0.04));
    }
  }

  // Add noise and baseline drift for realism.
  for (let i = 0; i < N; i++) {
    base[i] += noiseSigma * (rng() * 2 - 1);
    base[i] += 0.05 * Math.sin(2 * Math.PI * 0.3 * t[i]);
  }

  // 12 leads with mild amplitude variation (fake lead-specific morphology).
  const leads = LEAD_NAMES.map((name, idx) => {
    const amp = 0.85 + (idx / (LEAD_NAMES.length - 1)) * 0.3;
    const samples = new Array(N);
    for (let i = 0; i < N; i++) samples[i] = base[i] * amp;
    return { lead_name: name, samples, sample_rate: fs };
  });
  return leads;
}

/** Simple PRNG so the demo produces the same waveform on every click. */
function mulberry32(a) {
  return function () {
    let t = (a += 0x6D2B79F5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Convert per-lead JSON to a CSV string (for the CSV upload preview). */
export function leadsToCsv(leads) {
  const N = leads[0].samples.length;
  const header = leads.map((l) => l.lead_name).join(',');
  const rows = [];
  for (let i = 0; i < N; i++) {
    rows.push(leads.map((l) => l.samples[i].toFixed(4)).join(','));
  }
  return [header, ...rows].join('\n');
}
