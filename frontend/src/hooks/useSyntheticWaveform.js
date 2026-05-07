/**
 * useSyntheticWaveform — generates a realistic synthetic ECG waveform
 * buffer that can be treated as "recorded" data during demo mode.
 *
 * PQRST morphology at 500 Hz, 72 bpm baseline with slight jitter.
 * Returns a Float32Array of (durationSeconds × sampleRate) samples.
 */
import { SAMPLE_RATE } from '../constants/electrodeConfig.js';

function gaussian(t, center, sigma, amp) {
  return amp * Math.exp(-((t - center) ** 2) / (2 * sigma ** 2));
}

/** Generate one beat waveform (returns array of floats). */
function oneBeat(bpm = 72, noise = 0.02) {
  const fs   = SAMPLE_RATE;
  const rrMs = (60 / bpm) * 1000;
  const n    = Math.round((rrMs / 1000) * fs);
  const out  = new Float32Array(n);

  for (let i = 0; i < n; i++) {
    const t = i / fs;
    const P  = gaussian(t, 0.08,  0.025,  0.15);
    const Q  = gaussian(t, 0.155, 0.008, -0.10);
    const R  = gaussian(t, 0.17,  0.007,  1.00);
    const S  = gaussian(t, 0.187, 0.008, -0.15);
    const T  = gaussian(t, 0.35,  0.05,   0.30);
    const n_ = (Math.random() - 0.5) * 2 * noise;
    out[i]   = P + Q + R + S + T + n_;
  }
  return out;
}

/**
 * Generate a multi-beat ECG Float32Array.
 *
 * @param {number} durationS  — recording duration in seconds
 * @param {number} baseBpm    — baseline heart rate (default 72)
 * @returns {{ data: Float32Array, sqi: number }}
 */
export function generateSyntheticECG(durationS = 15, baseBpm = 72) {
  const totalSamples = durationS * SAMPLE_RATE;
  const buf = new Float32Array(totalSamples);
  let offset = 0;

  while (offset < totalSamples) {
    // Slight BPM jitter per beat (±4 bpm)
    const bpm  = baseBpm + (Math.random() - 0.5) * 8;
    const beat = oneBeat(bpm);
    const len  = Math.min(beat.length, totalSamples - offset);
    buf.set(beat.subarray(0, len), offset);
    offset += beat.length;
  }

  // SQI: randomize between 0.85 and 0.97 (demo always "good")
  const sqi = 0.85 + Math.random() * 0.12;

  return { data: buf, sqi };
}

/**
 * Compute a simple SQI estimate from a Float32Array by checking for
 * clipping, flatline, and noise-to-signal ratio.
 *
 * Returns a value between 0 (unusable) and 1 (perfect).
 */
export function computeSQI(data) {
  if (!data || data.length === 0) return 0;

  const n    = data.length;
  let   sum  = 0;
  let   sumSq = 0;
  let   clipped = 0;
  let   flat   = 0;

  for (let i = 0; i < n; i++) {
    sum   += data[i];
    sumSq += data[i] ** 2;
    if (Math.abs(data[i]) > 2.5)       clipped++;
    if (i > 0 && data[i] === data[i-1]) flat++;
  }

  const mean     = sum / n;
  const variance = sumSq / n - mean ** 2;
  const std      = Math.sqrt(Math.max(0, variance));

  const clipFrac = clipped / n;
  const flatFrac = flat    / n;

  // Penalise clipping and flatline
  let sqi = 1 - (clipFrac * 3 + flatFrac * 2);
  // Penalise very low amplitude (likely disconnected lead)
  if (std < 0.05) sqi -= 0.4;
  // Penalise very high noise (std > 0.8 without real signal)
  if (std > 0.8 && mean < 0.1) sqi -= 0.3;

  return Math.max(0, Math.min(1, sqi));
}
