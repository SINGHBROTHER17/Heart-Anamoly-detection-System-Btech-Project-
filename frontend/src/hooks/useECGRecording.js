/**
 * useECGRecording — orchestrates a single 15-second recording step.
 *
 * In demo mode (no hardware connected) it generates synthetic ECG data.
 * In live mode it would read from a WebSocket / Web Serial stream —
 * the interface is kept the same so hardware integration is a drop-in swap.
 *
 * Returns:
 *   status      — 'idle' | 'recording' | 'done' | 'error'
 *   startRecording(stepConfig) — begin a recording
 *   recordedLeads — { [leadName]: { data: Float32Array, sqi: number } }
 */
import { useCallback, useRef, useState } from 'react';
import { generateSyntheticECG } from './useSyntheticWaveform.js';
import { SAMPLE_RATE, RECORDING_DURATION_S } from '../constants/electrodeConfig.js';

const DEMO_MODE = true; // flip to false when hardware is connected

export function useECGRecording() {
  const [status,        setStatus]        = useState('idle');
  const [recordedLeads, setRecordedLeads] = useState({});
  const timerRef = useRef(null);

  const startRecording = useCallback((stepConfig) => {
    setStatus('recording');
    setRecordedLeads({});

    if (DEMO_MODE) {
      // Simulate the recording duration then produce synthetic data
      timerRef.current = setTimeout(() => {
        const leads = {};
        stepConfig.leads.forEach((leadName) => {
          // Slight BPM variation per lead to mimic real measurement variance
          const bpm   = 70 + Math.round(Math.random() * 8);
          leads[leadName] = generateSyntheticECG(RECORDING_DURATION_S, bpm);
        });
        setRecordedLeads(leads);
        setStatus('done');
      }, RECORDING_DURATION_S * 1000);

    } else {
      // ── Live hardware recording ─────────────────────────────────────────
      // TODO: open WebSocket / Web Serial port, stream samples for
      // RECORDING_DURATION_S seconds, then call setRecordedLeads / setStatus.
      // The stepConfig.electrodes array tells you which electrode to record from.
      console.warn('Hardware recording not implemented — switch DEMO_MODE to false once hardware is ready.');
      setStatus('error');
    }
  }, []);

  const reset = useCallback(() => {
    clearTimeout(timerRef.current);
    setStatus('idle');
    setRecordedLeads({});
  }, []);

  return { status, recordedLeads, startRecording, reset };
}
