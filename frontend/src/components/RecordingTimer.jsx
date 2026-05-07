/**
 * RecordingTimer — circular SVG countdown ring (iOS-style).
 *
 * - 15-second countdown (configurable via `duration` prop)
 * - Ring depletes clockwise as time passes
 * - Color transitions: green → yellow → red in last 3 seconds
 * - Plays a subtle Web Audio API chime on completion
 * - Calls onComplete() when finished
 */
import { useEffect, useRef, useState } from 'react';

const RADIUS = 52;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

function playChime() {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const osc  = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.type = 'sine';
    osc.frequency.setValueAtTime(880, ctx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(1174, ctx.currentTime + 0.15);
    gain.gain.setValueAtTime(0.25, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.5);
    osc.start(ctx.currentTime);
    osc.stop(ctx.currentTime + 0.5);
  } catch { /* Web Audio not available — silent */ }
}

export default function RecordingTimer({
  duration   = 15,   // seconds
  onComplete = null,
  autoStart  = false,
}) {
  const [secondsLeft, setSecondsLeft] = useState(duration);
  const [running,     setRunning]     = useState(autoStart);
  const [done,        setDone]        = useState(false);
  const intervalRef = useRef(null);

  // Allow parent to re-trigger by changing autoStart
  useEffect(() => {
    if (autoStart) start();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoStart]);

  function start() {
    setSecondsLeft(duration);
    setDone(false);
    setRunning(true);
  }

  useEffect(() => {
    if (!running) return;
    intervalRef.current = setInterval(() => {
      setSecondsLeft((prev) => {
        if (prev <= 1) {
          clearInterval(intervalRef.current);
          setRunning(false);
          setDone(true);
          playChime();
          onComplete?.();
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
    return () => clearInterval(intervalRef.current);
  }, [running, duration, onComplete]);

  // Fraction of ring remaining
  const fraction = secondsLeft / duration;
  const dashOffset = CIRCUMFERENCE * (1 - fraction);

  // Color: green → yellow (below 6s) → red (below 3s)
  const ringColor = secondsLeft <= 3 ? '#FC8181'     // red-400
                  : secondsLeft <= 6 ? '#F6AD55'     // orange-300
                  : '#68D391';                       // green-300

  return (
    <div className="flex flex-col items-center gap-4">
      <div className="relative w-36 h-36">
        <svg viewBox="0 0 120 120" className="w-full h-full -rotate-90">
          {/* Track */}
          <circle cx="60" cy="60" r={RADIUS}
                  fill="none" stroke="#334155" strokeWidth="8" />
          {/* Progress ring */}
          <circle
            cx="60" cy="60" r={RADIUS}
            fill="none"
            stroke={done ? '#68D391' : ringColor}
            strokeWidth="8"
            strokeLinecap="round"
            strokeDasharray={CIRCUMFERENCE}
            strokeDashoffset={done ? 0 : dashOffset}
            style={{ transition: 'stroke-dashoffset 0.9s linear, stroke 0.4s ease' }}
          />
        </svg>

        {/* Center content */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          {done ? (
            <div className="flex flex-col items-center gap-1 animate-[confirmed-pop_0.35s_ease-out_forwards]">
              <span className="text-3xl">✓</span>
              <span className="text-xs text-green-400 font-semibold">Done!</span>
            </div>
          ) : (
            <>
              <span className="text-4xl font-bold tabular-nums" style={{ color: ringColor }}>
                {secondsLeft}
              </span>
              <span className="text-xs text-slate-400">sec</span>
            </>
          )}
        </div>
      </div>

      {/* Status label */}
      {!done && (
        <p className="text-sm text-slate-300 text-center">
          {running ? (
            <span className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse inline-block" />
              Recording…
            </span>
          ) : (
            'Ready to record'
          )}
        </p>
      )}
      {done && (
        <p className="text-sm text-green-400 font-semibold">Recording complete ✓</p>
      )}
    </div>
  );
}
