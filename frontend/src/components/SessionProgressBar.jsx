/**
 * SessionProgressBar — 9-step progress dots connected by a line.
 *
 * Steps: Limb Leads, V1, V2, V3, V4, V5, V6, Confirm
 * (8 entries matching LEAD_SEQUENCE)
 */
import { LEAD_SEQUENCE, RECORDING_DURATION_S } from '../constants/electrodeConfig.js';

export default function SessionProgressBar({ currentStep }) {
  const recordingStepsRemaining = Math.max(
    0,
    LEAD_SEQUENCE.filter((s, i) => i > currentStep && s.id !== 'confirm').length,
  );
  const minutesLeft = Math.ceil((recordingStepsRemaining * RECORDING_DURATION_S) / 60);

  return (
    <div className="bg-slate-800/80 px-4 py-3 border-b border-slate-700">
      {/* Step dots + connecting line */}
      <div className="relative flex items-center justify-between">
        {/* Background line */}
        <div className="absolute left-0 right-0 top-1/2 -translate-y-1/2 h-0.5 bg-slate-700 z-0" />

        {/* Filled progress line */}
        <div
          className="absolute left-0 top-1/2 -translate-y-1/2 h-0.5 bg-brand-500 z-0 transition-all duration-500"
          style={{ width: `${(currentStep / (LEAD_SEQUENCE.length - 1)) * 100}%` }}
        />

        {/* Dots */}
        {LEAD_SEQUENCE.map((step, idx) => {
          const isDone    = idx < currentStep;
          const isCurrent = idx === currentStep;

          return (
            <div
              key={step.id}
              className="relative z-10 flex flex-col items-center"
              title={`${step.stepLabel}${step.duration ? ` (~${step.duration}s)` : ''}`}
            >
              <div
                className={[
                  'rounded-full border-2 transition-all duration-300 flex items-center justify-center',
                  isDone
                    ? 'w-4 h-4 bg-brand-500 border-brand-500'
                    : isCurrent
                    ? 'w-5 h-5 bg-brand-500 border-brand-300 ring-2 ring-brand-400/50 ring-offset-1 ring-offset-slate-900 animate-pulse'
                    : 'w-3.5 h-3.5 bg-slate-700 border-slate-600',
                ].join(' ')}
              >
                {isDone && (
                  <svg viewBox="0 0 8 8" className="w-2.5 h-2.5">
                    <path d="M1.5 4 L3 5.5 L6.5 2" stroke="white" strokeWidth="1.5" fill="none" strokeLinecap="round" />
                  </svg>
                )}
              </div>

              {/* Label below dot */}
              <span
                className={[
                  'absolute top-5 text-[8px] whitespace-nowrap',
                  isCurrent ? 'text-brand-400 font-bold' : 'text-slate-500',
                ].join(' ')}
              >
                {step.stepLabel}
              </span>
            </div>
          );
        })}
      </div>

      {/* Time remaining */}
      <p className="text-[10px] text-slate-500 mt-5 text-center">
        {currentStep >= LEAD_SEQUENCE.length - 1
          ? 'All leads recorded'
          : recordingStepsRemaining > 0
          ? `~${minutesLeft} min remaining · ${recordingStepsRemaining} step${recordingStepsRemaining !== 1 ? 's' : ''} left`
          : 'Almost done!'}
      </p>
    </div>
  );
}
