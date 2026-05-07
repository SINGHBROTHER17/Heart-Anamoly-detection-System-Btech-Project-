/**
 * ElectrodePlacementGuide — main 9-step electrode placement screen.
 *
 * Flow:
 *   welcome → step 0 (limb) → steps 1-6 (V1-V6) → step 7 (confirm) → analyze
 *
 * Each recording step:
 *   instructions phase → recording phase (timer + waveform) → review
 *
 * On completion: formats all lead data and calls analyzeJson(), then
 * navigates to /report/:id.
 */
import { useCallback, useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';

import TorsoSVGDiagram     from '../components/TorsoSVGDiagram.jsx';
import RecordingTimer       from '../components/RecordingTimer.jsx';
import LiveWaveformPreview  from '../components/LiveWaveformPreview.jsx';
import PlacementDetailCard  from '../components/PlacementDetailCard.jsx';
import SessionProgressBar   from '../components/SessionProgressBar.jsx';

import { useRecordingSession, ACTIONS } from '../context/RecordingSessionContext.jsx';
import { useECGRecording }              from '../hooks/useECGRecording.js';
import { LEAD_SEQUENCE, ELECTRODES }    from '../constants/electrodeConfig.js';
import { analyzeJson }                  from '../services/api.js';
import { useUserProfile }               from '../hooks/useUserProfile.js';

// SQI threshold below which we flag a lead as failed
const SQI_THRESHOLD = 0.60;

export default function ElectrodePlacementGuide() {
  const navigate = useNavigate();
  const [profile]       = useUserProfile();
  const { state, dispatch } = useRecordingSession();
  const { status: recStatus, recordedLeads, startRecording, reset: resetRec } = useECGRecording();

  const [phase,       setPhase]       = useState('welcome');   // 'welcome'|'instructions'|'recording'|'reviewing'|'confirm'|'submitting'
  const [submitError, setSubmitError] = useState(null);
  const [showHelp,    setShowHelp]    = useState(false);
  const [idleTimer,   setIdleTimer]   = useState(null);

  const currentStepCfg = LEAD_SEQUENCE[state.currentStep] ?? LEAD_SEQUENCE[LEAD_SEQUENCE.length - 1];
  const isConfirmStep  = currentStepCfg.id === 'confirm';

  // ── Start a fresh session when arriving at welcome ──────────────────────
  function handleBegin() {
    dispatch({ type: ACTIONS.START_SESSION });
    setPhase('instructions');
  }

  // ── Begin recording for the current step ────────────────────────────────
  function handleStartRecording() {
    clearIdleTimer();
    setPhase('recording');
    dispatch({ type: ACTIONS.BEGIN_LEAD_RECORDING });
    startRecording(currentStepCfg);
  }

  // ── When the recording hook produces data, save to context ──────────────
  useEffect(() => {
    if (recStatus !== 'done' || Object.keys(recordedLeads).length === 0) return;

    // Check SQI for each recorded lead
    const passed = {};
    const failed = [];
    for (const [leadName, { data, sqi }] of Object.entries(recordedLeads)) {
      if (sqi >= SQI_THRESHOLD) {
        passed[leadName] = { data, sqi };
      } else {
        failed.push(leadName);
      }
    }

    if (failed.length > 0) {
      dispatch({ type: ACTIONS.FAIL_LEAD, payload: { leads: failed } });
      setPhase('reviewing');
    } else {
      dispatch({ type: ACTIONS.COMPLETE_LEAD, payload: { leads: passed } });
      setPhase('reviewing');
    }
  }, [recStatus, recordedLeads, dispatch]);

  // ── Advance to next step after successful review ─────────────────────────
  function handleAdvance() {
    const anyFailed = currentStepCfg.leads.some((l) => state.failedLeads[l]);
    if (anyFailed) return;  // must retry first

    const nextStep = state.currentStep + 1;
    if (nextStep >= LEAD_SEQUENCE.length - 1) {
      // Move to confirm step
      dispatch({ type: ACTIONS.ADVANCE_STEP });
      setPhase('confirm');
    } else {
      dispatch({ type: ACTIONS.ADVANCE_STEP });
      resetRec();
      setPhase('instructions');
      // Idle timer: remind user if they haven't started in 60s
      setIdleTimer(setTimeout(() => {
        if (phase === 'instructions') alert('💬 Ready when you are — tap "Start Recording" to continue.');
      }, 60_000));
    }
  }

  // ── Retry a failed lead ─────────────────────────────────────────────────
  function handleRetry() {
    dispatch({ type: ACTIONS.RETRY_LEAD, payload: { leads: currentStepCfg.leads } });
    resetRec();
    setPhase('instructions');
  }

  // ── Submit all recorded leads to the API ────────────────────────────────
  async function handleSubmit() {
    setPhase('submitting');
    setSubmitError(null);
    dispatch({ type: ACTIONS.COMPLETE_SESSION });

    try {
      // Format: array of { lead_name, samples (plain array), sample_rate }
      const leads = Object.entries(state.completedLeads).map(([leadName, { data }]) => ({
        lead_name:   leadName,
        samples:     Array.from(data),
        sample_rate: 500,
      }));

      const result = await analyzeJson(leads, { patientId: profile.name || undefined });
      navigate(`/report/${result.report_id}`, { state: { report: result } });
    } catch (err) {
      setSubmitError(err.message || 'Analysis failed — please try again.');
      setPhase('confirm');
    }
  }

  function clearIdleTimer() {
    if (idleTimer) { clearTimeout(idleTimer); setIdleTimer(null); }
  }

  // Derived display data
  const confirmedLeadsMap = Object.fromEntries(
    Object.keys(state.completedLeads).map((k) => [k, true])
  );

  // ── Render ───────────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-slate-900 text-white flex flex-col">
      {/* ── Help modal ──────────────────────────────────────────────── */}
      {showHelp && (
        <HelpModal stepCfg={currentStepCfg} onClose={() => setShowHelp(false)} />
      )}

      {/* ── Welcome screen ──────────────────────────────────────────── */}
      {phase === 'welcome' && (
        <WelcomeScreen onBegin={handleBegin} />
      )}

      {/* ── Step screens (instructions / recording / reviewing) ──────── */}
      {phase !== 'welcome' && phase !== 'confirm' && phase !== 'submitting' && (
        <div className="flex flex-col flex-1">
          <SessionProgressBar currentStep={state.currentStep} />

          {/* Step header */}
          <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800">
            <h2 className="text-sm font-bold text-slate-100">{currentStepCfg.title}</h2>
            <button
              onClick={() => setShowHelp(true)}
              className="w-8 h-8 rounded-full bg-slate-700 flex items-center justify-center text-slate-300 text-sm font-bold"
              aria-label="Help — show detailed instructions"
            >
              ?
            </button>
          </div>

          {/* Scrollable content */}
          <div className="flex-1 overflow-y-auto px-4 py-4 flex flex-col gap-4">
            {/* Torso diagram */}
            <TorsoSVGDiagram
              activeElectrodes={phase === 'instructions' || phase === 'recording' ? currentStepCfg.electrodes : []}
              confirmedLeads={confirmedLeadsMap}
              failedLeads={state.failedLeads}
            />

            {/* Phase-specific content */}
            {phase === 'instructions' && (
              <>
                <PlacementDetailCard stepConfig={currentStepCfg} />
                <button
                  className="btn-primary w-full py-4 text-base"
                  onClick={handleStartRecording}
                >
                  {currentStepCfg.id === 'limb' || currentStepCfg.id === 'L1' || currentStepCfg.id === 'L2'
                    ? `✓ Electrodes attached — Start ${currentStepCfg.stepLabel} Recording`
                    : `Start Recording ${currentStepCfg.stepLabel}`}
                </button>
              </>
            )}

            {phase === 'recording' && (
              <RecordingPhase
                stepCfg={currentStepCfg}
                onComplete={() => { /* handled by useEffect watching recStatus */ }}
              />
            )}

            {phase === 'reviewing' && (
              <ReviewPhase
                stepCfg={currentStepCfg}
                completedLeads={state.completedLeads}
                failedLeads={state.failedLeads}
                onRetry={handleRetry}
                onAdvance={handleAdvance}
              />
            )}
          </div>
        </div>
      )}

      {/* ── Confirm / review all leads ───────────────────────────────── */}
      {phase === 'confirm' && (
        <ConfirmScreen
          leads={state.completedLeads}
          failedLeads={state.failedLeads}
          submitError={submitError}
          onRetry={(step) => {
            dispatch({ type: ACTIONS.RESET_SESSION });
            setPhase('welcome');
          }}
          onSubmit={handleSubmit}
        />
      )}

      {/* ── Submitting ───────────────────────────────────────────────── */}
      {phase === 'submitting' && (
        <div className="flex-1 flex flex-col items-center justify-center gap-6 px-6">
          <div className="w-16 h-16 rounded-full border-4 border-brand-500 border-t-transparent animate-spin" />
          <p className="text-lg font-semibold text-slate-200">Analyzing ECG…</p>
          <p className="text-sm text-slate-400 text-center">
            Running the AI model on your 12-lead recording. This takes a few seconds.
          </p>
        </div>
      )}
    </div>
  );
}

// ── Welcome screen ───────────────────────────────────────────────────────────
function WelcomeScreen({ onBegin }) {
  return (
    <div className="flex-1 flex flex-col px-6 py-8 gap-6">
      <div className="flex flex-col items-center gap-4 pt-6">
        <div className="w-20 h-20 rounded-full bg-brand-500/20 flex items-center justify-center text-4xl">
          🫀
        </div>
        <h1 className="text-2xl font-bold text-center">Electrode Placement Guide</h1>
        <p className="text-sm text-slate-400 text-center leading-relaxed">
          We'll walk you through placing all 10 electrodes step-by-step to record a
          complete 12-lead ECG. Each step takes only 15 seconds.
        </p>
      </div>

      {/* What you need */}
      <div className="card bg-slate-800 border-slate-700">
        <p className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">
          What you need
        </p>
        <ul className="flex flex-col gap-2 text-sm text-slate-300">
          {[
            '10-electrode ECG cable (4 limb clips + 6 chest stickers)',
            'Clean, dry skin — wipe with a dry cloth if needed',
            'Quiet room, sit or lie still during each 15-second recording',
            'Approx. 5–10 minutes total',
          ].map((item) => (
            <li key={item} className="flex items-start gap-2">
              <span className="text-brand-400 mt-0.5">✓</span>
              {item}
            </li>
          ))}
        </ul>
      </div>

      {/* Step overview */}
      <div className="card bg-slate-800 border-slate-700">
        <p className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">
          Recording steps
        </p>
        <ol className="flex flex-col gap-2 text-sm text-slate-300">
          {LEAD_SEQUENCE.filter(s => s.id !== 'confirm').map((s, i) => (
            <li key={s.id} className="flex items-center gap-2">
              <span className="w-5 h-5 rounded-full bg-slate-700 text-[10px] font-bold flex items-center justify-center text-slate-400">
                {i + 1}
              </span>
              <span>{s.stepLabel}</span>
              {s.duration > 0 && (
                <span className="ml-auto text-[10px] text-slate-500">{s.duration}s</span>
              )}
            </li>
          ))}
        </ol>
      </div>

      <div className="mt-auto flex flex-col gap-3">
        <button className="btn-primary w-full py-4 text-base" onClick={onBegin}>
          Begin Electrode Placement
        </button>
        <p className="text-[10px] text-slate-500 text-center">
          Demo mode active — uses synthetic ECG data (no hardware required)
        </p>
      </div>
    </div>
  );
}

// ── Recording phase (timer + waveform) ──────────────────────────────────────
function RecordingPhase({ stepCfg }) {
  return (
    <div className="flex flex-col gap-4">
      <div className="card bg-slate-800 border-slate-700 flex flex-col items-center gap-1 py-4">
        <p className="text-xs text-slate-400 mb-3">
          Hold still and breathe normally during the recording
        </p>
        <RecordingTimer duration={stepCfg.duration} autoStart />
      </div>
      <LiveWaveformPreview height={110} />
    </div>
  );
}

// ── Review phase (after recording) ──────────────────────────────────────────
function ReviewPhase({ stepCfg, completedLeads, failedLeads, onRetry, onAdvance }) {
  const anyFailed = stepCfg.leads.some((l) => failedLeads[l]);
  const allPassed = stepCfg.leads.every((l) => completedLeads[l]);

  return (
    <div className="flex flex-col gap-4">
      <div className="card bg-slate-800 border-slate-700">
        <p className="text-sm font-semibold text-slate-200 mb-3">Signal quality results:</p>
        {stepCfg.leads.map((lead) => {
          const rec  = completedLeads[lead];
          const fail = failedLeads[lead];
          const sqi  = rec?.sqi ?? 0;
          const pct  = Math.round(sqi * 100);

          return (
            <div key={lead} className="flex items-center gap-3 mb-2">
              <span
                className={`w-8 h-8 rounded-lg flex items-center justify-center text-xs font-bold ${
                  fail ? 'bg-red-900/50 text-red-400' : 'bg-green-900/50 text-green-400'
                }`}
              >
                {lead}
              </span>
              <div className="flex-1">
                <div className="h-1.5 rounded-full bg-slate-700 overflow-hidden">
                  <div
                    className={`h-full rounded-full ${fail ? 'bg-red-500' : 'bg-green-500'}`}
                    style={{ width: `${pct}%`, transition: 'width 0.5s ease' }}
                  />
                </div>
              </div>
              <span className={`text-xs font-mono font-bold ${fail ? 'text-red-400' : 'text-green-400'}`}>
                {pct}%
              </span>
              <span>{fail ? '✗' : '✓'}</span>
            </div>
          );
        })}

        {anyFailed && (
          <div className="mt-3 bg-red-900/30 border border-red-700 rounded-xl p-3">
            <p className="text-xs text-red-400 font-semibold mb-1">Signal too weak</p>
            <p className="text-xs text-red-300">
              Check electrode contact — make sure the skin is clean and dry,
              the clip/sticker is firmly attached, and you're not moving.
            </p>
          </div>
        )}

        {allPassed && (
          <div className="mt-3 bg-green-900/30 border border-green-700 rounded-xl p-3">
            <p className="text-xs text-green-400 font-semibold">
              ✓ Good signal quality — ready to continue!
            </p>
          </div>
        )}
      </div>

      {anyFailed ? (
        <button className="btn-primary w-full bg-amber-600 hover:bg-amber-700 py-4" onClick={onRetry}>
          Re-place electrode and retry
        </button>
      ) : (
        <button className="btn-primary w-full py-4" onClick={onAdvance}>
          {stepCfg.id === 'V6' ? 'Review all leads →' : `Continue to ${LEAD_SEQUENCE[LEAD_SEQUENCE.findIndex(s => s.id === stepCfg.id) + 1]?.stepLabel ?? 'next step'} →`}
        </button>
      )}
    </div>
  );
}

// ── Confirm screen — review all 12 leads ────────────────────────────────────
function ConfirmScreen({ leads, failedLeads, submitError, onRetry, onSubmit }) {
  const allGood = Object.keys(failedLeads).length === 0;
  const LEAD_NAMES = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6'];

  return (
    <div className="flex flex-col flex-1 px-4 py-5 gap-4">
      <div>
        <h2 className="text-xl font-bold">Review All Leads</h2>
        <p className="text-sm text-slate-400 mt-1">
          All 12 leads must show good signal quality before analysis.
        </p>
      </div>

      {/* Per-lead grid */}
      <div className="grid grid-cols-3 gap-2">
        {LEAD_NAMES.map((lead) => {
          const rec   = leads[lead];
          const fail  = failedLeads[lead];
          const pct   = rec ? Math.round(rec.sqi * 100) : 0;
          const ready = !!rec && !fail;

          return (
            <div
              key={lead}
              className={`rounded-xl p-3 flex flex-col items-center gap-1 border ${
                ready ? 'bg-green-900/30 border-green-700' : 'bg-slate-800 border-slate-700'
              }`}
            >
              <span className="text-xs font-bold text-slate-300">{lead}</span>
              <span className={`text-lg ${ready ? 'text-green-400' : 'text-slate-600'}`}>
                {ready ? '✓' : '–'}
              </span>
              {rec && (
                <span className="text-[10px] font-mono text-slate-400">{pct}%</span>
              )}
            </div>
          );
        })}
      </div>

      {submitError && (
        <div className="bg-red-900/40 border border-red-700 rounded-xl p-3 text-sm text-red-300">
          ⚠️ {submitError}
        </div>
      )}

      <div className="mt-auto flex flex-col gap-3">
        <button
          className="btn-primary w-full py-4 text-base"
          disabled={!allGood}
          onClick={onSubmit}
        >
          {allGood ? 'Analyze ECG →' : 'Fix failed leads first'}
        </button>
        <button className="btn-secondary w-full py-3" onClick={onRetry}>
          Start over
        </button>
        <p className="text-[10px] text-slate-500 text-center">
          aVR, aVL, aVF are computed automatically from limb leads — no separate recording needed.
        </p>
      </div>
    </div>
  );
}

// ── Help modal ───────────────────────────────────────────────────────────────
function HelpModal({ stepCfg, onClose }) {
  return (
    <div
      className="fixed inset-0 z-50 bg-black/80 flex flex-col"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label="Electrode placement help"
    >
      <div
        className="mt-auto bg-slate-900 rounded-t-3xl p-6 flex flex-col gap-5 max-h-[85vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-bold">Placement Help</h2>
          <button onClick={onClose} className="text-slate-400 text-xl leading-none">✕</button>
        </div>

        <p className="text-sm text-slate-300 leading-relaxed">
          {stepCfg.id === 'limb'
            ? 'Limb electrodes go on your wrists and ankles. They do not need to be on muscle tissue — the bony areas near joints give the cleanest signal.'
            : stepCfg.id === 'L2'
            ? 'Lead II measures the heart from the right arm down to the lower-left waist. Keep RA + LA attached and add the third electrode on the left side of the abdomen, just below the rib cage.'
            : stepCfg.id === 'L1'
            ? 'Lead I measures the heart between the right arm and the left arm. Remove the waist electrode — only RA and LA stay attached.'
            : ELECTRODES[stepCfg.electrodes[0]]?.instruction}
        </p>

        <div className="bg-slate-800 rounded-2xl p-4">
          <p className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Tips</p>
          <ul className="flex flex-col gap-2 text-sm text-slate-300">
            <li>• Clean the skin with a dry cloth before placing chest stickers.</li>
            <li>• Press firmly for 10 seconds after applying each sticker.</li>
            <li>• Stay as still as possible during the 15-second recording.</li>
            <li>• Breathe normally — don't hold your breath.</li>
          </ul>
        </div>

        <div className="bg-amber-900/30 border border-amber-700 rounded-2xl p-4">
          <p className="text-xs font-bold text-amber-400 uppercase tracking-wider mb-1">
            Video guide
          </p>
          <p className="text-sm text-amber-300">
            Video instructions will be available in a future update. For now, follow the written
            and diagram guidance on each step.
          </p>
        </div>

        <button className="btn-primary w-full py-3" onClick={onClose}>
          Got it
        </button>
      </div>
    </div>
  );
}
