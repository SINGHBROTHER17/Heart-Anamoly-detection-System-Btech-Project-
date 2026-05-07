/**
 * RecordingSessionContext — global state for the 9-step electrode recording
 * session.  Uses useReducer for predictable state transitions.
 *
 * aVR, aVL, aVF are computed automatically from I, II, III and never require
 * a separate recording step.
 *
 * State is persisted to localStorage so users can resume if the page reloads
 * mid-session.
 */
import { createContext, useContext, useEffect, useReducer } from 'react';
import { v4 as uuidv4 } from 'uuid';

// ── Helper: derive augmented leads from limb leads ──────────────────────────
function computeAugmentedLeads(I, II) {
  // III  = II − I
  // aVR  = −(I + II) / 2
  // aVL  = I − II / 2
  // aVF  = II − I / 2
  const len = I.length;

  const III = new Float32Array(len);
  const aVR = new Float32Array(len);
  const aVL = new Float32Array(len);
  const aVF = new Float32Array(len);

  for (let i = 0; i < len; i++) {
    III[i] = II[i] - I[i];
    aVR[i] = -(I[i] + II[i]) / 2;
    aVL[i] = I[i] - II[i] / 2;
    aVF[i] = II[i] - I[i] / 2;
  }
  return { III, aVR, aVL, aVF };
}

// ── Initial state ────────────────────────────────────────────────────────────
const INITIAL_STATE = {
  sessionId: null,
  currentStep: 0,           // 0-5=V1-V6, 6=L2, 7=L1, 8=confirm
  completedLeads: {},        // { "I": { data: Float32Array, sqi: 0.91 }, ... }
  failedLeads: {},           // { "V3": true }
  sessionStatus: 'idle',    // 'idle'|'instructions'|'recording'|'reviewing'|'complete'
  overallSQI: null,
  startedAt: null,
};

// ── Actions ──────────────────────────────────────────────────────────────────
export const ACTIONS = {
  START_SESSION:       'START_SESSION',
  BEGIN_INSTRUCTIONS:  'BEGIN_INSTRUCTIONS',
  BEGIN_LEAD_RECORDING:'BEGIN_LEAD_RECORDING',
  COMPLETE_LEAD:       'COMPLETE_LEAD',
  FAIL_LEAD:           'FAIL_LEAD',
  RETRY_LEAD:          'RETRY_LEAD',
  ADVANCE_STEP:        'ADVANCE_STEP',
  COMPLETE_SESSION:    'COMPLETE_SESSION',
  RESET_SESSION:       'RESET_SESSION',
};

// ── Reducer ──────────────────────────────────────────────────────────────────
function reducer(state, action) {
  switch (action.type) {
    case ACTIONS.START_SESSION:
      return {
        ...INITIAL_STATE,
        sessionId: uuidv4(),
        sessionStatus: 'instructions',
        startedAt: Date.now(),
      };

    case ACTIONS.BEGIN_LEAD_RECORDING:
      return { ...state, sessionStatus: 'recording' };

    case ACTIONS.COMPLETE_LEAD: {
      // action.payload: { leads: { "I": { data, sqi }, "II": {...}, ... } }
      const newCompleted = { ...state.completedLeads, ...action.payload.leads };

      // Auto-compute augmented limb leads once I and II are present
      if (newCompleted['I'] && newCompleted['II'] && !newCompleted['aVR']) {
        const aug = computeAugmentedLeads(
          newCompleted['I'].data,
          newCompleted['II'].data,
        );
        newCompleted['III'] = { data: aug.III, sqi: (newCompleted['I'].sqi + newCompleted['II'].sqi) / 2 };
        newCompleted['aVR'] = { data: aug.aVR, sqi: newCompleted['I'].sqi };
        newCompleted['aVL'] = { data: aug.aVL, sqi: newCompleted['I'].sqi };
        newCompleted['aVF'] = { data: aug.aVF, sqi: newCompleted['II'].sqi };
      }

      const newFailed = { ...state.failedLeads };
      Object.keys(action.payload.leads).forEach((k) => delete newFailed[k]);

      return {
        ...state,
        completedLeads: newCompleted,
        failedLeads: newFailed,
        sessionStatus: 'reviewing',
      };
    }

    case ACTIONS.FAIL_LEAD: {
      // action.payload: { leads: string[] }
      const newFailed = { ...state.failedLeads };
      action.payload.leads.forEach((l) => { newFailed[l] = true; });
      return { ...state, failedLeads: newFailed, sessionStatus: 'reviewing' };
    }

    case ACTIONS.RETRY_LEAD: {
      const newFailed = { ...state.failedLeads };
      action.payload.leads.forEach((l) => delete newFailed[l]);
      return { ...state, failedLeads: newFailed, sessionStatus: 'instructions' };
    }

    case ACTIONS.ADVANCE_STEP:
      return {
        ...state,
        currentStep: state.currentStep + 1,
        sessionStatus: 'instructions',
      };

    case ACTIONS.COMPLETE_SESSION: {
      const sqis = Object.values(state.completedLeads).map((l) => l.sqi);
      const overall = sqis.length ? sqis.reduce((a, b) => a + b, 0) / sqis.length : 0;
      return { ...state, sessionStatus: 'complete', overallSQI: overall };
    }

    case ACTIONS.RESET_SESSION:
      return { ...INITIAL_STATE };

    default:
      return state;
  }
}

// ── Persistence helpers ──────────────────────────────────────────────────────
const STORAGE_KEY = 'ecg_recording_session';

function loadPersistedState() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return INITIAL_STATE;
    const parsed = JSON.parse(raw);
    // Float32Arrays don't survive JSON — reconstruct from plain arrays
    const completed = {};
    for (const [k, v] of Object.entries(parsed.completedLeads || {})) {
      completed[k] = { data: new Float32Array(v.data), sqi: v.sqi };
    }
    return { ...parsed, completedLeads: completed };
  } catch {
    return INITIAL_STATE;
  }
}

function persistState(state) {
  try {
    const serializable = {
      ...state,
      completedLeads: Object.fromEntries(
        Object.entries(state.completedLeads).map(([k, v]) => [
          k, { data: Array.from(v.data), sqi: v.sqi },
        ])
      ),
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(serializable));
  } catch { /* quota exceeded — ignore */ }
}

// ── Context + Provider ───────────────────────────────────────────────────────
const RecordingSessionContext = createContext(null);

export function RecordingSessionProvider({ children }) {
  const [state, dispatch] = useReducer(reducer, undefined, loadPersistedState);

  // Persist on every state change
  useEffect(() => {
    if (state.sessionStatus !== 'idle') persistState(state);
  }, [state]);

  return (
    <RecordingSessionContext.Provider value={{ state, dispatch }}>
      {children}
    </RecordingSessionContext.Provider>
  );
}

export function useRecordingSession() {
  const ctx = useContext(RecordingSessionContext);
  if (!ctx) throw new Error('useRecordingSession must be used inside RecordingSessionProvider');
  return ctx;
}
