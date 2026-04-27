/**
 * Risk tier metadata: colors and plain-English descriptions per condition.
 * Keep in lockstep with api/app/schemas.py confidence_to_tier().
 */

export const TIERS = {
  none:     { color: 'bg-risk-none',     ring: 'ring-risk-none/30',     text: 'text-white' },
  possible: { color: 'bg-risk-possible', ring: 'ring-risk-possible/30', text: 'text-white' },
  likely:   { color: 'bg-risk-likely',   ring: 'ring-risk-likely/30',   text: 'text-white' },
  high:     { color: 'bg-risk-high',     ring: 'ring-risk-high/30',     text: 'text-white' },
};

/** Short plain-English description of each condition, shown on result cards. */
export const CONDITION_DESCRIPTIONS = {
  'Normal Sinus Rhythm':
    'A healthy, regular heartbeat originating from the sinoatrial node.',
  'Atrial Fibrillation':
    'An irregular, often rapid rhythm from the upper chambers; raises stroke risk.',
  'ST Elevation':
    'Abnormal ST segment elevation, often associated with an acute heart attack (STEMI).',
  'Left Bundle Branch Block':
    'A delay or block in the electrical pathway to the left ventricle.',
  'Right Bundle Branch Block':
    'A delay or block in the electrical pathway to the right ventricle.',
  'Left Ventricular Hypertrophy':
    'Thickening of the left ventricle wall, usually caused by long-term high blood pressure.',
  'Bradycardia':
    'A slower-than-normal heart rate (below 60 BPM at rest).',
  'Tachycardia':
    'A faster-than-normal heart rate (above 100 BPM at rest).',
  'First Degree AV Block':
    'A prolonged delay in conduction between the atria and ventricles; usually benign.',
  'Premature Ventricular Contraction':
    'Extra heartbeats originating from the ventricles; common and often harmless.',
};

/** SQI → quality-badge color. */
export function sqiColor(sqi) {
  if (sqi >= 0.8) return 'bg-emerald-500';
  if (sqi >= 0.6) return 'bg-amber-500';
  return 'bg-rose-500';
}

export function sqiLabel(sqi) {
  if (sqi >= 0.8) return 'Good';
  if (sqi >= 0.6) return 'Acceptable';
  return 'Poor';
}

/** Format an ISO timestamp to a short human-readable string. */
export function fmtTimestamp(iso) {
  const d = new Date(iso);
  return d.toLocaleString(undefined, {
    year: 'numeric', month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit',
  });
}
