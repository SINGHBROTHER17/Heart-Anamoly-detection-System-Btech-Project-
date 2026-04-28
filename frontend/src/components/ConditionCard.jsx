import { CONDITION_DESCRIPTIONS } from '../utils/riskTiers.js';

const TIER_STYLES = {
  none:     { dot: 'bg-green-500',  badge: 'bg-green-100 text-green-700',  bar: '#22c55e' },
  possible: { dot: 'bg-amber-400',  badge: 'bg-amber-100 text-amber-700',  bar: '#f59e0b' },
  likely:   { dot: 'bg-orange-500', badge: 'bg-orange-100 text-orange-700', bar: '#f97316' },
  high:     { dot: 'bg-red-500',    badge: 'bg-red-100 text-red-700',       bar: '#ef4444' },
};

const TIER_LABELS = {
  none:     'Normal',
  possible: 'Possible',
  likely:   'Likely',
  high:     'High risk',
};

export default function ConditionCard({ result }) {
  const { condition, confidence, risk_tier, tier_label } = result;
  const style = TIER_STYLES[risk_tier] ?? TIER_STYLES.none;
  const pct = Math.round(confidence * 100);
  const description = CONDITION_DESCRIPTIONS[condition] ?? '';

  return (
    <div className="card flex flex-col gap-2.5">
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className={`w-2.5 h-2.5 rounded-full flex-shrink-0 mt-0.5 ${style.dot}`} />
          <h3 className="font-semibold text-sm leading-tight text-slate-800">{condition}</h3>
        </div>
        <span className={`badge text-xs font-semibold px-2 py-0.5 rounded-full ${style.badge}`}>
          {TIER_LABELS[risk_tier] ?? risk_tier}
        </span>
      </div>

      {/* Confidence bar */}
      <div>
        <div className="flex justify-between text-xs text-slate-400 mb-1">
          <span>Confidence</span>
          <span className="font-semibold text-slate-600">{pct}%</span>
        </div>
        <div className="h-1.5 w-full rounded-full bg-slate-100 overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-500"
            style={{ width: `${pct}%`, backgroundColor: style.bar }}
          />
        </div>
      </div>

      <p className="text-xs text-slate-500 leading-snug">{description}</p>
    </div>
  );
}
