import { CONDITION_DESCRIPTIONS, TIERS } from '../utils/riskTiers.js';

export default function ConditionCard({ result }) {
  const { condition, confidence, risk_tier, tier_label } = result;
  const tierMeta = TIERS[risk_tier] || TIERS.none;
  const pct = Math.round(confidence * 100);
  const description = CONDITION_DESCRIPTIONS[condition] || '';

  return (
    <div className="card flex flex-col gap-3">
      <div className="flex items-start justify-between gap-2">
        <h3 className="font-semibold text-sm leading-tight">{condition}</h3>
        <span className={`badge ${tierMeta.color} ${tierMeta.text} ring-transparent capitalize`}>
          {risk_tier}
        </span>
      </div>

      {/* Confidence bar */}
      <div>
        <div className="flex items-center justify-between text-xs mb-1">
          <span className="text-slate-500 dark:text-slate-400">Confidence</span>
          <span className="font-mono font-medium">{pct}%</span>
        </div>
        <div className="h-2 w-full rounded-full bg-slate-200 dark:bg-slate-700 overflow-hidden">
          <div
            className={`h-full ${tierMeta.color} transition-all duration-500`}
            style={{ width: `${pct}%` }}
            role="progressbar"
            aria-valuenow={pct} aria-valuemin="0" aria-valuemax="100"
          />
        </div>
      </div>

      <p className="text-xs text-slate-600 dark:text-slate-400 leading-snug">
        {description}
      </p>

      <p className="text-xs font-medium text-slate-700 dark:text-slate-300">
        {tier_label}
      </p>
    </div>
  );
}
