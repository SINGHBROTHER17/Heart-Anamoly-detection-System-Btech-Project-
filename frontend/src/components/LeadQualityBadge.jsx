import { sqiColor, sqiLabel } from '../utils/riskTiers.js';

export default function LeadQualityBadge({ lead, sqi, flags = [] }) {
  return (
    <div className="flex items-center justify-between gap-2 px-3 py-2 rounded-lg border border-slate-200 dark:border-slate-700">
      <span className="font-mono text-sm font-medium w-10">{lead}</span>
      <div className="flex items-center gap-2 flex-1">
        <div className="h-1.5 flex-1 rounded-full bg-slate-200 dark:bg-slate-700 overflow-hidden">
          <div
            className={`h-full ${sqiColor(sqi)}`}
            style={{ width: `${Math.round(sqi * 100)}%` }}
          />
        </div>
        <span className="text-xs font-mono w-10 text-right">
          {(sqi * 100).toFixed(0)}%
        </span>
      </div>
      <span className="text-xs w-20 text-right text-slate-500 dark:text-slate-400">
        {flags.length > 0 ? flags[0] : sqiLabel(sqi)}
      </span>
    </div>
  );
}
