const TIER_META = {
  none:     { position: 8,  label: 'Low risk',      color: '#22c55e', textColor: 'text-green-600' },
  possible: { position: 38, label: 'Moderate risk',  color: '#f59e0b', textColor: 'text-amber-500' },
  likely:   { position: 65, label: 'High risk',      color: '#f97316', textColor: 'text-orange-500' },
  high:     { position: 90, label: 'High risk',      color: '#ef4444', textColor: 'text-red-500' },
};

export default function RiskMeter({ tier }) {
  const meta = TIER_META[tier] ?? TIER_META.none;

  return (
    <div className="flex flex-col gap-3">
      {/* Zone labels */}
      <div className="flex justify-between text-xs font-medium text-slate-500 px-1">
        <span>Low risk</span>
        <span>Moderate risk</span>
        <span>High risk</span>
      </div>

      {/* Gradient bar + indicator */}
      <div className="relative h-3 rounded-full overflow-visible"
           style={{ background: 'linear-gradient(to right, #22c55e 0%, #f59e0b 45%, #ef4444 100%)' }}>
        <div
          className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-5 h-5 rounded-full border-[3px] border-white shadow-md transition-all duration-500"
          style={{ left: `${meta.position}%`, backgroundColor: meta.color }}
        />
      </div>

      {/* Result label */}
      <div className="flex items-center gap-2 mt-1">
        <span
          className="text-xl font-bold"
          style={{ color: meta.color }}
        >
          You are at {meta.label}
        </span>
      </div>
    </div>
  );
}
