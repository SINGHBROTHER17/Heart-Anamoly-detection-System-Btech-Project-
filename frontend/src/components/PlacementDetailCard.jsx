/**
 * PlacementDetailCard — shows full placement instructions for one step.
 *
 * For limb-lead steps: renders one sub-card per electrode (RA, LA, RL, LL).
 * For chest-lead steps: renders a single focused card with schematic diagram.
 */
import { ELECTRODES } from '../constants/electrodeConfig.js';

const DIFFICULTY_STYLES = {
  Easy:     'bg-green-900/40 text-green-400',
  Moderate: 'bg-amber-900/40 text-amber-400',
  Tricky:   'bg-red-900/40 text-red-400',
};

// ── Limb electrode colour chips used in the "attach X clip" instructions
const CLIP_COLORS = {
  RA:  { bg: '#FEB2B2', text: '#C53030', chip: 'RED' },
  LA:  { bg: '#FEFCBF', text: '#B7791F', chip: 'YELLOW' },
  RL:  { bg: '#E2E8F0', text: '#2D3748', chip: 'BLACK' },
  LL:  { bg: '#C6F6D5', text: '#276749', chip: 'GREEN' },
  WST: { bg: '#E9D8FD', text: '#553C9A', chip: 'WAIST' },
};

export default function PlacementDetailCard({ stepConfig }) {
  const isLimb       = stepConfig.id === 'limb';
  const isLeadCombo  = stepConfig.id === 'L1' || stepConfig.id === 'L2';

  if (isLimb || isLeadCombo) {
    const intro = isLeadCombo
      ? (stepConfig.id === 'L2'
          ? 'Keep RA (red) and LA (yellow) attached. Add a 3rd electrode on your lower-LEFT waist:'
          : 'Remove the waist electrode — just RA (red) and LA (yellow) stay attached for this recording:')
      : 'Attach all four limb clips in any order before starting the recording:';

    return (
      <div className="flex flex-col gap-3">
        <p className="text-sm text-slate-300 font-semibold">{intro}</p>
        {stepConfig.electrodes.map((id) => (
          <LimbCard key={id} electrodeId={id} />
        ))}
      </div>
    );
  }

  // Chest lead
  const [id] = stepConfig.electrodes;
  const cfg  = ELECTRODES[id];

  return (
    <div className="card bg-slate-800 border-slate-700 flex flex-col gap-3"
         style={{ borderLeftWidth: 4, borderLeftColor: cfg.color }}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl flex items-center justify-center text-white font-black text-lg"
               style={{ backgroundColor: cfg.color + '33', color: cfg.color }}>
            {cfg.label}
          </div>
          <div>
            <p className="font-bold text-white text-sm">{cfg.fullName}</p>
            <p className="text-[11px] text-slate-400">{cfg.side}</p>
          </div>
        </div>
        <span className={`text-[10px] font-bold rounded-full px-2 py-0.5 ${DIFFICULTY_STYLES[cfg.difficulty]}`}>
          {cfg.difficulty}
        </span>
      </div>

      {/* Schematic cross-section + rib counting diagram */}
      <ChestSchematic electrodeId={id} />

      {/* Main instruction */}
      <div className="bg-slate-900/60 rounded-xl px-3 py-2.5">
        <p className="text-xs font-semibold text-slate-300 mb-1">📍 Where to place it</p>
        <p className="text-sm text-white leading-relaxed">{cfg.instruction}</p>
      </div>

      {/* Tip */}
      <div className="flex items-start gap-2">
        <span className="text-base mt-0.5">💡</span>
        <p className="text-xs text-slate-400 leading-relaxed">
          <span className="text-amber-400 font-semibold">Tip: </span>{cfg.tip}
        </p>
      </div>

      {/* Common mistake */}
      <div className="flex items-start gap-2">
        <span className="text-base mt-0.5">⚠️</span>
        <p className="text-xs text-slate-400 leading-relaxed">
          <span className="text-red-400 font-semibold">Common mistake: </span>{cfg.commonMistake}
        </p>
      </div>
    </div>
  );
}

// ── Limb electrode sub-card ─────────────────────────────────────────────────
function LimbCard({ electrodeId }) {
  const cfg  = ELECTRODES[electrodeId];
  const chip = CLIP_COLORS[electrodeId];

  return (
    <div className="bg-slate-800 rounded-2xl px-4 py-3 flex items-start gap-3 border border-slate-700">
      <span
        className="w-9 h-9 rounded-xl flex-shrink-0 flex items-center justify-center text-xs font-black mt-0.5"
        style={{ backgroundColor: cfg.color + '33', color: cfg.color }}
      >
        {cfg.label}
      </span>
      <div className="flex flex-col gap-1 flex-1">
        <div className="flex items-center gap-2">
          <span
            className="text-[10px] font-black rounded-full px-2 py-0.5"
            style={{ backgroundColor: chip.bg, color: chip.text }}
          >
            {chip.chip}
          </span>
          <span className="text-xs font-semibold text-white">{cfg.fullName}</span>
        </div>
        <p className="text-xs text-slate-400 leading-relaxed">{cfg.instruction}</p>
        <p className="text-[11px] text-amber-400 mt-0.5">💡 {cfg.tip}</p>
      </div>
    </div>
  );
}

// ── Inline SVG chest schematic showing intercostal position ─────────────────
function ChestSchematic({ electrodeId }) {
  const cfg = ELECTRODES[electrodeId];
  if (!cfg.ribCount) return null;  // V3 has no fixed rib — skip schematic

  // Show a simplified sternum + ribs cross-section
  return (
    <div className="bg-slate-900/40 rounded-xl p-3 flex items-center justify-between gap-3">
      <svg viewBox="0 0 100 70" className="w-28 flex-shrink-0" aria-label={`Rib position diagram for ${cfg.label}`}>
        {/* Sternum */}
        <rect x="42" y="2" width="16" height="66" rx="3" fill="#475569" />
        <text x="50" y="72" textAnchor="middle" fontSize="6" fill="#64748b">Sternum</text>

        {/* Rib pairs */}
        {[12, 24, 36, 48, 60].map((y, i) => {
          const isTarget = i + 1 === cfg.ribCount - 1;   // intercostal space above ribCount
          const isActive = i + 2 === cfg.ribCount;        // the target rib space
          return (
            <g key={y}>
              {/* ICS (intercostal space) highlight */}
              {isActive && (
                <rect x="0" y={y - 6} width="40" height="12" rx="3"
                      fill={cfg.color + '40'} stroke={cfg.color} strokeWidth="0.8" />
              )}
              {/* Left rib arc */}
              <path d={`M 42,${y} Q 20,${y} 5,${y + 6}`}
                    stroke={isActive ? cfg.color : '#334155'}
                    strokeWidth={isActive ? 1.5 : 1}
                    fill="none" />
              {/* Rib number */}
              <text x="48" y={y + 2} fontSize="5" fill="#64748b" textAnchor="middle">
                {i + 1}
              </text>
              {/* Electrode dot on target */}
              {isActive && (
                <>
                  <circle cx="18" cy={y + 2} r="4" fill={cfg.color} />
                  <text x="18" y={y + 4.5} textAnchor="middle" fontSize="4.5" fill="white" fontWeight="bold">
                    {cfg.label}
                  </text>
                </>
              )}
            </g>
          );
        })}

        {/* Arrow pointing to electrode */}
        {cfg.ribCount && (
          <text x="62" y={12 + (cfg.ribCount - 1) * 12 + 3} fontSize="5.5" fill={cfg.color}>
            ← {cfg.ribCount}th ICS
          </text>
        )}
      </svg>

      <p className="text-xs text-slate-400 leading-relaxed">
        Feel for your <span className="text-white font-semibold">collarbone</span>, count
        down <span className="text-white font-semibold">{cfg.ribCount} ribs</span>.
        {cfg.label === 'V1' && ' The electrode goes just to the RIGHT of your breastbone.'}
        {cfg.label === 'V2' && ' The electrode goes just to the LEFT of your breastbone.'}
        {cfg.label === 'V4' && ' Place at the midclavicular (nipple) line at that level.'}
      </p>
    </div>
  );
}
