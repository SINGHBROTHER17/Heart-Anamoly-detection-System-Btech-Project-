/**
 * TorsoSVGDiagram — front-view anatomical reference with electrode overlays.
 *
 * Renders the user-supplied 3D mannequin photograph as a background and
 * overlays electrode markers positioned in percentage units, so the layout
 * scales naturally with the container.
 *
 * IMPORTANT: This component expects the mannequin asset at:
 *   frontend/public/mannequin-front.png
 *
 * The image should be a clean front-view male/neutral mannequin on a white/
 * transparent background, full body or upper body. If the file is missing,
 * the component degrades to a plain coloured panel with the markers still
 * positioned correctly so you can see what's happening.
 *
 * Smooth transitions:
 *  - Pending → Active : pop-in scale + colour fade + animated pulsing ring
 *  - Active  → Confirmed : ✓ scales in, colour transitions to green
 *  - Active  → Failed : shake + red colour
 *
 * A subtle dashed "guide path" connects the previously confirmed electrode
 * to the new active one, so users can visually trace the placement order.
 */
import { useEffect, useRef, useState } from 'react';
import { ELECTRODES } from '../constants/electrodeConfig.js';

const MANNEQUIN_URL = '/mannequin-front.png';

// ── Marker positions as percentages of the photo (x%, y%) ───────────────────
// These are calibrated against a centred full-body mannequin image with the
// proportions of the reference asset (≈580×920). If you swap the asset, tune
// the percentages here.
//
// type: 'badge' = floating coloured pill outside the body (limb electrodes)
// type: 'ring'  = on-body open ring marker (chest leads + WST)
const POSITIONS = {
  RA:  { x: 18,  y: 47, type: 'badge' },     // right wrist (image left)
  LA:  { x: 82,  y: 47, type: 'badge' },     // left wrist  (image right)
  RL:  { x: 39,  y: 92, type: 'badge' },     // right ankle
  LL:  { x: 61,  y: 92, type: 'badge' },     // left ankle
  V1:  { x: 47,  y: 30, type: 'ring'  },     // 4th ICS, right of sternum
  V2:  { x: 53,  y: 30, type: 'ring'  },     // 4th ICS, left of sternum
  V3:  { x: 56,  y: 33, type: 'ring'  },     // mid V2-V4
  V4:  { x: 59,  y: 36, type: 'ring'  },     // 5th ICS, midclavicular
  V5:  { x: 64,  y: 36, type: 'ring'  },     // anterior axillary
  V6:  { x: 70,  y: 36, type: 'ring'  },     // midaxillary
  WST: { x: 42,  y: 45, type: 'ring'  },     // lower-left waist (Lead II)
};

// ── State → colour mapping ──────────────────────────────────────────────────
const COLOURS = {
  pending:   { stroke: '#94a3b8', fill: 'rgba(255,255,255,0.55)', text: '#475569' },
  active:    { stroke: null,      fill: null,                      text: null     }, // uses cfg.color
  confirmed: { stroke: '#22c55e', fill: '#22c55e',                 text: '#ffffff' },
  failed:    { stroke: '#ef4444', fill: '#ef4444',                 text: '#ffffff' },
};

export default function TorsoSVGDiagram({
  activeElectrodes = [],
  confirmedLeads   = {},
  failedLeads      = {},
  onElectrodeClick = null,
}) {
  const [imgError, setImgError] = useState(false);

  // Track previous active electrode so we can draw a "guide path" between them
  const prevActiveRef = useRef([]);
  useEffect(() => {
    prevActiveRef.current = activeElectrodes;
  }, [activeElectrodes]);

  const getState = (id) => {
    const cfg = ELECTRODES[id];
    const leadsFailed = cfg.group === 'limb'
      ? ['I', 'II', 'III'].some((l) => failedLeads[l])
      : failedLeads[id];
    if (leadsFailed) return 'failed';

    const leadsConfirmed = cfg.group === 'limb'
      ? ['I', 'II', 'III'].every((l) => confirmedLeads[l])
      : confirmedLeads[id];
    if (leadsConfirmed) return 'confirmed';

    if (activeElectrodes.includes(id)) return 'active';
    return 'pending';
  };

  return (
    <div className="relative select-none mx-auto" style={{ maxWidth: 280 }}>
      {/* ── Background image + electrode overlay ─────────────────────── */}
      <div
        className="relative rounded-2xl overflow-hidden"
        style={{ aspectRatio: '580 / 920', background: '#f5f5f3' }}
      >
        {!imgError ? (
          <img
            src={MANNEQUIN_URL}
            alt="Anatomical reference for electrode placement"
            className="absolute inset-0 w-full h-full object-contain pointer-events-none"
            style={{ filter: 'contrast(1.05) saturate(0.9)' }}
            onError={() => setImgError(true)}
          />
        ) : (
          <ImageMissingPlaceholder />
        )}

        {/* Connecting guide path between confirmed and active electrodes */}
        <GuidePathOverlay activeElectrodes={activeElectrodes} confirmedLeads={confirmedLeads} />

        {/* Electrode markers */}
        {Object.entries(POSITIONS).map(([id, pos]) => {
          const cfg   = ELECTRODES[id];
          const state = getState(id);
          if (!cfg) return null;

          return pos.type === 'badge' ? (
            <BadgeMarker
              key={id} id={id} pos={pos} cfg={cfg} state={state}
              onClick={() => onElectrodeClick?.(id)}
            />
          ) : (
            <RingMarker
              key={id} id={id} pos={pos} cfg={cfg} state={state}
              onClick={() => onElectrodeClick?.(id)}
            />
          );
        })}
      </div>

      {/* ── Legend ───────────────────────────────────────────────────── */}
      <div className="flex items-center justify-center gap-3 mt-2 text-[10px] text-slate-400 flex-wrap">
        <LegendItem variant="pending"   label="Pending" />
        <LegendItem variant="active"    label="Active" />
        <LegendItem variant="confirmed" label="Confirmed" />
        <LegendItem variant="failed"    label="Re-place" />
      </div>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// Floating limb-electrode badge (RA / LA / RL / LL)
// ────────────────────────────────────────────────────────────────────────────
function BadgeMarker({ id, pos, cfg, state, onClick }) {
  const isActive = state === 'active';
  const isConf   = state === 'confirmed';
  const isFail   = state === 'failed';
  const isPend   = state === 'pending';

  const bg = isConf ? COLOURS.confirmed.fill
    : isFail ? COLOURS.failed.fill
    : cfg.color;

  return (
    <div
      onClick={onClick}
      className="absolute cursor-pointer marker-transition"
      style={{
        left:      `${pos.x}%`,
        top:       `${pos.y}%`,
        transform: `translate(-50%, -50%) scale(${isActive ? 1.18 : 1})`,
        opacity:   isPend ? 0.55 : 1,
        zIndex:    isActive ? 10 : 5,
      }}
    >
      <div
        className="rounded-md px-2 py-0.5 text-[10px] font-extrabold text-white tracking-wider"
        style={{
          background: bg,
          boxShadow:  isActive
            ? `0 0 0 3px ${cfg.color}33, 0 4px 10px rgba(0,0,0,0.25)`
            : '0 2px 6px rgba(0,0,0,0.25)',
          animation:  isActive ? 'electrode-pulse 1.4s ease-in-out infinite' : 'none',
          minWidth:   24,
          textAlign:  'center',
        }}
      >
        {isConf ? '✓' : isFail ? '✗' : cfg.label}
      </div>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// On-body chest / WST ring marker
// ────────────────────────────────────────────────────────────────────────────
function RingMarker({ id, pos, cfg, state, onClick }) {
  const isActive = state === 'active';
  const isConf   = state === 'confirmed';
  const isFail   = state === 'failed';
  const isPend   = state === 'pending';

  const stroke = isConf ? COLOURS.confirmed.stroke
    : isFail ? COLOURS.failed.stroke
    : isActive ? cfg.color
    : COLOURS.pending.stroke;

  const fill = isConf ? COLOURS.confirmed.fill
    : isFail ? COLOURS.failed.fill
    : isActive ? cfg.color + '33'
    : COLOURS.pending.fill;

  return (
    <div
      onClick={onClick}
      className="absolute cursor-pointer marker-transition"
      style={{
        left:      `${pos.x}%`,
        top:       `${pos.y}%`,
        transform: `translate(-50%, -50%) scale(${isActive ? 1.25 : isConf ? 1.05 : 1})`,
        opacity:   isPend ? 0.5 : 1,
        zIndex:    isActive ? 10 : 5,
      }}
    >
      {/* Outer pulsing ring (active only) */}
      {isActive && (
        <span
          className="absolute inset-0 m-auto rounded-full pointer-events-none"
          style={{
            width: 26, height: 26,
            left: '50%', top: '50%',
            transform: 'translate(-50%, -50%)',
            border: `2px solid ${cfg.color}`,
            opacity: 0.55,
            animation: 'electrode-ring-expand 1.4s ease-out infinite',
          }}
        />
      )}

      {/* Main ring */}
      <div
        className="rounded-full flex items-center justify-center"
        style={{
          width:        16,
          height:       16,
          background:   fill,
          border:       `1.6px solid ${stroke}`,
          boxShadow:    isActive ? `0 0 8px ${cfg.color}99` : '0 1px 3px rgba(0,0,0,0.2)',
          transition:   'all 0.35s cubic-bezier(0.34, 1.56, 0.64, 1)',
        }}
      >
        {(isConf || isFail) && (
          <span className="text-[9px] font-extrabold text-white leading-none">
            {isConf ? '✓' : '✗'}
          </span>
        )}
      </div>

      {/* Label */}
      <span
        className="absolute left-1/2 -translate-x-1/2 text-[9px] font-bold whitespace-nowrap pointer-events-none"
        style={{
          top:        '105%',
          marginTop:  3,
          color:      isActive ? cfg.color : isConf ? '#22c55e' : '#475569',
          textShadow: '0 0 4px rgba(255,255,255,0.95), 0 0 2px rgba(255,255,255,0.95)',
          opacity:    isPend ? 0.7 : 1,
          transition: 'all 0.3s ease',
        }}
      >
        {cfg.label}
      </span>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// SVG path drawn between most-recently confirmed electrode and the active one
// — gives users a visual cue showing the placement journey.
// ────────────────────────────────────────────────────────────────────────────
function GuidePathOverlay({ activeElectrodes, confirmedLeads }) {
  if (!activeElectrodes?.length) return null;

  // Find the most recent confirmed chest electrode (V1..V6)
  // We treat WST as part of the limb-lead group (no path drawn for L2 → L1)
  const ORDER = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6'];
  const lastConfirmed = [...ORDER].reverse().find((id) => confirmedLeads[id]);
  const target = activeElectrodes.find((id) => POSITIONS[id]?.type === 'ring');

  if (!lastConfirmed || !target || lastConfirmed === target) return null;

  const from = POSITIONS[lastConfirmed];
  const to   = POSITIONS[target];
  if (!from || !to) return null;

  return (
    <svg
      className="absolute inset-0 w-full h-full pointer-events-none"
      viewBox="0 0 100 100"
      preserveAspectRatio="none"
      aria-hidden
    >
      <line
        x1={from.x} y1={from.y}
        x2={to.x}   y2={to.y}
        stroke="#e84c5a"
        strokeWidth="0.4"
        strokeDasharray="1.2,1.2"
        opacity="0.65"
        className="guide-path-anim"
      />
    </svg>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// Legend
// ────────────────────────────────────────────────────────────────────────────
function LegendItem({ variant, label }) {
  const styles = {
    pending:   { background: 'rgba(255,255,255,0.08)', border: '1.5px solid #94a3b8' },
    active:    { background: '#e84c5a',                animation: 'electrode-pulse 1.4s ease-in-out infinite' },
    confirmed: { background: '#22c55e' },
    failed:    { background: '#ef4444' },
  }[variant];

  return (
    <div className="flex items-center gap-1">
      <span className="w-3 h-3 rounded-full inline-block" style={styles} />
      {label}
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// Fallback panel shown when the mannequin asset is missing
// ────────────────────────────────────────────────────────────────────────────
function ImageMissingPlaceholder() {
  return (
    <div className="absolute inset-0 flex items-center justify-center p-6 text-center">
      <div className="text-slate-500">
        <p className="text-xs font-bold mb-2">Mannequin asset missing</p>
        <p className="text-[10px] leading-relaxed">
          Drop your front-view body image at<br />
          <code className="text-[9px] bg-slate-200/40 px-1 rounded">
            frontend/public/mannequin-front.png
          </code>
        </p>
      </div>
    </div>
  );
}
