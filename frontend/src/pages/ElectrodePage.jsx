import { useState } from 'react';

const LEAD_POSITIONS = {
  male: [
    { id: 'RA', label: 'RA', x: '35%', y: '22%', color: '#e84c5a' },
    { id: 'LA', label: 'LA', x: '62%', y: '22%', color: '#22c55e' },
    { id: 'RL', label: 'RL', x: '38%', y: '68%', color: '#000' },
    { id: 'LL', label: 'LL', x: '60%', y: '68%', color: '#f59e0b' },
    { id: 'V1', label: 'V1', x: '50%', y: '38%', color: '#3b82f6' },
  ],
  female: [
    { id: 'RA', label: 'RA', x: '34%', y: '24%', color: '#e84c5a' },
    { id: 'LA', label: 'LA', x: '63%', y: '24%', color: '#22c55e' },
    { id: 'RL', label: 'RL', x: '38%', y: '70%', color: '#000' },
    { id: 'LL', label: 'LL', x: '60%', y: '70%', color: '#f59e0b' },
    { id: 'V1', label: 'V1', x: '50%', y: '42%', color: '#3b82f6' },
  ],
};

export default function ElectrodePage() {
  const [gender, setGender] = useState('male');

  return (
    <div className="flex flex-col px-4 pt-5 gap-5">
      <div>
        <h1 className="text-xl font-bold text-slate-800">Electrode Positioning</h1>
        <p className="text-sm text-slate-500 mt-1">
          Place electrodes on the correct positions for accurate readings.
        </p>
      </div>

      {/* Gender toggle */}
      <div className="flex bg-slate-100 rounded-2xl p-1 gap-1">
        {['male', 'female'].map((g) => (
          <button
            key={g}
            onClick={() => setGender(g)}
            className={[
              'flex-1 py-2.5 rounded-xl text-sm font-semibold capitalize transition-all',
              gender === g
                ? 'bg-white text-brand-500 shadow-sm'
                : 'text-slate-500',
            ].join(' ')}
          >
            {g}
          </button>
        ))}
      </div>

      {/* 3D Model placeholder */}
      <div className="card relative flex items-center justify-center overflow-hidden"
           style={{ minHeight: 360 }}>
        {/* Placeholder body silhouette */}
        <div className="relative w-48 h-80 flex items-center justify-center">
          <svg viewBox="0 0 100 200" className="w-full h-full opacity-10" fill="currentColor">
            <ellipse cx="50" cy="20" rx="14" ry="18" />
            <rect x="30" y="36" width="40" height="70" rx="8" />
            <rect x="8" y="38" width="20" height="55" rx="8" />
            <rect x="72" y="38" width="20" height="55" rx="8" />
            <rect x="30" y="104" width="17" height="70" rx="8" />
            <rect x="53" y="104" width="17" height="70" rx="8" />
          </svg>

          {/* Electrode dots */}
          {LEAD_POSITIONS[gender].map((lead) => (
            <div
              key={lead.id}
              className="absolute flex flex-col items-center gap-0.5 cursor-pointer group"
              style={{ left: lead.x, top: lead.y, transform: 'translate(-50%, -50%)' }}
            >
              <div
                className="w-5 h-5 rounded-full border-2 border-white shadow-md flex items-center justify-center"
                style={{ backgroundColor: lead.color }}
              />
              <span className="text-[9px] font-bold bg-white rounded px-1 shadow-sm"
                    style={{ color: lead.color }}>
                {lead.label}
              </span>
            </div>
          ))}
        </div>

        {/* Asset coming soon overlay */}
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-white/70 backdrop-blur-sm rounded-2xl">
          <span className="text-4xl mb-3">🫀</span>
          <p className="font-bold text-slate-700 text-base">3D Guide Coming Soon</p>
          <p className="text-slate-500 text-xs mt-1 text-center px-6">
            Interactive 3D electrode placement guide will be available once assets are ready.
          </p>
        </div>
      </div>

      {/* Lead reference table */}
      <div className="card">
        <h2 className="font-semibold text-slate-700 mb-3">Lead Reference</h2>
        <div className="flex flex-col gap-2 text-sm">
          {[
            { label: 'RA (Right Arm)', color: '#e84c5a', desc: 'Right wrist or right shoulder' },
            { label: 'LA (Left Arm)',  color: '#22c55e', desc: 'Left wrist or left shoulder' },
            { label: 'RL (Right Leg)', color: '#1e293b', desc: 'Right ankle or right lower abdomen' },
            { label: 'LL (Left Leg)',  color: '#f59e0b', desc: 'Left ankle or left lower abdomen' },
            { label: 'V1–V6',          color: '#3b82f6', desc: 'Chest leads across the precordium' },
          ].map((row) => (
            <div key={row.label} className="flex items-start gap-3">
              <span className="w-3 h-3 rounded-full flex-shrink-0 mt-0.5" style={{ backgroundColor: row.color }} />
              <div>
                <p className="font-medium text-slate-700">{row.label}</p>
                <p className="text-xs text-slate-500">{row.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
