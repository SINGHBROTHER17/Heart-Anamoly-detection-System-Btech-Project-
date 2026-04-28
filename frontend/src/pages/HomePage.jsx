import { useNavigate } from 'react-router-dom';
import { useUserProfile } from '../hooks/useUserProfile.js';

const ECG_OPTIONS = [
  {
    id: '12lead',
    icon: '💗',
    bg: 'bg-red-50',
    iconBg: 'bg-brand-500',
    title: '12 Lead ECG',
    desc: 'Comprehensive heart analysis from 12 angles',
    route: '/analyze',
    active: true,
  },
  {
    id: 'risk',
    icon: '🫀',
    bg: 'bg-pink-50',
    iconBg: 'bg-pink-400',
    title: 'Heart Risk Calculator',
    desc: 'Heart risk assessment made easy',
    route: null,
  },
  {
    id: 'hrv',
    icon: '🧠',
    bg: 'bg-purple-50',
    iconBg: 'bg-purple-400',
    title: 'HRV',
    desc: 'Instant stress level readings',
    route: null,
  },
  {
    id: 'monitor',
    icon: '📡',
    bg: 'bg-blue-50',
    iconBg: 'bg-blue-400',
    title: 'Live Monitor',
    desc: 'Continuous, round-the-clock heart monitoring',
    route: null,
  },
  {
    id: 'lead2',
    icon: '⚡',
    bg: 'bg-yellow-50',
    iconBg: 'bg-yellow-400',
    title: 'Lead II ECG',
    desc: 'Quick basic arrhythmia screening',
    route: null,
  },
  {
    id: 'hyper',
    icon: '🔬',
    bg: 'bg-teal-50',
    iconBg: 'bg-teal-400',
    title: 'Hyperkalemia',
    desc: 'Quick potassium level checks',
    route: null,
  },
];

export default function HomePage() {
  const navigate = useNavigate();
  const [profile] = useUserProfile();
  const name = profile.name || 'there';

  const greeting = () => {
    const h = new Date().getHours();
    if (h < 12) return 'Good morning';
    if (h < 17) return 'Good afternoon';
    return 'Good evening';
  };

  const initials = profile.name
    ? profile.name.split(' ').map((w) => w[0]).join('').slice(0, 2).toUpperCase()
    : '?';

  return (
    <div className="flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-4 pt-5 pb-4">
        <div>
          <p className="text-slate-500 text-sm">{greeting()},</p>
          <h1 className="text-xl font-bold text-slate-800 capitalize">
            {name}! 👋
          </h1>
          <p className="text-xs text-slate-400 mt-0.5">Hope you have a great day!</p>
        </div>
        <button
          onClick={() => navigate('/settings')}
          className="w-11 h-11 rounded-full bg-brand-500 flex items-center justify-center text-white font-bold text-sm shadow-md"
        >
          {initials}
        </button>
      </div>

      {/* Record ECG section */}
      <div className="px-4">
        <h2 className="text-base font-bold text-slate-700 mb-3">Record your ECG</h2>
        <div className="grid grid-cols-2 gap-3">
          {ECG_OPTIONS.map((opt) => (
            <EcgCard key={opt.id} opt={opt} onClick={() => opt.active && navigate(opt.route)} />
          ))}
        </div>
      </div>

      {/* What's new */}
      <div className="px-4 mt-6">
        <h2 className="text-base font-bold text-slate-700 mb-3">What's new</h2>
        <div className="card bg-gradient-to-r from-brand-500 to-rose-400 text-white p-5 flex flex-col gap-1">
          <p className="text-xs font-semibold uppercase tracking-wide opacity-80">Coming soon</p>
          <h3 className="text-lg font-bold">Hardware ECG Device</h3>
          <p className="text-sm opacity-90">
            Connect a real-time ECG sensor for instant, on-device recordings.
          </p>
        </div>
      </div>

      <div className="px-4 mt-3 mb-2">
        <p className="text-[10px] text-center text-slate-400">
          Screening tool only — not a substitute for medical diagnosis.
        </p>
      </div>
    </div>
  );
}

function EcgCard({ opt, onClick }) {
  const isComingSoon = !opt.active;

  return (
    <button
      onClick={onClick}
      disabled={isComingSoon}
      className={[
        'relative text-left rounded-2xl p-4 flex flex-col gap-3 transition-all',
        opt.bg,
        isComingSoon
          ? 'opacity-60 cursor-not-allowed'
          : 'hover:shadow-md active:scale-95',
      ].join(' ')}
    >
      {isComingSoon && (
        <span className="absolute top-2 right-2 text-[9px] font-bold bg-slate-200 text-slate-500 rounded-full px-1.5 py-0.5">
          SOON
        </span>
      )}
      <div className={`w-10 h-10 rounded-xl flex items-center justify-center text-xl ${opt.iconBg} text-white shadow-sm`}>
        {opt.icon}
      </div>
      <div>
        <p className="font-bold text-sm text-slate-800 leading-tight">{opt.title}</p>
        <p className="text-xs text-slate-500 mt-0.5 leading-snug">{opt.desc}</p>
      </div>
    </button>
  );
}
