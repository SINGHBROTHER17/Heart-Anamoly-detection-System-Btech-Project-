import { useState } from 'react';
import { useUserProfile } from '../hooks/useUserProfile.js';

export default function SettingsPage() {
  const [profile, updateProfile] = useUserProfile();
  const [name, setName] = useState(profile.name ?? '');
  const [saved, setSaved] = useState(false);

  const handleSave = (e) => {
    e.preventDefault();
    updateProfile({ name: name.trim() });
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  const initials = name
    ? name.split(' ').map((w) => w[0]).join('').slice(0, 2).toUpperCase()
    : '?';

  return (
    <div className="flex flex-col px-4 pt-5 gap-5">
      {/* Page header */}
      <h1 className="text-xl font-bold text-slate-800">Settings</h1>

      {/* Avatar preview */}
      <div className="flex flex-col items-center gap-3 py-6">
        <div className="w-20 h-20 rounded-full bg-brand-500 flex items-center justify-center text-white text-2xl font-bold shadow-lg">
          {initials}
        </div>
        <p className="text-sm text-slate-500">Your profile picture is auto-generated from your initials</p>
      </div>

      {/* Name form */}
      <form onSubmit={handleSave} className="card flex flex-col gap-4">
        <h2 className="font-semibold text-slate-700">Profile</h2>
        <div className="flex flex-col gap-1.5">
          <label htmlFor="name" className="text-sm font-medium text-slate-600">
            Your name
          </label>
          <input
            id="name"
            type="text"
            value={name}
            onChange={(e) => { setName(e.target.value); setSaved(false); }}
            placeholder="e.g. Aditya Singh"
            className="w-full rounded-xl border border-slate-200 px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500 bg-slate-50"
          />
        </div>
        <button
          type="submit"
          disabled={!name.trim()}
          className="btn-primary w-full"
        >
          {saved ? '✓ Saved!' : 'Save profile'}
        </button>
      </form>

      {/* App info */}
      <div className="card flex flex-col gap-2 text-sm text-slate-500">
        <h2 className="font-semibold text-slate-700">About</h2>
        <div className="flex justify-between">
          <span>Version</span>
          <span className="font-medium text-slate-700">0.1.0</span>
        </div>
        <div className="flex justify-between">
          <span>Model</span>
          <span className="font-medium text-slate-700">1D-CNN + Transformer</span>
        </div>
        <div className="flex justify-between">
          <span>Conditions detected</span>
          <span className="font-medium text-slate-700">10</span>
        </div>
        <p className="text-xs text-slate-400 pt-1 border-t border-slate-100 mt-1">
          This is a screening tool only and does not constitute a medical diagnosis.
        </p>
      </div>
    </div>
  );
}
