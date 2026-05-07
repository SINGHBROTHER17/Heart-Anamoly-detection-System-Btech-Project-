export default function DeviceBanner() {
  return (
    <div className="bg-slate-800 text-white px-4 py-2.5 flex items-center justify-between text-xs">
      <div className="flex items-center gap-2">
        <span className="w-2 h-2 rounded-full bg-slate-400 inline-block" />
        <span className="font-medium">ECG Device Not Connected</span>
      </div>
      <button className="text-brand-300 font-semibold underline underline-offset-2">
        How to Connect
      </button>
    </div>
  );
}
