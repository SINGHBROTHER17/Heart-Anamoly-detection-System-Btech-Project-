export default function ArticlesPage() {
  return (
    <div className="flex flex-col px-4 pt-5 gap-5">
      <h1 className="text-xl font-bold text-slate-800">Articles</h1>

      {/* Coming soon hero */}
      <div className="card flex flex-col items-center text-center py-12 gap-4">
        <span className="text-5xl">📰</span>
        <div>
          <h2 className="text-lg font-bold text-slate-700">Coming Soon</h2>
          <p className="text-sm text-slate-500 mt-1 max-w-xs">
            Educational articles about heart health, ECG interpretation, and cardiac
            conditions will be available here.
          </p>
        </div>
      </div>

      {/* Preview topics */}
      <div className="card">
        <h2 className="font-semibold text-slate-700 mb-3">Upcoming topics</h2>
        <ul className="flex flex-col gap-3">
          {[
            { icon: '❤️', title: 'Understanding Your ECG Report' },
            { icon: '⚡', title: 'What is Atrial Fibrillation?' },
            { icon: '💊', title: 'Managing High Blood Pressure' },
            { icon: '🏃', title: 'Exercise and Heart Health' },
            { icon: '🩺', title: 'When to See a Cardiologist' },
          ].map((item) => (
            <li key={item.title} className="flex items-center gap-3 text-sm text-slate-600">
              <span className="text-xl">{item.icon}</span>
              <span>{item.title}</span>
              <span className="ml-auto text-xs bg-slate-100 text-slate-400 rounded-full px-2 py-0.5">
                Soon
              </span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
