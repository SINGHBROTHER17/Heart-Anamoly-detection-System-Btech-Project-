export default function ErrorBanner({ title = 'Something went wrong', message, perLeadSqi, onRetry }) {
  return (
    <div className="rounded-xl border border-rose-300 bg-rose-50 dark:bg-rose-950 dark:border-rose-900 p-4">
      <div className="flex items-start gap-3">
        <span className="text-xl leading-none mt-0.5" aria-hidden>⚠️</span>
        <div className="flex-1">
          <h3 className="font-semibold text-sm text-rose-900 dark:text-rose-200">{title}</h3>
          {message && (
            <p className="mt-1 text-sm text-rose-800 dark:text-rose-300">{message}</p>
          )}
          {perLeadSqi && (
            <div className="mt-3 grid grid-cols-2 sm:grid-cols-3 gap-1 text-xs">
              {Object.entries(perLeadSqi).map(([lead, sqi]) => (
                <div key={lead}
                     className={`px-2 py-1 rounded-md font-mono
                       ${sqi < 0.4 ? 'bg-rose-200 text-rose-900 dark:bg-rose-900 dark:text-rose-200'
                                    : 'bg-amber-100 text-amber-900 dark:bg-amber-900 dark:text-amber-200'}`}>
                  {lead}: {(sqi * 100).toFixed(0)}%
                </div>
              ))}
            </div>
          )}
          {onRetry && (
            <button className="btn-secondary mt-3" onClick={onRetry}>
              Try again
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
