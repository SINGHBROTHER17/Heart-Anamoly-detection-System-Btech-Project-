/** Skeleton screens shown while analysis runs. */

export function SkeletonCard() {
  return (
    <div className="card animate-pulse">
      <div className="h-4 bg-slate-200 dark:bg-slate-700 rounded w-3/4 mb-3" />
      <div className="h-2 bg-slate-200 dark:bg-slate-700 rounded w-full mb-4" />
      <div className="h-3 bg-slate-200 dark:bg-slate-700 rounded w-5/6 mb-2" />
      <div className="h-3 bg-slate-200 dark:bg-slate-700 rounded w-4/6" />
    </div>
  );
}

export function SkeletonPlot({ height = 200 }) {
  return (
    <div
      className="skeleton rounded-lg border border-slate-200 dark:border-slate-700"
      style={{ height }}
      aria-label="Loading plot"
    />
  );
}

export function SkeletonResultList({ count = 6 }) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
      {Array.from({ length: count }).map((_, i) => (
        <SkeletonCard key={i} />
      ))}
    </div>
  );
}
