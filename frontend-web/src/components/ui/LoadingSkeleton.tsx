import React from 'react';

interface SkeletonProps {
  className?: string;
}

export const Skeleton: React.FC<SkeletonProps> = ({ className = '' }) => (
  <div
    className={`animate-pulse rounded bg-slate-200/80 dark:bg-[#1E2630]/80 ${className}`}
  />
);

export const CardSkeleton: React.FC<{ count?: number }> = ({ count = 3 }) => (
  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 my-4">
    {Array.from({ length: count }).map((_, i) => (
      <div
        key={i}
        className="p-5 rounded-xl border border-[#E3E8EE] dark:border-[#1E2630] bg-white dark:bg-[#11171E] space-y-3"
      >
        <Skeleton className="h-4 w-28" />
        <Skeleton className="h-8 w-36" />
        <Skeleton className="h-3 w-48" />
      </div>
    ))}
  </div>
);

export const TableSkeleton: React.FC<{ rows?: number }> = ({ rows = 5 }) => (
  <div className="w-full rounded-xl border border-[#E3E8EE] dark:border-[#1E2630] bg-white dark:bg-[#11171E] overflow-hidden my-4">
    <div className="p-4 border-b border-[#E3E8EE] dark:border-[#1E2630] flex justify-between items-center">
      <Skeleton className="h-5 w-40" />
      <Skeleton className="h-8 w-24" />
    </div>
    <div className="divide-y divide-[#E3E8EE] dark:divide-[#1E2630]">
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="p-4 flex items-center justify-between gap-4">
          <Skeleton className="h-4 w-24" />
          <Skeleton className="h-4 w-32" />
          <Skeleton className="h-4 w-20" />
          <Skeleton className="h-4 w-16" />
          <Skeleton className="h-6 w-20 rounded-md" />
        </div>
      ))}
    </div>
  </div>
);

export const HeroLoadingSkeleton: React.FC = () => (
  <div className="max-w-4xl mx-auto py-10 px-4 space-y-6 animate-pulse text-center">
    <div className="space-y-3 flex flex-col items-center">
      <Skeleton className="h-12 w-3/4 max-w-2xl rounded-lg" />
      <Skeleton className="h-5 w-1/2 max-w-lg rounded" />
    </div>
    <div className="flex justify-center gap-3">
      <Skeleton className="h-10 w-36 rounded-lg" />
      <Skeleton className="h-10 w-36 rounded-lg" />
    </div>
    <div className="h-[150px] w-full max-w-3xl mx-auto rounded-xl border border-[#E3E8EE] dark:border-[#1E2630] bg-white dark:bg-[#11171E]" />
  </div>
);

export const PageLoadingSkeleton: React.FC = () => (
  <div className="space-y-6 py-4 max-w-7xl mx-auto">
    <div className="space-y-2 pb-5 border-b border-[#E3E8EE] dark:border-[#1E2630]">
      <Skeleton className="h-8 w-64" />
      <Skeleton className="h-4 w-96 max-w-full" />
    </div>
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      {Array.from({ length: 4 }).map((_, i) => (
        <div
          key={i}
          className="p-5 rounded-xl border border-[#E3E8EE] dark:border-[#1E2630] bg-white dark:bg-[#11171E] space-y-3"
        >
          <Skeleton className="h-3 w-24" />
          <Skeleton className="h-7 w-32" />
          <Skeleton className="h-3 w-40" />
        </div>
      ))}
    </div>
    <TableSkeleton rows={6} />
  </div>
);
