import React, { useState, useEffect } from 'react';
import { Database, X } from 'lucide-react';

export const ProvenanceBanner: React.FC = () => {
  const [dismissed, setDismissed] = useState<boolean>(true); // start hidden to avoid layout shift before checking sessionStorage

  useEffect(() => {
    try {
      const isDismissed = sessionStorage.getItem('fraudshield_provenance_dismissed');
      if (isDismissed !== 'true') {
        setDismissed(false);
      }
    } catch {
      setDismissed(false);
    }
  }, []);

  const handleDismiss = () => {
    try {
      sessionStorage.setItem('fraudshield_provenance_dismissed', 'true');
    } catch {
      // ignore
    }
    setDismissed(true);
  };

  if (dismissed) return null;

  return (
    <div
      role="region"
      aria-label="Data Provenance Notice"
      className="w-full border-b border-border bg-surface-subtle text-text transition-colors"
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-2 flex items-center justify-between gap-3 text-xs">
        <div className="flex items-center gap-2 min-w-0">
          <Database className="w-3.5 h-3.5 shrink-0 text-stone-600 dark:text-stone-300" />
          <p className="truncate sm:whitespace-normal font-sans leading-tight">
            <span className="font-semibold text-stone-900 dark:text-stone-100">Data Provenance: </span>
            <span className="text-stone-700 dark:text-stone-300">
              Evaluated strictly on public academic benchmarks (ULB Card, PaySim, BAF) &amp; synthetic Indian UPI streams. All metrics derived on held-out test splits with 95% bootstrap confidence intervals.
            </span>
          </p>
        </div>
        <button
          onClick={handleDismiss}
          aria-label="Dismiss data provenance notice for this session"
          className="shrink-0 p-1 rounded hover:bg-stone-200 dark:hover:bg-stone-800 text-stone-600 dark:text-stone-300 hover:text-stone-900 dark:hover:text-stone-100 transition-colors"
          title="Dismiss for this session"
        >
          <X className="w-3.5 h-3.5" />
        </button>
      </div>
    </div>
  );
};
