import React from 'react';
import { AlertCircle, RefreshCw } from 'lucide-react';

interface ErrorStateProps {
  title?: string;
  message?: string;
  onRetry?: () => void;
  className?: string;
}

export const ErrorState: React.FC<ErrorStateProps> = ({
  title = 'Failed to Load Data',
  message = 'An error occurred while connecting to the decision API.',
  onRetry,
  className = '',
}) => {
  return (
    <div
      className={`p-8 text-center rounded-lg border border-red-200 dark:border-red-900/30 bg-red-50/40 dark:bg-red-950/10 ${className}`}
    >
      <div className="w-10 h-10 rounded-full bg-red-100 dark:bg-red-900/40 text-red-600 dark:text-red-400 flex items-center justify-center mx-auto mb-3">
        <AlertCircle className="w-5 h-5" />
      </div>
      <h4 className="font-sans font-semibold text-sm text-text mb-1">
        {title}
      </h4>
      <p className="text-xs text-muted max-w-sm mx-auto mb-4">
        {message}
      </p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-semibold border border-border bg-surface hover:bg-surface-subtle text-text transition-colors"
        >
          <RefreshCw className="w-3 h-3" />
          <span>Retry Request</span>
        </button>
      )}
    </div>
  );
};

