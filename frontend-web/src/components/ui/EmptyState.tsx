import React from 'react';
import { Inbox } from 'lucide-react';

interface EmptyStateProps {
  title?: string;
  description?: string;
  icon?: React.ComponentType<{ className?: string }>;
  actionLabel?: string;
  onAction?: () => void;
  className?: string;
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  title = 'No Records Found',
  description = 'No transactions or data entries match the selected filters.',
  icon: Icon = Inbox,
  actionLabel,
  onAction,
  className = '',
}) => {
  return (
    <div
      className={`p-12 text-center rounded-lg border border-dashed border-border bg-surface/50 ${className}`}
    >
      <div className="w-12 h-12 rounded-full bg-surface-subtle text-muted flex items-center justify-center mx-auto mb-4">
        <Icon className="w-6 h-6" />
      </div>
      <h3 className="font-sans font-semibold text-base text-text mb-1">
        {title}
      </h3>
      <p className="text-xs text-muted max-w-sm mx-auto mb-5">
        {description}
      </p>
      {actionLabel && onAction && (
        <button
          onClick={onAction}
          className="inline-flex items-center px-3.5 py-1.5 rounded text-xs font-semibold bg-stone-900 text-stone-50 dark:bg-stone-100 dark:text-stone-900 hover:opacity-90 transition-opacity"
        >
          {actionLabel}
        </button>
      )}
    </div>
  );
};
