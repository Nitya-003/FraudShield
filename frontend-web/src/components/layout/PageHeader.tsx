import React from 'react';

export interface PageHeaderProps {
  title: string;
  purpose: string;
  action?: React.ReactNode;
  badge?: React.ReactNode;
}

export const PageHeader: React.FC<PageHeaderProps> = ({
  title,
  purpose,
  action,
  badge,
}) => {
  return (
    <header className="border-b border-border pb-6 mb-6 flex flex-col md:flex-row md:items-end justify-between gap-4">
      <div className="space-y-1.5">
        <div className="flex items-center gap-3">
          <h1 className="text-xl font-semibold font-sans tracking-normal text-text">
            {title}
          </h1>
          {badge}
        </div>
        <p className="text-sm text-muted font-sans font-normal leading-relaxed max-w-2xl">
          {purpose}
        </p>
      </div>

      {action && (
        <div className="flex items-center gap-2.5 shrink-0 w-full sm:w-auto">
          {action}
        </div>
      )}
    </header>
  );
};
