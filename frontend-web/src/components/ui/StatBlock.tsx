import React from 'react';
import { Tooltip } from './Tooltip';
import { HelpCircle } from 'lucide-react';

export interface StatBlockProps {
  label: string;
  value: string | number;
  delta?: string;
  deltaType?: 'positive' | 'negative' | 'neutral';
  subtext?: string;
  trend?: string;
  definition?: string;
  variant?: 'default' | 'vermilion' | 'amber' | 'green';
}

export const StatBlock: React.FC<StatBlockProps> = ({
  label,
  value,
  delta,
  deltaType = 'neutral',
  subtext,
  trend,
  definition,
  variant = 'default',
}) => {
  const valueColor = {
    default: 'text-text',
    vermilion: 'text-danger',
    amber: 'text-warn',
    green: 'text-accent',
  }[variant];

  const deltaColor = {
    positive: 'text-accent font-semibold',
    negative: 'text-danger font-semibold',
    neutral: 'text-muted font-normal',
  }[deltaType];

  return (
    <div className="bg-surface border border-border rounded-lg p-5 shadow-card transition-colors">
      <div className="flex items-center justify-between gap-1.5 mb-2">
        <span className="text-xs font-normal text-muted normal-case">
          {label}
        </span>
        {definition && (
          <Tooltip content={definition} side="top">
            <HelpCircle className="w-3.5 h-3.5 text-muted hover:text-text transition-colors" />
          </Tooltip>
        )}
      </div>

      <div className={`text-xl sm:text-2xl font-semibold font-sans tracking-normal tabular-nums ${valueColor}`}>
        {value}
      </div>

      {(delta || subtext || trend) && (
        <div className="flex flex-wrap items-center gap-2 mt-2 text-xs text-muted">
          {delta && <span className={`tabular-nums ${deltaColor}`}>{delta}</span>}
          {trend && <span className="tabular-nums font-semibold">{trend}</span>}
          {subtext && <span>{subtext}</span>}
        </div>
      )}
    </div>
  );
};
