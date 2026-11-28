import * as React from 'react';
import { cva, type VariantProps } from 'class-variance-authority';
import { CheckCircle2, AlertTriangle, AlertOctagon, Info } from 'lucide-react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: (string | undefined | null | false)[]) {
  return twMerge(clsx(inputs));
}

// eslint-disable-next-line react-refresh/only-export-components
export const badgeVariants = cva(
  'inline-flex items-center gap-1.5 px-2 py-0.5 rounded-sm text-xs font-semibold border select-none transition-colors',
  {
    variants: {
      variant: {
        low: 'bg-emerald-500/10 text-emerald-800 border-emerald-500/30 dark:bg-emerald-500/15 dark:text-emerald-300 dark:border-emerald-500/40',
        medium: 'bg-amber-500/10 text-amber-900 border-amber-500/30 dark:bg-amber-500/15 dark:text-amber-300 dark:border-amber-500/40',
        high: 'bg-rose-500/10 text-rose-800 border-rose-500/30 dark:bg-rose-500/15 dark:text-rose-300 dark:border-rose-500/40',
        accent: 'bg-accent-subtle text-accent border-accent/30',
        neutral: 'bg-surface-subtle text-muted border-border',
        outline: 'bg-transparent text-text border-border',
      },
    },
    defaultVariants: {
      variant: 'neutral',
    },
  }
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badgeVariants> {
  icon?: React.ReactNode;
  showIcon?: boolean;
}

export const Badge: React.FC<BadgeProps> = ({
  className,
  variant = 'neutral',
  icon,
  showIcon = true,
  children,
  ...props
}) => {
  const getDefaultIcon = () => {
    switch (variant) {
      case 'low':
        return <CheckCircle2 className="w-3 h-3 shrink-0 text-current" />;
      case 'medium':
        return <AlertTriangle className="w-3 h-3 shrink-0 text-current" />;
      case 'high':
        return <AlertOctagon className="w-3 h-3 shrink-0 text-current" />;
      case 'accent':
        return <CheckCircle2 className="w-3 h-3 shrink-0 text-current" />;
      case 'neutral':
      default:
        return <Info className="w-3 h-3 shrink-0 text-current" />;
    }
  };

  return (
    <span className={cn(badgeVariants({ variant }), className)} {...props}>
      {showIcon && (icon || getDefaultIcon())}
      <span>{children}</span>
    </span>
  );
};

export interface RiskBadgeProps {
  band: 'Low' | 'Medium' | 'High' | string;
  className?: string;
}

/**
 * Standardized Risk Badge: Icon + Label + Contrast border (never color alone)
 */
export const RiskBadge: React.FC<RiskBadgeProps> = ({ band, className }) => {
  const normalized = (band || '').toLowerCase();
  if (normalized.includes('low') || normalized.includes('approve')) {
    return (
      <Badge variant="low" className={className}>
        {band || 'Low'}
      </Badge>
    );
  }
  if (normalized.includes('med') || normalized.includes('step')) {
    return (
      <Badge variant="medium" className={className}>
        {band || 'Medium'}
      </Badge>
    );
  }
  if (normalized.includes('high') || normalized.includes('decline') || normalized.includes('block')) {
    return (
      <Badge variant="high" className={className}>
        {band || 'High'}
      </Badge>
    );
  }
  return (
    <Badge variant="neutral" className={className}>
      {band}
    </Badge>
  );
};
