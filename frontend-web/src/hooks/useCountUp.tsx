import React, { useState, useEffect, useRef } from 'react';

export interface UseCountUpOptions {
  duration?: number;
  decimals?: number;
  format?: 'inr' | 'pct' | 'num' | 'decimal';
  prefix?: string;
  suffix?: string;
}

function formatValue(
  val: number,
  format: 'inr' | 'pct' | 'num' | 'decimal',
  decimals: number,
  prefix: string,
  suffix: string
): string {
  if (format === 'inr') {
    const intVal = Math.round(val);
    return `${prefix}₹${intVal.toLocaleString('en-IN')}${suffix}`;
  }
  if (format === 'pct') {
    return `${prefix}${val.toFixed(decimals)}%${suffix}`;
  }
  if (format === 'decimal') {
    return `${prefix}${val.toLocaleString('en-US', {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    })}${suffix}`;
  }
  return `${prefix}${Math.round(val).toLocaleString('en-IN')}${suffix}`;
}

export function useCountUp(
  endValue: number | undefined | null,
  options: UseCountUpOptions = {}
): string {
  const { duration = 800, decimals = 0, format = 'num', prefix = '', suffix = '' } = options;
  const [displayValue, setDisplayValue] = useState<string>(() => {
    if (endValue === undefined || endValue === null) return '—';
    return formatValue(endValue, format, decimals, prefix, suffix);
  });
  const prevValueRef = useRef<number | null>(null);

  useEffect(() => {
    if (endValue === undefined || endValue === null) {
      setDisplayValue('—');
      return;
    }

    const prefersReducedMotion =
      typeof window !== 'undefined' &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    if (prefersReducedMotion) {
      setDisplayValue(formatValue(endValue, format, decimals, prefix, suffix));
      prevValueRef.current = endValue;
      return;
    }

    const startValue = prevValueRef.current === null ? 0 : prevValueRef.current;
    prevValueRef.current = endValue;

    if (startValue === endValue && prevValueRef.current !== null) {
      setDisplayValue(formatValue(endValue, format, decimals, prefix, suffix));
      return;
    }

    const animDuration = Math.min(duration, 400);
    const startTime = performance.now();
    let animationFrameId: number;

    const tick = (now: number) => {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / animDuration, 1);
      // Smooth cubic ease-out
      const easeProgress = 1 - Math.pow(1 - progress, 3);
      const current = startValue + (endValue - startValue) * easeProgress;

      setDisplayValue(formatValue(current, format, decimals, prefix, suffix));

      if (progress < 1) {
        animationFrameId = requestAnimationFrame(tick);
      } else {
        setDisplayValue(formatValue(endValue, format, decimals, prefix, suffix));
      }
    };

    animationFrameId = requestAnimationFrame(tick);

    return () => {
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
    };
  }, [endValue, duration, decimals, format, prefix, suffix]);

  return displayValue;
}

export interface AnimatedNumberProps extends UseCountUpOptions {
  value: number | undefined | null;
  className?: string;
}

export const AnimatedNumber: React.FC<AnimatedNumberProps> = ({
  value,
  duration,
  decimals,
  format,
  prefix,
  suffix,
  className = '',
}) => {
  const display = useCountUp(value, { duration, decimals, format, prefix, suffix });
  return <span className={`tabular-nums ${className}`}>{display}</span>;
};

