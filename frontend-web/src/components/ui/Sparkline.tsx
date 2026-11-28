import React, { useMemo } from 'react';

export interface SparklineProps {
  data: number[];
  width?: number;
  height?: number;
  className?: string;
  color?: string;
}

export const Sparkline: React.FC<SparklineProps> = ({
  data,
  width = 100,
  height = 32,
  className = '',
  color = 'currentColor',
}) => {
  const gradientId = useMemo(
    () => `sparkline-grad-${Math.random().toString(36).substring(2, 9)}`,
    []
  );

  const { points, areaPoints } = useMemo(() => {
    if (!data || data.length < 2) {
      return { points: '', areaPoints: '' };
    }

    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;
    const padding = 3;
    const usableHeight = height - padding * 2;
    const stepX = (width - padding * 2) / (data.length - 1);

    const coords = data.map((val, idx) => {
      const x = padding + idx * stepX;
      // Invert Y because SVG 0 is top
      const y = height - padding - ((val - min) / range) * usableHeight;
      return [x.toFixed(1), y.toFixed(1)];
    });

    const pts = coords.map(([x, y]) => `${x},${y}`).join(' ');
    const lastX = coords[coords.length - 1][0];
    const firstX = coords[0][0];
    const bottomY = (height - padding).toFixed(1);
    const area = `${firstX},${bottomY} ${pts} ${lastX},${bottomY}`;

    return { points: pts, areaPoints: area };
  }, [data, width, height]);

  if (!points) {
    return null;
  }

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className={`overflow-visible select-none ${className}`}
      width={width}
      height={height}
      aria-hidden="true"
    >
      <defs>
        <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.18" />
          <stop offset="100%" stopColor={color} stopOpacity="0.0" />
        </linearGradient>
      </defs>

      {/* Subtle area fill */}
      <polygon points={areaPoints} fill={`url(#${gradientId})`} />

      {/* Main crisp line */}
      <polyline
        fill="none"
        stroke={color}
        strokeWidth="2.2"
        strokeLinecap="round"
        strokeLinejoin="round"
        points={points}
      />
    </svg>
  );
};

