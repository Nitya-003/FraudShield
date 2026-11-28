/* eslint-disable react-refresh/only-export-components */
import React, { useMemo, useState, useEffect, useRef } from 'react';
import ReactEChartsCore from 'echarts-for-react/lib/core';
import * as echarts from 'echarts/core';
import {
  BarChart,
  LineChart,
  PieChart,
  GraphChart,
  ScatterChart,
} from 'echarts/charts';
import {
  TitleComponent,
  TooltipComponent,
  GridComponent,
  LegendComponent,
  MarkLineComponent,
  MarkAreaComponent,
} from 'echarts/components';
import { CanvasRenderer, SVGRenderer } from 'echarts/renderers';
import type { EChartsOption } from 'echarts';

// Register strictly modular chart types and components required by FraudShield
echarts.use([
  TitleComponent,
  TooltipComponent,
  GridComponent,
  LegendComponent,
  MarkLineComponent,
  MarkAreaComponent,
  BarChart,
  LineChart,
  PieChart,
  GraphChart,
  ScatterChart,
  CanvasRenderer,
  SVGRenderer,
]);

interface ChartWrapperProps {
  option: EChartsOption;
  height?: number | string;
  className?: string;
  isDark?: boolean;
  onEvents?: Record<string, (params: unknown) => void>;
}

export const ChartWrapper: React.FC<ChartWrapperProps> = ({
  option,
  height = 320,
  className = '',
  isDark = false,
  onEvents,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [isVisible, setIsVisible] = useState(false);

  // Lazy-mount ECharts only when the chart scrolls into view
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    // If already visible or no IntersectionObserver, render directly
    if (isVisible || typeof window === 'undefined' || !('IntersectionObserver' in window)) {
      setIsVisible(true);
      return;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0]?.isIntersecting) {
          setIsVisible(true);
          observer.disconnect();
        }
      },
      { rootMargin: '100px 0px' }
    );

    observer.observe(el);
    return () => observer.disconnect();
  }, [isVisible]);

  // Rebuilt design tokens: calm security control room palette
  const mergedOption = useMemo<EChartsOption>(() => {
    const textColor = isDark ? '#E8EDF2' : '#0F172A';
    const mutedColor = isDark ? '#8A96A3' : '#52606D';
    const gridlineColor = isDark ? '#1E2630' : '#E3E8EE';
    const tooltipBg = isDark ? '#11171E' : '#FFFFFF';
    const tooltipBorder = isDark ? '#1E2630' : '#E3E8EE';

    return {
      backgroundColor: 'transparent',
      textStyle: {
        fontFamily: 'Inter, system-ui, -apple-system, sans-serif',
        color: textColor,
      },
      grid: {
        top: 36,
        right: 24,
        bottom: 32,
        left: 48,
        containLabel: true,
        borderColor: gridlineColor,
        ...((option.grid as object) || {}),
      },
      tooltip: {
        trigger: 'axis',
        backgroundColor: tooltipBg,
        borderColor: tooltipBorder,
        borderWidth: 1,
        padding: [10, 14],
        textStyle: {
          color: textColor,
          fontSize: 12,
          fontFamily: 'JetBrains Mono, GeistMono, Menlo, monospace',
        },
        extraCssText:
          'border-radius: 8px; box-shadow: 0 4px 16px -2px rgba(0,0,0,0.25); font-feature-settings: "tnum";',
        ...((option.tooltip as object) || {}),
      },
      xAxis: Array.isArray(option.xAxis)
        ? option.xAxis.map((x) => ({
            axisLine: { lineStyle: { color: gridlineColor } },
            axisLabel: {
              color: mutedColor,
              fontFamily: 'JetBrains Mono, monospace',
              fontSize: 12,
            },
            splitLine: { show: false },
            ...x,
          }))
        : option.xAxis
        ? {
            axisLine: { lineStyle: { color: gridlineColor } },
            axisLabel: {
              color: mutedColor,
              fontFamily: 'JetBrains Mono, monospace',
              fontSize: 12,
            },
            splitLine: { show: false },
            ...option.xAxis,
          }
        : undefined,
      yAxis: Array.isArray(option.yAxis)
        ? option.yAxis.map((y) => ({
            axisLine: { show: false },
            axisLabel: {
              color: mutedColor,
              fontFamily: 'JetBrains Mono, monospace',
              fontSize: 12,
            },
            splitLine: {
              lineStyle: {
                color: gridlineColor,
                type: 'dashed',
              },
            },
            ...y,
          }))
        : option.yAxis
        ? {
            axisLine: { show: false },
            axisLabel: {
              color: mutedColor,
              fontFamily: 'JetBrains Mono, monospace',
              fontSize: 12,
            },
            splitLine: {
              lineStyle: {
                color: gridlineColor,
                type: 'dashed',
              },
            },
            ...option.yAxis,
          }
        : undefined,
      ...option,
    };
  }, [option, isDark]);

  return (
    <div
      ref={containerRef}
      className={`w-full overflow-hidden ${className}`}
      style={{ minHeight: typeof height === 'number' ? `${height}px` : height }}
    >
      {isVisible ? (
        <ReactEChartsCore
          echarts={echarts}
          option={mergedOption}
          style={{ height, width: '100%' }}
          opts={{ renderer: 'svg' }}
          notMerge={false}
          lazyUpdate={true}
          onEvents={onEvents}
        />
      ) : (
        <div
          className="w-full flex items-center justify-center text-xs text-muted font-mono animate-pulse bg-surface-subtle/50 rounded-lg"
          style={{ height }}
        >
          Loading chart view...
        </div>
      )}
    </div>
  );
};

/**
 * Standard Threshold Lines for Risk Score Charts (decline at 0.85, review at 0.45)
 */
export const THRESHOLD_MARK_LINES = {
  data: [
    {
      yAxis: 0.85,
      name: 'Decline threshold',
      lineStyle: { color: '#E11D48', type: 'dashed' as const, width: 1.5 },
      label: {
        show: true,
        position: 'insideEndTop' as const,
        formatter: 'Decline (0.85)',
        fontSize: 12,
        fontFamily: 'JetBrains Mono, monospace',
        color: '#E11D48',
      },
    },
    {
      yAxis: 0.45,
      name: 'Review threshold',
      lineStyle: { color: '#D97706', type: 'dashed' as const, width: 1.5 },
      label: {
        show: true,
        position: 'insideEndTop' as const,
        formatter: 'Review (0.45)',
        fontSize: 12,
        fontFamily: 'JetBrains Mono, monospace',
        color: '#D97706',
      },
    },
  ],
};
