import React, { useMemo, useState, useCallback } from 'react';
import { useQuery, keepPreviousData } from '@tanstack/react-query';
import { motion } from 'framer-motion';
import { fetchOverviewKPIs, fetchDailySeries, fetchRiskBands } from '../api/client';
import { Badge } from '../components/ui/Badge';
import { ChartWrapper } from '../components/ui/ChartWrapper';
import { Tooltip } from '../components/ui/Tooltip';
import { PageHeader } from '../components/layout/PageHeader';
import { Button } from '../components/ui/Button';
import { Sparkline } from '../components/ui/Sparkline';
import { AnimatedNumber } from '../hooks/useCountUp';
import { useChannel } from '../context/ChannelContext';
import {
  ArrowRight,
  WarningCircle,
  Question,
  ArrowClockwise,
} from '@phosphor-icons/react';
import type { EChartsOption } from 'echarts';
import { ChannelBenchmarkTable } from '../components/common/ChannelBenchmarkTable';

export interface OverviewPageProps {
  onNavigate: (tab: string) => void;
  isDark: boolean;
}

export const OverviewPage: React.FC<OverviewPageProps> = ({ onNavigate, isDark }) => {
  const { channel, setChannel, channelMeta } = useChannel();
  const [, setLastRefreshed] = useState<Date>(() => new Date());

  // Queries driven by active navbar channel with 5-minute staleTime and keepPreviousData
  const {
    data: kpis,
    isLoading: kpisLoading,
    isError: kpisError,
    refetch: refetchKPIs,
    isFetching: kpisFetching,
  } = useQuery({
    queryKey: ['overviewKPIs', channel],
    queryFn: () => fetchOverviewKPIs(channel),
    placeholderData: keepPreviousData,
    staleTime: 300000,
  });

  const {
    data: dailyData,
    isLoading: dailyLoading,
    refetch: refetchDaily,
    isFetching: dailyFetching,
  } = useQuery({
    queryKey: ['dailySeries', channel],
    queryFn: () => fetchDailySeries(channel),
    placeholderData: keepPreviousData,
    staleTime: 300000,
  });

  const {
    data: riskBands,
    isLoading: bandsLoading,
    refetch: refetchBands,
  } = useQuery({
    queryKey: ['riskBands', channel],
    queryFn: () => fetchRiskBands(channel),
    placeholderData: keepPreviousData,
    staleTime: 300000,
  });

  const isRefreshing = kpisFetching || dailyFetching;

  const handleManualRefresh = useCallback(async () => {
    await Promise.all([refetchKPIs(), refetchDaily(), refetchBands()]);
    setLastRefreshed(new Date());
  }, [refetchKPIs, refetchDaily, refetchBands]);

  // Sparkline data for hero metric from daily data
  const sparklineData = useMemo(() => {
    if (!dailyData?.series || dailyData.series.length === 0) {
      return [];
    }
    return dailyData.series.map((s) => s.tx_count);
  }, [dailyData]);

  // ECharts Option for Daily Volume & High-Risk Alert Trajectory
  // Two-axis theme with light gridlines, direct end labels, no legend box,
  // tooltip with exact values, zero-based volume axis, alerts as distinct markers
  const chartOption: EChartsOption = useMemo(() => {
    const hours = dailyData?.series?.map((s) => s.hour_label) || [];
    const txCounts = dailyData?.series?.map((s) => s.tx_count) || [];
    const alertCounts = dailyData?.series?.map((s) => s.high_risk_count) || [];

    const textColor = isDark ? '#94A3B8' : '#52606D';
    const gridLineColor = isDark ? 'rgba(255, 255, 255, 0.06)' : 'rgba(0, 0, 0, 0.06)';
    const barFill = isDark ? 'rgba(45, 212, 160, 0.18)' : 'rgba(4, 120, 87, 0.12)';
    const barBorder = isDark ? '#2DD4A0' : '#047857';
    const alertColor = isDark ? '#F43F5E' : '#B91C1C';

    return {
      tooltip: {
        trigger: 'axis',
        backgroundColor: isDark ? '#11171E' : '#FFFFFF',
        borderColor: isDark ? '#1E2630' : '#E2E8F0',
        textStyle: { color: isDark ? '#F1F5F9' : '#0F172A', fontSize: 12 },
        formatter: (params: unknown) => {
          const arr = Array.isArray(params) ? params : [params];
          if (arr.length === 0) return '';
          const time = (arr[0] as { axisValueLabel?: string }).axisValueLabel || '';
          let html = `<div style="font-weight:600;margin-bottom:6px;font-family:Geist Mono,monospace;font-size:12px;">${time}</div>`;
          arr.forEach((item: unknown) => {
            const it = item as { seriesName?: string; value?: number; color?: string };
            const val = typeof it.value === 'number' ? it.value.toLocaleString() : it.value;
            html += `<div style="display:flex;align-items:center;justify-content:space-between;gap:14px;font-size:12px;margin-top:2px;">
              <span style="display:inline-flex;align-items:center;gap:6px;">
                <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${it.color};"></span>
                <span>${it.seriesName}</span>
              </span>
              <span style="font-weight:600;font-family:Geist Mono,monospace;">${val}</span>
            </div>`;
          });
          return html;
        },
      },
      grid: {
        top: 40,
        right: 48,
        bottom: 24,
        left: 48,
        containLabel: true,
      },
      xAxis: {
        type: 'category',
        data: hours,
        axisLine: { lineStyle: { color: gridLineColor } },
        axisLabel: {
          color: textColor,
          fontSize: 12,
          fontFamily: 'Geist Mono, monospace',
          interval: Math.max(0, Math.floor(hours.length / 6)),
        },
      },
      yAxis: [
        {
          type: 'value',
          min: 0, // Zero-based volume axis
          name: 'Volume (Tx)',
          nameTextStyle: { fontSize: 12, color: textColor, padding: [0, 0, 4, 0] },
          nameGap: 14,
          splitLine: { lineStyle: { color: gridLineColor, type: 'dashed' } },
          axisLabel: { color: textColor, fontSize: 12, fontFamily: 'Geist Mono, monospace' },
        },
        {
          type: 'value',
          min: 0, // Zero-based alerts axis
          name: 'Alerts',
          nameTextStyle: { fontSize: 12, color: alertColor, padding: [0, 0, 4, 0] },
          nameGap: 14,
          splitLine: { show: false },
          axisLabel: { color: alertColor, fontSize: 12, fontFamily: 'Geist Mono, monospace' },
          minInterval: 1,
        },
      ],
      series: [
        {
          name: 'Evaluated Volume',
          type: 'bar',
          barWidth: '38%',
          data: txCounts,
          itemStyle: {
            color: barFill,
            borderColor: barBorder,
            borderWidth: 1,
            borderRadius: [2, 2, 0, 0],
          },
        },
        {
          name: 'High-Risk Alerts',
          type: 'line',
          yAxisIndex: 1,
          smooth: false,
          data: alertCounts,
          lineStyle: { color: alertColor, width: 2 },
          itemStyle: { color: alertColor },
          symbol: 'circle',
          symbolSize: 7, // Alerts as distinct markers
          endLabel: {
            show: true,
            formatter: '{a}',
            color: alertColor,
            fontSize: 12,
            fontFamily: 'Geist Sans, sans-serif',
            offset: [8, 0],
          },
        },
      ],
    };
  }, [dailyData, isDark]);

  const headline = kpis?.headline_metric;
  const supporting = kpis?.supporting_metrics || [];
  const perChannelTable = useMemo(() => {
    return kpis?.per_channel_table || [];
  }, [kpis?.per_channel_table]);

  // Channel meta lookup
  const isAll = channel === 'ALL' || !channel;

  // Calculate sum of alerts across channels for "ALL" view
  const totalAlertsAcrossRails = useMemo(() => {
    return perChannelTable.reduce((acc, row) => acc + (row.alerts_per_day || 0), 0);
  }, [perChannelTable]);

  return (
    <div className="space-y-6 pb-[calc(3rem+env(safe-area-inset-bottom,0px))] font-sans content-auto">
      {/* Consistent Page Header: Title + One Line Purpose + Actions */}
      <PageHeader
        title="Overview"
        purpose="Transaction evaluation volume, protected capital, and operational alert distribution."
        action={
          <Button
            variant="primary"
            size="sm"
            onClick={() => onNavigate('transactions')}
            className="w-full sm:w-auto justify-center min-h-[36px]"
          >
            <span>View Ledger</span>
            <ArrowRight className="w-3.5 h-3.5 ml-1" weight="thin" />
          </Button>
        }
      />

      {/* Short Muted Provenance Line + Small Refresh Icon Button */}
      <div className="flex items-center justify-between text-xs text-muted pt-0 pb-1">
        <p className="font-normal truncate">
          <span>{channelMeta.name}</span>
          <span className="mx-1.5 opacity-60">·</span>
          <span>{channelMeta.dataSource}</span>
        </p>

        <Tooltip content="Updated just now · Click to refresh" side="left">
          <button
            type="button"
            onClick={handleManualRefresh}
            disabled={isRefreshing}
            className="p-1.5 rounded-md border border-border bg-surface hover:bg-surface-subtle text-muted hover:text-text transition-colors disabled:opacity-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent"
            aria-label="Refresh overview metrics"
          >
            <ArrowClockwise
              className={`w-3.5 h-3.5 ${isRefreshing ? 'animate-spin' : ''}`}
              weight="bold"
            />
          </button>
        </Tooltip>
      </div>

      {/* Error State with Retry Action */}
      {kpisError && (
        <div className="p-4 rounded-xl border border-danger/30 bg-danger/10 text-danger text-xs flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <WarningCircle className="w-4 h-4 shrink-0" weight="thin" />
            <span>Metrics endpoint unreachable. Check backend connection.</span>
          </div>
          <Button variant="secondary" size="sm" onClick={() => handleManualRefresh()}>
            Retry
          </Button>
        </div>
      )}

      {/* =========================================================================
          TOP ROW KPIS: Frauds caught, Recall, False alarms per 100k, Alerts per day
          (For All channels: shows channel count & tiny per-channel recall dot strip)
          ========================================================================= */}
      <motion.div
        key={`kpis-${channel}`}
        initial={{ opacity: 0.85 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
        className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4"
      >
        {kpisLoading ? (
          Array.from({ length: 4 }).map((_, idx) => (
            <div key={idx} className="p-5 rounded-xl border border-border bg-surface min-h-[148px] flex flex-col justify-between">
              <div className="h-3 w-28 bg-border/60 rounded animate-pulse" />
              <div className="h-7 w-36 bg-border/60 rounded animate-pulse" />
              <div className="h-3 w-20 bg-border/60 rounded animate-pulse" />
            </div>
          ))
        ) : (
          <>
            {/* KPI 1: FRAUDS CAUGHT */}
            <div className="p-5 rounded-xl border border-border bg-surface min-h-[148px] flex flex-col justify-between relative overflow-hidden shadow-card">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-muted">
                  Frauds caught
                </span>
                <Tooltip
                  content={
                    headline?.definition ||
                    'Estimated monetary fraud value or incident count averted through model interventions.'
                  }
                  side="top"
                >
                  <Question className="w-3.5 h-3.5 text-muted hover:text-text transition-colors cursor-pointer" weight="thin" />
                </Tooltip>
              </div>

              <div className="text-xl font-semibold font-sans tracking-normal text-text tabular-nums">
                {headline?.format === 'currency' ? (
                  headline?.currency_symbol === '€' ? (
                    <span>€<AnimatedNumber value={headline.value} format="decimal" decimals={2} duration={900} /></span>
                  ) : headline?.currency_symbol === '£' ? (
                    <span>£<AnimatedNumber value={headline.value} format="decimal" decimals={0} duration={900} /></span>
                  ) : (
                    <AnimatedNumber value={headline.value} format="inr" duration={900} />
                  )
                ) : (
                  <AnimatedNumber value={headline?.value} format="num" duration={900} />
                )}
              </div>

              <div className="flex items-start justify-between gap-2 pt-1">
                <span className="text-xs text-muted font-sans leading-snug line-clamp-1" title={headline?.sub_label}>
                  {isAll ? 'Native currencies (EUR, INR, GBP)' : (headline?.sub_label || 'Held-out test split')}
                </span>

                {sparklineData.length > 0 && (
                  <Sparkline
                    data={sparklineData}
                    width={70}
                    height={20}
                    className="text-accent shrink-0 mt-0.5"
                  />
                )}
              </div>
            </div>

            {/* KPI 2: RECALL */}
            <div className="p-5 rounded-xl border border-border bg-surface min-h-[148px] flex flex-col justify-between shadow-card">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-muted">
                  Recall
                </span>
                <Tooltip
                  content={
                    isAll
                      ? 'Per-channel recall evaluated strictly on held-out benchmark splits. Never blended.'
                      : (supporting[0]?.definition || 'Percentage of ground-truth fraudulent transactions caught.')
                  }
                  side="top"
                >
                  <Question className="w-3.5 h-3.5 text-muted hover:text-text transition-colors cursor-pointer" weight="thin" />
                </Tooltip>
              </div>

              {isAll ? (
                <div className="space-y-2">
                  <div className="flex items-baseline justify-between">
                    <span className="text-xl font-semibold text-text tabular-nums">4 rails</span>
                    <span className="text-xs text-muted font-mono">Unblended</span>
                  </div>
                  {/* Tiny per-channel recall dot strip */}
                  <div className="grid grid-cols-4 gap-1.5 pt-0.5">
                    {perChannelTable.map((ch) => (
                      <div
                        key={ch.channel_key}
                        onClick={() => setChannel(ch.channel_key)}
                        className="p-1 rounded bg-surface-subtle border border-border flex flex-col items-center hover:border-accent/40 transition-colors cursor-pointer"
                        title={`${ch.channel_name}: ${ch.recall_pct.toFixed(1)}% recall`}
                      >
                        <span className="text-xs text-muted font-mono">
                          {ch.channel_key === 'transfer_wallet' ? 'Wal' : ch.channel_key.slice(0, 3)}
                        </span>
                        <span className="font-mono font-semibold text-xs tabular-nums text-accent">
                          {ch.recall_pct.toFixed(0)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <>
                  <div className="text-xl font-semibold font-sans tracking-normal text-accent tabular-nums">
                    <AnimatedNumber value={supporting[0]?.value ?? 0} format="pct" decimals={1} duration={800} />
                  </div>
                  <div className="text-xs text-muted font-sans leading-snug truncate" title={supporting[0]?.sub_label}>
                    {supporting[0]?.sub_label || 'Held-out test split'}
                  </div>
                </>
              )}
            </div>

            {/* KPI 3: FALSE ALARMS PER 100K */}
            <div className="p-5 rounded-xl border border-border bg-surface min-h-[148px] flex flex-col justify-between shadow-card">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-muted">
                  False alarms / 100k
                </span>
                <Tooltip
                  content={
                    isAll
                      ? 'False alarm rates depend on rail base rates and are evaluated per channel without blending.'
                      : (supporting[1]?.definition || 'Legitimate transactions falsely flagged per 100,000 evaluations.')
                  }
                  side="top"
                >
                  <Question className="w-3.5 h-3.5 text-muted hover:text-text transition-colors cursor-pointer" weight="thin" />
                </Tooltip>
              </div>

              <div className="text-xl font-semibold font-sans tracking-normal text-text tabular-nums">
                {isAll ? (
                  <span>1.6 – 4,903.8</span>
                ) : (
                  <AnimatedNumber value={supporting[1]?.value ?? 0} format="decimal" decimals={1} duration={800} />
                )}
              </div>

              <div className="text-xs text-muted font-sans leading-snug truncate">
                {isAll ? 'Preserved per rail distribution' : (supporting[1]?.sub_label || 'Held-out test split')}
              </div>
            </div>

            {/* KPI 4: ALERTS PER DAY */}
            <div className="p-5 rounded-xl border border-border bg-surface min-h-[148px] flex flex-col justify-between shadow-card">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-muted">
                  Alerts per day
                </span>
                <Tooltip
                  content={
                    isAll
                      ? 'Summed triage queue volume routed to analysts across all 4 rails.'
                      : (supporting[2]?.definition || 'Average daily alerts generated on current channel traffic.')
                  }
                  side="top"
                >
                  <Question className="w-3.5 h-3.5 text-muted hover:text-text transition-colors cursor-pointer" weight="thin" />
                </Tooltip>
              </div>

              <div className="text-xl font-semibold font-sans tracking-normal text-text tabular-nums">
                {isAll ? (
                  <AnimatedNumber value={totalAlertsAcrossRails} format="decimal" decimals={1} duration={800} />
                ) : (
                  <AnimatedNumber value={supporting[2]?.value ?? 0} format="decimal" decimals={1} duration={800} />
                )}
              </div>

              <div className="text-xs text-muted font-sans leading-snug truncate">
                {isAll ? 'Combined analyst queue' : (supporting[2]?.sub_label || 'Current traffic distribution')}
              </div>
            </div>
          </>
        )}
      </motion.div>

      {/* =========================================================================
          MAIN ELEMENT: CHANNELS BENCHMARK TABLE
          (Sentence-case headers, right-aligned numeric columns, tabular numbers,
          small horizontal bar in Recall & PR-AUC, coloured left marker, native currency suffix,
          row hover, selected-row state, title simply "Channels", currency note in tooltip)
          ========================================================================= */}
      <ChannelBenchmarkTable
        onSelectChannel={setChannel}
        selectedChannel={channel}
        showTitle={true}
      />

      {/* =========================================================================
          ANALYTICAL SECTION: Volume and alerts chart (7 cols) + Risk Bands (5 cols)
          ========================================================================= */}
      <motion.div
        key={`analytics-${channel}`}
        initial={{ opacity: 0.85 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
        className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start"
      >
        {/* Left (7 cols): Daily Volume & Alerts Chart */}
        <div className="lg:col-span-7 p-5 rounded-xl border border-border bg-surface space-y-4 shadow-card">
          <div className="flex items-center justify-between border-b border-border pb-3">
            <div>
              <h2 className="font-semibold text-sm text-text">
                Volume &amp; Risk Trajectory
              </h2>
              <p className="text-xs text-muted">
                Evaluated transaction volume vs flagged high-risk alerts over time.
              </p>
            </div>
            <div className="flex items-center gap-3 text-xs font-sans">
              <span className="flex items-center gap-1.5 text-muted">
                <span className="w-2.5 h-2.5 bg-accent/30 border border-accent rounded-sm" /> Tx Count
              </span>
              <span className="flex items-center gap-1.5 text-danger font-medium">
                <span className="w-2.5 h-2.5 bg-danger rounded-full" /> High Risk
              </span>
            </div>
          </div>

          {dailyLoading ? (
            /* Skeleton shaped like the chart (CLS 0) */
            <div className="h-[280px] w-full flex flex-col justify-end p-2 space-y-3">
              <div className="flex items-end justify-between gap-2 h-48 w-full border-b border-l border-border/80 pl-2 pb-1">
                {[45, 60, 30, 80, 55, 90, 70, 40, 85, 65, 95, 50].map((h, i) => (
                  <div
                    key={i}
                    className="w-full bg-border/60 rounded-t animate-pulse"
                    style={{ height: `${h}%` }}
                  />
                ))}
              </div>
              <div className="flex justify-between text-xs text-muted px-2">
                <span className="w-12 h-3 bg-border/60 rounded animate-pulse" />
                <span className="w-12 h-3 bg-border/60 rounded animate-pulse" />
                <span className="w-12 h-3 bg-border/60 rounded animate-pulse" />
              </div>
            </div>
          ) : (
            <div className="space-y-2">
              <ChartWrapper option={chartOption} height={280} isDark={isDark} />
              <p className="text-xs text-muted text-center sm:text-left">
                Data source: {dailyData?.data_source || 'Held-out test split'}. {dailyData?.is_sample ? 'Sampled chronological ledger slice.' : ''}
              </p>
            </div>
          )}
        </div>

        {/* Right (5 cols): Risk Bands Distribution & Table */}
        <div className="lg:col-span-5 p-5 rounded-xl border border-border bg-surface space-y-4 min-h-[360px] shadow-card">
          <div className="flex items-center justify-between border-b border-border pb-3">
            <div>
              <h2 className="font-semibold text-sm text-text">
                Risk Band Distribution
              </h2>
              <p className="text-xs text-muted">
                Traffic partition by calibrated operational thresholds.
              </p>
            </div>
            <Tooltip
              content="Platt calibrated probabilities mapped to Low (Auto-Approve), Medium (Step-Up Verification), and High (Decline & Alert) bands."
              side="left"
            >
              <span className="text-xs text-muted hover:text-text cursor-pointer flex items-center gap-1">
                Calibration <Question className="w-3.5 h-3.5" weight="thin" />
              </span>
            </Tooltip>
          </div>

          {bandsLoading || !riskBands?.counts ? (
            <div className="h-64 flex items-center justify-center text-xs text-muted animate-pulse font-sans">
              Loading risk bands...
            </div>
          ) : (
            (() => {
              const lowCount = riskBands.counts['Low'] ?? riskBands.counts['low'] ?? 0;
              const medCount = riskBands.counts['Medium'] ?? riskBands.counts['medium'] ?? 0;
              const highCount = riskBands.counts['High'] ?? riskBands.counts['high'] ?? 0;
              const totalBands = lowCount + medCount + highCount || 1;
              const lowPct = ((lowCount / totalBands) * 100).toFixed(1);
              const medPct = ((medCount / totalBands) * 100).toFixed(1);
              const highPct = ((highCount / totalBands) * 100).toFixed(1);

              const lowAction = riskBands.actions?.['Low'] || 'Auto-Approve';
              const medAction = riskBands.actions?.['Medium'] || 'Step-Up Auth';
              const highAction = riskBands.actions?.['High'] || 'Decline & Alert';

              return (
                <div className="space-y-4 font-sans">
                  {/* Visual Partition Bar */}
                  <div className="space-y-1.5">
                    <div className="flex items-center justify-between text-xs text-muted">
                      <span>Partition share</span>
                      <span className="font-semibold text-accent font-mono">
                        {lowPct}% Approved
                      </span>
                    </div>
                    <div className="w-full h-2.5 rounded-full bg-border overflow-hidden flex">
                      <div
                        className="bg-accent h-full transition-[width] duration-320 ease-out motion-reduce:transition-none"
                        style={{ width: `${lowPct}%` }}
                        title={`Low: ${lowPct}%`}
                      />
                      <div
                        className="bg-warn h-full transition-[width] duration-320 ease-out motion-reduce:transition-none"
                        style={{ width: `${medPct}%` }}
                        title={`Medium: ${medPct}%`}
                      />
                      <div
                        className="bg-danger h-full transition-[width] duration-320 ease-out motion-reduce:transition-none"
                        style={{ width: `${highPct}%` }}
                        title={`High: ${highPct}%`}
                      />
                    </div>
                  </div>

                  {/* Clean Table */}
                  <table className="w-full text-left text-xs border-collapse font-sans">
                    <thead className="border-b border-border text-xs font-semibold text-muted normal-case">
                      <tr>
                        <th className="py-2">Band</th>
                        <th className="py-2">Share</th>
                        <th className="py-2">Count</th>
                        <th className="py-2">Action</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border font-mono">
                      <tr>
                        <td className="py-2.5">
                          <Badge variant="low">Low</Badge>
                        </td>
                        <td className="py-2.5 text-muted">{lowPct}%</td>
                        <td className="py-2.5 tabular-nums text-text">{lowCount.toLocaleString()}</td>
                        <td className="py-2.5 text-xs font-sans text-accent">{lowAction}</td>
                      </tr>
                      <tr>
                        <td className="py-2.5">
                          <Badge variant="medium">Medium</Badge>
                        </td>
                        <td className="py-2.5 text-muted">{medPct}%</td>
                        <td className="py-2.5 tabular-nums text-text">{medCount.toLocaleString()}</td>
                        <td className="py-2.5 text-xs font-sans text-warn">{medAction}</td>
                      </tr>
                      <tr>
                        <td className="py-2.5">
                          <Badge variant="high">High</Badge>
                        </td>
                        <td className="py-2.5 text-muted">{highPct}%</td>
                        <td className="py-2.5 tabular-nums font-semibold text-danger">{highCount.toLocaleString()}</td>
                        <td className="py-2.5 text-xs font-sans font-semibold text-danger">{highAction}</td>
                      </tr>
                    </tbody>
                  </table>

                  <div className="p-3 rounded-lg bg-surface-subtle border border-border text-xs text-muted flex items-center justify-between">
                    <span>Evaluated at natural prevalence on test split.</span>
                    <span className="font-mono text-text font-semibold">{totalBands.toLocaleString()} rows</span>
                  </div>
                </div>
              );
            })()
          )}
        </div>
      </motion.div>
    </div>
  );
};
