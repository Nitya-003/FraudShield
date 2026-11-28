import React from 'react';
import { useQuery, keepPreviousData } from '@tanstack/react-query';
import { fetchOverviewKPIs } from '../../api/client';
import { Question } from '@phosphor-icons/react';
import { Tooltip } from '../ui/Tooltip';
import { Skeleton } from '../ui/Skeleton';

export interface ChannelBenchmarkTableProps {
  onSelectChannel?: (channel: string) => void;
  selectedChannel?: string;
  showTitle?: boolean;
}

const CHANNEL_COLORS: Record<string, string> = {
  card: 'bg-emerald-500',
  transfer_wallet: 'bg-sky-500',
  account: 'bg-amber-500',
  upi: 'bg-indigo-500',
};

export const ChannelBenchmarkTable: React.FC<ChannelBenchmarkTableProps> = ({
  onSelectChannel,
  selectedChannel,
  showTitle = true,
}) => {
  const { data: kpis, isLoading } = useQuery({
    queryKey: ['overviewKPIs', 'all'],
    queryFn: () => fetchOverviewKPIs('all'),
    placeholderData: keepPreviousData,
    staleTime: 60000,
  });

  const perChannelTable = kpis?.per_channel_table || [];

  return (
    <div className="w-full space-y-3 font-sans">
      {showTitle && (
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <h2 className="text-base font-semibold text-text">Channels</h2>
            <Tooltip
              content="Currencies and operational loss thresholds are preserved natively in each rail currency and are never blended."
              side="right"
            >
              <Question
                className="w-3.5 h-3.5 text-muted hover:text-text cursor-pointer transition-colors"
                weight="thin"
              />
            </Tooltip>
          </div>
          {onSelectChannel && (
            <span className="text-xs text-muted font-normal hidden sm:inline">
              Click a row to inspect rail
            </span>
          )}
        </div>
      )}

      {/* Desktop / Tablet Table */}
      <div className="hidden md:block overflow-x-auto rounded-lg border border-border bg-surface shadow-card">
        {isLoading ? (
          <div className="p-6 space-y-3">
            <Skeleton className="h-4 w-36" />
            <Skeleton className="h-9 w-full" />
            <Skeleton className="h-9 w-full" />
            <Skeleton className="h-9 w-full" />
            <Skeleton className="h-9 w-full" />
          </div>
        ) : (
          <table className="w-full min-w-[760px] text-left text-xs border-collapse">
            <thead className="border-b border-border bg-surface-subtle text-xs font-semibold text-muted normal-case font-sans">
              <tr>
                <th className="py-2.5 px-3.5">Channel</th>
                <th className="py-2.5 px-3">Benchmark</th>
                <th className="py-2.5 px-3 text-right">PR-AUC</th>
                <th className="py-2.5 px-3 text-right">Recall</th>
                <th className="py-2.5 px-3 text-right">False alarms / 100k</th>
                <th className="py-2.5 px-3 text-right">Alerts / day</th>
                <th className="py-2.5 px-3 text-right">Test rows</th>
                <th className="py-2.5 px-3.5 text-right">Currency</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border font-sans">
              {perChannelTable.map((row) => {
                const isSelected = selectedChannel?.toLowerCase() === row.channel_key.toLowerCase();
                const markerColor = CHANNEL_COLORS[row.channel_key] || 'bg-accent';

                return (
                  <tr
                    key={row.channel_key}
                    onClick={() => {
                      if (onSelectChannel) {
                        // Toggle: if already selected, switch to 'ALL', otherwise select channel
                        onSelectChannel(isSelected ? 'ALL' : row.channel_key);
                      }
                    }}
                    className={`transition-colors group relative ${
                      onSelectChannel ? 'cursor-pointer' : ''
                    } ${
                      isSelected
                        ? 'bg-surface-subtle font-semibold shadow-[inset_3px_0_0_0_#047857] dark:shadow-[inset_3px_0_0_0_#2DD4A0]'
                        : 'hover:bg-surface-subtle'
                    }`}
                  >
                    {/* Channel with Coloured Left Marker */}
                    <td className="py-3 px-3.5 font-medium text-text flex items-center gap-2.5">
                      <span className={`w-2 h-2 rounded-full ${markerColor} shrink-0`} />
                      <span>{row.channel_name}</span>
                    </td>

                    {/* Benchmark dataset citation */}
                    <td className="py-3 px-3 text-muted text-xs">
                      {row.benchmark}
                    </td>

                    {/* PR-AUC with Horizontal Bar */}
                    <td className="py-3 px-3 text-right">
                      <div className="flex items-center justify-end gap-2">
                        <div className="w-14 h-1.5 rounded-full bg-border overflow-hidden hidden sm:block">
                          <div
                            className="h-full rounded-full bg-text/50"
                            style={{ width: `${Math.min(row.pr_auc * 100, 100)}%` }}
                          />
                        </div>
                        <span className="font-semibold tabular-nums text-text">
                          {row.pr_auc.toFixed(5)}
                        </span>
                      </div>
                    </td>

                    {/* Recall with Horizontal Bar */}
                    <td className="py-3 px-3 text-right">
                      <div className="flex items-center justify-end gap-2">
                        <div className="w-14 h-1.5 rounded-full bg-border overflow-hidden hidden sm:block">
                          <div
                            className="h-full rounded-full bg-accent"
                            style={{ width: `${Math.min(row.recall_pct, 100)}%` }}
                          />
                        </div>
                        <span className="font-semibold tabular-nums text-accent">
                          {row.recall_pct.toFixed(1)}%
                        </span>
                      </div>
                    </td>

                    {/* False alarms / 100k */}
                    <td className="py-3 px-3 text-right text-text tabular-nums font-normal">
                      {row.false_alarms_per_100k.toFixed(1)}
                    </td>

                    {/* Alerts / day */}
                    <td className="py-3 px-3 text-right text-text tabular-nums font-normal">
                      {row.alerts_per_day.toFixed(1)}
                    </td>

                    {/* Test rows with native currency as muted suffix */}
                    <td className="py-3 px-3 text-right tabular-nums">
                      <span className="text-text font-normal">{row.test_rows.toLocaleString()}</span>
                    </td>

                    {/* Native currency suffix */}
                    <td className="py-3 px-3.5 text-right font-normal">
                      <span className="text-muted text-xs font-sans">{row.currency_name}</span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>

      {/* Mobile Stacked Cards (Tap targets >= 44px, no horizontal scroll) */}
      <div className="md:hidden space-y-2.5">
        {isLoading ? (
          Array.from({ length: 4 }).map((_, idx) => (
            <div key={`m-skel-${idx}`} className="p-3.5 rounded-lg border border-border bg-surface space-y-2 animate-pulse">
              <Skeleton className="h-4 w-32" />
              <Skeleton className="h-3 w-48" />
              <div className="grid grid-cols-2 gap-2 pt-1">
                <Skeleton className="h-4 w-20" />
                <Skeleton className="h-4 w-20" />
              </div>
            </div>
          ))
        ) : (
          perChannelTable.map((row) => {
            const isSelected = selectedChannel?.toLowerCase() === row.channel_key.toLowerCase();
            const markerColor = CHANNEL_COLORS[row.channel_key] || 'bg-accent';

            return (
              <div
                key={`m-${row.channel_key}`}
                onClick={() => {
                  if (onSelectChannel) {
                    onSelectChannel(isSelected ? 'ALL' : row.channel_key);
                  }
                }}
                className={`p-3.5 rounded-lg border min-h-[44px] transition-colors cursor-pointer space-y-2 ${
                  isSelected
                    ? 'border-accent/40 bg-surface-subtle shadow-xs'
                    : 'border-border bg-surface hover:bg-surface-subtle'
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <span className={`w-2 h-2 rounded-full ${markerColor} shrink-0`} />
                    <span className="font-semibold text-xs text-text">{row.channel_name}</span>
                  </div>
                  <span className="text-xs text-muted font-sans">{row.currency_name}</span>
                </div>

                <div className="text-xs text-muted truncate">{row.benchmark}</div>

                <div className="grid grid-cols-2 gap-2 pt-1 border-t border-border/40 text-xs">
                  <div>
                    <span className="text-muted">PR-AUC: </span>
                    <span className="font-semibold tabular-nums text-text font-mono">{row.pr_auc.toFixed(4)}</span>
                  </div>
                  <div>
                    <span className="text-muted">Recall: </span>
                    <span className="font-semibold tabular-nums text-accent font-mono">{row.recall_pct.toFixed(1)}%</span>
                  </div>
                  <div>
                    <span className="text-muted">False alarms: </span>
                    <span className="font-semibold tabular-nums text-text font-mono">{row.false_alarms_per_100k.toFixed(1)}</span>
                  </div>
                  <div>
                    <span className="text-muted">Test rows: </span>
                    <span className="font-semibold tabular-nums text-text font-mono">{row.test_rows.toLocaleString()}</span>
                  </div>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};
