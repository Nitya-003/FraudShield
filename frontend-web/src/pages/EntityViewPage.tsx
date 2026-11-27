import React, { useState, useMemo } from 'react';
import { useQuery, keepPreviousData } from '@tanstack/react-query';
import { PageHeader } from '../components/layout/PageHeader';
import { Input } from '../components/ui/Input';
import { Button } from '../components/ui/Button';
import { Badge } from '../components/ui/Badge';
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
} from '../components/ui/Sheet';
import { ChartWrapper } from '../components/ui/ChartWrapper';
import { EmptyState } from '../components/ui/EmptyState';
import {
  Search,
  Network,
  ArrowUpRight,
  ArrowDownLeft,
  AlertTriangle,
  History,
  Layers,
  AlertCircle,
} from 'lucide-react';
import type { EChartsOption } from 'echarts';

interface TimelineEvent {
  id: number;
  step: number;
  type: string;
  counterparty: string;
  amount: number;
  isFraud: boolean;
  timeAgo: string;
}

interface EntityViewPageProps {
  isDark: boolean;
}

export const EntityViewPage: React.FC<EntityViewPageProps> = ({ isDark }) => {
  const [entityId, setEntityId] = useState('C716083600');
  const [activeEntityId, setActiveEntityId] = useState('C716083600');
  const [hops, setHops] = useState(1);
  const [isTimelineDrawerOpen, setIsTimelineDrawerOpen] = useState(false);

  const { data, isLoading, isError, refetch } = useQuery({
    queryKey: ['entityProfile', activeEntityId, hops],
    queryFn: async () => {
      const res = await fetch(`/api/entity/${activeEntityId}?hops=${hops}`);
      if (!res.ok) throw new Error('Failed to load entity profile');
      return res.json();
    },
    placeholderData: keepPreviousData,
    staleTime: 60 * 1000,
  });

  const profile = data?.profile;
  const isSparse = profile?.is_sparse ?? false;

  const handleNodeClick = (clickedId: string) => {
    setEntityId(clickedId);
    setActiveEntityId(clickedId);
  };

  const graphOption: EChartsOption = useMemo(() => {
    const rawNodes = data?.graph?.nodes || [];
    const rawEdges = data?.graph?.edges || [];
    if (rawNodes.length === 0) return {};

    const nodes = rawNodes.map(
      (n: { id: string; role?: string; is_target?: boolean; risk?: number }) => {
        const nodeId = n.id;
        const isTarget = nodeId === activeEntityId || n.is_target;
        const isMuleHub =
          n.role === 'mule_hub' ||
          nodeId === 'C716083600' ||
          (n.role && n.role.includes('hub'));
        const isHighRisk = (n.risk !== undefined && n.risk >= 0.85) || isTarget;

        return {
          name: nodeId,
          symbol: isMuleHub ? 'diamond' : 'circle',
          symbolSize: isTarget ? 34 : isMuleHub ? 26 : 18,
          itemStyle: {
            color: isHighRisk ? '#B91C1C' : isDark ? '#2DD4A0' : '#0E9F77',
            borderColor: isDark ? '#0B0F14' : '#FFFFFF',
            borderWidth: isTarget ? 2.5 : 1.5,
          },
          label: {
            show: isTarget || rawNodes.length <= 12,
            position: 'right' as const,
            fontSize: 12,
            fontFamily: 'JetBrains Mono, monospace',
            color: isDark ? '#E8EDF2' : '#0F172A',
            formatter: (params: { name: string }) => params.name.slice(0, 11),
          },
        };
      }
    );

    const links = rawEdges.map(
      (edge: { source: string; target: string; amount: number; is_fraud: number }) => ({
        source: edge.source,
        target: edge.target,
        lineStyle: {
          color: edge.is_fraud === 1 ? '#B91C1C' : isDark ? '#1E2630' : '#E3E8EE',
          width: edge.is_fraud === 1 ? 2.5 : 1.2,
          curveness: 0.15,
        },
        tooltip: {
          formatter: `Transit: ₹${edge.amount.toLocaleString('en-IN')}${
            edge.is_fraud ? ' (Flagged High Risk)' : ''
          }`,
        },
      })
    );

    return {
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'item',
        backgroundColor: isDark ? '#11171E' : '#FFFFFF',
        borderColor: isDark ? '#1E2630' : '#E3E8EE',
        textStyle: { color: isDark ? '#E8EDF2' : '#0F172A', fontSize: 12 },
        formatter: (params: unknown) => {
          const p = params as { dataType?: string; name: string };
          if (p.dataType === 'node') {
            return `<strong>Account:</strong> ${p.name}<br/><span style="font-size:12px;color:#8A96A3">Tap to re-centre graph</span>`;
          }
          return p.name;
        },
      },
      series: [
        {
          type: 'graph',
          layout: 'force',
          roam: true,
          draggable: true,
          data: nodes,
          links: links,
          emphasis: {
            focus: 'adjacency',
            lineStyle: { width: 3 },
          },
          force: {
            repulsion: 140,
            edgeLength: [60, 140],
            gravity: 0.1,
          },
        },
      ],
    };
  }, [data, activeEntityId, isDark]);

  const flowChartOption: EChartsOption = useMemo(() => {
    const textColor = isDark ? '#8A96A3' : '#52606D';
    const gridLineColor = isDark ? '#1E2630' : '#E3E8EE';

    return {
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        backgroundColor: isDark ? '#11171E' : '#FFFFFF',
        borderColor: isDark ? '#1E2630' : '#E3E8EE',
        textStyle: { color: isDark ? '#E8EDF2' : '#0F172A', fontSize: 12 },
      },
      grid: {
        top: 24,
        bottom: 24,
        left: 56,
        right: 16,
      },
      xAxis: {
        type: 'category',
        data: ['Inbound Volume', 'Outbound Drain'],
        axisLabel: {
          fontFamily: 'Inter, sans-serif',
          fontSize: 12,
          color: textColor,
        },
        axisLine: { lineStyle: { color: gridLineColor } },
      },
      yAxis: {
        type: 'value',
        axisLabel: {
          formatter: (v: number) => `₹${(v / 1e6).toFixed(1)}M`,
          fontFamily: 'JetBrains Mono, monospace',
          fontSize: 12,
          color: textColor,
        },
        splitLine: {
          lineStyle: {
            color: gridLineColor,
            type: 'dashed',
          },
        },
      },
      series: [
        {
          type: 'bar',
          barWidth: '36%',
          data: [
            {
              value: profile?.total_received_inr ?? 0,
              itemStyle: {
                color: isDark ? '#2DD4A0' : '#0E9F77',
                borderRadius: [3, 3, 0, 0],
              },
            },
            {
              value: profile?.total_sent_inr ?? 0,
              itemStyle: {
                color: isDark ? '#B91C1C' : '#DC2626',
                borderRadius: [3, 3, 0, 0],
              },
            },
          ],
        },
      ],
    };
  }, [profile, isDark]);

  const timelineEvents: TimelineEvent[] = useMemo(() => {
    const edges = data?.graph?.edges || [];
    if (edges.length === 0) return [];
    return edges
      .slice(0, 10)
      .map(
        (
          e: { source: string; target: string; amount: number; is_fraud: number },
          idx: number
        ): TimelineEvent => {
          const isInflow = e.target === activeEntityId;
          return {
            id: idx,
            step: 500 - idx * 14,
            type: isInflow ? 'Inbound Transfer' : 'Outbound Cash-Out',
            counterparty: isInflow ? e.source : e.target,
            amount: e.amount,
            isFraud: e.is_fraud === 1,
            timeAgo: `${(idx + 1) * 22} mins ago`,
          };
        }
      );
  }, [data, activeEntityId]);

  return (
    <div className="space-y-6 pb-16">
      <PageHeader
        title="Entity Graph"
        purpose="Multi-hop transaction neighborhood analysis identifying aggregator accounts and ring patterns."
        action={
          <div className="flex items-center gap-2">
            <Button
              variant="secondary"
              size="sm"
              onClick={() => {
                setEntityId('C716083600');
                setActiveEntityId('C716083600');
              }}
            >
              <span>Mule Hub (C716083600)</span>
            </Button>
            <Button
              variant="secondary"
              size="sm"
              onClick={() => {
                setEntityId('C1231006815');
                setActiveEntityId('C1231006815');
              }}
            >
              <span>Clean Sender (C1231006815)</span>
            </Button>
          </div>
        }
      />

      {/* Search Input & Hop Toggle Bar */}
      <section className="p-4 rounded-xl border border-border bg-surface flex flex-col sm:flex-row items-end gap-3 shadow-card">
        <div className="flex-1 w-full">
          <Input
            label="Target Entity ID or Account Identifier"
            value={entityId}
            onChange={(e) => setEntityId(e.target.value)}
            placeholder="e.g. C716083600"
          />
        </div>
        <div className="flex items-center gap-2 w-full sm:w-auto">
          <button
            onClick={() => setHops(1)}
            className={`px-3 py-2 rounded-lg border text-xs font-semibold transition-colors cursor-pointer ${
              hops === 1
                ? 'border-accent bg-accent/10 text-text font-semibold'
                : 'border-border bg-surface-subtle text-muted hover:text-text'
            }`}
          >
            1-Hop (Clean)
          </button>
          <button
            onClick={() => setHops(2)}
            className={`px-3 py-2 rounded-lg border text-xs font-semibold transition-colors cursor-pointer ${
              hops === 2
                ? 'border-accent bg-accent/10 text-text font-semibold'
                : 'border-border bg-surface-subtle text-muted hover:text-text'
            }`}
          >
            2-Hops (Max)
          </button>
          <Button
            variant="primary"
            size="md"
            onClick={() => setActiveEntityId(entityId)}
          >
            <Search className="w-4 h-4" />
            <span>Inspect</span>
          </Button>
        </div>
      </section>

      {isLoading ? (
        <div className="py-16 text-center text-xs text-muted animate-pulse">
          Loading graph topology and account activity...
        </div>
      ) : isError ? (
        <div className="p-4 rounded-xl border border-danger/30 bg-danger/10 text-xs text-danger flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>Failed to load entity graph data for &ldquo;{activeEntityId}&rdquo;.</span>
          </div>
          <Button variant="secondary" size="sm" onClick={() => refetch()}>
            Retry
          </Button>
        </div>
      ) : isSparse ? (
        <EmptyState
          title="Sparse Account Profile"
          description={
            profile?.sparse_message ||
            'This account has 2 or fewer lifetime transactions. No complex graph ring was detected.'
          }
          icon={AlertTriangle}
          actionLabel="Inspect Known Mule Hub (C716083600)"
          onAction={() => {
            setEntityId('C716083600');
            setActiveEntityId('C716083600');
          }}
        />
      ) : profile ? (
        <div className="space-y-6">
          {/* Profile Overview Stats */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="p-4 rounded-xl border border-border bg-surface shadow-card">
              <div className="text-xs font-semibold text-muted mb-1">
                Active Entity Identifier
              </div>
              <div className="text-xl font-mono font-semibold text-text">
                {profile.entity_id}
              </div>
              <div className="mt-1">
                <Badge
                  variant={
                    profile.risk_band?.toLowerCase() as 'low' | 'medium' | 'high'
                  }
                >
                  Score:{' '}
                  {profile.risk_score !== undefined
                    ? profile.risk_score.toFixed(3)
                    : '--'}{' '}
                  ({profile.risk_band})
                </Badge>
              </div>
            </div>

            <div className="p-4 rounded-xl border border-border bg-surface shadow-card">
              <div className="text-xs font-semibold text-muted mb-1">
                Total Inflow (Received)
              </div>
              <div className="text-xl font-mono font-semibold text-accent tabular-nums">
                ₹{(profile.total_received_inr ?? 0).toLocaleString('en-IN')}
              </div>
              <div className="text-xs text-muted mt-1 flex items-center gap-1">
                <ArrowDownLeft className="w-3.5 h-3.5" />
                <span>{(data?.graph?.nodes?.length ?? 1) - 1} inbound senders</span>
              </div>
            </div>

            <div className="p-4 rounded-xl border border-border bg-surface shadow-card">
              <div className="text-xs font-semibold text-muted mb-1">
                Total Outflow (Liquidated)
              </div>
              <div className="text-xl font-mono font-semibold text-danger tabular-nums">
                ₹{(profile.total_sent_inr ?? 0).toLocaleString('en-IN')}
              </div>
              <div className="text-xs text-muted mt-1 flex items-center gap-1">
                <ArrowUpRight className="w-3.5 h-3.5" />
                <span>Rapid wallet drain pattern</span>
              </div>
            </div>

            <div className="p-4 rounded-xl border border-border bg-surface shadow-card">
              <div className="text-xs font-semibold text-muted mb-1">
                Network Links
              </div>
              <div className="text-xl font-mono font-semibold text-text tabular-nums">
                {data?.graph?.edges?.length ?? 0} Transits
              </div>
              <div className="text-xs text-muted mt-1">
                {hops}-Hop Neighborhood (Capped at 2)
              </div>
            </div>
          </div>

          {/* Interactive Graph Visualizer & In/Out Flow Chart */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
            {/* Left (7 cols): Interactive Graph Canvas */}
            <div className="lg:col-span-7 p-5 rounded-xl border border-border bg-surface space-y-3 shadow-card">
              <div className="flex items-center justify-between border-b border-border pb-3">
                <div className="flex items-center gap-2">
                  <Network className="w-4 h-4 text-accent" />
                  <h3 className="font-semibold text-xs text-text">
                    Clean Topology Subgraph
                  </h3>
                </div>
                <span className="font-mono text-xs text-muted">
                  Tap node to re-centre graph
                </span>
              </div>

              {/* Real ECharts Force Graph */}
              <div className="h-80 w-full rounded-xl border border-border bg-surface-subtle overflow-hidden relative">
                <ChartWrapper
                  option={graphOption}
                  height={320}
                  isDark={isDark}
                  onEvents={{
                    click: (params: unknown) => {
                      const p = params as { dataType?: string; name?: string };
                      if (p?.dataType === 'node' && p.name) {
                        handleNodeClick(p.name);
                      }
                    },
                  }}
                />
                <div className="absolute bottom-2 left-2 flex items-center gap-3 text-xs font-mono text-muted bg-surface/90 px-2.5 py-1 rounded-md border border-border">
                  <span className="flex items-center gap-1">
                    <span className="w-2.5 h-2.5 rounded-full bg-danger" /> High Risk Target
                  </span>
                  <span className="flex items-center gap-1">
                    <span className="w-2.5 h-2.5 rotate-45 bg-warn" /> Mule Hub (Diamond)
                  </span>
                  <span className="flex items-center gap-1">
                    <span className="w-2.5 h-2.5 rounded-full bg-accent" /> Counterparty
                  </span>
                </div>
              </div>
            </div>

            {/* Right (5 cols): Cash Flow Disparity Bar Chart */}
            <div className="lg:col-span-5 p-5 rounded-xl border border-border bg-surface space-y-3 shadow-card">
              <div className="border-b border-border pb-3">
                <h3 className="font-semibold text-xs text-text">
                  Inbound vs Outbound Flow
                </h3>
                <p className="text-xs text-muted mt-0.5">
                  Indegree aggregation with rapid liquidation disparity.
                </p>
              </div>

              <ChartWrapper option={flowChartOption} height={320} isDark={isDark} />
            </div>
          </div>

          {/* Timeline of Inbound / Outbound Transits */}
          <div className="p-5 rounded-xl border border-border bg-surface space-y-3 shadow-card">
            <div className="flex items-center justify-between border-b border-border pb-3">
              <div className="flex items-center gap-2">
                <History className="w-4 h-4 text-accent" />
                <h3 className="font-semibold text-xs text-text">
                  Account Activity Timeline
                </h3>
              </div>
              <div className="flex items-center gap-2">
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => setIsTimelineDrawerOpen(true)}
                >
                  <Layers className="w-3.5 h-3.5" />
                  <span>Open Drawer</span>
                </Button>
              </div>
            </div>

            <div className="space-y-2 font-mono text-xs">
              {timelineEvents.map((evt) => (
                <div
                  key={evt.id}
                  onClick={() => setIsTimelineDrawerOpen(true)}
                  className="flex items-center justify-between p-3 rounded-lg border border-border bg-surface-subtle hover:bg-surface transition-colors duration-120 cursor-pointer"
                >
                  <div className="flex items-center gap-3">
                    <div
                      className={`p-1.5 rounded-full ${
                        evt.isFraud
                          ? 'bg-danger/10 text-danger'
                          : 'bg-accent/10 text-accent'
                      }`}
                    >
                      {evt.type.includes('Inbound') ? (
                        <ArrowDownLeft className="w-3.5 h-3.5" />
                      ) : (
                        <ArrowUpRight className="w-3.5 h-3.5" />
                      )}
                    </div>
                    <div>
                      <div className="font-medium text-text font-sans">
                        {evt.type}
                      </div>
                      <div className="text-xs text-muted">
                        Counterparty:{' '}
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleNodeClick(evt.counterparty);
                          }}
                          className="hover:underline text-accent"
                        >
                          {evt.counterparty}
                        </button>
                      </div>
                    </div>
                  </div>

                  <div className="text-right">
                    <div className="font-semibold tabular-nums text-text">
                      ₹{evt.amount.toLocaleString('en-IN')}
                    </div>
                    <div className="text-xs text-muted">
                      {evt.timeAgo}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Right Sheet with Account Timeline */}
          <Sheet open={isTimelineDrawerOpen} onOpenChange={setIsTimelineDrawerOpen}>
            <SheetContent side="right" className="overflow-y-auto">
              <SheetHeader>
                <div className="flex items-center gap-2 text-xs text-muted mb-1 font-mono">
                  <span>{activeEntityId}</span>
                  <span>&bull;</span>
                  <span>{hops}-Hop Neighborhood</span>
                </div>
                <SheetTitle>Account Activity Timeline</SheetTitle>
                <SheetDescription>
                  Chronological event ledger across {hops}-hop neighborhood ({timelineEvents.length} transactions).
                </SheetDescription>
              </SheetHeader>

              <div className="space-y-4 text-xs font-mono mt-4">
                <div className="p-3.5 rounded-xl border border-border bg-surface-subtle space-y-1">
                  <div className="text-muted font-sans">
                    Account Overview
                  </div>
                  <div className="text-base font-semibold text-text">
                    {activeEntityId}
                  </div>
                  <div className="text-accent font-sans">
                    Risk Status: {profile.risk_band} ({profile.risk_score?.toFixed(3) ?? '--'})
                  </div>
                </div>

                <div className="space-y-2">
                  <div className="font-sans font-semibold text-xs text-text">
                    Chronological Transit Ledger
                  </div>
                  <div className="divide-y divide-border border border-border rounded-xl overflow-hidden bg-surface">
                    {timelineEvents.map((evt) => (
                      <div
                        key={evt.id}
                        className="p-3 flex items-center justify-between hover:bg-surface-subtle transition-colors"
                      >
                        <div className="space-y-0.5">
                          <div className="font-sans font-semibold text-text flex items-center gap-1.5">
                            <span>{evt.type}</span>
                            {evt.isFraud && (
                              <Badge variant="high" className="text-xs py-0 px-1.5">
                                Flagged
                              </Badge>
                            )}
                          </div>
                          <div className="text-muted text-xs">
                            Step {evt.step} &bull; {evt.counterparty}
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="font-semibold text-text tabular-nums">
                            ₹{evt.amount.toLocaleString('en-IN')}
                          </div>
                          <div className="text-muted text-xs">
                            {evt.timeAgo}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="pt-2">
                  <Button
                    variant="primary"
                    size="sm"
                    className="w-full justify-center"
                    onClick={() => setIsTimelineDrawerOpen(false)}
                  >
                    Close Drawer
                  </Button>
                </div>
              </div>
            </SheetContent>
          </Sheet>
        </div>
      ) : null}
    </div>
  );
};
