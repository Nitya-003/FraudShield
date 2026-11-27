import React, { useState } from 'react';
import { useQuery, keepPreviousData } from '@tanstack/react-query';
import { fetchModelRegistry } from '../api/client';
import { PageHeader } from '../components/layout/PageHeader';
import { Button } from '../components/ui/Button';
import { Tooltip } from '../components/ui/Tooltip';
import {
  ChevronDown,
  ChevronUp,
  HelpCircle,
  ShieldCheck,
  Activity,
  Layers,
  AlertCircle,
  Inbox,
} from 'lucide-react';
import { getChannelReadableName } from '../lib/channelUtils';
import { Skeleton } from '../components/ui/Skeleton';

export const ModelsPage: React.FC = () => {
  const [expandedChannel, setExpandedChannel] = useState<string | null>('card');

  const {
    data: registryData,
    isLoading,
    isError,
    refetch,
  } = useQuery({
    queryKey: ['models'],
    queryFn: fetchModelRegistry,
    placeholderData: keepPreviousData,
    staleTime: 10 * 60 * 1000,
  });

  const toggleExpand = (channel: string) => {
    setExpandedChannel((prev) => (prev === channel ? null : channel));
  };

  return (
    <div className="space-y-6 pb-[calc(3rem+env(safe-area-inset-bottom,0px))]">
      <PageHeader
        title="Models"
        purpose="Calibrated champion model registry, evaluation metrics, and operational thresholds."
      />

      {isError && (
        <div className="p-4 rounded-xl border border-danger/30 bg-danger/10 text-xs text-danger flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>Failed to load model registry from API.</span>
          </div>
          <Button variant="secondary" size="sm" onClick={() => refetch()} className="min-h-[36px]">
            Retry
          </Button>
        </div>
      )}

      {/* Champion Table */}
      {/* Desktop / Tablet Table */}
      <div className="hidden md:block overflow-x-auto bg-surface border border-border rounded-xl shadow-card">
        {isLoading ? (
          <table className="w-full min-w-[760px] text-left text-xs border-collapse font-sans">
            <thead className="border-b border-border bg-surface-subtle text-xs font-semibold text-muted normal-case">
              <tr className="h-10">
                <th className="py-2.5 px-4">Channel</th>
                <th className="py-2.5 px-4">Champion model</th>
                <th className="py-2.5 px-4">Algorithm</th>
                <th className="py-2.5 px-4">PR-AUC</th>
                <th className="py-2.5 px-4">ROC-AUC</th>
                <th className="py-2.5 px-4">Thresholds</th>
                <th className="py-2.5 px-4 text-right">Details</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {Array.from({ length: 4 }).map((_, idx) => (
                <tr key={`model-skel-${idx}`} className="h-[44px] animate-pulse">
                  <td className="py-2.5 px-4"><div className="h-3.5 bg-border rounded w-24" /></td>
                  <td className="py-2.5 px-4"><div className="h-3.5 bg-border rounded w-36" /></td>
                  <td className="py-2.5 px-4"><div className="h-3.5 bg-border rounded w-44" /></td>
                  <td className="py-2.5 px-4"><div className="h-3.5 bg-border rounded w-16" /></td>
                  <td className="py-2.5 px-4"><div className="h-3.5 bg-border rounded w-16" /></td>
                  <td className="py-2.5 px-4"><div className="h-3.5 bg-border rounded w-24" /></td>
                  <td className="py-2.5 px-4"><div className="h-3.5 bg-border rounded w-10 ml-auto" /></td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : !registryData?.champions || Object.keys(registryData.champions).length === 0 ? (
          <div className="py-16 text-center text-xs text-muted space-y-2">
            <Inbox className="w-8 h-8 mx-auto opacity-40 text-muted" />
            <div className="font-semibold text-sm text-text">No models found</div>
            <p className="text-muted">Champion model configurations could not be loaded.</p>
            <Button variant="secondary" size="sm" onClick={() => refetch()} className="mt-2">
              Refresh registry
            </Button>
          </div>
        ) : (
          <table className="w-full min-w-[760px] text-left text-xs border-collapse font-sans">
            <thead className="border-b border-border bg-surface-subtle text-xs font-semibold text-muted normal-case">
              <tr className="h-10">
                <th className="py-2.5 px-4 sticky left-0 bg-surface-subtle z-20">Channel</th>
                <th className="py-2.5 px-4">Champion model</th>
                <th className="py-2.5 px-4">Algorithm</th>
                <th className="py-2.5 px-4">
                  <div className="flex items-center gap-1">
                    <span>PR-AUC</span>
                    <Tooltip content="Precision-Recall Area Under Curve: primary metric for imbalanced fraud distributions." side="top">
                      <HelpCircle className="w-3.5 h-3.5 text-muted cursor-help" />
                    </Tooltip>
                  </div>
                </th>
                <th className="py-2.5 px-4">
                  <div className="flex items-center gap-1">
                    <span>ROC-AUC</span>
                    <Tooltip content="Receiver Operating Characteristic ranking score across thresholds." side="top">
                      <HelpCircle className="w-3.5 h-3.5 text-muted cursor-help" />
                    </Tooltip>
                  </div>
                </th>
                <th className="py-2.5 px-4">
                  <div className="flex items-center gap-1">
                    <span>Thresholds (Approve / Decline)</span>
                    <Tooltip content="Operating score thresholds applied to incoming transactions." side="top">
                      <HelpCircle className="w-3.5 h-3.5 text-muted cursor-help" />
                    </Tooltip>
                  </div>
                </th>
                <th className="py-2.5 px-4 text-right">Details</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border font-sans">
              {Object.entries(registryData.champions).map(([channelKey, m]) => {
                const isExpanded = expandedChannel === channelKey;
                return (
                  <React.Fragment key={channelKey}>
                    <tr
                      onClick={() => toggleExpand(channelKey)}
                      className={`h-[44px] cursor-pointer transition-colors ${
                        isExpanded
                          ? 'bg-surface-subtle'
                          : 'hover:bg-surface-subtle'
                      }`}
                    >
                      <td className="py-2.5 px-4 font-semibold text-xs text-accent sticky left-0 bg-surface">
                        {getChannelReadableName(channelKey)}
                      </td>
                      <td className="py-2.5 px-4 font-semibold text-xs text-text">
                        {m.model_name}
                      </td>
                      <td className="py-2.5 px-4 text-muted text-xs">
                        {m.algorithm}
                      </td>
                      <td className="py-2.5 px-4 font-semibold font-mono tabular-nums text-xs text-text">
                        {typeof m.pr_auc === 'number' ? m.pr_auc.toFixed(4) : '--'}
                      </td>
                      <td className="py-2.5 px-4 font-mono tabular-nums text-xs text-muted">
                        {typeof m.roc_auc === 'number' ? m.roc_auc.toFixed(4) : '--'}
                      </td>
                      <td className="py-2.5 px-4 font-mono tabular-nums text-xs">
                        <span className="text-accent">{typeof m.thresholds?.low_risk_approve === 'number' ? m.thresholds.low_risk_approve.toFixed(2) : '--'}</span>
                        <span className="text-muted mx-1.5 font-sans">/</span>
                        <span className="text-danger">{typeof m.thresholds?.high_risk_decline === 'number' ? m.thresholds.high_risk_decline.toFixed(2) : '--'}</span>
                      </td>
                      <td className="py-2.5 px-4 text-right" onClick={(e) => e.stopPropagation()}>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => toggleExpand(channelKey)}
                          title={isExpanded ? 'Collapse detail' : 'Expand detail'}
                          className="h-7 w-7 p-0 ml-auto"
                        >
                          {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                        </Button>
                      </td>
                    </tr>

                    {/* Expandable Detail Panel */}
                    {isExpanded && (
                      <tr>
                        <td colSpan={7} className="p-5 bg-surface-subtle">
                          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 font-sans">
                            {/* 1. Validation CI */}
                            <div className="p-4 rounded-xl border border-border bg-surface space-y-2">
                              <div className="text-xs font-semibold text-text flex items-center justify-between">
                                <span>Bootstrap confidence intervals</span>
                                <ShieldCheck className="w-3.5 h-3.5 text-accent" />
                              </div>
                              <p className="text-xs text-muted">
                                95% confidence intervals from 1,000 bootstrap test resamples:
                              </p>
                              <div className="font-mono tabular-nums text-xs text-text pt-1">
                                PR-AUC: [{m.bootstrap_ci_95?.pr_auc?.[0] != null ? m.bootstrap_ci_95.pr_auc[0].toFixed(4) : '--'},{' '}
                                {m.bootstrap_ci_95?.pr_auc?.[1] != null ? m.bootstrap_ci_95.pr_auc[1].toFixed(4) : '--'}]
                              </div>
                            </div>

                            {/* 2. Calibration */}
                            <div className="p-4 rounded-xl border border-border bg-surface space-y-2">
                              <div className="text-xs font-semibold text-text flex items-center justify-between">
                                <span>Probability calibration</span>
                                <Activity className="w-3.5 h-3.5 text-accent" />
                              </div>
                              <p className="text-xs text-muted">
                                Platt sigmoid calibration mapping raw margins to posterior risk probabilities.
                              </p>
                              <div className="font-mono tabular-nums text-xs text-text pt-1">
                                Calibrated Brier score: {typeof m.brier_score === 'number' ? m.brier_score.toFixed(4) : '--'}
                              </div>
                            </div>

                            {/* 3. Provenance & Artifacts */}
                            <div className="p-4 rounded-xl border border-border bg-surface space-y-2">
                              <div className="text-xs font-semibold text-text flex items-center justify-between">
                                <span>Model artifact</span>
                                <Layers className="w-3.5 h-3.5 text-accent" />
                              </div>
                              <div className="font-sans text-xs space-y-1 text-muted">
                                <div>Version: <span className="font-mono text-text">v{m.version}</span></div>
                                <div>Trained: <span className="font-mono text-text">{m.trained_date}</span></div>
                                <div className="truncate">File: <span className="font-mono text-text">{m.model_file || `champion_${channelKey}.joblib`}</span></div>
                              </div>
                            </div>
                          </div>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                );
              })}
            </tbody>
          </table>
        )}
      </div>

      {/* Mobile Stacked Cards (Zero horizontal scroll, tap targets >= 44px) */}
      <div className="md:hidden space-y-2.5">
        {isLoading ? (
          Array.from({ length: 4 }).map((_, idx) => (
            <div key={`m-mod-skel-${idx}`} className="p-3.5 rounded-lg border border-border bg-surface space-y-2 animate-pulse">
              <Skeleton className="h-4 w-32" />
              <Skeleton className="h-3 w-48" />
            </div>
          ))
        ) : !registryData?.champions || Object.keys(registryData.champions).length === 0 ? (
          <div className="p-8 text-center text-xs text-muted">No models found.</div>
        ) : (
          Object.entries(registryData.champions).map(([channelKey, m]) => {
            const isExpanded = expandedChannel === channelKey;
            return (
              <div
                key={`m-mod-${channelKey}`}
                className="p-3.5 rounded-lg border border-border bg-surface space-y-2.5"
              >
                <div
                  onClick={() => toggleExpand(channelKey)}
                  className="flex items-center justify-between cursor-pointer min-h-[44px]"
                >
                  <div>
                    <div className="font-semibold text-xs text-text">{getChannelReadableName(channelKey)}</div>
                    <div className="text-xs text-muted font-sans">{m.model_name}</div>
                  </div>
                  <Button variant="ghost" size="sm" className="h-8 w-8 p-0" aria-label={`Toggle details for ${getChannelReadableName(channelKey)}`}>
                    <ChevronDown className={`w-4 h-4 transition-transform duration-200 ${isExpanded ? 'rotate-180' : ''}`} />
                  </Button>
                </div>

                <div className="grid grid-cols-2 gap-2 text-xs pt-1 border-t border-border/40">
                  <div>
                    <span className="text-muted">PR-AUC: </span>
                    <span className="font-semibold text-text font-mono tabular-nums">{typeof m.pr_auc === 'number' ? m.pr_auc.toFixed(4) : '--'}</span>
                  </div>
                  <div>
                    <span className="text-muted">ROC-AUC: </span>
                    <span className="font-semibold text-text font-mono tabular-nums">{typeof m.roc_auc === 'number' ? m.roc_auc.toFixed(4) : '--'}</span>
                  </div>
                  <div className="col-span-2">
                    <span className="text-muted">Algorithm: </span>
                    <span className="text-text">{m.algorithm || 'LightGBM'}</span>
                  </div>
                </div>

                {isExpanded && (
                  <div className="pt-2 border-t border-border/40 space-y-2 text-xs">
                    <div className="p-2.5 rounded-lg bg-surface-subtle border border-border space-y-1">
                      <div className="font-semibold text-text">Thresholds</div>
                      <div className="font-mono text-muted">
                        Low: {typeof m.thresholds?.low_risk_approve === 'number' ? m.thresholds.low_risk_approve.toFixed(2) : '--'} | High: {typeof m.thresholds?.high_risk_decline === 'number' ? m.thresholds.high_risk_decline.toFixed(2) : '--'}
                      </div>
                    </div>
                    <div className="p-2.5 rounded-lg bg-surface-subtle border border-border space-y-1">
                      <div className="font-semibold text-text">Model artifact</div>
                      <div className="text-muted font-mono">v{m.version} &bull; {m.trained_date}</div>
                    </div>
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};
