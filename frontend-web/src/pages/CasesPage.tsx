import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { useQuery, useMutation, useQueryClient, keepPreviousData } from '@tanstack/react-query';
import { fetchCases, fetchCaseDetail, updateCaseStatus } from '../api/client';
import type { CaseItem, CaseAuditLog } from '../types/api';
import { Button } from '../components/ui/Button';
import { Badge } from '../components/ui/Badge';
import { Select } from '../components/ui/Select';
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
} from '../components/ui/Sheet';
import { PageHeader } from '../components/layout/PageHeader';
import { useToast } from '../components/ui/Toast';
import { Skeleton } from '../components/ui/Skeleton';
import {
  Clock,
  Keyboard,
  AlertCircle,
  Inbox,
  CheckCircle2,
  XCircle,
} from 'lucide-react';
import { getChannelReadableName, formatChannelAmount } from '../lib/channelUtils';

export const CasesPage: React.FC = () => {
  const queryClient = useQueryClient();
  const { toast } = useToast();

  const [statusFilter, setStatusFilter] = useState('ALL');
  const [channelFilter, setChannelFilter] = useState('ALL');
  const [selectedCaseId, setSelectedCaseId] = useState<string | null>(null);
  const [highlightedIndex, setHighlightedIndex] = useState<number>(0);
  const [notesInput, setNotesInput] = useState('');
  const analystName = 'Risk Investigator';

  const {
    data: casesData,
    isLoading,
    isFetching,
    isError,
    refetch,
  } = useQuery({
    queryKey: ['cases', statusFilter, channelFilter],
    queryFn: () => fetchCases({ status: statusFilter, channel: channelFilter, limit: 50 }),
    placeholderData: keepPreviousData,
    staleTime: 60 * 1000,
  });

  const { data: detailData } = useQuery({
    queryKey: ['caseDetail', selectedCaseId],
    queryFn: () => (selectedCaseId ? fetchCaseDetail(selectedCaseId) : null),
    enabled: !!selectedCaseId,
    staleTime: 60 * 1000,
  });

  // Optimistic update for case triage with automatic rollback on error
  const mutation = useMutation({
    mutationFn: ({ caseId, status, notes }: { caseId: string; status: string; notes: string }) =>
      updateCaseStatus(caseId, { status, notes, analyst_name: analystName }),
    onMutate: async ({ caseId, status, notes }) => {
      await queryClient.cancelQueries({ queryKey: ['cases', statusFilter, channelFilter] });
      await queryClient.cancelQueries({ queryKey: ['caseDetail', caseId] });

      const previousCases = queryClient.getQueryData<{ cases: CaseItem[]; count: number }>([
        'cases',
        statusFilter,
        channelFilter,
      ]);

      if (previousCases) {
        queryClient.setQueryData(['cases', statusFilter, channelFilter], {
          ...previousCases,
          cases: previousCases.cases.map((c) =>
            c.case_id === caseId
              ? { ...c, status, analyst_notes: notes, updated_at: new Date().toISOString() }
              : c
          ),
        });
      }

      toast({
        title: 'Updating Disposition',
        description: `Case ${caseId} updating to ${status}`,
        variant: 'info',
      });

      return { previousCases, caseId };
    },
    onError: (_err, _vars, context) => {
      if (context?.previousCases) {
        queryClient.setQueryData(['cases', statusFilter, channelFilter], context.previousCases);
      }
      toast({
        title: 'Update Failed',
        description: 'Failed to update case disposition. Changes rolled back.',
        variant: 'error',
      });
    },
    onSuccess: (data) => {
      toast({
        title: 'Case Disposition Recorded',
        description: `Case ${data.case.case_id} marked as ${data.case.status}. Feedback loop synced.`,
        variant: 'success',
      });
      setSelectedCaseId(null);
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ['cases'] });
    },
  });

  const cases = useMemo(() => casesData?.cases || [], [casesData?.cases]);

  // Keyboard navigation & triage hotkeys [J/K to move, F to confirm fraud, C to clear]
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      const activeEl = document.activeElement;
      if (activeEl && (activeEl.tagName === 'INPUT' || activeEl.tagName === 'TEXTAREA')) {
        return;
      }

      if (cases.length === 0) return;

      if (e.key === 'j' || e.key === 'ArrowDown') {
        e.preventDefault();
        setHighlightedIndex((prev) => Math.min(cases.length - 1, prev + 1));
      } else if (e.key === 'k' || e.key === 'ArrowUp') {
        e.preventDefault();
        setHighlightedIndex((prev) => Math.max(0, prev - 1));
      } else if (e.key === 'Enter') {
        e.preventDefault();
        const target = cases[highlightedIndex];
        if (target) {
          setSelectedCaseId(target.case_id);
          setNotesInput(target.analyst_notes || '');
        }
      } else if (e.key === 'f' || e.key === 'F') {
        const target = cases[highlightedIndex];
        if (target && target.status === 'OPEN') {
          e.preventDefault();
          mutation.mutate({
            caseId: target.case_id,
            status: 'CONFIRMED_FRAUD',
            notes: 'Confirmed fraud via keyboard hotkey [F]',
          });
        }
      } else if (e.key === 'c' || e.key === 'C') {
        const target = cases[highlightedIndex];
        if (target && target.status === 'OPEN') {
          e.preventDefault();
          mutation.mutate({
            caseId: target.case_id,
            status: 'FALSE_POSITIVE',
            notes: 'False positive cleared via keyboard hotkey [C]',
          });
        }
      }
    },
    [cases, highlightedIndex, mutation]
  );

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  const handleResetFilters = () => {
    setStatusFilter('ALL');
    setChannelFilter('ALL');
    setHighlightedIndex(0);
  };

  return (
    <div className="space-y-6 pb-[calc(3rem+env(safe-area-inset-bottom,0px))]">
      <PageHeader
        title="Case Management"
        purpose="Analyst review queue with active feedback loop to model training data."
        action={
          <div className="flex items-center gap-1.5 text-xs text-muted border border-border rounded-lg px-2.5 py-1 bg-surface">
            <Keyboard className="w-3.5 h-3.5 text-muted" />
            <span className="font-mono text-xs">J/K: Navigate &bull; F: Fraud &bull; C: Clear</span>
          </div>
        }
      />

      {/* Error state with Retry */}
      {isError && (
        <div className="p-4 rounded-xl border border-danger/30 bg-danger/10 text-xs text-danger flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>Failed to load case queue from API.</span>
          </div>
          <Button variant="secondary" size="sm" onClick={() => refetch()}>
            Retry
          </Button>
        </div>
      )}

      {/* Filter Bar */}
      <div className="p-3.5 rounded-xl border border-border bg-surface flex flex-col md:flex-row items-center justify-between gap-3 shadow-card">
        <div className="flex flex-wrap items-center gap-3 w-full md:w-auto">
          <Select
            label="Disposition"
            value={statusFilter}
            onChange={(e) => {
              setStatusFilter(e.target.value);
              setHighlightedIndex(0);
            }}
            options={[
              { value: 'ALL', label: 'All Dispositions' },
              { value: 'OPEN', label: 'Open Queue' },
              { value: 'CONFIRMED_FRAUD', label: 'Confirmed Fraud' },
              { value: 'FALSE_POSITIVE', label: 'False Positive' },
            ]}
          />

          <Select
            label="Channel"
            value={channelFilter}
            onChange={(e) => {
              setChannelFilter(e.target.value);
              setHighlightedIndex(0);
            }}
            options={[
              { value: 'ALL', label: 'All Channels' },
              { value: 'upi', label: 'UPI Instant' },
              { value: 'transfer_wallet', label: 'Wallet Transfer' },
              { value: 'card', label: 'Card Payment' },
              { value: 'account', label: 'Account Opening' },
            ]}
          />
        </div>

        <div className="flex items-center gap-3 text-xs text-muted">
          {isFetching && !isLoading && (
            <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs text-accent bg-accent/10 border border-accent/20 animate-pulse">
              <span className="w-1.5 h-1.5 rounded-full bg-accent" />
              Updating
            </span>
          )}
          <div>
            Queue: <strong className="font-mono tabular-nums text-text font-semibold">{cases.length}</strong> alerts
          </div>
        </div>
      </div>

      {/* Case Queue Table */}
      <div className="bg-surface border border-border rounded-xl shadow-card overflow-hidden">
        {/* Desktop / Tablet Table */}
        <div className="hidden md:block overflow-x-auto max-h-[600px]">
          <table className="w-full min-w-[800px] text-left text-xs border-collapse">
            <thead className="bg-surface-subtle border-b border-border sticky top-0 z-20">
              <tr className="h-10">
                <th scope="col" className="py-2.5 px-3.5 font-semibold text-muted text-xs tracking-normal w-[110px]">
                  Case ID
                </th>
                <th scope="col" className="py-2.5 px-3.5 font-semibold text-muted text-xs tracking-normal w-[130px]">
                  Channel
                </th>
                <th scope="col" className="py-2.5 px-3.5 font-semibold text-muted text-xs tracking-normal w-[180px]">
                  Counterparty ID
                </th>
                <th scope="col" className="py-2.5 px-3.5 font-semibold text-muted text-xs tracking-normal text-right w-[110px]">
                  Amount
                </th>
                <th scope="col" className="py-2.5 px-3.5 font-semibold text-muted text-xs tracking-normal w-[90px]">
                  Risk score
                </th>
                <th scope="col" className="py-2.5 px-3.5 font-semibold text-muted text-xs tracking-normal w-[120px]">
                  Status
                </th>
                <th scope="col" className="py-2.5 px-3.5 font-semibold text-muted text-xs tracking-normal text-right min-w-[160px]">
                  Quick triage
                </th>
              </tr>
            </thead>

            <tbody className="divide-y divide-border font-sans">
              {isLoading && cases.length === 0 ? (
                Array.from({ length: 6 }).map((_, idx) => (
                  <tr key={`case-skel-${idx}`} className="h-[44px] animate-pulse">
                    <td className="py-2.5 px-3.5"><div className="h-3 bg-border rounded w-16" /></td>
                    <td className="py-2.5 px-3.5"><div className="h-3 bg-border rounded w-20" /></td>
                    <td className="py-2.5 px-3.5"><div className="h-3 bg-border rounded w-24" /></td>
                    <td className="py-2.5 px-3.5 text-right"><div className="h-3 bg-border rounded w-14 ml-auto" /></td>
                    <td className="py-2.5 px-3.5"><div className="h-3 bg-border rounded w-12" /></td>
                    <td className="py-2.5 px-3.5"><div className="h-3 bg-border rounded w-16" /></td>
                    <td className="py-2.5 px-3.5 text-right"><div className="h-3 bg-border rounded w-28 ml-auto" /></td>
                  </tr>
                ))
              ) : cases.length === 0 ? (
                <tr>
                  <td colSpan={7} className="py-16 text-center text-xs text-muted">
                    <div className="max-w-xs mx-auto space-y-2">
                      <Inbox className="w-8 h-8 mx-auto opacity-40 text-muted" />
                      <div className="font-semibold text-sm text-text">No cases in queue</div>
                      <p className="text-muted">All alerts matching your filter criteria have been triaged.</p>
                      <Button variant="secondary" size="sm" onClick={handleResetFilters} className="mt-2">
                        Reset filters
                      </Button>
                    </div>
                  </td>
                </tr>
              ) : (
                cases.map((c, idx) => {
                  const isSelected = highlightedIndex === idx;
                  return (
                    <tr
                      key={c.case_id}
                      onClick={() => {
                        setHighlightedIndex(idx);
                        setSelectedCaseId(c.case_id);
                        setNotesInput(c.analyst_notes || '');
                      }}
                      className={`h-[44px] cursor-pointer transition-colors duration-120 select-none ${isSelected
                          ? 'bg-surface-subtle ring-1 ring-inset ring-accent/40'
                          : 'hover:bg-surface-subtle'
                        }`}
                    >
                      <td className="py-2.5 px-3.5 font-mono text-xs font-semibold text-text">
                        {c.case_id}
                      </td>
                      <td className="py-2.5 px-3.5 text-xs text-text">
                        {getChannelReadableName(c.channel)}
                      </td>
                      <td className="py-2.5 px-3.5 font-mono text-xs text-muted truncate max-w-[160px]">
                        {c.entity_id || 'n/a'}
                      </td>
                      <td className="py-2.5 px-3.5 text-right font-mono text-xs tabular-nums font-semibold text-text">
                        {formatChannelAmount(c.amount, c.channel)}
                      </td>
                      <td className="py-2.5 px-3.5 font-mono text-xs tabular-nums">
                        <span className={c.risk_score >= 0.85 ? 'text-danger font-semibold' : 'text-warn font-semibold'}>
                          {c.risk_score.toFixed(3)}
                        </span>
                      </td>
                      <td className="py-2.5 px-3.5">
                        <Badge
                          variant={
                            c.status === 'CONFIRMED_FRAUD'
                              ? 'high'
                              : c.status === 'FALSE_POSITIVE'
                                ? 'low'
                                : 'medium'
                          }
                        >
                          {c.status.replace('_', ' ')}
                        </Badge>
                      </td>
                      <td className="py-2.5 px-3.5 text-right" onClick={(e) => e.stopPropagation()}>
                        {c.status === 'OPEN' ? (
                          <div className="flex items-center justify-end gap-1.5">
                            <Button
                              variant="destructive"
                              size="sm"
                              className="text-xs py-1 px-2.5 h-7"
                              onClick={() =>
                                mutation.mutate({
                                  caseId: c.case_id,
                                  status: 'CONFIRMED_FRAUD',
                                  notes: 'Confirmed fraud via row action',
                                })
                              }
                            >
                              <XCircle className="w-3 h-3 mr-1" />
                              Fraud
                            </Button>
                            <Button
                              variant="secondary"
                              size="sm"
                              className="text-xs py-1 px-2.5 h-7 text-accent"
                              onClick={() =>
                                mutation.mutate({
                                  caseId: c.case_id,
                                  status: 'FALSE_POSITIVE',
                                  notes: 'False positive cleared via row action',
                                })
                              }
                            >
                              <CheckCircle2 className="w-3 h-3 mr-1" />
                              Clear
                            </Button>
                          </div>
                        ) : (
                          <span className="text-xs text-muted font-sans">
                            Resolved
                          </span>
                        )}
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>

        {/* Mobile Stacked Cards View (No horizontal scroll, tap targets >= 44px) */}
        <div className="md:hidden p-3 space-y-2.5">
          {isLoading && cases.length === 0 ? (
            Array.from({ length: 4 }).map((_, idx) => (
              <div key={`m-case-skel-${idx}`} className="p-3.5 rounded-lg border border-border bg-surface space-y-2 animate-pulse">
                <div className="flex justify-between"><Skeleton className="h-4 w-24" /><Skeleton className="h-4 w-16" /></div>
                <Skeleton className="h-4 w-32" />
                <Skeleton className="h-3 w-full" />
              </div>
            ))
          ) : cases.length === 0 ? (
            <div className="p-8 text-center text-xs text-muted">No cases in queue.</div>
          ) : (
            cases.map((c) => (
              <div
                key={`m-case-${c.case_id}`}
                onClick={() => setSelectedCaseId(c.case_id)}
                className="p-3.5 rounded-lg border border-border bg-surface hover:bg-surface-subtle active:bg-surface-subtle transition-colors cursor-pointer space-y-2.5 min-h-[44px]"
              >
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-xs font-semibold text-text">{c.case_id}</span>
                    <span className="text-xs text-muted">&bull;</span>
                    <span className="text-xs text-muted">{getChannelReadableName(c.channel)}</span>
                  </div>
                  <Badge
                    variant={
                      c.status === 'OPEN'
                        ? 'medium'
                        : c.status === 'CONFIRMED_FRAUD'
                        ? 'high'
                        : 'low'
                    }
                  >
                    {c.status.replace('_', ' ')}
                  </Badge>
                </div>

                <div className="flex items-baseline justify-between gap-2">
                  <span className="font-mono text-sm font-semibold tabular-nums text-text">
                    {formatChannelAmount(c.amount, c.channel)}
                  </span>
                  <div className="flex items-center gap-1.5 font-mono text-xs text-muted">
                    <span>Risk Score</span>
                    <span className="font-semibold text-text">{c.risk_score.toFixed(2)}</span>
                  </div>
                </div>

                <div className="text-xs text-muted font-mono truncate">
                  Counterparty: {c.counterparty_id}
                </div>

                {c.status === 'OPEN' && (
                  <div className="flex items-center gap-2 pt-1 border-t border-border/40" onClick={(e) => e.stopPropagation()}>
                    <Button
                      variant="destructive"
                      size="sm"
                      className="flex-1 min-h-[44px] text-xs py-2"
                      onClick={() =>
                        mutation.mutate({
                          caseId: c.case_id,
                          status: 'CONFIRMED_FRAUD',
                          notes: 'Confirmed fraud via card action',
                        })
                      }
                    >
                      <XCircle className="w-3.5 h-3.5 mr-1.5" />
                      Fraud
                    </Button>
                    <Button
                      variant="secondary"
                      size="sm"
                      className="flex-1 min-h-[44px] text-xs py-2 text-accent"
                      onClick={() =>
                        mutation.mutate({
                          caseId: c.case_id,
                          status: 'FALSE_POSITIVE',
                          notes: 'False positive cleared via card action',
                        })
                      }
                    >
                      <CheckCircle2 className="w-3.5 h-3.5 mr-1.5" />
                      Clear
                    </Button>
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      </div>

      {/* Sheet Drawer for Case Investigation & Audit Trail */}
      <Sheet open={!!selectedCaseId} onOpenChange={(open) => !open && setSelectedCaseId(null)}>
        <SheetContent side="right" className="overflow-y-auto">
          {detailData && (
            <div className="space-y-6 text-xs">
              <SheetHeader>
                <div className="flex items-center gap-2 text-xs text-muted mb-1 font-mono">
                  <span>{detailData.case.case_id}</span>
                  <span>&bull;</span>
                  <span>{getChannelReadableName(detailData.case.channel)}</span>
                </div>
                <SheetTitle>Case Investigation</SheetTitle>
                <SheetDescription>
                  Review counterparty risk telemetry and register disposition feedback.
                </SheetDescription>
              </SheetHeader>

              {/* Case Parameters Card */}
              <div className="p-4 rounded-xl border border-border bg-surface-subtle space-y-2 font-mono">
                <div className="flex justify-between items-center">
                  <span className="text-muted font-sans">Amount:</span>
                  <span className="font-semibold text-text tabular-nums">
                    {formatChannelAmount(detailData.case.amount, detailData.case.channel)}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-muted font-sans">Calibrated Risk Score:</span>
                  <span className="font-semibold text-danger tabular-nums">
                    {detailData.case.risk_score.toFixed(4)}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-muted font-sans">Entity ID:</span>
                  <span className="text-text">{detailData.case.entity_id || 'n/a'}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-muted font-sans">Status:</span>
                  <Badge
                    variant={
                      detailData.case.status === 'CONFIRMED_FRAUD'
                        ? 'high'
                        : detailData.case.status === 'FALSE_POSITIVE'
                          ? 'low'
                          : 'medium'
                    }
                  >
                    {detailData.case.status.replace('_', ' ')}
                  </Badge>
                </div>
              </div>

              {/* Investigator Notes */}
              <div className="space-y-2">
                <div className="font-semibold text-xs text-text">
                  Investigator Disposition Notes
                </div>
                <textarea
                  value={notesInput}
                  onChange={(e) => setNotesInput(e.target.value)}
                  placeholder="Record observations, evidence, and audit rationale..."
                  className="w-full h-24 p-3 rounded-lg border border-border bg-surface-subtle text-xs text-text focus:outline-none focus:ring-2 focus:ring-accent font-sans"
                />
              </div>

              {/* Action Buttons */}
              <div className="flex items-center gap-2 pt-1">
                <Button
                  variant="destructive"
                  size="sm"
                  className="flex-1 justify-center min-h-[38px]"
                  onClick={() =>
                    mutation.mutate({
                      caseId: detailData.case.case_id,
                      status: 'CONFIRMED_FRAUD',
                      notes: notesInput || 'Confirmed fraud disposition recorded',
                    })
                  }
                >
                  <XCircle className="w-3.5 h-3.5 mr-1.5" />
                  Confirm Fraud
                </Button>
                <Button
                  variant="secondary"
                  size="sm"
                  className="flex-1 justify-center min-h-[38px] text-accent"
                  onClick={() =>
                    mutation.mutate({
                      caseId: detailData.case.case_id,
                      status: 'FALSE_POSITIVE',
                      notes: notesInput || 'False positive disposition recorded',
                    })
                  }
                >
                  <CheckCircle2 className="w-3.5 h-3.5 mr-1.5" />
                  False Positive
                </Button>
              </div>

              {/* Audit Trail */}
              <div className="pt-4 border-t border-border space-y-2.5">
                <div className="font-semibold text-xs text-text flex items-center gap-1.5">
                  <Clock className="w-3.5 h-3.5 text-muted" />
                  <span>Audit Trail</span>
                </div>
                <div className="space-y-2 font-mono text-xs">
                  {detailData.audit_logs?.map((log: CaseAuditLog) => (
                    <div
                      key={log.log_id}
                      className="p-2.5 rounded-lg border border-border bg-surface-subtle space-y-1"
                    >
                      <div className="flex justify-between text-muted text-xs">
                        <span>{log.analyst_name}</span>
                        <span>{log.timestamp}</span>
                      </div>
                      <div className="text-text font-sans">
                        Action: <strong className="text-accent">{log.action}</strong>
                      </div>
                      {log.notes && (
                        <div className="text-muted font-sans text-xs pt-0.5">
                          {log.notes}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </SheetContent>
      </Sheet>
    </div>
  );
};
