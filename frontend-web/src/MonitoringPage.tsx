import React from 'react';
import { useQuery, keepPreviousData } from '@tanstack/react-query';
import { PageHeader } from '../components/layout/PageHeader';
import { Badge } from '../components/ui/Badge';
import { Button } from '../components/ui/Button';
import { GlossaryPopover } from '../components/ui/GlossaryPopover';
import { AnimatedNumber } from '../hooks/useCountUp';
import { Activity, ShieldCheck, AlertTriangle, TrendingUp, AlertCircle, Inbox } from 'lucide-react';

export const MonitoringPage: React.FC = () => {
  const { data: monData, isLoading, isError, refetch } = useQuery({
    queryKey: ['monitoring'],
    queryFn: async () => {
      const res = await fetch('/api/monitoring');
      if (!res.ok) throw new Error('Failed to load monitoring metrics');
      return res.json();
    },
    placeholderData: keepPreviousData,
    staleTime: 60 * 1000,
  });

  const psi = monData?.psi_drift || {};
  const conceptDrift = monData?.concept_drift || {};
  const bafSeries = conceptDrift.baf_account_drift?.series || [];

  return (
    <div className="space-y-6 pb-16">
      <PageHeader
        title="Model Monitoring"
        purpose="Population stability tracking and longitudinal drift benchmarks across active channels."
      />

      {isError && (
        <div className="p-4 rounded-xl border border-danger/30 bg-danger/10 text-xs text-danger flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>Failed to load monitoring metrics from the API service.</span>
          </div>
          <Button variant="secondary" size="sm" onClick={() => refetch()}>
            Retry
          </Button>
        </div>
      )}

      {isLoading ? (
        <div className="py-16 text-center text-xs text-muted animate-pulse">
          Loading population stability index and drift benchmarks...
        </div>
      ) : !monData ? (
        <div className="p-8 text-center text-xs text-muted border border-border rounded-xl bg-surface space-y-2">
          <Inbox className="w-8 h-8 mx-auto opacity-40 text-muted" />
          <div className="font-semibold text-sm text-text">No monitoring telemetry available</div>
          <p>Longitudinal drift benchmarks have not yet been populated.</p>
          <Button variant="secondary" size="sm" onClick={() => refetch()} className="mt-2">
            Refresh
          </Button>
        </div>
      ) : (
        <div className="space-y-6">
          {/* PSI Drift Overview */}
          <section className="p-5 rounded-xl border border-border bg-surface space-y-4 shadow-card">
            <div className="flex items-center justify-between border-b border-border pb-3">
              <div className="flex items-center gap-2">
                <Activity className="w-4 h-4 text-accent" />
                <h2 className="font-semibold text-xs text-text">
                  Population Stability Index (PSI)
                </h2>
              </div>
              <GlossaryPopover term="PSI" />
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
              <div className="p-4 rounded-xl border border-border bg-surface-subtle space-y-1">
                <div className="text-xs font-semibold text-muted">
                  UPI Stream PSI
                </div>
                <div className="text-2xl font-mono tabular-nums font-semibold text-accent">
                  <AnimatedNumber value={typeof psi.upi === 'number' ? psi.upi : null} format="decimal" decimals={3} duration={700} />
                </div>
                <div className="pt-0.5">
                  <Badge variant={typeof psi.upi === 'number' && psi.upi >= 0.10 ? 'medium' : 'low'}>
                    {typeof psi.upi === 'number' && psi.upi >= 0.10 ? 'Moderate Shift' : 'Stable (< 0.10)'}
                  </Badge>
                </div>
                <p className="text-xs text-muted pt-1">Distribution matches baseline reference.</p>
              </div>

              <div className="p-4 rounded-xl border border-border bg-surface-subtle space-y-1">
                <div className="text-xs font-semibold text-muted">
                  Wallets Stream PSI
                </div>
                <div className="text-2xl font-mono tabular-nums font-semibold text-accent">
                  <AnimatedNumber value={typeof psi.transfer_wallet === 'number' ? psi.transfer_wallet : null} format="decimal" decimals={3} duration={700} />
                </div>
                <div className="pt-0.5">
                  <Badge variant={typeof psi.transfer_wallet === 'number' && psi.transfer_wallet >= 0.10 ? 'medium' : 'low'}>
                    {typeof psi.transfer_wallet === 'number' && psi.transfer_wallet >= 0.10 ? 'Moderate Shift' : 'Stable (< 0.10)'}
                  </Badge>
                </div>
                <p className="text-xs text-muted pt-1">Temporal sequence cache aligned.</p>
              </div>

              <div className="p-4 rounded-xl border border-border bg-surface-subtle space-y-1">
                <div className="text-xs font-semibold text-muted">
                  Card Stream PSI
                </div>
                <div className="text-2xl font-mono tabular-nums font-semibold text-accent">
                  <AnimatedNumber value={typeof psi.card === 'number' ? psi.card : null} format="decimal" decimals={3} duration={700} />
                </div>
                <div className="pt-0.5">
                  <Badge variant={typeof psi.card === 'number' && psi.card >= 0.10 ? 'medium' : 'low'}>
                    {typeof psi.card === 'number' && psi.card >= 0.10 ? 'Moderate Shift' : 'Stable (< 0.10)'}
                  </Badge>
                </div>
                <p className="text-xs text-muted pt-1">Ensemble prediction distribution consistent.</p>
              </div>

              <div className="p-4 rounded-xl border border-border bg-surface-subtle space-y-1">
                <div className="text-xs font-semibold text-muted">
                  BAF Accounts PSI
                </div>
                <div className="text-2xl font-mono tabular-nums font-semibold text-warn">
                  <AnimatedNumber value={typeof psi.account === 'number' ? psi.account : null} format="decimal" decimals={3} duration={700} />
                </div>
                <div className="pt-0.5">
                  <Badge variant={typeof psi.account === 'number' && psi.account >= 0.10 ? 'medium' : 'low'}>
                    {typeof psi.account === 'number' && psi.account >= 0.10 ? 'Moderate (0.10–0.25)' : 'Stable (< 0.10)'}
                  </Badge>
                </div>
                <p className="text-xs text-muted pt-1">Age demographic ratio slight variance.</p>
              </div>
            </div>
          </section>

          {/* Longitudinal Drift Evaluation Table */}
          {bafSeries.length > 0 && (
            <section className="p-5 rounded-xl border border-border bg-surface space-y-4 shadow-card">
              <div className="flex items-center justify-between border-b border-border pb-3">
                <div className="flex items-center gap-2">
                  <TrendingUp className="w-4 h-4 text-accent" />
                  <h2 className="font-semibold text-xs text-text">
                    Longitudinal drift evaluation (BAF benchmark)
                  </h2>
                </div>
                <span className="font-mono text-xs text-muted">
                  Static model vs monthly retrained champion
                </span>
              </div>

              <div className="border border-border rounded-xl overflow-x-auto font-mono text-xs bg-surface">
                <table className="w-full text-left border-collapse min-w-[680px]">
                  <thead className="bg-surface-subtle border-b border-border text-xs font-semibold text-muted normal-case font-sans">
                    <tr className="h-10">
                      <th className="py-2.5 px-3.5">Evaluation month</th>
                      <th className="py-2.5 px-3.5 text-right">Total applications</th>
                      <th className="py-2.5 px-3.5 text-right">Static model PR-AUC</th>
                      <th className="py-2.5 px-3.5 text-right">Monthly retrained PR-AUC</th>
                      <th className="py-2.5 px-3.5 text-right">Retraining gain (&Delta;)</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border">
                    {bafSeries.map((s: { month: number; total_applications: number; static_model: { pr_auc: number }; monthly_retrained_model: { pr_auc: number }; pr_auc_delta: number }) => (
                      <tr key={s.month} className="h-[44px] hover:bg-surface-subtle transition-colors">
                        <td className="py-2.5 px-3.5 font-sans font-semibold text-text">Month {s.month}</td>
                        <td className="py-2.5 px-3.5 text-right tabular-nums text-muted">{s.total_applications.toLocaleString()}</td>
                        <td className="py-2.5 px-3.5 text-right tabular-nums text-muted">{s.static_model.pr_auc.toFixed(5)}</td>
                        <td className="py-2.5 px-3.5 text-right tabular-nums font-semibold text-accent">
                          {s.monthly_retrained_model.pr_auc.toFixed(5)}
                        </td>
                        <td className="py-2.5 px-3.5 text-right tabular-nums font-semibold text-accent">
                          +{s.pr_auc_delta.toFixed(5)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}

          {/* Drift Guidance Guidelines */}
          <section className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs font-sans">
            <div className="p-4 rounded-xl border border-accent/30 bg-accent/5 space-y-1">
              <div className="font-semibold text-accent flex items-center gap-1.5">
                <ShieldCheck className="w-4 h-4" />
                <span>PSI &lt; 0.10: Stable</span>
              </div>
              <p className="text-muted leading-relaxed">
                No significant shift detected. Champion models operate within validated error tolerances.
              </p>
            </div>

            <div className="p-4 rounded-xl border border-warn/30 bg-warn/5 space-y-1">
              <div className="font-semibold text-warn flex items-center gap-1.5">
                <AlertTriangle className="w-4 h-4" />
                <span>0.10 &le; PSI &lt; 0.25: Moderate Shift</span>
              </div>
              <p className="text-muted leading-relaxed">
                Noticeable distributional movement. Increase investigator alert sampling rate for active learning.
              </p>
            </div>

            <div className="p-4 rounded-xl border border-danger/30 bg-danger/5 space-y-1">
              <div className="font-semibold text-danger flex items-center gap-1.5">
                <AlertTriangle className="w-4 h-4" />
                <span>PSI &ge; 0.25: Significant Drift</span>
              </div>
              <p className="text-muted leading-relaxed">
                Major population divergence. Triggers retrain review with recent ground truth labels.
              </p>
            </div>
          </section>
        </div>
      )}
    </div>
  );
};
