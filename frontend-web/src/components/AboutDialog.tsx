import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { fetchOverviewKPIs } from '../api/client';
import type { OverviewKPIs } from '../types/api';
import { Dialog } from './ui/Dialog';
import { Button } from './ui/Button';
import { Shield, Cpu, Database, Activity, FileCode } from 'lucide-react';

interface AboutDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onNavigate?: (tab: string) => void;
}

export const AboutDialog: React.FC<AboutDialogProps> = ({
  open,
  onOpenChange,
  onNavigate,
}) => {
  const { data: kpis } = useQuery<OverviewKPIs>({
    queryKey: ['overviewKPIs'],
    queryFn: () => fetchOverviewKPIs(),
    staleTime: 60000,
  });
  return (
    <Dialog
      open={open}
      onOpenChange={onOpenChange}
      title="About FraudShield"
      description="Multi-channel payment fraud detection platform"
    >
      <div className="space-y-4 text-xs text-[#0F172A] dark:text-[#E8EDF2] leading-relaxed">
        {/* Core Identity */}
        <div className="flex items-center gap-3 p-3 rounded-lg border border-[#E3E8EE] dark:border-[#1E2630] bg-slate-50 dark:bg-[#11171E]">
          <div className="w-9 h-9 rounded-md bg-[#0F172A] text-white dark:bg-[#161F28] dark:text-[#2DD4A0] flex items-center justify-center font-semibold text-base shrink-0 border border-[#E3E8EE] dark:border-[#1E2630]">
            <Shield className="w-4 h-4" />
          </div>
          <div>
            <div className="font-sans font-semibold text-sm tracking-normal text-text">FraudShield Decision Gate</div>
            <div className="text-xs text-[#52606D] dark:text-[#8A96A3]">
              Pre-authorization scoring & explainable triage across cards, wallets, accounts, and UPI.
            </div>
          </div>
        </div>

        {/* Provenance & Citations */}
        <div>
          <div className="flex items-center gap-1.5 text-xs font-semibold text-[#52606D] dark:text-[#8A96A3] mb-2">
            <Database className="w-3.5 h-3.5" />
            <span>Benchmark Citations & Data Sources</span>
          </div>
          <div className="space-y-2 border border-[#E3E8EE] dark:border-[#1E2630] rounded-lg p-2.5 bg-slate-50/50 dark:bg-[#0B0F14]/50">
            <div className="text-xs">
              <span className="font-semibold text-[#0F172A] dark:text-[#E8EDF2]">UPI India: </span>
              <span className="text-[#52606D] dark:text-[#8A96A3]">
                Synthetic Indian UPI Transaction Stream (50,000 txns, behavioral simulator).
              </span>
            </div>
            <div className="text-xs">
              <span className="font-semibold text-[#0F172A] dark:text-[#E8EDF2]">Card Payments: </span>
              <span className="text-[#52606D] dark:text-[#8A96A3]">
                European Cardholders Benchmark (Dal Pozzolo et al., 2015, ULB Machine Learning Group).
              </span>
            </div>
            <div className="text-xs">
              <span className="font-semibold text-[#0F172A] dark:text-[#E8EDF2]">Wallets & Transfers: </span>
              <span className="text-[#52606D] dark:text-[#8A96A3]">
                PaySim Synthetic Mobile Money (Lopez-Rojas et al., 2016).
              </span>
            </div>
            <div className="text-xs">
              <span className="font-semibold text-[#0F172A] dark:text-[#E8EDF2]">Bank Accounts: </span>
              <span className="text-[#52606D] dark:text-[#8A96A3]">
                Bank Account Fraud Benchmark (BAF, Jesus et al., NeurIPS 2022).
              </span>
            </div>
          </div>
        </div>

        {/* Architecture & Metrics */}
        <div>
          <div className="flex items-center gap-1.5 text-xs font-semibold text-[#52606D] dark:text-[#8A96A3] mb-2">
            <Cpu className="w-3.5 h-3.5" />
            <span>Architecture & Measured Latency</span>
          </div>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="p-2.5 rounded-lg border border-border bg-surface">
              <div className="text-xs text-muted font-sans">p50 Latency</div>
              <div className="text-sm font-semibold font-mono text-text">
                {kpis?.latency_p50_ms ? `${kpis.latency_p50_ms.toFixed(2)} ms` : '-- ms'}
              </div>
            </div>
            <div className="p-2.5 rounded-lg border border-border bg-surface">
              <div className="text-xs text-muted font-sans">p95 Latency</div>
              <div className="text-sm font-semibold font-mono text-text">
                {kpis?.latency_p95_ms ? `${kpis.latency_p95_ms.toFixed(2)} ms` : '-- ms'}
              </div>
            </div>
          </div>
        </div>

        {/* Data Integrity Notice */}
        <div className="p-2.5 rounded-lg border border-amber-300/60 dark:border-amber-900/40 bg-amber-500/5 text-xs text-amber-900 dark:text-amber-200 leading-snug">
          <span className="font-semibold">Data Integrity Notice: </span>
          All data used for training and evaluation is sourced strictly from public academic research benchmarks and synthetic simulation models. No proprietary bank data or live private customer records are stored or processed.
        </div>

        {/* Action Buttons */}
        <div className="flex items-center justify-between pt-2 border-t border-[#E3E8EE] dark:border-[#1E2630]">
          <div className="flex items-center gap-2">
            {onNavigate && (
              <Button
                variant="secondary"
                size="sm"
                onClick={() => {
                  onNavigate('api_docs');
                  onOpenChange(false);
                }}
              >
                <FileCode className="w-3.5 h-3.5" />
                <span>API Docs</span>
              </Button>
            )}
            {onNavigate && (
              <Button
                variant="secondary"
                size="sm"
                onClick={() => {
                  onNavigate('models');
                  onOpenChange(false);
                }}
              >
                <Activity className="w-3.5 h-3.5" />
                <span>Model Specs</span>
              </Button>
            )}
            <Button
              variant="secondary"
              size="sm"
              onClick={() => {
                onOpenChange(false);
                window.dispatchEvent(new CustomEvent('fraudshield:replay-intro'));
              }}
            >
              <Shield className="w-3.5 h-3.5 text-emerald-500" />
              <span>Replay Opening</span>
            </Button>
          </div>
          <Button variant="primary" size="sm" onClick={() => onOpenChange(false)}>
            Close
          </Button>
        </div>
      </div>
    </Dialog>
  );
};
