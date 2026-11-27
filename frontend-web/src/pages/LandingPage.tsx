import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Button } from '../components/ui/Button';
import { Dialog } from '../components/ui/Dialog';
import {
  ArrowRight,
  Sparkle,
  WarningCircle,
  Info,
  ArrowSquareOut,
  SlidersHorizontal,
  Gauge,
  ShieldCheck,
} from '@phosphor-icons/react';
import { ProductPreviewHero } from '../components/landing/ProductPreviewHero';
import { ChannelBenchmarkTable } from '../components/common/ChannelBenchmarkTable';

export interface LandingPageProps {
  onNavigate: (tab: string) => void;
  isDark: boolean;
}

export const LandingPage: React.FC<LandingPageProps> = ({ onNavigate, isDark }) => {
  const [limitationOpen, setLimitationOpen] = useState(false);

  return (
    <div className="space-y-4 pb-12 font-sans overflow-x-hidden">
      {/* =========================================================================
          HERO & PRODUCT PREVIEW: Headline, Subtitle, 2 CTAs, Decision Card + Gate
          ========================================================================= */}
      <section className="pt-2 sm:pt-4 pb-8 flex flex-col items-center">
        <ProductPreviewHero onNavigate={onNavigate} isDark={isDark} />
      </section>

      {/* =========================================================================
          THREE SHORT STEPS: Features, Score, Action as a clean minimal row
          ========================================================================= */}
      <motion.section
        initial={{ opacity: 0, y: 8 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: '-40px' }}
        transition={{ duration: 0.32, ease: [0.16, 1, 0.3, 1] }}
        className="py-12 border-t border-border"
      >
        <div className="max-w-5xl mx-auto px-4">
          <div className="mb-8">
            <h2 className="text-xl font-semibold tracking-normal text-text">
              How the gate evaluates transactions.
            </h2>
            <p className="text-sm text-muted mt-1">
              Three synchronous phases completed in under 15 milliseconds.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 divide-y md:divide-y-0 md:divide-x divide-border">
            {/* Step 1: Features */}
            <div className="space-y-3 pt-4 md:pt-0 md:pr-6">
              <div className="w-9 h-9 rounded-md bg-surface-subtle border border-border flex items-center justify-center text-text">
                <SlidersHorizontal className="w-4 h-4 text-accent" weight="thin" />
              </div>
              <h3 className="text-sm font-semibold text-text">1. Extract features</h3>
              <p className="text-xs text-muted leading-relaxed">
                Computes entity behavioral velocity, cross-channel graph degrees, and historical baseline deviations.
              </p>
            </div>

            {/* Step 2: Score */}
            <div className="space-y-3 pt-6 md:pt-0 md:px-6">
              <div className="w-9 h-9 rounded-md bg-surface-subtle border border-border flex items-center justify-center text-text">
                <Gauge className="w-4 h-4 text-accent" weight="thin" />
              </div>
              <h3 className="text-sm font-semibold text-text">2. Calibrate score</h3>
              <p className="text-xs text-muted leading-relaxed">
                Evaluates champion tree ensembles with isotonic and Platt probability scaling for honest risk estimates.
              </p>
            </div>

            {/* Step 3: Action */}
            <div className="space-y-3 pt-6 md:pt-0 md:pl-6">
              <div className="w-9 h-9 rounded-md bg-surface-subtle border border-border flex items-center justify-center text-text">
                <ShieldCheck className="w-4 h-4 text-accent" weight="thin" />
              </div>
              <h3 className="text-sm font-semibold text-text">3. Route decision</h3>
              <p className="text-xs text-muted leading-relaxed">
                Enforces per-rail loss thresholds: auto-approve, step-up verification challenge, or immediate block.
              </p>
            </div>
          </div>
        </div>
      </motion.section>

      {/* =========================================================================
          PER-CHANNEL RESULTS TABLE (Reusing Overview Table Component)
          ========================================================================= */}
      <motion.section
        initial={{ opacity: 0, y: 8 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: '-40px' }}
        transition={{ duration: 0.32, ease: [0.16, 1, 0.3, 1] }}
        className="py-12 border-t border-border"
      >
        <div className="max-w-5xl mx-auto px-4 space-y-6">
          <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-3">
            <div>
              <h2 className="text-xl font-semibold tracking-normal text-text">
                Per-channel benchmark performance
              </h2>
              <p className="text-sm text-muted mt-1">
                Validated on held-out test splits with native rail currencies. Zero blended metrics.
              </p>
            </div>

            {/* Operational Limitations Link */}
            <button
              type="button"
              onClick={() => setLimitationOpen(true)}
              className="inline-flex items-center gap-1.5 text-xs text-muted hover:text-text transition-colors self-start sm:self-auto focus:outline-none focus-visible:ring-2 focus-visible:ring-accent rounded-sm"
            >
              <Info className="w-3.5 h-3.5 text-warn" weight="thin" />
              <span className="underline underline-offset-4 font-medium">What this can't do</span>
            </button>
          </div>

          {/* Tidy Table Component */}
          <ChannelBenchmarkTable showTitle={false} />
        </div>
      </motion.section>

      {/* =========================================================================
          CLOSING CALL TO ACTION
          ========================================================================= */}
      <motion.section
        initial={{ opacity: 0, y: 8 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: '-40px' }}
        transition={{ duration: 0.32, ease: [0.16, 1, 0.3, 1] }}
        className="py-14 border-t border-border"
      >
        <div className="max-w-2xl mx-auto px-4 text-center space-y-4">
          <h3 className="text-2xl font-semibold tracking-normal text-text">
            Ready to evaluate payment streams?
          </h3>
          <p className="text-sm text-muted max-w-md mx-auto">
            Inspect real ledger distributions, analyze decision latencies, or simulate custom transaction payloads.
          </p>
          <div className="pt-2 flex flex-col sm:flex-row items-center justify-center gap-3">
            <Button
              variant="primary"
              size="md"
              onClick={() => onNavigate('overview')}
              className="w-full sm:w-auto justify-center px-5 min-h-[44px]"
            >
              <span>Open dashboard</span>
              <ArrowRight className="w-3.5 h-3.5 ml-1.5" weight="thin" />
            </Button>
            <Button
              variant="secondary"
              size="md"
              onClick={() => onNavigate('live_check')}
              className="w-full sm:w-auto justify-center px-5 min-h-[44px]"
            >
              <Sparkle className="w-3.5 h-3.5 mr-1.5 text-accent" weight="thin" />
              <span>Try a live check</span>
            </Button>
          </div>
        </div>
      </motion.section>

      {/* =========================================================================
          SLIM FOOTER WITH DATASET CITATIONS
          ========================================================================= */}
      <footer className="py-6 border-t border-border text-center text-xs text-muted">
        <div className="max-w-5xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-3">
          <div>
            <span>FraudShield Decision Engine · Benchmark evaluations on public and simulated ledgers.</span>
          </div>
          <div className="flex items-center gap-4 text-xs font-mono">
            <a
              href="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
              target="_blank"
              rel="noreferrer"
              className="hover:text-accent transition-colors inline-flex items-center gap-1"
            >
              <span>ULB</span>
              <ArrowSquareOut className="w-2.5 h-2.5" weight="thin" />
            </a>
            <a
              href="https://www.kaggle.com/datasets/ealaxi/paysim1"
              target="_blank"
              rel="noreferrer"
              className="hover:text-accent transition-colors inline-flex items-center gap-1"
            >
              <span>PaySim</span>
              <ArrowSquareOut className="w-2.5 h-2.5" weight="thin" />
            </a>
            <a
              href="https://github.com/feedzai/bank-account-fraud"
              target="_blank"
              rel="noreferrer"
              className="hover:text-accent transition-colors inline-flex items-center gap-1"
            >
              <span>BAF (NeurIPS)</span>
              <ArrowSquareOut className="w-2.5 h-2.5" weight="thin" />
            </a>
          </div>
        </div>
      </footer>

      {/* "What this can't do" Dialog (Documented Operational Limitations) */}
      <Dialog
        open={limitationOpen}
        onOpenChange={setLimitationOpen}
        title="What This Platform Can't Do"
        description="Honest operational boundaries and documented blind spots."
      >
        <div className="space-y-4 pt-2 text-xs leading-relaxed text-muted">
          <div className="p-3 rounded-lg border border-warn/20 bg-warn/5 space-y-1.5">
            <div className="font-semibold text-text flex items-center gap-1.5">
              <WarningCircle className="w-3.5 h-3.5 text-warn" weight="thin" />
              <span>1. Cold-Start Entities</span>
            </div>
            <p>
              First-time VPAs and fresh accounts lack historical baseline distributions. Detection relies on general population priors until entity history accumulates.
            </p>
          </div>

          <div className="p-3 rounded-lg border border-warn/20 bg-warn/5 space-y-1.5">
            <div className="font-semibold text-text flex items-center gap-1.5">
              <WarningCircle className="w-3.5 h-3.5 text-warn" weight="thin" />
              <span>2. Delayed Multi-Hour Split Evasion</span>
            </div>
            <p>
              When attackers split balances into small transfers spaced over 4+ hours, detection recall drops as temporal window caches expire.
            </p>
          </div>

          <div className="p-3 rounded-lg border border-warn/20 bg-warn/5 space-y-1.5">
            <div className="font-semibold text-text flex items-center gap-1.5">
              <WarningCircle className="w-3.5 h-3.5 text-warn" weight="thin" />
              <span>3. Public &amp; Simulated Data Prevalences</span>
            </div>
            <p>
              Public academic benchmarks have empirical prevalence distributions that may diverge from live production issuer traffic. True issuer deployments require continual fine-tuning.
            </p>
          </div>

          <div className="flex justify-end pt-2">
            <Button variant="secondary" size="sm" onClick={() => setLimitationOpen(false)}>
              Close
            </Button>
          </div>
        </div>
      </Dialog>
    </div>
  );
};
