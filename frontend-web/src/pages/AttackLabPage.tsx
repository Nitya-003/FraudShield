import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { fetchAttackPresets, simulateAttack } from '../api/client';
import type { AttackPreset, AttackSimulationResult } from '../types/api';
import { PageHeader } from '../components/layout/PageHeader';
import { Button } from '../components/ui/Button';
import { Badge } from '../components/ui/Badge';
import { Slider } from '../components/ui/Slider';
import { Input } from '../components/ui/Input';
import { InlineRiskBar } from '../components/ui/DataTable';
import {
  FlaskConical,
  ShieldAlert,
  ShieldCheck,
  Split,
  Clock,
  Coins,
  History,
  Play,
  Pause,
  SkipForward,
  SkipBack,
  RotateCcw,
  Sparkles,
  AlertCircle,
} from 'lucide-react';

export const AttackLabPage: React.FC = () => {
  const { data: presetsData, isLoading: presetsLoading, isError: presetsError, refetch } = useQuery({
    queryKey: ['attackPresets'],
    queryFn: fetchAttackPresets,
    staleTime: 10 * 60 * 1000,
  });

  const [selectedPresetId, setSelectedPresetId] = useState('paysim_full_drain');
  const [activeAttackType, setActiveAttackType] = useState('split_transfers');
  const [splitCount, setSplitCount] = useState(3);
  const [delayHours, setDelayHours] = useState(4);
  const [drainPct, setDrainPct] = useState(50);
  const [hopCount, setHopCount] = useState(2);

  const [simulationResult, setSimulationResult] = useState<AttackSimulationResult | null>(null);
  const [isSimulating, setIsSimulating] = useState(false);
  const [activeStepIndex, setActiveStepIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  const [historyLog, setHistoryLog] = useState<
    Array<{
      id: string;
      timestamp: string;
      attack: string;
      baseScore: number;
      peakScore: number;
      verdict: string;
    }>
  >([]);

  const currentPreset: AttackPreset | undefined = presetsData?.presets.find(
    (p) => p.id === selectedPresetId
  );

  useEffect(() => {
    if (!isPlaying || !simulationResult?.transfers?.length) return;
    const timer = setInterval(() => {
      setActiveStepIndex((prev) => (prev + 1) % simulationResult.transfers.length);
    }, 1400);
    return () => clearInterval(timer);
  }, [isPlaying, simulationResult]);

  const handleSimulate = async () => {
    if (!currentPreset) return;
    setIsSimulating(true);
    setIsPlaying(false);
    setActiveStepIndex(0);
    try {
      let params: Record<string, unknown> = {};
      if (activeAttackType === 'split_transfers') {
        params = { num_splits: splitCount };
      } else if (activeAttackType === 'timing_delay') {
        params = { delay_hours: delayHours };
      } else if (activeAttackType === 'partial_drain') {
        params = { drain_fraction: drainPct / 100.0 };
      } else {
        params = { hop_count: hopCount };
      }

      const res = await simulateAttack({
        channel: currentPreset.channel,
        base_payload: currentPreset.payload,
        attack_type: activeAttackType,
        params,
      });

      setSimulationResult(res.simulation);
      setHistoryLog((prev) => [
        {
          id: Math.random().toString(36).substring(2, 7).toUpperCase(),
          timestamp: new Date().toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
          }),
          attack: `${activeAttackType.replace('_', ' ')} (${JSON.stringify(params)})`,
          baseScore: res.simulation.base_score,
          peakScore: res.simulation.peak_attack_score,
          verdict: res.simulation.verdict,
        },
        ...prev.slice(0, 9),
      ]);
    } catch (err) {
      console.error(err);
    } finally {
      setIsSimulating(false);
    }
  };

  const activeTransfer = simulationResult?.transfers?.[activeStepIndex];

  return (
    <div className="space-y-6 pb-16">
      <PageHeader
        title="Attack Lab"
        purpose="Simulate evasion scenarios and adversarial perturbations against production models."
        action={
          <Button
            variant="primary"
            size="md"
            onClick={handleSimulate}
            disabled={isSimulating || presetsLoading}
          >
            <FlaskConical className="w-4 h-4" />
            <span>{isSimulating ? 'Simulating...' : 'Execute simulation'}</span>
          </Button>
        }
      />

      {presetsError && (
        <div className="p-4 rounded-xl border border-danger/30 bg-danger/10 text-xs text-danger flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>Failed to load attack presets from the API.</span>
          </div>
          <Button variant="secondary" size="sm" onClick={() => refetch()}>
            Retry
          </Button>
        </div>
      )}

      {/* 1. Base Fraud Pattern Selection */}
      <section className="space-y-3">
        <h2 className="text-xs font-semibold text-muted">
          1. Select Base Fraud Pattern
        </h2>
        {presetsLoading ? (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-3 animate-pulse">
            {[1, 2, 3, 4].map((i) => (
              <div
                key={i}
                className="h-20 rounded-xl bg-surface-subtle border border-border"
              />
            ))}
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3">
            {presetsData?.presets.map((p) => {
              const isSelected = selectedPresetId === p.id;
              return (
                <button
                  key={p.id}
                  onClick={() => {
                    setSelectedPresetId(p.id);
                    setSimulationResult(null);
                    setActiveStepIndex(0);
                    setIsPlaying(false);
                  }}
                  className={`p-3.5 rounded-xl border text-left transition-colors cursor-pointer ${
                    isSelected
                      ? 'border-accent bg-accent/10 text-text font-semibold shadow-card'
                      : 'border-border bg-surface text-text hover:bg-surface-subtle'
                  }`}
                >
                  <div className="font-semibold text-xs mb-1">{p.name}</div>
                  <div className="text-xs leading-relaxed text-muted">
                    {p.description}
                  </div>
                </button>
              );
            })}
          </div>
        )}
      </section>

      {/* 2. Adversarial Perturbation Controls */}
      <section className="p-5 rounded-xl border border-border bg-surface space-y-5 shadow-card">
        <h2 className="text-xs font-semibold text-muted">
          2. Configure Adversarial Perturbation
        </h2>

        {/* Attack Type Segment Selector */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
          {[
            { id: 'split_transfers', label: 'Split Transfers', icon: Split },
            { id: 'timing_delay', label: 'Timing Delay', icon: Clock },
            { id: 'partial_drain', label: 'Partial Drain', icon: Coins },
            { id: 'mule_hop', label: 'Mule Hops', icon: FlaskConical },
          ].map((att) => {
            const Icon = att.icon;
            const active = activeAttackType === att.id;
            return (
              <button
                key={att.id}
                onClick={() => {
                  setActiveAttackType(att.id);
                  setActiveStepIndex(0);
                  setIsPlaying(false);
                }}
                className={`flex items-center justify-center gap-2 p-3 rounded-lg border text-xs transition-colors cursor-pointer ${
                  active
                    ? 'border-accent bg-accent/10 text-text font-semibold shadow-card'
                    : 'border-border bg-surface-subtle text-muted hover:text-text'
                }`}
              >
                <Icon className="w-4 h-4" />
                <span>{att.label}</span>
              </button>
            );
          })}
        </div>

        {/* Dynamic Parameter Sliders with Editable Numeric Input Fields */}
        <div className="p-4 rounded-xl border border-border bg-surface-subtle">
          {activeAttackType === 'split_transfers' && (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div>
                  <span className="font-semibold text-xs text-text">Split count</span>
                  <p className="text-xs text-muted">Decompose transaction into smaller micro-transfers</p>
                </div>
                <div className="w-20">
                  <Input
                    type="number"
                    min={2}
                    max={5}
                    value={splitCount}
                    onChange={(e) => setSplitCount(Math.min(5, Math.max(2, Number(e.target.value))))}
                    aria-label="Split transfers count input"
                  />
                </div>
              </div>
              <Slider
                value={[splitCount]}
                min={2}
                max={5}
                step={1}
                onValueChange={([val]) => setSplitCount(val)}
                aria-label="Split count slider"
              />
              <div className="flex justify-between text-xs font-mono text-muted">
                <span>2 transfers (50% each)</span>
                <span>3 transfers (33% each)</span>
                <span>5 transfers (20% each)</span>
              </div>
            </div>
          )}

          {activeAttackType === 'timing_delay' && (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div>
                  <span className="font-semibold text-xs text-text">Timing delay</span>
                  <p className="text-xs text-muted">Elapsed duration between micro-transfers to age feature caches</p>
                </div>
                <div className="w-24">
                  <Input
                    type="number"
                    min={1}
                    max={12}
                    value={delayHours}
                    onChange={(e) => setDelayHours(Math.min(12, Math.max(1, Number(e.target.value))))}
                    aria-label="Timing delay hours input"
                  />
                </div>
              </div>
              <Slider
                value={[delayHours]}
                min={1}
                max={12}
                step={1}
                onValueChange={([val]) => setDelayHours(val)}
                aria-label="Timing delay slider"
              />
              <div className="flex justify-between text-xs font-mono text-muted">
                <span>1 hour (within window)</span>
                <span>4 hours</span>
                <span>12 hours (beyond cache)</span>
              </div>
            </div>
          )}

          {activeAttackType === 'partial_drain' && (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div>
                  <span className="font-semibold text-xs text-text">Drain fraction</span>
                  <p className="text-xs text-muted">Percentage of available balance liquidated in the attack</p>
                </div>
                <div className="w-24">
                  <Input
                    type="number"
                    min={10}
                    max={95}
                    step={5}
                    value={drainPct}
                    onChange={(e) => setDrainPct(Math.min(95, Math.max(10, Number(e.target.value))))}
                    aria-label="Drain percentage input"
                  />
                </div>
              </div>
              <Slider
                value={[drainPct]}
                min={10}
                max={95}
                step={5}
                onValueChange={([val]) => setDrainPct(val)}
                aria-label="Drain fraction slider"
              />
              <div className="flex justify-between text-xs font-mono text-muted">
                <span>10% partial drain</span>
                <span>50% standard</span>
                <span>95% full balance liquidation</span>
              </div>
            </div>
          )}

          {activeAttackType === 'mule_hop' && (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div>
                  <span className="font-semibold text-xs text-text">Mule hops</span>
                  <p className="text-xs text-muted">Intermediate accounts in layering transit before cash-out</p>
                </div>
                <div className="w-20">
                  <Input
                    type="number"
                    min={1}
                    max={3}
                    value={hopCount}
                    onChange={(e) => setHopCount(Math.min(3, Math.max(1, Number(e.target.value))))}
                    aria-label="Mule hops count input"
                  />
                </div>
              </div>
              <Slider
                value={[hopCount]}
                min={1}
                max={3}
                step={1}
                onValueChange={([val]) => setHopCount(val)}
                aria-label="Mule hops slider"
              />
              <div className="flex justify-between text-xs font-mono text-muted">
                <span>1 intermediary hop</span>
                <span>2 hops</span>
                <span>3 multi-tier ring hops</span>
              </div>
            </div>
          )}
        </div>
      </section>

      {/* 3. Simulation Outcome & Interactive Replay */}
      {simulationResult ? (
        <section className="space-y-5 animate-fade-in-row">
          <div className="flex items-center justify-between">
            <h2 className="text-xs font-semibold text-muted">
              3. Adversarial Simulation Metrics
            </h2>
            <div className="flex items-center gap-2">
              <span className="text-xs font-mono text-muted">
                Perturbation:
              </span>
              <Badge variant="neutral" className="text-xs font-mono">
                {simulationResult.attack_type.toUpperCase()}
              </Badge>
            </div>
          </div>

          {/* Outcome Metric Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-4 rounded-xl border border-border bg-surface space-y-1 shadow-card">
              <div className="text-xs font-semibold text-muted">
                Unperturbed base score
              </div>
              <div className="text-2xl font-mono tabular-nums font-semibold text-text">
                {simulationResult.base_score.toFixed(3)}
              </div>
              <div className="pt-1">
                <Badge
                  variant={
                    simulationResult.base_band.toLowerCase() as 'low' | 'medium' | 'high'
                  }
                >
                  {simulationResult.base_band} ({simulationResult.base_action})
                </Badge>
              </div>
            </div>

            <div className="p-4 rounded-xl border border-border bg-surface space-y-1 shadow-card">
              <div className="text-xs font-semibold text-muted">
                Perturbed peak score
              </div>
              <div className="text-2xl font-mono tabular-nums font-semibold text-text">
                {simulationResult.peak_attack_score.toFixed(3)}
              </div>
              <div className="text-xs font-mono tabular-nums text-muted pt-1">
                Mean across splits: {simulationResult.mean_attack_score.toFixed(3)}
              </div>
            </div>

            <div
              className={`p-4 rounded-xl border space-y-1 shadow-card ${
                simulationResult.verdict === 'BLOCKED'
                  ? 'border-accent/40 bg-accent/10'
                  : simulationResult.verdict === 'STEP_UP_CHALLENGED'
                  ? 'border-warn/40 bg-warn/10'
                  : 'border-danger/40 bg-danger/10'
              }`}
            >
              <div className="text-xs font-semibold text-muted">
                Attack verdict
              </div>
              <div className="text-xl font-mono font-semibold flex items-center gap-2 text-text">
                {simulationResult.verdict === 'BLOCKED' ? (
                  <ShieldCheck className="w-5 h-5 text-accent" />
                ) : (
                  <ShieldAlert className="w-5 h-5 text-danger" />
                )}
                <span>{simulationResult.verdict}</span>
              </div>
              <p className="text-xs leading-relaxed text-text pt-1">
                {simulationResult.verdict_summary}
              </p>
            </div>
          </div>

          {/* SVG Transit Path Diagram */}
          <div className="p-5 rounded-xl border border-border bg-surface space-y-3 shadow-card">
            <div className="flex items-center justify-between border-b border-border pb-2.5">
              <div className="flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-accent" />
                <h3 className="font-semibold text-xs text-text">
                  Transit Path Diagram &amp; Flow Vector
                </h3>
              </div>
              <span className="font-mono text-xs text-muted">
                Step-by-step telemetry
              </span>
            </div>

            <div className="p-4 rounded-xl border border-border bg-surface-subtle overflow-x-auto">
              <div className="min-w-[620px] py-2">
                <svg viewBox="0 0 700 100" className="w-full h-24 overflow-visible">
                  <defs>
                    <linearGradient id="flowGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                      <stop offset="0%" stopColor="#0E9F77" />
                      <stop offset="50%" stopColor="#D97706" />
                      <stop offset="100%" stopColor="#B91C1C" />
                    </linearGradient>
                  </defs>

                  <line
                    x1="60"
                    y1="50"
                    x2="640"
                    y2="50"
                    stroke="url(#flowGrad)"
                    strokeWidth="3"
                    strokeDasharray="8 6"
                    className="opacity-80"
                  />

                  {/* Node 1 */}
                  <g transform="translate(60, 50)">
                    <circle r="22" className="fill-surface stroke-accent" strokeWidth="2.5" />
                    <text textAnchor="middle" dy="4" className="text-xs font-mono font-semibold fill-text">
                      ORIG
                    </text>
                    <text textAnchor="middle" dy="38" className="text-xs font-medium fill-muted">
                      Payer Wallet
                    </text>
                  </g>

                  {/* Node 2 */}
                  <g transform="translate(250, 50)">
                    <rect
                      x="-20"
                      y="-20"
                      width="40"
                      height="40"
                      rx="8"
                      className={`fill-surface ${
                        activeStepIndex >= 0 ? 'stroke-warn stroke-[2.5px]' : 'stroke-border'
                      }`}
                    />
                    <text textAnchor="middle" dy="4" className="text-xs font-mono font-semibold fill-text">
                      HOP-1
                    </text>
                    <text textAnchor="middle" dy="38" className="text-xs font-medium fill-muted">
                      Intermediate Mule
                    </text>
                  </g>

                  {/* Node 3 */}
                  <g transform="translate(450, 50)">
                    <polygon
                      points="0,-22 22,0 0,22 -22,0"
                      className={`fill-surface ${
                        activeStepIndex >= 1 ? 'stroke-warn stroke-[2.5px]' : 'stroke-border'
                      }`}
                    />
                    <text textAnchor="middle" dy="4" className="text-xs font-mono font-semibold fill-text">
                      HUB
                    </text>
                    <text textAnchor="middle" dy="38" className="text-xs font-medium fill-muted">
                      Mule Aggregator
                    </text>
                  </g>

                  {/* Node 4 */}
                  <g transform="translate(640, 50)">
                    <circle r="22" className="fill-surface stroke-danger" strokeWidth="2.5" />
                    <text textAnchor="middle" dy="4" className="text-xs font-mono font-semibold fill-text">
                      EXIT
                    </text>
                    <text textAnchor="middle" dy="38" className="text-xs font-medium fill-muted">
                      Cash-out Exit
                    </text>
                  </g>
                </svg>
              </div>
            </div>
          </div>

          {/* Replay Controls & Table */}
          {simulationResult.transfers && simulationResult.transfers.length > 0 && (
            <div className="p-5 rounded-xl border border-border bg-surface space-y-4 shadow-card">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-border pb-3">
                <div>
                  <h3 className="font-semibold text-xs text-text">
                    Transfer chain step-through replay
                  </h3>
                  <p className="text-xs text-muted">
                    Step forward or replay across each perturbed micro-transfer.
                  </p>
                </div>

                <div className="flex items-center gap-1.5 self-start sm:self-auto">
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => {
                      setIsPlaying(false);
                      setActiveStepIndex(0);
                    }}
                    title="Reset to Step 1"
                  >
                    <RotateCcw className="w-3.5 h-3.5" />
                  </Button>
                  <Button
                    variant="secondary"
                    size="sm"
                    disabled={activeStepIndex <= 0}
                    onClick={() => {
                      setIsPlaying(false);
                      setActiveStepIndex((p) => Math.max(0, p - 1));
                    }}
                    title="Previous Step"
                  >
                    <SkipBack className="w-3.5 h-3.5" />
                  </Button>
                  <Button
                    variant="primary"
                    size="sm"
                    onClick={() => setIsPlaying((p) => !p)}
                    title={isPlaying ? 'Pause replay' : 'Auto-play step replay'}
                  >
                    {isPlaying ? <Pause className="w-3.5 h-3.5 mr-1" /> : <Play className="w-3.5 h-3.5 mr-1" />}
                    <span>{isPlaying ? 'Pause' : 'Play'}</span>
                  </Button>
                  <Button
                    variant="secondary"
                    size="sm"
                    disabled={activeStepIndex >= simulationResult.transfers.length - 1}
                    onClick={() => {
                      setIsPlaying(false);
                      setActiveStepIndex((p) =>
                        Math.min(simulationResult.transfers.length - 1, p + 1)
                      );
                    }}
                    title="Next Step"
                  >
                    <SkipForward className="w-3.5 h-3.5" />
                  </Button>
                </div>
              </div>

              {/* Active Step Highlight Card */}
              {activeTransfer && (
                <div className="p-4 rounded-xl border border-border bg-surface-subtle flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 font-mono text-xs">
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <span className="font-semibold text-text text-sm">
                        Step {activeTransfer.transfer_index} of {simulationResult.transfers.length}
                      </span>
                      <Badge
                        variant={
                          activeTransfer.band.toLowerCase() as 'low' | 'medium' | 'high'
                        }
                      >
                        {activeTransfer.band}
                      </Badge>
                    </div>
                    <div className="text-muted font-sans">
                      Signal: <span className="text-text font-semibold">{activeTransfer.primary_signal}</span>
                    </div>
                  </div>

                  <div className="text-right sm:self-center">
                    <div className="text-base font-semibold text-text tabular-nums">
                      ₹{activeTransfer.amount.toLocaleString('en-IN')}
                    </div>
                    <div className="text-muted text-xs">
                      Fraction: {(activeTransfer.amount_fraction * 100).toFixed(0)}% of total
                    </div>
                  </div>
                </div>
              )}

              {/* Transfer Table */}
              <div className="border border-border rounded-xl overflow-x-auto bg-surface">
                <table className="w-full text-left text-xs border-collapse font-sans min-w-[700px]">
                  <thead className="bg-surface-subtle border-b border-border text-xs font-semibold text-muted normal-case font-sans">
                    <tr className="h-10">
                      <th className="py-2.5 px-3.5">Tx #</th>
                      <th className="py-2.5 px-3.5 text-right">Amount</th>
                      <th className="py-2.5 px-3.5 text-right">Fraction</th>
                      <th className="py-2.5 px-3.5">Risk score</th>
                      <th className="py-2.5 px-3.5">Risk band</th>
                      <th className="py-2.5 px-3.5">Signal triggered</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border font-mono">
                    {simulationResult.transfers.map((tx, idx) => {
                      const isActive = activeStepIndex === idx;
                      return (
                        <tr
                          key={tx.transfer_index}
                          onClick={() => {
                            setIsPlaying(false);
                            setActiveStepIndex(idx);
                          }}
                          className={`h-[44px] cursor-pointer transition-colors duration-120 ${
                            isActive
                              ? 'bg-surface-subtle ring-1 ring-inset ring-accent/30'
                              : 'hover:bg-surface-subtle'
                          }`}
                        >
                          <td className="py-2.5 px-3.5 font-semibold text-text">
                            #{tx.transfer_index}
                          </td>
                          <td className="py-2.5 px-3.5 text-right font-semibold text-text tabular-nums">
                            ₹{tx.amount.toLocaleString('en-IN')}
                          </td>
                          <td className="py-2.5 px-3.5 text-right text-muted tabular-nums">
                            {(tx.amount_fraction * 100).toFixed(0)}%
                          </td>
                          <td className="py-2.5 px-3.5 font-semibold">
                            <InlineRiskBar score={tx.risk_score} />
                          </td>
                          <td className="py-2.5 px-3.5">
                            <Badge variant={tx.band.toLowerCase() as 'low' | 'medium' | 'high'}>
                              {tx.band}
                            </Badge>
                          </td>
                          <td className="py-2.5 px-3.5 font-sans text-xs text-muted">
                            {tx.primary_signal}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </section>
      ) : (
        <div className="p-12 text-center border border-dashed border-border rounded-xl bg-surface space-y-2">
          <FlaskConical className="w-8 h-8 text-muted mx-auto opacity-40" />
          <h3 className="font-semibold text-sm text-text">
            No simulation executed
          </h3>
          <p className="text-xs text-muted max-w-sm mx-auto">
            Choose a base fraud pattern and perturbation above to simulate evasion efficacy.
          </p>
          <Button variant="primary" size="sm" onClick={handleSimulate} disabled={isSimulating || presetsLoading} className="mt-2">
            Execute simulation
          </Button>
        </div>
      )}

      {/* 4. Session History Log */}
      {historyLog.length > 0 && (
        <section className="space-y-3 pt-2">
          <div className="flex items-center gap-2 text-xs font-semibold text-muted">
            <History className="w-4 h-4" />
            <span>Simulation Session Audit Log</span>
          </div>

          <div className="border border-border rounded-xl overflow-x-auto font-mono text-xs bg-surface shadow-card">
            <table className="w-full text-left border-collapse min-w-[640px]">
              <thead className="bg-surface-subtle border-b border-border text-xs font-semibold text-muted normal-case font-sans">
                <tr className="h-10">
                  <th className="py-2.5 px-3.5">Sim ID</th>
                  <th className="py-2.5 px-3.5">Time</th>
                  <th className="py-2.5 px-3.5">Perturbation type</th>
                  <th className="py-2.5 px-3.5 text-right">Base score</th>
                  <th className="py-2.5 px-3.5 text-right">Peak score</th>
                  <th className="py-2.5 px-3.5">Verdict</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {historyLog.map((log) => (
                  <tr key={log.id} className="h-[44px] hover:bg-surface-subtle transition-colors">
                    <td className="py-2.5 px-3.5 font-semibold text-text">{log.id}</td>
                    <td className="py-2.5 px-3.5 text-muted">{log.timestamp}</td>
                    <td className="py-2.5 px-3.5 font-sans text-xs text-text">{log.attack}</td>
                    <td className="py-2.5 px-3.5 text-right tabular-nums">{log.baseScore.toFixed(3)}</td>
                    <td className="py-2.5 px-3.5 text-right tabular-nums font-semibold text-text">{log.peakScore.toFixed(3)}</td>
                    <td className="py-2.5 px-3.5">
                      <span className={`font-semibold ${log.verdict === 'BLOCKED' ? 'text-accent' : 'text-danger'}`}>
                        {log.verdict}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  );
};
