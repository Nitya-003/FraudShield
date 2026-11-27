import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { motion } from 'framer-motion';
import { Button } from '../components/ui/Button';
import { Input } from '../components/ui/Input';
import { Select } from '../components/ui/Select';
import { Slider } from '../components/ui/Slider';
import { Checkbox } from '../components/ui/Checkbox';
import { PageHeader } from '../components/layout/PageHeader';
import { useToast } from '../components/ui/Toast';
import { AnimatedNumber } from '../hooks/useCountUp';
import { scoreTransaction, previewStepUpMessage } from '../api/client';
import type { ScoreResponse, StepUpMessageData } from '../types/api';
import {
  ShieldCheck,
  Warning,
  WarningCircle,
  ArrowCounterClockwise,
  Sparkle,
  DeviceMobile,
  Clock,
  ArrowUp,
  ArrowDown,
  CaretDown,
} from '@phosphor-icons/react';

interface Preset {
  id: string;
  name: string;
  badge: 'Low' | 'Medium' | 'High';
  description: string;
  channel: string;
  amount: number;
  hour: number;
  payerId: string;
  payeeId: string;
  isFirstTimePayee: boolean;
  isCollectRequest: boolean;
  isNewDevice: boolean;
  payeeVpaAgeDays?: number;
}

const PRESETS: Preset[] = [
  {
    id: 'normal',
    name: 'Routine Payment',
    badge: 'Low',
    description: 'Daytime peer transfer between established handles with typical ticket size.',
    channel: 'upi',
    amount: 3500,
    hour: 14,
    payerId: 'user_4102@okhdfcbank',
    payeeId: 'user_9812@paytm',
    isFirstTimePayee: false,
    isCollectRequest: false,
    isNewDevice: false,
  },
  {
    id: 'suspicious',
    name: 'High-Value Nocturnal Split',
    badge: 'Medium',
    description: 'Transfer at 02:00 IST to a recently linked beneficiary account.',
    channel: 'upi',
    amount: 20000,
    hour: 2,
    payerId: 'user_7719@okhdfcbank',
    payeeId: 'merchant_rapid@paytm',
    isFirstTimePayee: false,
    isCollectRequest: false,
    isNewDevice: false,
    payeeVpaAgeDays: 2.0,
  },
  {
    id: 'scam',
    name: 'UPI Collect Scam',
    badge: 'High',
    description: 'Inbound high-velocity collect request from an unverified handle.',
    channel: 'upi',
    amount: 145000,
    hour: 15,
    payerId: 'user_8821@okhdfcbank',
    payeeId: 'cashback_agent@paytm',
    isFirstTimePayee: true,
    isCollectRequest: true,
    isNewDevice: true,
  },
];

export const LiveCheckPage: React.FC = () => {
  const { toast } = useToast();

  // Form fields
  const [channel, setChannel] = useState('upi');
  const [amount, setAmount] = useState(145000);
  const [hour, setHour] = useState(15);
  const [payerId, setPayerId] = useState('user_8821@okhdfcbank');
  const [payeeId, setPayeeId] = useState('cashback_agent@upi');
  const [isFirstTimePayee, setIsFirstTimePayee] = useState(true);
  const [isCollectRequest, setIsCollectRequest] = useState(true);
  const [isNewDevice, setIsNewDevice] = useState(true);
  const [payeeVpaAgeDays, setPayeeVpaAgeDays] = useState<number | undefined>(undefined);

  // Scoring response state
  const [scoreResult, setScoreResult] = useState<ScoreResponse | null>(null);
  const [stepUpData, setStepUpData] = useState<StepUpMessageData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Core scoring executor
  const scoreWithParams = useCallback(
    async (params: {
      channel: string;
      amount: number;
      hour: number;
      payerId: string;
      payeeId: string;
      isFirstTimePayee: boolean;
      isCollectRequest: boolean;
      isNewDevice: boolean;
      payeeVpaAgeDays?: number;
    }) => {
      setIsLoading(true);
      setErrorMessage(null);
      try {
        let payload: Record<string, unknown> = {};

        if (params.channel === 'upi') {
          payload = {
            amount: params.amount,
            hour: Number(params.hour),
            txn_type: params.isCollectRequest ? 'COLLECT_REQUEST' : 'PAY',
            is_first_time_payee: params.isFirstTimePayee ? 1 : 0,
            is_new_device: params.isNewDevice ? 1 : 0,
            payer_vpa: params.payerId,
            payee_vpa: params.payeeId,
            payee_vpa_age_days:
              params.payeeVpaAgeDays !== undefined
                ? params.payeeVpaAgeDays
                : params.isFirstTimePayee
                ? 2.0
                : 365.0,
          };
        } else if (params.channel === 'transfer_wallet') {
          payload = {
            type: 'TRANSFER',
            amount: params.amount,
            oldbalanceOrg: params.amount * 1.05,
            newbalanceOrig: params.amount * 0.05,
            oldbalanceDest: 0,
            newbalanceDest: 0,
            nameOrig: params.payerId,
            nameDest: params.payeeId,
          };
        } else if (params.channel === 'card') {
          payload = {
            Amount: params.amount,
            Hour: Number(params.hour),
            Use_Chip: 1,
            V1: -1.2,
            V2: 2.3,
            V3: -0.8,
            V4: 1.5,
          };
        } else {
          payload = {
            payment_type: 'AB',
            proposed_credit_limit: params.amount,
            income: 0.6,
            customer_age: 35,
            has_other_cards: 0,
            email_is_free: 1,
          };
        }

        const res = await scoreTransaction({ channel: params.channel, payload });
        setScoreResult(res);

        if (res.band !== 'Low' || params.isCollectRequest || params.channel === 'upi') {
          try {
            const stepUpRes = await previewStepUpMessage({
              channel: params.channel,
              payload: {
                amount: params.amount,
                payer_id: params.payerId,
                payee_id: params.payeeId,
              },
            });
            setStepUpData(stepUpRes.messaging);
          } catch {
            setStepUpData(null);
          }
        } else {
          setStepUpData(null);
        }
      } catch (err: unknown) {
        const e = err as { response?: { data?: { detail?: string } }; message?: string };
        setErrorMessage(e?.response?.data?.detail || e?.message || 'Transaction scoring request failed.');
      } finally {
        setIsLoading(false);
      }
    },
    []
  );

  const handleScore = useCallback(() => {
    scoreWithParams({
      channel,
      amount,
      hour,
      payerId,
      payeeId,
      isFirstTimePayee,
      isCollectRequest,
      isNewDevice,
      payeeVpaAgeDays,
    });
  }, [scoreWithParams, channel, amount, hour, payerId, payeeId, isFirstTimePayee, isCollectRequest, isNewDevice, payeeVpaAgeDays]);

  const applyPreset = (preset: Preset) => {
    setChannel(preset.channel);
    setAmount(preset.amount);
    setHour(preset.hour);
    setPayerId(preset.payerId);
    setPayeeId(preset.payeeId);
    setIsFirstTimePayee(preset.isFirstTimePayee);
    setIsCollectRequest(preset.isCollectRequest);
    setIsNewDevice(preset.isNewDevice);
    setPayeeVpaAgeDays(preset.payeeVpaAgeDays);
    toast({
      title: 'Preset Applied',
      description: `Loaded "${preset.name}" scenario.`,
      variant: 'info',
    });
    scoreWithParams(preset);
  };

  useEffect(() => {
    handleScore();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const score = scoreResult?.risk_score ?? 0;
  const lowThresh = scoreResult?.thresholds_applied?.low_threshold ?? 0.15;
  const highThresh = scoreResult?.thresholds_applied?.high_threshold ?? 0.87;

  // Semicircle Arc geometry: Center: 100, 100; Radius: 75
  const getArcPoint = useCallback((val: number) => {
    const clamped = Math.min(Math.max(val, 0), 1);
    const angle = Math.PI * (1 - clamped);
    return {
      x: Number((100 + 75 * Math.cos(angle)).toFixed(2)),
      y: Number((100 - 75 * Math.sin(angle)).toFixed(2)),
    };
  }, []);

  const pLow = useMemo(() => getArcPoint(lowThresh), [getArcPoint, lowThresh]);
  const pHigh = useMemo(() => getArcPoint(highThresh), [getArcPoint, highThresh]);

  // Needle angle: -90deg at 0.0 to +90deg at 1.0
  const needleDeg = useMemo(() => {
    const clamped = Math.min(Math.max(score, 0), 1);
    return clamped * 180 - 90;
  }, [score]);

  const needleColor = useMemo(() => {
    if (score >= highThresh) return '#B91C1C';
    if (score >= lowThresh) return '#D97706';
    return '#0E9F77';
  }, [score, lowThresh, highThresh]);

  const top3Reasons = useMemo(() => {
    const list = scoreResult?.top_reasons || [];
    return list.slice(0, 3);
  }, [scoreResult?.top_reasons]);

  return (
    <div className="space-y-6 pb-12 font-sans">
      <PageHeader
        title="Live Check"
        purpose="Pre-authorization risk scoring and factor attribution across payment rails."
      />

      {/* Error state with Retry */}
      {errorMessage && (
        <div className="p-3.5 rounded-xl border border-danger/30 bg-danger/10 text-xs text-danger flex items-center justify-between gap-3">
          <span>{errorMessage}</span>
          <Button variant="secondary" size="sm" onClick={handleScore}>
            Retry
          </Button>
        </div>
      )}

      {/* Compact Preset Selector */}
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs font-semibold text-muted mr-1">Presets:</span>
        {PRESETS.map((p) => {
          const isActive = channel === p.channel && amount === p.amount && hour === p.hour;
          return (
            <button
              key={p.id}
              type="button"
              onClick={() => applyPreset(p)}
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full border text-xs font-medium transition-colors cursor-pointer focus:outline-none focus-visible:ring-2 focus-visible:ring-accent ${
                isActive
                  ? 'border-accent bg-accent/10 text-text font-semibold'
                  : 'border-border bg-surface hover:bg-surface-subtle text-muted hover:text-text'
              }`}
            >
              <span
                className={`w-2 h-2 rounded-full ${
                  p.badge === 'Low'
                    ? 'bg-accent'
                    : p.badge === 'Medium'
                    ? 'bg-warn'
                    : 'bg-danger'
                }`}
              />
              <span>{p.name}</span>
            </button>
          );
        })}
      </div>

      {/* Two Balanced Columns Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Column (7 cols): Input Form with Grouped Fields and Editable Sliders */}
        <div className="lg:col-span-7 space-y-4">
          <div className="p-5 rounded-xl border border-border bg-surface space-y-4 shadow-card">
            {/* Group 1: Rail & Amount */}
            <div className="space-y-3">
              <div className="text-xs font-semibold text-muted">
                Transaction parameters
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5">
                <Select
                  label="Channel"
                  value={channel}
                  onChange={(e) => setChannel(e.target.value)}
                  options={[
                    { value: 'upi', label: 'UPI Instant' },
                    { value: 'card', label: 'Card Payment' },
                    { value: 'transfer_wallet', label: 'Wallet Transfer' },
                    { value: 'account', label: 'Account Opening' },
                  ]}
                />
                <Input
                  label={`Amount (${channel === 'card' ? '€' : channel === 'account' ? '£' : '₹'})`}
                  type="number"
                  min={1}
                  max={500000}
                  value={amount}
                  onChange={(e) => setAmount(Math.max(1, Number(e.target.value)))}
                />
              </div>

              {/* Amount Slider with Thicker Track and Visible Thumb */}
              <div className="pt-1 space-y-2">
                <div className="flex justify-between text-xs text-muted">
                  <span>What-if amount slider</span>
                  <span className="font-mono tabular-nums font-semibold text-text">
                    {channel === 'card' ? `€${amount.toLocaleString()}` : channel === 'account' ? `£${amount.toLocaleString()}` : `₹${amount.toLocaleString('en-IN')}`}
                  </span>
                </div>
                <Slider
                  value={[amount]}
                  min={500}
                  max={500000}
                  step={500}
                  onValueChange={([val]) => setAmount(val)}
                  aria-label="What-if amount slider"
                />
              </div>
            </div>

            {/* Group 2: Counterparties */}
            <div className="space-y-3 pt-2 border-t border-border/50">
              <div className="text-xs font-semibold text-muted">
                Counterparties
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5">
                <Input
                  label="Sender / Payer ID"
                  value={payerId}
                  onChange={(e) => setPayerId(e.target.value)}
                />
                <Input
                  label="Beneficiary / Payee ID"
                  value={payeeId}
                  onChange={(e) => setPayeeId(e.target.value)}
                />
              </div>
            </div>

            {/* Group 3: Temporal Context */}
            <div className="space-y-3 pt-2 border-t border-border/50">
              <div className="flex items-center justify-between">
                <div className="text-xs font-semibold text-muted">
                  Temporal context
                </div>
                <div className="w-24">
                  <Input
                    label="Hour"
                    type="number"
                    min={0}
                    max={23}
                    value={hour}
                    onChange={(e) => setHour(Math.min(23, Math.max(0, Number(e.target.value))))}
                    aria-label="Hour of day numeric input"
                  />
                </div>
              </div>

              <div className="space-y-2">
                <div className="flex justify-between text-xs text-muted">
                  <span>Hour of day (0–23)</span>
                  <span className="font-mono tabular-nums font-semibold text-text">
                    {String(hour).padStart(2, '0')}:00 hrs
                  </span>
                </div>
                <Slider
                  value={[hour]}
                  min={0}
                  max={23}
                  step={1}
                  onValueChange={([val]) => setHour(val)}
                  aria-label="Hour of day slider"
                />
                <div className="flex justify-between text-xs text-muted font-mono pt-0.5">
                  <span>00:00 (Night)</span>
                  <span>12:00 (Noon)</span>
                  <span>23:00 (Night)</span>
                </div>
              </div>
            </div>

            {/* Group 4: Risk Indicators */}
            <div className="space-y-3 pt-2 border-t border-border/50">
              <div className="text-xs font-semibold text-muted">
                Behavioral flags
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
                <div className="flex items-center p-2.5 rounded-lg border border-border bg-surface-subtle">
                  <Checkbox
                    id="flag-first-time-payee"
                    label="First-time payee"
                    checked={isFirstTimePayee}
                    onCheckedChange={(checked) => {
                      setIsFirstTimePayee(Boolean(checked));
                      setPayeeVpaAgeDays(undefined);
                    }}
                  />
                </div>

                <div className="flex items-center p-2.5 rounded-lg border border-border bg-surface-subtle">
                  <Checkbox
                    id="flag-collect-request"
                    label="Collect request"
                    checked={isCollectRequest}
                    onCheckedChange={(checked) => setIsCollectRequest(Boolean(checked))}
                  />
                </div>

                <div className="flex items-center p-2.5 rounded-lg border border-border bg-surface-subtle">
                  <Checkbox
                    id="flag-new-device"
                    label="New device flag"
                    checked={isNewDevice}
                    onCheckedChange={(checked) => setIsNewDevice(Boolean(checked))}
                  />
                </div>
              </div>
            </div>

            {/* Form Action Controls */}
            <div className="flex justify-end gap-2.5 pt-3 border-t border-border/50">
              <Button
                variant="secondary"
                size="sm"
                onClick={() => applyPreset(PRESETS[0])}
              >
                <ArrowCounterClockwise className="w-3.5 h-3.5 mr-1" />
                <span>Reset to nominal</span>
              </Button>
              <Button
                variant="primary"
                size="sm"
                onClick={handleScore}
                disabled={isLoading}
              >
                <Sparkle className="w-3.5 h-3.5 mr-1" />
                <span>{isLoading ? 'Evaluating...' : 'Score transaction'}</span>
              </Button>
            </div>
          </div>
        </div>

        {/* Right Column (5 cols): Sticky Card that Fits in One Screen */}
        <div className="lg:col-span-5 lg:sticky lg:top-20 space-y-4">
          <div className="p-5 rounded-xl border border-border bg-surface space-y-4 shadow-card">
            {/* Header: Latency & Title */}
            <div className="flex items-center justify-between border-b border-border/60 pb-2.5">
              <span className="text-xs font-semibold text-text">
                Risk verdict
              </span>
              <span className="font-mono text-xs text-muted flex items-center gap-1 tabular-nums">
                <Clock className="w-3.5 h-3.5" />
                <span>{scoreResult?.latency_ms !== undefined ? `${scoreResult.latency_ms} ms` : '—'}</span>
              </span>
            </div>

            {/* Clean SVG Semicircle Gauge with Tick Labels at Ends Only & Centred Score */}
            <div className="flex flex-col items-center justify-center pt-1">
              <div className="relative w-64 h-34 flex items-end justify-center">
                <svg viewBox="0 0 200 120" className="w-full h-full overflow-visible">
                  {/* Background Arc Track */}
                  <path
                    d="M 25 100 A 75 75 0 0 1 175 100"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="10"
                    className="text-border-strong/30"
                  />

                  {/* Zone 1: Low Risk (0.0 to lowThresh) */}
                  <path
                    d={`M 25 100 A 75 75 0 0 1 ${pLow.x} ${pLow.y}`}
                    fill="none"
                    stroke="#0E9F77"
                    strokeWidth="10"
                    strokeLinecap="round"
                    className="opacity-90"
                  />

                  {/* Zone 2: Medium Risk (lowThresh to highThresh) */}
                  <path
                    d={`M ${pLow.x} ${pLow.y} A 75 75 0 0 1 ${pHigh.x} ${pHigh.y}`}
                    fill="none"
                    stroke="#D97706"
                    strokeWidth="10"
                    className="opacity-90"
                  />

                  {/* Zone 3: High Risk (highThresh to 1.0) */}
                  <path
                    d={`M ${pHigh.x} ${pHigh.y} A 75 75 0 0 1 175 100`}
                    fill="none"
                    stroke="#B91C1C"
                    strokeWidth="10"
                    strokeLinecap="round"
                    className="opacity-90"
                  />

                  {/* End Ticks Outside the Arc Only */}
                  <text x="18" y="116" textAnchor="middle" className="text-xs font-mono fill-muted">
                    0
                  </text>
                  <text x="182" y="116" textAnchor="middle" className="text-xs font-mono fill-muted">
                    1.0
                  </text>

                  {/* Needle Spring Animation Under 400ms */}
                  <motion.g
                    initial={false}
                    animate={{ rotate: needleDeg }}
                    transition={{
                      type: 'spring',
                      stiffness: 280,
                      damping: 22,
                      mass: 0.5,
                    }}
                    style={{ transformOrigin: '100px 100px' }}
                  >
                    <line
                      x1="100"
                      y1="100"
                      x2="100"
                      y2="36"
                      stroke={needleColor}
                      strokeWidth="3.5"
                      strokeLinecap="round"
                    />
                    <circle cx="100" cy="100" r="4.5" className="fill-text" />
                  </motion.g>
                </svg>
              </div>

              {/* Score Centred Below the Arc with No Overlap */}
              <div className="text-center mt-1 mb-2">
                <div className="text-2xl font-semibold font-sans tracking-normal tabular-nums text-text">
                  <AnimatedNumber value={score} format="decimal" decimals={3} duration={350} />
                </div>
                <div className="text-xs text-muted">calibrated risk score</div>
              </div>

              {/* Threshold Zones Breakdown */}
              <div className="w-full flex items-center justify-between text-xs text-muted px-1 pt-2 border-t border-border/50">
                <span className="flex items-center gap-1 text-accent font-medium text-xs">
                  <span className="w-1.5 h-1.5 rounded-full bg-accent" />
                  &lt; {lowThresh.toFixed(2)} Approve
                </span>
                <span className="flex items-center gap-1 text-warn font-medium text-xs">
                  <span className="w-1.5 h-1.5 rounded-full bg-warn" />
                  {lowThresh.toFixed(2)}–{highThresh.toFixed(2)} Step-up
                </span>
                <span className="flex items-center gap-1 text-danger font-medium text-xs">
                  <span className="w-1.5 h-1.5 rounded-full bg-danger" />
                  &ge; {highThresh.toFixed(2)} Decline
                </span>
              </div>
            </div>

            {/* One Verdict Line with Icon and Sentence */}
            <div
              className={`p-3 rounded-lg border text-xs flex items-center gap-2.5 font-medium ${
                scoreResult?.band === 'Low'
                  ? 'border-accent/40 bg-accent/10 text-accent'
                  : scoreResult?.band === 'Medium'
                  ? 'border-warn/40 bg-warn/10 text-warn'
                  : 'border-danger/40 bg-danger/10 text-danger'
              }`}
            >
              {scoreResult?.band === 'Low' ? (
                <ShieldCheck className="w-4 h-4 shrink-0 text-accent" weight="bold" />
              ) : scoreResult?.band === 'Medium' ? (
                <Warning className="w-4 h-4 shrink-0 text-warn" weight="bold" />
              ) : (
                <WarningCircle className="w-4 h-4 shrink-0 text-danger" weight="bold" />
              )}
              <span className="leading-snug">
                {scoreResult?.band === 'Low'
                  ? 'Approve: Risk score falls within nominal behavioral thresholds.'
                  : scoreResult?.band === 'Medium'
                  ? 'Step-up: Elevated risk detected; initiate two-factor verification.'
                  : 'Decline: High risk threshold breached; block authorization.'}
              </span>
            </div>

            {/* Top 3 Factor Drivers */}
            <div className="space-y-1.5 pt-1">
              <div className="text-xs font-semibold text-muted">
                Top factor drivers
              </div>
              <div className="space-y-1.5 text-xs">
                {top3Reasons.length > 0 ? (
                  top3Reasons.map((reason, idx) => {
                    const isDown =
                      reason.toLowerCase().includes('down') ||
                      reason.toLowerCase().includes('lower') ||
                      reason.toLowerCase().includes('decrease') ||
                      reason.toLowerCase().includes('reduced') ||
                      reason.toLowerCase().includes('nominal');
                    const isUp = !isDown;
                    return (
                      <div
                        key={idx}
                        className="p-2 rounded-lg bg-surface-subtle border border-border flex items-start gap-2 text-xs text-text"
                      >
                        <span
                          className={`w-1 self-stretch rounded-full shrink-0 ${
                            isUp ? 'bg-danger' : 'bg-accent'
                          }`}
                        />
                        {isUp ? (
                          <ArrowUp className="w-3.5 h-3.5 text-danger shrink-0 mt-0.5" weight="bold" />
                        ) : (
                          <ArrowDown className="w-3.5 h-3.5 text-accent shrink-0 mt-0.5" weight="bold" />
                        )}
                        <span className="leading-snug flex-1 break-words">
                          {reason}
                        </span>
                      </div>
                    );
                  })
                ) : (
                  <div className="text-muted text-xs p-2">No factor drivers triggered.</div>
                )}
              </div>
            </div>

            {/* Collapsible Customer Message Section */}
            {stepUpData && (scoreResult?.band === 'Medium' || isCollectRequest) && (
              <details open className="group border border-border rounded-lg bg-surface-subtle p-3 open:pb-3.5 transition-colors">
                <summary className="flex items-center justify-between text-xs font-semibold text-text cursor-pointer select-none">
                  <span className="flex items-center gap-1.5">
                    <DeviceMobile className="w-3.5 h-3.5 text-accent" />
                    <span>Customer message</span>
                  </span>
                  <CaretDown className="w-3.5 h-3.5 text-muted transition-transform duration-200 group-open:rotate-180" />
                </summary>
                <div className="mt-3 space-y-2 text-xs font-sans">
                  <div className="p-2.5 rounded-md border border-border bg-surface space-y-1">
                    <div className="text-xs font-semibold text-muted">
                      English SMS / notification
                    </div>
                    <p className="leading-snug text-text">
                      {stepUpData.english_message}
                    </p>
                  </div>
                  <div className="p-2.5 rounded-md border border-border bg-surface space-y-1">
                    <div className="text-xs font-semibold text-muted">
                      Hindi SMS / notification (हिंदी संदेश)
                    </div>
                    <p className="leading-snug text-text">
                      {stepUpData.hindi_message}
                    </p>
                  </div>
                </div>
              </details>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
