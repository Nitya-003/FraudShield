import React, { useState, useEffect, useRef, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ArrowRight,
  Play,
  Pause,
  Sparkle,
  Clock,
  CreditCard,
  DeviceMobile,
  Bank,
  Wallet,
} from '@phosphor-icons/react';
import { Button } from '../ui/Button';
import { Badge } from '../ui/Badge';
import { useQuery } from '@tanstack/react-query';
import { fetchStreamReplay } from '../../api/client';

export interface ProductPreviewHeroProps {
  onNavigate: (tab: string) => void;
  isDark: boolean;
}

interface StreamItem {
  id: string;
  channel: string;
  amount: number;
  risk_score: number;
  band: 'Low' | 'Medium' | 'High';
  action: string;
  top_reasons?: string[];
  latency_ms?: number;
}

interface ParticleDebris {
  x: number;
  y: number;
  vx: number;
  vy: number;
  alpha: number;
  size: number;
}

interface Particle {
  id: string;
  item: StreamItem;
  x: number;
  y: number;
  baseY: number;
  vx: number;
  phase: 'approaching' | 'orbiting' | 'exiting' | 'shattered' | 'finished';
  orbitAngle: number;
  orbitRadius: number;
  orbitSpeed: number;
  color: string;
  debris?: ParticleDebris[];
}

function formatCurrency(amount: number, channel: string): string {
  const ch = (channel || '').toLowerCase();
  if (ch.includes('upi') || ch.includes('wallet')) {
    return `₹${Math.round(amount).toLocaleString('en-IN')}`;
  }
  if (ch.includes('account')) {
    return `£${Math.round(amount).toLocaleString('en-GB')}`;
  }
  if (ch.includes('card')) {
    return `€${amount.toFixed(2)}`;
  }
  return `₹${Math.round(amount).toLocaleString()}`;
}

function getChannelMeta(channel: string) {
  const ch = (channel || '').toLowerCase();
  if (ch.includes('upi')) {
    return { name: 'UPI Payment', icon: DeviceMobile };
  }
  if (ch.includes('card')) {
    return { name: 'Card Payment', icon: CreditCard };
  }
  if (ch.includes('account')) {
    return { name: 'Account Opening', icon: Bank };
  }
  return { name: 'Wallet Transfer', icon: Wallet };
}

export const ProductPreviewHero: React.FC<ProductPreviewHeroProps> = ({
  onNavigate,
  isDark,
}) => {
  const canvasContainerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Animation lifecycle and accessibility controls
  const [isPaused, setIsPaused] = useState(false);
  const [isInView, setIsInView] = useState(true);
  const [isTabVisible, setIsTabVisible] = useState(true);
  const [canvasWidth, setCanvasWidth] = useState(540);
  const canvasHeight = 250;

  // Active Decision Card transaction index
  const [activeItemIndex, setActiveItemIndex] = useState(0);

  // Fetch real stream replay items from API
  const { data: streamData } = useQuery({
    queryKey: ['streamReplayHero'],
    queryFn: () => fetchStreamReplay(30),
    staleTime: 30000,
    refetchInterval: 30000,
  });

  const streamItems: StreamItem[] = useMemo(() => {
    return (streamData?.items as StreamItem[]) || [];
  }, [streamData?.items]);

  const activeTx: StreamItem | null = useMemo(() => {
    if (streamItems.length === 0) return null;
    return streamItems[activeItemIndex % streamItems.length];
  }, [streamItems, activeItemIndex]);

  // Rotate Decision Card every 3.8s when active
  useEffect(() => {
    if (isPaused || !isInView || !isTabVisible || streamItems.length === 0) return;
    const interval = setInterval(() => {
      setActiveItemIndex((prev) => (prev + 1) % streamItems.length);
    }, 3800);
    return () => clearInterval(interval);
  }, [isPaused, isInView, isTabVisible, streamItems.length]);

  // Responsive canvas width observer
  useEffect(() => {
    const container = canvasContainerRef.current;
    if (!container) return;

    const updateWidth = () => {
      const width = Math.min(Math.max(container.clientWidth, 280), 640);
      setCanvasWidth(width);
    };

    updateWidth();
    const observer = new ResizeObserver(updateWidth);
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  // Intersection & Tab Visibility Observers
  useEffect(() => {
    const container = canvasContainerRef.current;
    if (!container) return;

    const intersectionObserver = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          setIsInView(entry.isIntersecting);
        });
      },
      { threshold: 0.1 }
    );
    intersectionObserver.observe(container);

    const handleVisibilityChange = () => {
      setIsTabVisible(!document.hidden);
    };
    document.addEventListener('visibilitychange', handleVisibilityChange);

    return () => {
      intersectionObserver.disconnect();
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, []);

  // Honor prefers-reduced-motion
  useEffect(() => {
    if (typeof window === 'undefined') return;
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    if (mediaQuery.matches) {
      setIsPaused(true);
    }
    const handler = (e: MediaQueryListEvent) => {
      if (e.matches) setIsPaused(true);
    };
    mediaQuery.addEventListener('change', handler);
    return () => mediaQuery.removeEventListener('change', handler);
  }, []);

  // Canvas particle animation state refs
  const particlesRef = useRef<Particle[]>([]);
  const itemIndexRef = useRef(0);
  const frameCounterRef = useRef(0);
  const isPausedRef = useRef(isPaused);
  const isInViewRef = useRef(isInView);
  const isTabVisibleRef = useRef(isTabVisible);

  useEffect(() => {
    isPausedRef.current = isPaused;
    isInViewRef.current = isInView;
    isTabVisibleRef.current = isTabVisible;
  }, [isPaused, isInView, isTabVisible]);

  // High contrast color tokens
  const greenColor = isDark ? '#2DD4A0' : '#047857';
  const amberColor = isDark ? '#F59E0B' : '#B45309';
  const redColor = isDark ? '#F43F5E' : '#DC2626';
  const gateStroke = isDark ? 'rgba(148, 163, 184, 0.32)' : 'rgba(82, 96, 109, 0.32)';
  const gridColor = isDark ? 'rgba(255, 255, 255, 0.04)' : 'rgba(15, 23, 42, 0.04)';
  const axisLineColor = isDark ? 'rgba(255, 255, 255, 0.08)' : 'rgba(15, 23, 42, 0.08)';

  // Canvas Animation Engine
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.floor(canvasWidth * dpr);
    canvas.height = Math.floor(canvasHeight * dpr);

    const gateX = canvasWidth / 2;
    const gateY = canvasHeight / 2;
    const gateRadius = 52;

    // Seed initial particles
    if (particlesRef.current.length === 0 && streamItems.length > 0) {
      const initialParticles: Particle[] = [];
      const count = 3;
      for (let i = 0; i < count; i++) {
        const item = streamItems[i % streamItems.length];
        const band = item.band;
        const color = band === 'Low' ? greenColor : band === 'Medium' ? amberColor : redColor;
        const startX = (canvasWidth / (count + 1)) * (i + 0.5);
        initialParticles.push({
          id: `init-${i}-${item.id}`,
          item,
          x: startX,
          y: gateY,
          baseY: gateY,
          vx: 1.1,
          phase: startX >= gateX ? 'exiting' : 'approaching',
          orbitAngle: 0,
          orbitRadius: gateRadius,
          orbitSpeed: 0.04,
          color,
        });
      }
      particlesRef.current = initialParticles;
      itemIndexRef.current = count;
    }

    let isVisible = true;
    let isDocVisible = typeof document !== 'undefined' ? !document.hidden : true;
    let animationId: number;

    const render = () => {
      if (isPaused || !isVisible || !isDocVisible) {
        return;
      }
      ctx.save();
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, canvasWidth, canvasHeight);

      // 1. Subtle Background Grid (Faint 24px grid)
      ctx.beginPath();
      ctx.strokeStyle = gridColor;
      ctx.lineWidth = 1;
      for (let x = 0; x < canvasWidth; x += 24) {
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvasHeight);
      }
      for (let y = 0; y < canvasHeight; y += 24) {
        ctx.moveTo(0, y);
        ctx.lineTo(canvasWidth, y);
      }
      ctx.stroke();

      // 2. Horizontal axis line
      ctx.beginPath();
      ctx.setLineDash([4, 6]);
      ctx.moveTo(0, gateY);
      ctx.lineTo(canvasWidth, gateY);
      ctx.strokeStyle = axisLineColor;
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.setLineDash([]);

      // 3. Crisp Decision Gate Aperture
      ctx.save();
      ctx.beginPath();
      ctx.arc(gateX, gateY, gateRadius, 0, Math.PI * 2);
      ctx.strokeStyle = gateStroke;
      ctx.lineWidth = 2;
      ctx.stroke();

      // Gate center subtle indicator
      ctx.beginPath();
      ctx.arc(gateX, gateY, 5, 0, Math.PI * 2);
      ctx.fillStyle = gateStroke;
      ctx.fill();

      // Aperture notches
      const notchLength = 8;
      const angles = [0, Math.PI / 2, Math.PI, (3 * Math.PI) / 2];
      angles.forEach((angle) => {
        const x1 = gateX + Math.cos(angle) * (gateRadius - notchLength);
        const y1 = gateY + Math.sin(angle) * (gateRadius - notchLength);
        const x2 = gateX + Math.cos(angle) * gateRadius;
        const y2 = gateY + Math.sin(angle) * gateRadius;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.strokeStyle = gateStroke;
        ctx.lineWidth = 1.5;
        ctx.stroke();
      });

      // Gate Labels (>= 12px font size, sentence case)
      ctx.font = '600 12px Geist Mono, ui-monospace, monospace';
      ctx.fillStyle = isDark ? '#94A3B8' : '#52606D';
      ctx.textAlign = 'center';
      ctx.fillText('Decision Gate', gateX, gateY - gateRadius - 10);

      if (canvasWidth >= 400) {
        ctx.font = '12px Geist Mono, ui-monospace, monospace';
        ctx.textAlign = 'left';
        ctx.fillText('Payload Inflow', 16, 24);

        ctx.textAlign = 'right';
        ctx.fillText('Routed Verdict', canvasWidth - 16, 24);
      }
      ctx.restore();

      // 4. Update & Spawn Particles
      const running = !isPausedRef.current && isInViewRef.current && isTabVisibleRef.current;

      if (running) {
        frameCounterRef.current += 1;
        if (frameCounterRef.current % 110 === 0 && streamItems.length > 0) {
          const item = streamItems[itemIndexRef.current % streamItems.length];
          itemIndexRef.current += 1;
          const band = item.band;
          const color = band === 'Low' ? greenColor : band === 'Medium' ? amberColor : redColor;

          particlesRef.current.push({
            id: `pt-${Date.now()}-${item.id}`,
            item,
            x: -16,
            y: gateY,
            baseY: gateY,
            vx: 1.15,
            phase: 'approaching',
            orbitAngle: Math.PI,
            orbitRadius: gateRadius,
            orbitSpeed: 0.04,
            color,
          });
        }
      }

      // 5. Render Active Particles
      const updatedParticles: Particle[] = [];
      for (let i = 0; i < particlesRef.current.length; i++) {
        const p = particlesRef.current[i];

        if (running) {
          if (p.phase === 'approaching') {
            p.x += p.vx;
            if (p.x >= gateX - gateRadius) {
              if (p.item.band === 'High') {
                p.phase = 'shattered';
                p.x = gateX - gateRadius;
                p.debris = Array.from({ length: 8 }).map((_, dIdx) => {
                  const angle = (dIdx / 8) * Math.PI * 2 + (Math.random() - 0.5) * 0.5;
                  const speed = 0.8 + Math.random() * 1.6;
                  return {
                    x: p.x,
                    y: p.y,
                    vx: Math.cos(angle) * speed * 0.7 - 0.4,
                    vy: Math.sin(angle) * speed * 0.7,
                    alpha: 1.0,
                    size: 2 + Math.random() * 1.5,
                  };
                });
              } else if (p.item.band === 'Medium') {
                p.phase = 'orbiting';
                p.orbitAngle = Math.PI;
              } else {
                p.phase = 'exiting';
              }
            }
          } else if (p.phase === 'orbiting') {
            p.orbitAngle += p.orbitSpeed;
            p.x = gateX + Math.cos(p.orbitAngle) * p.orbitRadius;
            p.y = gateY + Math.sin(p.orbitAngle) * p.orbitRadius;
            if (p.orbitAngle >= Math.PI * 2) {
              p.phase = 'exiting';
              p.x = gateX + gateRadius;
              p.y = gateY;
            }
          } else if (p.phase === 'exiting') {
            p.x += p.vx * 1.25;
            if (p.x > canvasWidth + 30) {
              p.phase = 'finished';
            }
          } else if (p.phase === 'shattered') {
            if (p.debris) {
              let activeCount = 0;
              p.debris.forEach((d) => {
                d.x += d.vx;
                d.y += d.vy;
                d.alpha -= 0.024;
                if (d.alpha > 0) activeCount++;
              });
              if (activeCount === 0) p.phase = 'finished';
            } else {
              p.phase = 'finished';
            }
          }
        }

        // Draw particle if not finished
        if (p.phase !== 'finished') {
          if (p.phase === 'shattered' && p.debris) {
            p.debris.forEach((d) => {
              if (d.alpha > 0) {
                ctx.save();
                ctx.globalAlpha = Math.max(0, d.alpha);
                ctx.fillStyle = p.color;
                ctx.beginPath();
                ctx.arc(d.x, d.y, d.size, 0, Math.PI * 2);
                ctx.fill();
                ctx.restore();
              }
            });
          } else {
            // Draw crisp particle dot (radius 5.5px)
            ctx.save();
            ctx.beginPath();
            ctx.arc(p.x, p.y, 5.5, 0, Math.PI * 2);
            ctx.fillStyle = p.color;
            ctx.fill();

            // Subtle ring indicator
            ctx.lineWidth = 1.5;
            ctx.strokeStyle = isDark ? 'rgba(0,0,0,0.4)' : 'rgba(255,255,255,0.6)';
            ctx.stroke();

            // Item amount tag above dot
            ctx.font = '500 12px Geist Sans, sans-serif';
            ctx.fillStyle = isDark ? '#F1F5F9' : '#0F172A';
            ctx.textAlign = 'center';
            ctx.fillText(formatCurrency(p.item.amount, p.item.channel), p.x, p.y - 12);
            ctx.restore();
          }
          updatedParticles.push(p);
        }
      }

      particlesRef.current = updatedParticles;
      ctx.restore();

      animationId = requestAnimationFrame(render);
    };

    const handleVisibility = () => {
      isDocVisible = !document.hidden;
      if (isDocVisible && isVisible && !isPaused) {
        cancelAnimationFrame(animationId);
        animationId = requestAnimationFrame(render);
      }
    };
    document.addEventListener('visibilitychange', handleVisibility);

    let observer: IntersectionObserver | null = null;
    if (typeof IntersectionObserver !== 'undefined') {
      observer = new IntersectionObserver(
        ([entry]) => {
          isVisible = entry.isIntersecting;
          if (isVisible && isDocVisible && !isPaused) {
            cancelAnimationFrame(animationId);
            animationId = requestAnimationFrame(render);
          }
        },
        { threshold: 0.05 }
      );
      observer.observe(canvas);
    }

    animationId = requestAnimationFrame(render);
    return () => {
      document.removeEventListener('visibilitychange', handleVisibility);
      if (observer) observer.disconnect();
      cancelAnimationFrame(animationId);
    };
  }, [canvasWidth, isDark, isPaused, streamItems, greenColor, amberColor, redColor, gateStroke, gridColor, axisLineColor]);

  const ChannelIcon = activeTx ? getChannelMeta(activeTx.channel).icon : CreditCard;
  const channelTitle = activeTx ? getChannelMeta(activeTx.channel).name : 'Payment';

  return (
    <div className="w-full flex flex-col items-center">
      {/* Hero Header: Headline max 2 lines at desktop with text-balance, 1 sentence, 2 buttons */}
      <div className="max-w-3xl mx-auto text-center px-4 pt-6 sm:pt-10 pb-8 flex flex-col items-center">
        <h1 className="text-[clamp(36px,4.5vw,52px)] font-semibold tracking-normal text-text leading-[1.15] text-balance">
          Decide in milliseconds whether a payment should go through.
        </h1>

        <p className="mt-4 text-base sm:text-lg text-muted max-w-xl leading-relaxed">
          Pre-authorization risk engine scoring transactions across card, wallet, account, and UPI rails.
        </p>

        {/* Two Call-to-Action Buttons */}
        <div className="mt-6 sm:mt-8 flex flex-col sm:flex-row items-center justify-center gap-3 w-full sm:w-auto px-4 sm:px-0">
          <Button
            variant="primary"
            size="lg"
            onClick={() => onNavigate('overview')}
            className="w-full sm:w-auto justify-center px-6 min-h-[44px]"
          >
            <span>Open dashboard</span>
            <ArrowRight className="w-4 h-4 ml-2" weight="thin" />
          </Button>

          <Button
            variant="secondary"
            size="lg"
            onClick={() => onNavigate('live_check')}
            className="w-full sm:w-auto justify-center px-6 min-h-[44px]"
          >
            <Sparkle className="w-4 h-4 mr-2 text-accent" weight="thin" />
            <span>Try a live check</span>
          </Button>
        </div>
      </div>

      {/* Real Product Preview: Framed Decision Card + Crisp Gate Animation Canvas */}
      <div className="w-full max-w-5xl px-4 mt-2">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
          {/* 1. Framed, Tilted Decision Card (5 cols on desktop) */}
          <div className="lg:col-span-5 flex flex-col justify-center">
            <div className="relative">
              {/* Subtle background glow card for depth */}
              <div className="absolute -inset-1 rounded-2xl bg-gradient-to-r from-accent/10 via-transparent to-accent/5 blur-sm opacity-60" />

              <div className="relative rounded-xl border border-border bg-surface p-5 shadow-overlay transform lg:-rotate-1 hover:rotate-0 transition-transform duration-300 ease-out space-y-4">
                {/* Header: Live status + ID */}
                <div className="flex items-center justify-between border-b border-border/60 pb-3">
                  <div className="flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-accent animate-pulse" />
                    <span className="text-xs font-semibold text-muted">
                      Live Stream Replay
                    </span>
                  </div>
                  <span className="font-mono text-xs text-muted">
                    {activeTx?.id || 'TX-CAR-06150'}
                  </span>
                </div>

                {/* Animated Decision Content */}
                <AnimatePresence mode="wait">
                  {activeTx && (
                    <motion.div
                      key={activeTx.id}
                      initial={{ opacity: 0, y: 6 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -6 }}
                      transition={{ duration: 0.22, ease: [0.16, 1, 0.3, 1] }}
                      className="space-y-4"
                    >
                      {/* Channel + Amount */}
                      <div className="flex items-start justify-between gap-3">
                        <div className="space-y-1">
                          <div className="flex items-center gap-1.5 text-xs text-muted">
                            <ChannelIcon className="w-3.5 h-3.5" weight="thin" />
                            <span>{channelTitle}</span>
                          </div>
                          <div className="text-2xl font-semibold font-sans tracking-normal tabular-nums text-text">
                            {formatCurrency(activeTx.amount, activeTx.channel)}
                          </div>
                        </div>

                        {/* Band Badge */}
                        <div className="pt-1">
                          <Badge
                            variant={
                              activeTx.band === 'Low'
                                ? 'low'
                                : activeTx.band === 'Medium'
                                ? 'medium'
                                : 'high'
                            }
                          >
                            {activeTx.band} Risk
                          </Badge>
                        </div>
                      </div>

                      {/* Calibrated Risk Score & Meter */}
                      <div className="space-y-1.5 p-3 rounded-md bg-surface-subtle border border-border">
                        <div className="flex items-center justify-between text-xs">
                          <span className="text-muted font-normal">Calibrated Score</span>
                          <span className="font-mono font-semibold tabular-nums text-text">
                            {activeTx.risk_score.toFixed(4)}
                          </span>
                        </div>
                        {/* Thin progress bar */}
                        <div className="w-full h-1.5 rounded-full bg-border overflow-hidden">
                          <div
                            className={`h-full transition-[width,background-color] duration-500 rounded-full ${
                              activeTx.band === 'Low'
                                ? 'bg-accent'
                                : activeTx.band === 'Medium'
                                ? 'bg-warn'
                                : 'bg-danger'
                            }`}
                            style={{
                              width: `${Math.min(Math.max(activeTx.risk_score * 100, 4), 100)}%`,
                            }}
                          />
                        </div>
                        <div className="flex items-center justify-between text-xs pt-0.5">
                          <span className="text-muted font-normal">Decision:</span>
                          <span
                            className={`font-semibold ${
                              activeTx.band === 'Low'
                                ? 'text-accent'
                                : activeTx.band === 'Medium'
                                ? 'text-warn'
                                : 'text-danger'
                            }`}
                          >
                            {activeTx.action || (activeTx.band === 'Low' ? 'Approve' : activeTx.band === 'Medium' ? 'Step-Up' : 'Decline')}
                          </span>
                        </div>
                      </div>

                      {/* Top Factor Attribution Reason */}
                      <div className="text-xs space-y-1">
                        <span className="text-muted font-normal">Primary factor driver:</span>
                        <p className="text-text font-normal leading-relaxed line-clamp-2">
                          {activeTx.top_reasons?.[0] || 'Behavioral transaction pattern consistent with historical profile.'}
                        </p>
                      </div>

                      {/* Execution Latency footer */}
                      <div className="flex items-center justify-between text-xs text-muted pt-1 border-t border-border/50">
                        <span className="flex items-center gap-1 font-mono">
                          <Clock className="w-3 h-3" weight="thin" />
                          <span>{activeTx.latency_ms?.toFixed(1) || '6.8'} ms evaluation</span>
                        </span>
                        <span className="font-mono text-accent">SLA &lt; 50 ms</span>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </div>
          </div>

          {/* 2. Crisp Decision Gate Aperture Canvas (7 cols on desktop) */}
          <div className="lg:col-span-7 flex flex-col justify-between rounded-xl border border-border bg-surface p-4 shadow-card">
            <div
              ref={canvasContainerRef}
              className="relative w-full rounded-lg bg-surface-subtle border border-border overflow-hidden flex items-center justify-center shadow-[inset_0_1px_3px_rgba(0,0,0,0.06)] dark:shadow-[inset_0_1px_4px_rgba(0,0,0,0.3)]"
            >
              <canvas
                ref={canvasRef}
                className="w-full block"
                style={{ height: `${canvasHeight}px` }}
                aria-label="Calibrated decision gate aperture animation"
              />
            </div>

            {/* Animation controls footer */}
            <div className="mt-3 flex items-center justify-between px-1 text-xs text-muted">
              <div className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-accent animate-pulse" />
                <span>Live replay stream from held-out test splits</span>
              </div>

              <button
                type="button"
                onClick={() => setIsPaused((prev) => !prev)}
                className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-sm border border-border bg-surface hover:bg-surface-subtle hover:text-text transition-colors text-xs font-normal"
                aria-label={isPaused ? 'Resume animation' : 'Pause animation'}
              >
                {isPaused ? <Play className="w-3 h-3" weight="thin" /> : <Pause className="w-3 h-3" weight="thin" />}
                <span>{isPaused ? 'Resume' : 'Pause'}</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

