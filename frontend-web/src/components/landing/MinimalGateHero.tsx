import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { motion } from 'framer-motion';
import { ArrowRight, Play, Pause } from '@phosphor-icons/react';
import { Button } from '../ui/Button';
import { useQuery } from '@tanstack/react-query';
import { fetchStreamReplay } from '../../api/client';

export interface MinimalGateHeroProps {
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
  const prefix = ch.includes('upi') || ch.includes('wallet') ? '₹' : ch.includes('card') ? '€' : '$';
  return `${prefix}${Math.round(amount).toLocaleString('en-US')}`;
}

function formatChannelName(channel: string): string {
  const ch = (channel || '').toLowerCase();
  if (ch.includes('upi')) return 'UPI';
  if (ch.includes('card')) return 'Cards';
  if (ch.includes('account')) return 'Account';
  if (ch.includes('wallet')) return 'Wallets';
  return channel || 'Payment';
}

export const MinimalGateHero: React.FC<MinimalGateHeroProps> = ({ onNavigate, isDark }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Animation lifecycle and accessibility controls
  const [isPaused, setIsPaused] = useState(false);
  const [isInView, setIsInView] = useState(true);
  const [isTabVisible, setIsTabVisible] = useState(true);
  const [canvasWidth, setCanvasWidth] = useState(800);
  const canvasHeight = 150;

  // Hovered item tooltip state
  const [activeTooltip, setActiveTooltip] = useState<{
    x: number;
    y: number;
    item: StreamItem;
  } | null>(null);

  // Fetch real stream replay items from API
  const { data: streamData } = useQuery({
    queryKey: ['streamReplayHero'],
    queryFn: () => fetchStreamReplay(30),
    staleTime: 30000,
    refetchInterval: 30000,
  });

  const streamItems = useMemo(() => streamData?.items || [], [streamData?.items]);
  const provenanceLabel = streamData?.mode === 'test_set_replay'
    ? 'Replay of the test split'
    : 'Simulated data';

  // Responsive canvas width observer
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const updateWidth = () => {
      const width = Math.min(Math.max(container.clientWidth, 320), 840);
      setCanvasWidth(width);
    };

    updateWidth();
    const observer = new ResizeObserver(updateWidth);
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  // Intersection & Tab Visibility Observers (Auto-pause when offscreen or tab hidden)
  useEffect(() => {
    const container = containerRef.current;
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

  // Canvas particle animation state refs (Zero React re-render per frame)
  const particlesRef = useRef<Particle[]>([]);
  const itemIndexRef = useRef(0);
  const frameCounterRef = useRef(0);
  const animationFrameIdRef = useRef<number | null>(null);
  const isPausedRef = useRef(isPaused);
  const isInViewRef = useRef(isInView);
  const isTabVisibleRef = useRef(isTabVisible);

  useEffect(() => {
    isPausedRef.current = isPaused;
    isInViewRef.current = isInView;
    isTabVisibleRef.current = isTabVisible;
  }, [isPaused, isInView, isTabVisible]);

  // Color tokens
  const greenColor = isDark ? '#2DD4A0' : '#047857';
  const amberColor = isDark ? '#F59E0B' : '#D97706';
  const redColor = isDark ? '#F43F5E' : '#DC2626';
  const neutralColor = isDark ? '#94A3B8' : '#64748B';
  const gateStroke = isDark ? 'rgba(148, 163, 184, 0.28)' : 'rgba(100, 116, 139, 0.28)';
  const axisLineColor = isDark ? 'rgba(255, 255, 255, 0.08)' : 'rgba(15, 23, 42, 0.08)';

  // Animation Engine
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Handle high DPI displays
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(canvasWidth * dpr);
    canvas.height = Math.floor(canvasHeight * dpr);

    const gateX = canvasWidth / 2;
    const gateY = canvasHeight / 2;
    const gateRadius = 42;

    // Seed initial particles if empty
    if (particlesRef.current.length === 0 && streamItems.length > 0) {
      const initialParticles: Particle[] = [];
      const count = 4;
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
          orbitSpeed: 0.045,
          color,
        });
      }
      particlesRef.current = initialParticles;
      itemIndexRef.current = count;
    }

    const render = () => {
      ctx.save();
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, canvasWidth, canvasHeight);

      // 1. Draw central horizontal axis guide line
      ctx.beginPath();
      ctx.setLineDash([4, 6]);
      ctx.moveTo(0, gateY);
      ctx.lineTo(canvasWidth, gateY);
      ctx.strokeStyle = axisLineColor;
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.setLineDash([]);

      // 2. Draw Decision Gate (Aperture Ring)
      ctx.save();
      // Outer subtle ring
      ctx.beginPath();
      ctx.arc(gateX, gateY, gateRadius, 0, Math.PI * 2);
      ctx.strokeStyle = gateStroke;
      ctx.lineWidth = 1.75;
      ctx.stroke();

      // Gate center subtle indicator
      ctx.beginPath();
      ctx.arc(gateX, gateY, 4.5, 0, Math.PI * 2);
      ctx.fillStyle = gateStroke;
      ctx.fill();

      // Aperture notches
      const notchLength = 6;
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
        ctx.lineWidth = 1.25;
        ctx.stroke();
      });

      // Gate Labels (Minimum 12px font size)
      ctx.font = '12px Geist Mono, ui-monospace, monospace';
      ctx.fillStyle = isDark ? '#64748B' : '#94A3B8';
      ctx.textAlign = 'center';
      ctx.fillText('Gate', gateX, gateY - gateRadius - 8);

      // Only show side labels on wider viewports to avoid mobile crowding
      if (canvasWidth >= 520) {
        ctx.font = '12px Geist Mono, ui-monospace, monospace';
        ctx.textAlign = 'left';
        ctx.fillText('Incoming', 18, gateY - 14);

        ctx.textAlign = 'right';
        ctx.fillText('Authorized', canvasWidth - 18, gateY - 14);
      }
      ctx.restore();

      // 3. Update & Spawn Particles if active
      const running = !isPausedRef.current && isInViewRef.current && isTabVisibleRef.current;

      if (running) {
        frameCounterRef.current += 1;

        // Spawn a new particle every ~100 frames if stream items available
        if (frameCounterRef.current % 100 === 0 && streamItems.length > 0) {
          const item = streamItems[itemIndexRef.current % streamItems.length];
          itemIndexRef.current += 1;

          const band = item.band;
          const color = band === 'Low' ? greenColor : band === 'Medium' ? amberColor : redColor;

          particlesRef.current.push({
            id: `pt-${Date.now()}-${item.id}`,
            item,
            x: -12,
            y: gateY,
            baseY: gateY,
            vx: 1.15,
            phase: 'approaching',
            orbitAngle: Math.PI, // Starts at left edge of gate
            orbitRadius: gateRadius,
            orbitSpeed: 0.045,
            color,
          });
        }
      }

      // 4. Update and render active particles
      const updatedParticles: Particle[] = [];

      for (let i = 0; i < particlesRef.current.length; i++) {
        const p = particlesRef.current[i];

        if (running) {
          if (p.phase === 'approaching') {
            p.x += p.vx;

            // Check if particle reaches the gate ring
            if (p.x >= gateX - gateRadius) {
              if (p.item.band === 'High') {
                // High Risk: Shatter at gate entry!
                p.phase = 'shattered';
                p.x = gateX - gateRadius;
                p.debris = Array.from({ length: 8 }).map((_, dIdx) => {
                  const angle = (dIdx / 8) * Math.PI * 2 + (Math.random() - 0.5) * 0.5;
                  const speed = 0.8 + Math.random() * 1.5;
                  return {
                    x: p.x,
                    y: p.y,
                    vx: Math.cos(angle) * speed * 0.7 - 0.4,
                    vy: Math.sin(angle) * speed * 0.7,
                    alpha: 1.0,
                    size: 1.5 + Math.random() * 1.5,
                  };
                });
              } else if (p.item.band === 'Medium') {
                // Medium Risk: Pause & orbit once
                p.phase = 'orbiting';
                p.orbitAngle = Math.PI; // 180 degrees (left side)
              } else {
                // Low Risk: Pass straight through
                p.phase = 'exiting';
              }
            }
          } else if (p.phase === 'orbiting') {
            // Orbit around the aperture ring
            p.orbitAngle += p.orbitSpeed;
            p.x = gateX + Math.cos(p.orbitAngle) * p.orbitRadius;
            p.y = gateY + Math.sin(p.orbitAngle) * p.orbitRadius;

            // Complete full orbit: starts at PI, once it completes 360 deg + reaches right side (3*PI)
            if (p.orbitAngle >= Math.PI + Math.PI * 2) {
              p.phase = 'exiting';
              p.x = gateX + p.orbitRadius;
              p.y = gateY;
            }
          } else if (p.phase === 'exiting') {
            p.x += p.vx;
            p.y = gateY;
            if (p.x > canvasWidth + 20) {
              p.phase = 'finished';
            }
          } else if (p.phase === 'shattered' && p.debris) {
            let activeDebrisCount = 0;
            p.debris.forEach((d) => {
              d.x += d.vx;
              d.y += d.vy;
              d.alpha -= 0.025;
              d.size = Math.max(0.5, d.size * 0.98);
              if (d.alpha > 0) activeDebrisCount += 1;
            });
            if (activeDebrisCount === 0) {
              p.phase = 'finished';
            }
          }
        }

        // Draw particle if not finished
        if (p.phase !== 'finished') {
          if (p.phase === 'shattered' && p.debris) {
            // Draw red shatter sparks
            p.debris.forEach((d) => {
              if (d.alpha <= 0) return;
              ctx.save();
              ctx.beginPath();
              ctx.arc(d.x, d.y, d.size, 0, Math.PI * 2);
              ctx.fillStyle = `rgba(${isDark ? '244, 63, 94' : '220, 38, 38'}, ${Math.max(0, d.alpha)})`;
              ctx.fill();
              ctx.restore();
            });
          } else {
            // Draw payment dot
            ctx.save();
            // Subtle glow
            ctx.beginPath();
            ctx.arc(p.x, p.y, 11, 0, Math.PI * 2);
            ctx.fillStyle = p.phase === 'approaching'
              ? (isDark ? 'rgba(148, 163, 184, 0.08)' : 'rgba(100, 116, 139, 0.08)')
              : (p.color === greenColor ? 'rgba(45, 212, 160, 0.18)' : 'rgba(245, 158, 11, 0.18)');
            ctx.fill();

            // Core dot
            ctx.beginPath();
            ctx.arc(p.x, p.y, 5.5, 0, Math.PI * 2);
            ctx.fillStyle = p.phase === 'approaching' ? neutralColor : p.color;
            ctx.fill();
            ctx.restore();
          }

          updatedParticles.push(p);
        }
      }

      particlesRef.current = updatedParticles;
      ctx.restore();

      animationFrameIdRef.current = requestAnimationFrame(render);
    };

    animationFrameIdRef.current = requestAnimationFrame(render);

    return () => {
      if (animationFrameIdRef.current) {
        cancelAnimationFrame(animationFrameIdRef.current);
      }
    };
  }, [canvasWidth, isDark, streamItems, greenColor, amberColor, redColor, neutralColor, gateStroke, axisLineColor]);

  // Handle Mouse Hover / Touch to inspect dot details
  const handlePointerMove = useCallback((e: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    let closest: Particle | null = null;
    let minDist = 18;

    for (const p of particlesRef.current) {
      if (p.phase === 'finished' || p.phase === 'shattered') continue;
      const dist = Math.hypot(p.x - mx, p.y - my);
      if (dist < minDist) {
        minDist = dist;
        closest = p;
      }
    }

    if (closest) {
      setActiveTooltip({
        x: closest.x,
        y: closest.y,
        item: closest.item,
      });
    } else {
      setActiveTooltip(null);
    }
  }, []);

  const handlePointerLeave = useCallback(() => {
    setActiveTooltip(null);
  }, []);

  return (
    <div className="w-full flex flex-col items-center">
      {/* 400ms Staggered Fade-in Header Section */}
      <div className="max-w-4xl mx-auto text-center px-4 pt-6 sm:pt-10 pb-8 flex flex-col items-center">
        {/* Headline with comfortable tracking and balanced two lines */}
        <motion.h1
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.32, ease: [0.16, 1, 0.3, 1] }}
          className="text-3xl sm:text-4xl md:text-5xl lg:text-6xl font-semibold tracking-normal text-text max-w-2xl leading-[1.18]"
        >
          Decide in milliseconds <br className="hidden sm:inline" />
          whether a payment{' '}
          <span className="text-accent font-semibold">should go through.</span>
        </motion.h1>

        {/* Subtitle: truthful, 13 words (<=18 words, removed 'Calibrated') */}
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.32, delay: 0.08, ease: [0.16, 1, 0.3, 1] }}
          className="mt-4 sm:mt-5 text-base sm:text-lg text-muted max-w-xl leading-relaxed"
        >
          Pre-authorization risk engine deciding transactions across card, wallet, transfer, and UPI payment rails.
        </motion.p>

        {/* Two Call to Action Buttons */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.32, delay: 0.16, ease: [0.16, 1, 0.3, 1] }}
          className="mt-6 sm:mt-8 flex flex-col sm:flex-row items-center justify-center gap-3 w-full sm:w-auto px-4 sm:px-0"
        >
          <Button
            variant="primary"
            size="lg"
            onClick={() => onNavigate('overview')}
            className="w-full sm:w-auto justify-center font-medium px-5"
          >
            Open dashboard
            <ArrowRight size={16} weight="thin" className="ml-2 inline-block" />
          </Button>

          <Button
            variant="outline"
            size="lg"
            onClick={() => onNavigate('live_check')}
            className="w-full sm:w-auto justify-center font-medium px-5"
          >
            Try a live check
          </Button>
        </motion.div>
      </div>

      {/* ONE Visual: Central Payment Decision Gate Animation */}
      <motion.div
        ref={containerRef}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5, delay: 0.3 }}
        className="w-full max-w-4xl px-4 flex flex-col items-center mt-2"
      >
        <div className="relative w-full rounded-xl border border-border bg-surface overflow-hidden flex items-center justify-center">
          <canvas
            ref={canvasRef}
            onPointerMove={handlePointerMove}
            onPointerLeave={handlePointerLeave}
            className="w-full block cursor-crosshair touch-none"
            style={{ height: `${canvasHeight}px` }}
            aria-label="Payment risk decision gate visualization"
          />

          {/* Floating Data-Driven Tooltip on Hover / Focus */}
          {activeTooltip && (
            <div
              className="absolute pointer-events-none z-30 transform -translate-x-1/2 -translate-y-full mb-3 px-3 py-2 rounded-lg bg-surface border border-border-strong shadow-xl text-left font-sans"
              style={{
                left: `${activeTooltip.x}px`,
                top: `${activeTooltip.y}px`,
              }}
            >
              <div className="flex items-center justify-between gap-3 text-xs mb-1">
                <span className="font-semibold text-text">
                  {formatCurrency(activeTooltip.item.amount, activeTooltip.item.channel)}
                </span>
                <span className="text-xs font-sans px-1.5 py-0.5 rounded bg-surface-subtle text-muted">
                  {formatChannelName(activeTooltip.item.channel)}
                </span>
              </div>
              <div className="flex items-center gap-2 text-xs">
                <span className="text-muted font-mono">
                  Score: {(activeTooltip.item.risk_score * 100).toFixed(1)}%
                </span>
                <span
                  className={`px-1.5 py-0.5 rounded text-xs font-semibold ${
                    activeTooltip.item.band === 'Low'
                      ? 'text-accent bg-accent/10'
                      : activeTooltip.item.band === 'Medium'
                      ? 'text-warn bg-warn/10'
                      : 'text-danger bg-danger/10'
                  }`}
                >
                  {activeTooltip.item.action || activeTooltip.item.band}
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Tiny honest caption & animation pause control */}
        <div className="w-full mt-3 flex items-center justify-between px-2 text-xs font-mono text-muted">
          <div className="flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-accent animate-pulse" />
            <span>{provenanceLabel}</span>
          </div>

          <button
            type="button"
            onClick={() => setIsPaused((prev) => !prev)}
            className="inline-flex items-center gap-1.5 px-2 py-1 rounded border border-border hover:border-border-strong bg-surface hover:text-text transition-colors"
            aria-label={isPaused ? 'Resume animation' : 'Pause animation'}
          >
            {isPaused ? <Play size={12} weight="thin" /> : <Pause size={12} weight="thin" />}
            <span>{isPaused ? 'Resume' : 'Pause'}</span>
          </button>
        </div>
      </motion.div>
    </div>
  );
};
