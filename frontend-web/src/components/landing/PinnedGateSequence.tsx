import React, { useEffect, useRef, useState } from 'react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

export interface PinnedGateSequenceProps {
  isDark: boolean;
}

interface StepData {
  title: string;
  caption: string;
  detail: string;
  badge: string;
}

const STEPS: StepData[] = [
  {
    title: 'Features',
    caption: 'Extracts velocity, outflow ratios, and counterparty graph signals in memory without leakage.',
    detail: '42 streaming behavioral features computed with temporal safety',
    badge: 'Stage 1',
  },
  {
    title: 'Score',
    caption: 'Calibrated probability scaling maps raw model predictions directly to empirical fraud risk.',
    detail: 'Platt calibrated probability distribution minimizing Brier score',
    badge: 'Stage 2',
  },
  {
    title: 'Action',
    caption: 'Deterministic policy routing partitions payments into auto-approval, step-up verification, or decline.',
    detail: 'Fast policy routing across tuned operating risk thresholds',
    badge: 'Stage 3',
  },
];

export const PinnedGateSequence: React.FC<PinnedGateSequenceProps> = ({ isDark }) => {
  const triggerRef = useRef<HTMLDivElement>(null);
  const pinTargetRef = useRef<HTMLDivElement>(null);
  const [activeStep, setActiveStep] = useState(0);

  useEffect(() => {
    const triggerEl = triggerRef.current;
    const pinTargetEl = pinTargetRef.current;
    if (!triggerEl || !pinTargetEl) return;

    // Check prefers-reduced-motion
    const prefersReducedMotion =
      typeof window !== 'undefined' &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    if (prefersReducedMotion) {
      return;
    }

    const st = ScrollTrigger.create({
      trigger: triggerEl,
      start: 'top top+=64',
      end: '+=1400',
      pin: pinTargetEl,
      pinSpacing: true,
      scrub: 0.5,
      onUpdate: (self) => {
        const progress = self.progress;
        if (progress < 0.33) {
          setActiveStep(0);
        } else if (progress < 0.66) {
          setActiveStep(1);
        } else {
          setActiveStep(2);
        }
      },
    });

    return () => {
      st.kill();
    };
  }, []);

  const greenColor = isDark ? '#2DD4A0' : '#047857';
  const amberColor = isDark ? '#F59E0B' : '#D97706';
  const redColor = isDark ? '#F43F5E' : '#DC2626';

  return (
    <div ref={triggerRef} className="w-full relative my-16">
      <div
        ref={pinTargetRef}
        className="w-full max-w-4xl mx-auto px-4 py-8 flex flex-col items-center"
      >
        {/* Step Indicator Pills */}
        <div className="flex items-center gap-2 mb-8">
          {STEPS.map((step, idx) => {
            const isActive = activeStep === idx;
            return (
              <button
                key={step.title}
                type="button"
                onClick={() => setActiveStep(idx)}
                className={`px-3 py-1.5 rounded-full text-xs font-mono transition-colors duration-200 border ${
                  isActive
                    ? 'bg-accent/10 border-accent text-accent font-semibold shadow-sm'
                    : 'bg-surface border-border text-muted hover:text-text'
                }`}
              >
                <span>{step.title}</span>
              </button>
            );
          })}
        </div>

        {/* Central Aperture Gate Stage Visualization */}
        <div className="relative w-full max-w-2xl h-52 sm:h-60 rounded-xl border border-border bg-surface p-6 flex flex-col items-center justify-center overflow-hidden">
          {/* Subtle Background Guide Grid */}
          <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-accent/5 via-transparent to-transparent opacity-60 pointer-events-none" />

          {/* Decision Pipeline Visual */}
          <div className="relative z-10 w-full flex items-center justify-between px-2 sm:px-10">
            {/* 1. Feature Ingestion Node */}
            <div
              className={`flex flex-col items-center gap-1.5 sm:gap-2 transition-transform duration-300 ${
                activeStep === 0 ? 'scale-105' : ''
              }`}
            >
              <div
                className={`w-10 h-10 sm:w-12 sm:h-12 rounded-xl flex items-center justify-center border font-mono text-xs transition-colors ${
                  activeStep === 0
                    ? 'border-accent bg-accent/15 text-accent shadow-md shadow-accent/10'
                    : 'border-border bg-surface text-muted'
                }`}
              >
                Φ
              </div>
              <span className={`text-xs font-sans ${activeStep === 0 ? 'text-accent font-medium' : 'text-muted'}`}>
                Signals
              </span>
            </div>

            {/* Connecting Track 1 */}
            <div className="flex-1 mx-2 sm:mx-6 h-[1.5px] bg-border relative overflow-hidden">
              <div
                className={`absolute inset-0 transition-[width,background-color] duration-300 ${
                  activeStep >= 1 ? 'bg-accent w-full' : 'bg-transparent w-0'
                }`}
              />
            </div>

            {/* 2. Calibration Gate Aperture Ring */}
            <div
              className={`flex flex-col items-center gap-1.5 sm:gap-2 transition-transform duration-300 ${
                activeStep === 1 ? 'scale-110' : ''
              }`}
            >
              <div
                className={`relative w-14 h-14 sm:w-16 sm:h-16 rounded-full flex items-center justify-center border-2 transition-[border-color,background-color] duration-300 ${
                  activeStep === 1
                    ? 'border-accent shadow-lg shadow-accent/20 bg-accent/10'
                    : 'border-border bg-surface'
                }`}
              >
                {/* Aperture Center Crosshair */}
                <div className="w-2 h-2 rounded-full bg-accent" />
                <div className="absolute inset-1 border border-dashed border-border rounded-full animate-spin-slow" />
              </div>
              <span className={`text-xs font-sans ${activeStep === 1 ? 'text-accent font-medium' : 'text-muted'}`}>
                Gate
              </span>
            </div>

            {/* Connecting Track 2 */}
            <div className="flex-1 mx-2 sm:mx-6 h-[1.5px] bg-border relative overflow-hidden">
              <div
                className={`absolute inset-0 transition-[width,background-color] duration-300 ${
                  activeStep >= 2 ? 'bg-accent w-full' : 'bg-transparent w-0'
                }`}
              />
            </div>

            {/* 3. Action Routing Nodes (Approve / Step-up / Decline) */}
            <div
              className={`flex flex-col items-center gap-1.5 sm:gap-2 transition-transform duration-300 ${
                activeStep === 2 ? 'scale-105' : ''
              }`}
            >
              <div className="flex flex-col gap-1">
                <div
                  className={`w-16 sm:w-20 px-2 py-0.5 rounded text-xs font-sans font-medium text-center border ${
                    activeStep === 2 ? '' : 'border-border bg-surface text-muted'
                  }`}
                  style={
                    activeStep === 2
                      ? {
                          borderColor: greenColor,
                          color: greenColor,
                          backgroundColor: `${greenColor}15`,
                        }
                      : undefined
                  }
                >
                  Approve
                </div>
                <div
                  className={`w-16 sm:w-20 px-2 py-0.5 rounded text-xs font-sans font-medium text-center border ${
                    activeStep === 2 ? '' : 'border-border bg-surface text-muted'
                  }`}
                  style={
                    activeStep === 2
                      ? {
                          borderColor: amberColor,
                          color: amberColor,
                          backgroundColor: `${amberColor}15`,
                        }
                      : undefined
                  }
                >
                  Step-Up
                </div>
                <div
                  className={`w-16 sm:w-20 px-2 py-0.5 rounded text-xs font-sans font-medium text-center border ${
                    activeStep === 2 ? '' : 'border-border bg-surface text-muted'
                  }`}
                  style={
                    activeStep === 2
                      ? {
                          borderColor: redColor,
                          color: redColor,
                          backgroundColor: `${redColor}15`,
                        }
                      : undefined
                  }
                >
                  Decline
                </div>
              </div>
              <span className={`text-xs font-sans ${activeStep === 2 ? 'text-accent font-medium' : 'text-muted'}`}>
                Action
              </span>
            </div>
          </div>
        </div>

        {/* Dynamic Caption Container (Max 12 words per caption, plain language) */}
        <div className="mt-8 text-center max-w-lg min-h-[80px] flex flex-col items-center justify-center">
          <div className="text-xs font-semibold text-accent mb-1.5">
            {STEPS[activeStep].title}
          </div>
          <p className="text-base sm:text-lg text-text font-medium leading-relaxed transition-opacity duration-200">
            {STEPS[activeStep].caption}
          </p>
          <div className="mt-2 text-xs font-mono text-muted">
            {STEPS[activeStep].detail}
          </div>
        </div>
      </div>
    </div>
  );
};
