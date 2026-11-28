import React, { useState, useEffect } from 'react';

interface IntroAnimationOverlayProps {
  onComplete: () => void;
  isOpen: boolean;
}

export const IntroAnimationOverlay: React.FC<IntroAnimationOverlayProps> = ({
  onComplete,
  isOpen,
}) => {
  const [phase, setPhase] = useState<'locked' | 'splitting' | 'done'>('locked');

  useEffect(() => {
    if (!isOpen) return;

    // Immediately skip if user prefers reduced motion
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      onComplete();
      return;
    }

    setPhase('locked');

    // 0ms - 220ms: Focused center aperture glow + laser horizon ignite
    const t1 = setTimeout(() => {
      setPhase('splitting');
    }, 220);

    // 750ms: Splitting curtain completes, transition unmounts
    const t2 = setTimeout(() => {
      setPhase('done');
      onComplete();
    }, 780);

    // Fast-skip on any interaction
    const handleKeyDown = () => {
      clearTimeout(t1);
      clearTimeout(t2);
      onComplete();
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [isOpen, onComplete]);

  if (!isOpen || phase === 'done') return null;

  const isSplitting = phase === 'splitting';

  return (
    <div
      onClick={onComplete}
      role="dialog"
      aria-modal="true"
      aria-label="Website Opening Transition"
      className="fixed inset-0 z-50 pointer-events-auto select-none overflow-hidden cursor-pointer"
    >
      {/* =======================================================================
          TOP HALF SHUTTER
          ======================================================================= */}
      <div
        className={`absolute top-0 inset-x-0 h-1/2 bg-[#080C11] border-b border-emerald-400/80 shadow-[0_4px_24px_rgba(45,212,160,0.35)] transition-transform ease-[cubic-bezier(0.16,1,0.3,1)] duration-500 will-change-transform flex items-end justify-center ${
          isSplitting ? '-translate-y-full' : 'translate-y-0'
        }`}
      >
        {/* Subtle geometric grid on shutter surface */}
        <div className="absolute inset-0 opacity-15 bg-[radial-gradient(ellipse_at_bottom,rgba(45,212,160,0.25)_0%,transparent_70%)]" />
      </div>

      {/* =======================================================================
          BOTTOM HALF SHUTTER
          ======================================================================= */}
      <div
        className={`absolute bottom-0 inset-x-0 h-1/2 bg-[#080C11] border-t border-emerald-400/80 shadow-[0_-4px_24px_rgba(45,212,160,0.35)] transition-transform ease-[cubic-bezier(0.16,1,0.3,1)] duration-500 will-change-transform flex items-start justify-center ${
          isSplitting ? 'translate-y-full' : 'translate-y-0'
        }`}
      >
        <div className="absolute inset-0 opacity-15 bg-[radial-gradient(ellipse_at_top,rgba(45,212,160,0.25)_0%,transparent_70%)]" />
      </div>

      {/* =======================================================================
          CENTER HORIZON LASER BEAM
          ======================================================================= */}
      <div
        className={`absolute top-1/2 inset-x-0 -translate-y-1/2 h-[2px] bg-gradient-to-r from-transparent via-[#2DD4A0] to-transparent pointer-events-none transition-[transform,opacity] duration-300 ease-out ${
          isSplitting
            ? 'opacity-0 scale-y-0 blur-sm'
            : 'opacity-100 scale-x-100 shadow-[0_0_16px_#2DD4A0]'
        }`}
      />

      {/* =======================================================================
          CENTRAL APERTURE SHIELD GLYPH
          ======================================================================= */}
      <div
        className={`absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 pointer-events-none transition-[transform,opacity] duration-500 ease-[cubic-bezier(0.16,1,0.3,1)] ${
          isSplitting
            ? 'opacity-0 scale-150 blur-md'
            : 'opacity-100 scale-100'
        }`}
      >
        <div className="relative w-20 h-20 sm:w-24 sm:h-24 flex items-center justify-center">
          {/* Radial Shockwave Halo */}
          <div className="absolute inset-0 rounded-full bg-emerald-400/20 blur-xl animate-ping" />

          {/* Aperture Shield SVG */}
          <svg
            className="w-16 h-16 sm:w-20 sm:h-20 drop-shadow-[0_0_20px_#2DD4A0]"
            viewBox="0 0 32 32"
            fill="none"
          >
            {/* Outer Faceted Shield */}
            <polygon
              points="16,2 28,7 28,15 22,25 16,30 10,25 4,15 4,7"
              fill="#0B0F14"
              stroke="#2DD4A0"
              strokeWidth="1.8"
            />
            {/* Inner Aperture Hexagon */}
            <polygon
              points="16,10 21,13 21,19 16,22 11,19 11,13"
              fill="#0F1720"
              stroke="#2DD4A0"
              strokeWidth="1.4"
            />
            {/* Core Glowing Singularity */}
            <circle cx="16" cy="16" r="3" fill="#2DD4A0" className="animate-pulse" />
          </svg>
        </div>
      </div>
    </div>
  );
};

