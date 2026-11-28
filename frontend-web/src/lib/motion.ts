export const DURATIONS = {
  fast: 0.12, 
  normal: 0.2, 
  slow: 0.32, 
} as const;

export const DURATIONS_MS = {
  fast: 120,
  normal: 200,
  slow: 320,
} as const;

export const EASING = [0.16, 1, 0.3, 1] as const; 
export const EASING_CSS = 'cubic-bezier(0.16, 1, 0.3, 1)';

export const pageFadeVariants = {
  initial: { opacity: 0 },
  animate: {
    opacity: 1,
    transition: { duration: DURATIONS.fast, ease: EASING },
  },
  exit: {
    opacity: 0,
    transition: { duration: DURATIONS.fast, ease: EASING },
  },
};

export const standardTransition = {
  duration: DURATIONS.normal,
  ease: EASING,
};

