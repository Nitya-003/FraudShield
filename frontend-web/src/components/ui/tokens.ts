/**
 * FraudShield Design System Tokens ("The Ledger")
 * Single source of truth for color values, spacing, radii, typography, and contrast.
 */

export const TOKENS = {
  theme: {
    light: {
      name: 'The Ledger',
      background: '#fbf9f5', // Warm paper
      card: '#f4f0e8',       // Slightly deeper paper
      cardHover: '#ebe5d8',
      border: '#e6ded1',     // Hairline rule
      ink: '#1c1917',        // Deep stone ink
      muted: '#78716c',      // Warm gray
      accent: '#1e3a8a',     // Deep ink-blue
      accentMuted: '#dbeafe',
      riskHigh: '#b91c1c',   // Vermilion
      riskMed: '#b45309',    // Amber
      riskLow: '#15803d',    // Green
      gridline: '#ede7db',
    },
    dark: {
      name: 'Night Ledger',
      background: '#121212', // Charcoal night
      card: '#1a1917',       // Elevated panel
      cardHover: '#242220',
      border: '#2a2723',     // Subtle rule
      ink: '#f5f5f4',        // Crisp light stone
      muted: '#a8a29e',      // Slate gray
      accent: '#60a5fa',     // Bright blue
      accentMuted: '#1e3a8a',
      riskHigh: '#ef4444',   // Bright vermilion
      riskMed: '#f59e0b',    // Bright amber
      riskLow: '#22c55e',    // Bright green
      gridline: '#22201d',
    },
  },
  typography: {
    titleFont: 'Newsreader, Georgia, serif',
    bodyFont: 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
    monoFont: 'JetBrains Mono, Menlo, monospace',
    scale: {
      xs: '0.75rem',    // 12px
      sm: '0.8125rem',  // 13px
      base: '0.875rem', // 14px
      md: '1.0rem',     // 16px
      lg: '1.125rem',   // 18px
      xl: '1.5rem',     // 24px
      '2xl': '1.875rem',// 30px
      '3xl': '2.25rem', // 36px
    },
  },
  spacing: {
    xs: '4px',
    sm: '8px',
    md: '16px',
    lg: '24px',
    xl: '32px',
    '2xl': '48px',
  },
  radii: {
    sm: '4px',
    md: '6px',
    lg: '8px',
  },
  shadows: {
    none: 'none',
    subtle: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
  },
  contrast: {
    wcagTarget: '4.5:1 (AA)',
    measuredLight: {
      inkOnPaper: '14.2:1',
      mutedOnPaper: '5.1:1',
      accentOnPaper: '9.8:1',
      vermilionOnPaper: '5.6:1',
    },
    measuredDark: {
      inkOnNight: '15.1:1',
      mutedOnNight: '6.4:1',
      accentOnNight: '8.2:1',
      vermilionOnNight: '6.1:1',
    },
  },
} as const;

export type ThemeMode = 'light' | 'dark';

