/* eslint-disable react-refresh/only-export-components */
import React from 'react';
import { HelpCircle } from 'lucide-react';
import { Popover } from './Popover';

export const GLOSSARY_TERMS: Record<string, { title: string; definition: string; formula?: string }> = {
  'PR-AUC': {
    title: 'Precision-Recall Area Under Curve (PR-AUC)',
    definition:
      'The primary benchmark metric for highly imbalanced fraud datasets. Unlike ROC-AUC, it ignores massive true-negative counts and directly reflects the trade-off between precision (minimizing false alarms) and recall (catching frauds).',
  },
  'ROC-AUC': {
    title: 'Receiver Operating Characteristic AUC (ROC-AUC)',
    definition:
      'The probability that a randomly chosen fraud case ranks higher in predicted risk than a randomly chosen legitimate case. Insensitive to class imbalance.',
  },
  'False Alarms / 100k': {
    title: 'False Alarms per 100,000 Transactions',
    definition:
      'The number of innocent customers challenged or declined per 100,000 valid transactions at the chosen operating threshold. Directly measures customer friction.',
  },
  'Platt Calibration': {
    title: 'Platt Calibration (Sigmoid Scaling)',
    definition:
      'A logistic regression fit over raw model outputs to transform raw tree margins into calibrated empirical probabilities P(fraud | x), ensuring 80% scored alerts have an 80% empirical fraud rate.',
    formula: 'P(y=1 | s) = 1 / (1 + exp(A * s + B))',
  },
  'TreeSHAP': {
    title: 'TreeSHAP Feature Attributions',
    definition:
      'Game-theoretic Shapley values computed efficiently for tree ensembles. Provides local accuracy and consistency, breaking down a transaction score into the exact positive and negative contributions of each input feature.',
  },
  'PSI': {
    title: 'Population Stability Index (PSI)',
    definition:
      'Measures distributional shift between baseline reference and production inference data. Values < 0.10 denote stability; 0.10 - 0.25 indicate slight shift; > 0.25 trigger retraining alerts.',
    formula: 'PSI = sum((Actual% - Expected%) * ln(Actual% / Expected%))',
  },
  'Brier Score': {
    title: 'Brier Score',
    definition:
      'The mean squared error between calibrated probability predictions and binary ground truth (0 or 1). Lower values indicate superior probabilistic calibration.',
    formula: 'Brier = (1/N) * sum((p_i - y_i)^2)',
  },
};

interface GlossaryPopoverProps {
  term: keyof typeof GLOSSARY_TERMS | string;
  children?: React.ReactNode;
  side?: 'top' | 'right' | 'bottom' | 'left';
}

export const GlossaryPopover: React.FC<GlossaryPopoverProps> = ({
  term,
  children,
  side = 'top',
}) => {
  const item = GLOSSARY_TERMS[term] || {
    title: term,
    definition: 'Statistical or operational parameter in the FraudShield pipeline.',
  };

  return (
    <Popover
      side={side}
      align="start"
      className="max-w-xs"
      trigger={
        <button
          type="button"
          aria-label={`Glossary definition for ${item.title}`}
          className="inline-flex items-center gap-1 cursor-help group underline decoration-dotted decoration-stone-400 dark:decoration-stone-600 underline-offset-2 text-inherit font-inherit text-left bg-transparent border-0 p-0"
        >
          {children || term}
          <HelpCircle className="w-3 h-3 text-muted group-hover:text-text" />
        </button>
      }
    >
      <div className="space-y-1.5">
        <h4 className="font-sans font-semibold text-xs text-text tracking-normal">
          {item.title}
        </h4>
        <p className="text-xs leading-relaxed text-muted">
          {item.definition}
        </p>
        {item.formula && (
          <div className="font-mono text-xs p-1.5 rounded bg-surface-subtle border border-border text-text overflow-x-auto">
            {item.formula}
          </div>
        )}
      </div>
    </Popover>
  );
};
