import React, { useState, useRef } from 'react';
import { PageHeader } from '../components/layout/PageHeader';
import { Button } from '../components/ui/Button';
import { Badge } from '../components/ui/Badge';
import { Select } from '../components/ui/Select';
import { InlineRiskBar } from '../components/ui/DataTable';
import { useToast } from '../components/ui/Toast';
import { AnimatedNumber } from '../hooks/useCountUp';
import { Play, Sparkles, Layers, Upload, AlertCircle, FileText } from 'lucide-react';
import { formatChannelAmount } from '../lib/channelUtils';

interface BatchResult {
  status: string;
  channel: string;
  total_scored: number;
  high_risk_count: number;
  medium_risk_count: number;
  low_risk_count: number;
  mean_risk_score: number;
  total_duration_ms: number;
  results: Array<{
    index: number;
    amount: number;
    risk_score: number;
    band: string;
    action: string;
  }>;
}

export const BatchScoringPage: React.FC = () => {
  const { toast } = useToast();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [channel, setChannel] = useState('upi');
  const [batchSize, setBatchSize] = useState(25);
  const [isProcessing, setIsProcessing] = useState(false);
  const [batchResult, setBatchResult] = useState<BatchResult | null>(null);
  const [validationErrors, setValidationErrors] = useState<string[]>([]);
  const [uploadedFileName, setUploadedFileName] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const validateTransactions = (txs: Array<Record<string, unknown>>): string[] => {
    const errors: string[] = [];
    if (!Array.isArray(txs) || txs.length === 0) {
      errors.push('File contains no transaction records or is an empty array.');
      return errors;
    }

    if (txs.length > 500) {
      errors.push(`Batch size exceeds maximum limit of 500 records (found ${txs.length} records).`);
    }

    txs.slice(0, 50).forEach((tx, idx) => {
      const rowNum = idx + 1;
      if (tx.amount === undefined && tx.Amount === undefined) {
        errors.push(`Row ${rowNum}: missing required field "amount" (or "Amount").`);
      } else {
        const amt = Number(tx.amount ?? tx.Amount);
        if (isNaN(amt) || amt <= 0) {
          errors.push(`Row ${rowNum}: amount must be a positive number; found "${tx.amount ?? tx.Amount}".`);
        }
      }

      if (channel === 'upi') {
        if (tx.hour !== undefined) {
          const h = Number(tx.hour);
          if (isNaN(h) || h < 0 || h > 23) {
            errors.push(`Row ${rowNum}: "hour" must be between 0 and 23; found "${tx.hour}".`);
          }
        }
      }
    });

    return errors;
  };

  const handleFileProcess = async (file: File) => {
    setValidationErrors([]);
    setUploadedFileName(file.name);

    try {
      const text = await file.text();
      let parsedTxs: Array<Record<string, unknown>> = [];

      if (file.name.endsWith('.json')) {
        try {
          const parsed = JSON.parse(text);
          parsedTxs = Array.isArray(parsed) ? parsed : (parsed.transactions || []);
        } catch {
          setValidationErrors([`File syntax error: unable to parse JSON file "${file.name}". Please ensure valid JSON format.`]);
          return;
        }
      } else if (file.name.endsWith('.csv')) {
        const lines = text.trim().split('\n');
        if (lines.length < 2) {
          setValidationErrors(['CSV file must contain a header row and at least one data row.']);
          return;
        }
        const headers = lines[0].split(',').map((h) => h.trim().replace(/^"|"$/g, ''));
        parsedTxs = lines.slice(1).map((line) => {
          const cols = line.split(',').map((c) => c.trim().replace(/^"|"$/g, ''));
          const rowObj: Record<string, unknown> = {};
          headers.forEach((h, i) => {
            rowObj[h] = cols[i];
          });
          return rowObj;
        });
      } else {
        setValidationErrors(['Unsupported file type. Please upload a .csv or .json file.']);
        return;
      }

      const errors = validateTransactions(parsedTxs);
      if (errors.length > 0) {
        setValidationErrors(errors);
        return;
      }

      toast({
        title: 'File Parsed Successfully',
        description: `Ready to score ${parsedTxs.length} records from ${file.name}.`,
        variant: 'info',
      });

      setIsProcessing(true);
      const res = await fetch(`/api/batch-score?channel=${channel}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ transactions: parsedTxs }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || 'Batch scoring failed');
      }

      const result: BatchResult = await res.json();
      setBatchResult(result);
      toast({
        title: 'Batch Complete',
        description: `Processed ${result.total_scored} transactions in ${result.total_duration_ms.toFixed(1)}ms.`,
        variant: 'success',
      });
    } catch (err: unknown) {
      const e = err as { message?: string };
      setValidationErrors([e?.message || 'Error processing batch upload']);
      toast({
        title: 'Upload Processing Failed',
        description: e?.message || 'Failed to complete batch scoring',
        variant: 'error',
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileProcess(e.dataTransfer.files[0]);
    }
  };

  const runBatchScoring = async () => {
    setIsProcessing(true);
    setValidationErrors([]);
    try {
      const res = await fetch(`/api/batch-score/sample?channel=${channel}&size=${batchSize}`, {
        method: 'POST',
      });
      if (!res.ok) throw new Error('Batch scoring request failed');
      const data: BatchResult = await res.json();
      setBatchResult(data);
      toast({
        title: 'Batch Evaluation Complete',
        description: `Scored ${data.total_scored} ${channel.toUpperCase()} transactions in ${data.total_duration_ms.toFixed(1)}ms.`,
        variant: 'success',
      });
    } catch {
      toast({
        title: 'Scoring Failed',
        description: 'Failed to complete batch scoring.',
        variant: 'error',
      });
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="space-y-6 pb-16">
      <PageHeader
        title="Batch Scoring"
        purpose="High-throughput array inference for bulk transaction CSV and JSON payloads."
        action={
          <Button
            variant="primary"
            size="md"
            onClick={runBatchScoring}
            disabled={isProcessing}
          >
            <Play className="w-4 h-4" />
            <span>{isProcessing ? 'Processing batch...' : 'Run synthetic batch'}</span>
          </Button>
        }
      />

      {/* Drag & Drop Upload Zone */}
      <section
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`p-8 rounded-xl border-2 border-dashed transition-colors text-center cursor-pointer ${
          isDragging
            ? 'border-accent bg-accent/10'
            : 'border-border hover:border-border-strong bg-surface'
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv,.json"
          className="hidden"
          onChange={(e) => {
            if (e.target.files && e.target.files[0]) {
              handleFileProcess(e.target.files[0]);
            }
          }}
        />
        <Upload className="w-8 h-8 text-muted mx-auto mb-2 opacity-60" />
        <h3 className="font-semibold text-sm text-text">
          Drag and drop transaction CSV or JSON here
        </h3>
        <p className="text-xs text-muted mt-1">
          Supports arrays of transactions with schema validation. Click to browse files.
        </p>
        {uploadedFileName && (
          <div className="mt-3 inline-flex items-center gap-1.5 px-3 py-1 rounded bg-surface-subtle border border-border text-xs font-mono text-text">
            <FileText className="w-3.5 h-3.5 text-muted" />
            <span>{uploadedFileName}</span>
          </div>
        )}
      </section>

      {/* Validation Errors */}
      {validationErrors.length > 0 && (
        <section className="p-4 rounded-xl border border-danger/30 bg-danger/10 text-danger text-xs space-y-2">
          <div className="flex items-center gap-2 font-semibold text-sm">
            <AlertCircle className="w-4 h-4" />
            <span>Batch Input Validation Issues ({validationErrors.length})</span>
          </div>
          <p className="text-xs">
            Please resolve the following formatting discrepancies before batch scoring:
          </p>
          <ul className="list-disc list-inside space-y-1 font-mono text-xs pt-1">
            {validationErrors.map((err, idx) => (
              <li key={idx}>{err}</li>
            ))}
          </ul>
        </section>
      )}

      {/* Channel & Preset Controls */}
      <section className="p-4 rounded-xl border border-border bg-surface grid grid-cols-1 sm:grid-cols-3 gap-4 items-end shadow-card">
        <Select
          label="Target Channel Model"
          value={channel}
          onChange={(e) => setChannel(e.target.value)}
          options={[
            { value: 'upi', label: 'UPI Instant' },
            { value: 'card', label: 'Card Payment' },
            { value: 'transfer_wallet', label: 'Wallet Transfer' },
            { value: 'account', label: 'Account Opening' },
          ]}
        />

        <Select
          label="Synthetic Batch Size"
          value={batchSize.toString()}
          onChange={(e) => setBatchSize(Number(e.target.value))}
          options={[
            { value: '10', label: '10 Transactions (Quick Test)' },
            { value: '25', label: '25 Transactions (Standard)' },
            { value: '50', label: '50 Transactions (Stress Test)' },
          ]}
        />

        <div>
          <Button
            variant="secondary"
            size="md"
            className="w-full"
            onClick={runBatchScoring}
            disabled={isProcessing}
          >
            <Sparkles className="w-4 h-4" />
            <span>Generate &amp; score sample</span>
          </Button>
        </div>
      </section>

      {/* Batch Results Output */}
      {batchResult && (
        <section className="space-y-5">
          {/* Summary KPIs */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="p-4 rounded-xl border border-border bg-surface shadow-card">
              <div className="text-xs font-semibold text-muted">
                Total scored
              </div>
              <div className="text-2xl font-mono tabular-nums font-semibold text-text mt-0.5">
                <AnimatedNumber value={batchResult.total_scored} format="num" suffix=" txs" duration={600} />
              </div>
              <div className="text-xs font-mono tabular-nums text-muted mt-1">
                Completed in {batchResult.total_duration_ms.toFixed(1)} ms
              </div>
            </div>

            <div className="p-4 rounded-xl border border-border bg-surface shadow-card">
              <div className="text-xs font-semibold text-danger">
                High risk (decline)
              </div>
              <div className="text-2xl font-mono tabular-nums font-semibold text-danger mt-0.5">
                <AnimatedNumber value={batchResult.high_risk_count} format="num" duration={600} />
              </div>
              <div className="text-xs font-mono tabular-nums text-muted mt-1">
                {((batchResult.high_risk_count / batchResult.total_scored) * 100).toFixed(1)}% of batch
              </div>
            </div>

            <div className="p-4 rounded-xl border border-border bg-surface shadow-card">
              <div className="text-xs font-semibold text-warn">
                Medium (step-up)
              </div>
              <div className="text-2xl font-mono tabular-nums font-semibold text-warn mt-0.5">
                <AnimatedNumber value={batchResult.medium_risk_count} format="num" duration={600} />
              </div>
              <div className="text-xs font-mono tabular-nums text-muted mt-1">
                {((batchResult.medium_risk_count / batchResult.total_scored) * 100).toFixed(1)}% of batch
              </div>
            </div>

            <div className="p-4 rounded-xl border border-border bg-surface shadow-card">
              <div className="text-xs font-semibold text-accent">
                Low risk (approve)
              </div>
              <div className="text-2xl font-mono tabular-nums font-semibold text-accent mt-0.5">
                <AnimatedNumber value={batchResult.low_risk_count} format="num" duration={600} />
              </div>
              <div className="text-xs font-mono tabular-nums text-muted mt-1">
                Mean score: {batchResult.mean_risk_score.toFixed(3)}
              </div>
            </div>
          </div>

          {/* Results Table */}
          <div className="border border-border rounded-xl overflow-x-auto font-mono text-xs bg-surface shadow-card">
            <table className="w-full text-left border-collapse min-w-[640px]">
              <thead className="bg-surface-subtle border-b border-border text-xs font-semibold text-muted normal-case font-sans">
                <tr className="h-10">
                  <th className="py-2.5 px-3.5">Batch index</th>
                  <th className="py-2.5 px-3.5 text-right">Amount</th>
                  <th className="py-2.5 px-3.5">Risk score</th>
                  <th className="py-2.5 px-3.5">Risk band</th>
                  <th className="py-2.5 px-3.5">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {batchResult.results.map((r, idx) => (
                  <tr key={idx} className="h-[44px] hover:bg-surface-subtle transition-colors">
                    <td className="py-2.5 px-3.5 font-semibold text-text">
                      TX-BATCH-{(r.index ?? idx + 1).toString().padStart(3, '0')}
                    </td>
                    <td className="py-2.5 px-3.5 text-right font-semibold text-text tabular-nums">
                      {typeof r.amount === 'number' ? formatChannelAmount(r.amount, channel) : 'Evaluated'}
                    </td>
                    <td className="py-2.5 px-3.5 font-semibold">
                      <InlineRiskBar score={r.risk_score} />
                    </td>
                    <td className="py-2.5 px-3.5">
                      <Badge variant={r.band.toLowerCase() as 'low' | 'medium' | 'high'}>
                        {r.band}
                      </Badge>
                    </td>
                    <td className="py-2.5 px-3.5 font-sans text-xs text-text">{r.action}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {!batchResult && !isProcessing && (
        <div className="p-12 text-center border border-dashed border-border rounded-xl bg-surface space-y-2">
          <Layers className="w-8 h-8 text-muted mx-auto opacity-40" />
          <h3 className="font-semibold text-sm text-text">No batch scored yet</h3>
          <p className="text-xs text-muted max-w-sm mx-auto">
            Drag and drop a CSV/JSON file above or click &ldquo;Run synthetic batch&rdquo; to test high-throughput inference.
          </p>
          <Button variant="primary" size="sm" onClick={runBatchScoring} className="mt-2">
            Run synthetic batch
          </Button>
        </div>
      )}
    </div>
  );
};
