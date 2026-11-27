import React, { useState, useEffect } from 'react';
import { ExternalLink, Copy, Check, Terminal, Play, Loader2 } from 'lucide-react';
import { fetchOverviewKPIs } from '../api/client';
import type { OverviewKPIs } from '../types/api';
import { PageHeader } from '../components/layout/PageHeader';
import { Button } from '../components/ui/Button';
import { useToast } from '../components/ui/Toast';

interface EndpointDef {
  id: string;
  method: 'GET' | 'POST';
  path: string;
  tag: string;
  description: string;
  examplePayload?: string;
  canLiveTest?: boolean;
}

export const ApiDocsPage: React.FC = () => {
  const { toast } = useToast();
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [kpis, setKpis] = useState<OverviewKPIs | null>(null);

  const [testingEndpointId, setTestingEndpointId] = useState<string | null>(null);
  const [liveTestResponse, setLiveTestResponse] = useState<Record<string, { status: number; durationMs: number; data: unknown }>>({});

  useEffect(() => {
    fetchOverviewKPIs().then(setKpis).catch(() => {});
  }, []);

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
    toast({
      title: 'Copied to Clipboard',
      description: id.startsWith('curl') ? 'cURL command ready to run in terminal.' : 'JSON payload copied.',
      variant: 'info',
    });
  };

  const getCurlCommand = (ep: EndpointDef) => {
    const origin = typeof window !== 'undefined' ? window.location.origin : 'http://localhost:8000';
    if (ep.method === 'GET') {
      return `curl -X GET "${origin}${ep.path.replace('{entity_id}', 'C716083600')}" \\\n  -H "Accept: application/json"`;
    }
    return `curl -X POST "${origin}${ep.path}" \\\n  -H "Content-Type: application/json" \\\n  -d '${ep.examplePayload?.replace(/\n\s*/g, ' ')}'`;
  };

  const handleLiveTest = async (ep: EndpointDef) => {
    setTestingEndpointId(ep.id);
    const start = performance.now();
    try {
      let res: Response;
      if (ep.method === 'GET') {
        const url = ep.path.replace('{entity_id}', 'C716083600');
        res = await fetch(url);
      } else {
        res = await fetch(ep.path, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: ep.examplePayload,
        });
      }
      const data = await res.json();
      const durationMs = Math.round(performance.now() - start);
      setLiveTestResponse((prev) => ({
        ...prev,
        [ep.id]: { status: res.status, durationMs, data },
      }));
      toast({
        title: 'Endpoint Executed',
        description: `${ep.method} ${ep.path} responded (${res.status}) in ${durationMs}ms`,
        variant: res.ok ? 'success' : 'warning',
      });
    } catch (err) {
      setLiveTestResponse((prev) => ({
        ...prev,
        [ep.id]: { status: 500, durationMs: Math.round(performance.now() - start), data: { error: String(err) } },
      }));
      toast({
        title: 'Execution Failed',
        description: `Could not connect to ${ep.path}`,
        variant: 'error',
      });
    } finally {
      setTestingEndpointId(null);
    }
  };

  const endpoints: EndpointDef[] = [
    {
      id: 'score',
      method: 'POST',
      path: '/score',
      tag: 'Scoring',
      description: 'Score a single transaction with calibrated risk probability and local TreeSHAP attributions.',
      canLiveTest: true,
      examplePayload: `{
  "channel": "upi",
  "payload": {
    "amount": 145000,
    "hour": 14,
    "txn_type": "COLLECT_REQUEST",
    "is_first_time_payee": 1,
    "is_new_device": 1
  },
  "top_k": 3,
  "update_history": true
}`,
    },
    {
      id: 'batch',
      method: 'POST',
      path: '/api/batch-score',
      tag: 'Scoring',
      description: 'High-throughput vectorized batch scoring of multiple transactions across any channel.',
      canLiveTest: true,
      examplePayload: `{
  "channel": "upi",
  "transactions": [
    { "amount": 3500, "hour": 14, "txn_type": "PAY", "is_first_time_payee": 0 },
    { "amount": 145000, "hour": 2, "txn_type": "COLLECT_REQUEST", "is_first_time_payee": 1 }
  ],
  "top_k": 2
}`,
    },
    {
      id: 'overview',
      method: 'GET',
      path: '/api/overview',
      tag: 'Analytics',
      description: 'Return high-level KPIs, total volume, alert counts, and measured p50/p95 latency metrics.',
      canLiveTest: true,
    },
    {
      id: 'transactions',
      method: 'GET',
      path: '/api/transactions',
      tag: 'Ledger',
      description: 'Query paginated transactions with channel, risk band, and account handle filters.',
      canLiveTest: true,
    },
    {
      id: 'entity',
      method: 'GET',
      path: '/api/entity/{entity_id}',
      tag: 'Graph',
      description: 'Retrieve multi-hop network neighborhood, cash flow breakdowns, and mule ring associations.',
      canLiveTest: true,
    },
    {
      id: 'simulate',
      method: 'POST',
      path: '/api/attack-lab/simulate',
      tag: 'Adversarial',
      description: 'Simulate adversarial evasions (split transfers, timing delays, mule hops) against models.',
      canLiveTest: true,
      examplePayload: `{
  "channel": "transfers",
  "base_payload": { "amount": 820000, "type": "TRANSFER", "oldbalanceOrg": 820000, "newbalanceOrig": 0 },
  "attack_type": "split_transfers",
  "params": { "num_splits": 3 }
}`,
    },
    {
      id: 'stepup',
      method: 'POST',
      path: '/api/step-up/preview',
      tag: 'Messaging',
      description: 'Generate authentic bilingual English and Hindi customer challenge notifications.',
      canLiveTest: true,
      examplePayload: `{
  "channel": "upi",
  "amount": 145000,
  "action": "Step-Up Challenge",
  "risk_score": 0.65
}`,
    },
  ];

  return (
    <div className="space-y-6 pb-16">
      <PageHeader
        title="API Documentation"
        purpose="REST endpoints for transaction scoring, batch inference, and entity intelligence."
        action={
          <a
            href="/docs"
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-primary text-primary-foreground shadow-card hover:opacity-90 transition-opacity"
          >
            <span>Open Swagger UI</span>
            <ExternalLink className="w-3.5 h-3.5" />
          </a>
        }
      />

      {/* Latency Benchmarks */}
      <section className="p-4 rounded-xl border border-border bg-surface grid grid-cols-1 sm:grid-cols-3 gap-4 shadow-card">
        <div>
          <div className="text-xs font-semibold text-muted">
            Median Server Latency (p50)
          </div>
          <div className="text-2xl font-mono tabular-nums font-semibold text-text mt-0.5">
            {kpis?.latency_p50_ms ? `${kpis.latency_p50_ms.toFixed(2)} ms` : '-- ms'}
          </div>
          <div className="text-xs text-muted mt-1">
            Measured on local benchmark host under load
          </div>
        </div>

        <div>
          <div className="text-xs font-semibold text-muted">
            95th Percentile Latency (p95)
          </div>
          <div className="text-2xl font-mono tabular-nums font-semibold text-text mt-0.5">
            {kpis?.latency_p95_ms ? `${kpis.latency_p95_ms.toFixed(2)} ms` : '-- ms'}
          </div>
          <div className="text-xs text-muted mt-1">
            Under 20 concurrent scoring clients
          </div>
        </div>

        <div>
          <div className="text-xs font-semibold text-muted">
            Pre-Auth SLA Target
          </div>
          <div className="text-2xl font-mono tabular-nums font-semibold text-accent mt-0.5">
            &lt; 50 ms
          </div>
          <div className="text-xs text-muted mt-1">
            Deterministic pre-settlement decision budget
          </div>
        </div>
      </section>

      {/* Endpoints Reference List */}
      <section className="space-y-4">
        <h2 className="text-xs font-semibold text-muted">
          Available REST Endpoints
        </h2>

        <div className="space-y-4 font-sans text-xs">
          {endpoints.map((ep) => {
            const curlCmd = getCurlCommand(ep);
            const liveResult = liveTestResponse[ep.id];
            const isRunningTest = testingEndpointId === ep.id;

            return (
              <div
                key={ep.id}
                className="p-5 rounded-xl border border-border bg-surface space-y-4 shadow-card"
              >
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-border pb-3">
                  <div className="flex items-center gap-3">
                    <span
                      className={`font-mono font-semibold text-xs px-2 py-0.5 rounded ${
                        ep.method === 'POST'
                          ? 'bg-blue-500/10 text-blue-600 dark:text-blue-400 border border-blue-500/20'
                          : 'bg-accent/10 text-accent border border-accent/20'
                      }`}
                    >
                      {ep.method}
                    </span>
                    <span className="font-mono font-semibold text-sm text-text">
                      {ep.path}
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-xs text-muted">
                      {ep.tag}
                    </span>
                    {ep.canLiveTest && (
                      <Button
                        variant="secondary"
                        size="sm"
                        onClick={() => handleLiveTest(ep)}
                        disabled={isRunningTest}
                      >
                        {isRunningTest ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
                        <span>Live test</span>
                      </Button>
                    )}
                  </div>
                </div>

                <p className="text-muted leading-relaxed">{ep.description}</p>

                {/* cURL Command Block */}
                <div className="space-y-1.5 pt-1">
                  <div className="flex items-center justify-between text-xs font-mono text-muted">
                    <span className="flex items-center gap-1.5">
                      <Terminal className="w-3.5 h-3.5" />
                      <span>cURL Command</span>
                    </span>
                    <button
                      onClick={() => copyToClipboard(curlCmd, `curl-${ep.id}`)}
                      className="flex items-center gap-1 hover:text-text transition-colors cursor-pointer"
                    >
                      {copiedId === `curl-${ep.id}` ? (
                        <>
                          <Check className="w-3.5 h-3.5 text-accent" />
                          <span className="text-accent font-semibold">Copied</span>
                        </>
                      ) : (
                        <>
                          <Copy className="w-3.5 h-3.5" />
                          <span>Copy</span>
                        </>
                      )}
                    </button>
                  </div>
                  <pre
                    tabIndex={0}
                    aria-label="cURL command example"
                    className="p-3 rounded-lg bg-surface-subtle border border-border text-text text-xs font-mono overflow-x-auto leading-relaxed"
                  >
                    {curlCmd}
                  </pre>
                </div>

                {/* Example Request Payload (if POST) */}
                {ep.examplePayload && (
                  <div className="space-y-1.5 pt-1">
                    <div className="flex items-center justify-between text-xs font-mono text-muted">
                      <span>Example Request JSON Payload</span>
                      <button
                        onClick={() => copyToClipboard(ep.examplePayload!, `json-${ep.id}`)}
                        className="flex items-center gap-1 hover:text-text transition-colors cursor-pointer"
                      >
                        {copiedId === `json-${ep.id}` ? (
                          <>
                            <Check className="w-3.5 h-3.5 text-accent" />
                            <span className="text-accent font-semibold">Copied</span>
                          </>
                        ) : (
                          <>
                            <Copy className="w-3.5 h-3.5" />
                            <span>Copy</span>
                          </>
                        )}
                      </button>
                    </div>
                    <pre
                      tabIndex={0}
                      aria-label="Request JSON payload example"
                      className="p-3 rounded-lg bg-surface-subtle border border-border text-text text-xs font-mono overflow-x-auto leading-relaxed"
                    >
                      {ep.examplePayload}
                    </pre>
                  </div>
                )}

                {/* Live Test Response */}
                {liveResult && (
                  <div className="p-3.5 rounded-xl border border-border bg-surface-subtle space-y-2">
                    <div className="flex items-center justify-between font-mono text-xs">
                      <div className="flex items-center gap-2">
                        <span className="font-semibold text-text">Live response:</span>
                        <span className={`px-2 py-0.5 rounded text-xs font-semibold ${liveResult.status === 200 ? 'bg-accent/10 text-accent' : 'bg-danger/10 text-danger'}`}>
                          {liveResult.status} OK
                        </span>
                      </div>
                      <span className="text-muted text-xs tabular-nums">
                        Latency: {liveResult.durationMs} ms
                      </span>
                    </div>
                    <pre
                      tabIndex={0}
                      aria-label="Live test JSON response output"
                      className="p-2.5 rounded bg-surface border border-border text-text text-xs font-mono overflow-x-auto max-h-48"
                    >
                      {JSON.stringify(liveResult.data, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </section>
    </div>
  );
};
