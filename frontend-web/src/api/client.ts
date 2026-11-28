import type {
  OverviewKPIs,
  RiskBandsSummary,
  DailySeriesResponse,
  PagedTransactions,
  CaseItem,
  CaseAuditLog,
  ModelRegistryData,
  ScoreResponse,
  AttackPreset,
  AttackSimulationResult,
  StepUpMessageData,
} from '../types/api';

const API_BASE = '/api';

export async function fetchOverviewKPIs(channel?: string): Promise<OverviewKPIs> {
  const query = channel && channel !== 'ALL' ? `?channel=${encodeURIComponent(channel)}` : '';
  const res = await fetch(`${API_BASE}/overview${query}`);
  if (!res.ok) throw new Error('Failed to load overview KPIs');
  return res.json();
}

export async function fetchRiskBands(channel?: string): Promise<RiskBandsSummary> {
  const query = channel && channel !== 'ALL' ? `?channel=${encodeURIComponent(channel)}` : '';
  const res = await fetch(`${API_BASE}/risk-bands${query}`);
  if (!res.ok) throw new Error('Failed to load risk bands');
  return res.json();
}

export async function fetchDailySeries(channel?: string): Promise<DailySeriesResponse> {
  const query = channel && channel !== 'ALL' ? `?channel=${encodeURIComponent(channel)}` : '';
  const res = await fetch(`${API_BASE}/daily-series${query}`);
  if (!res.ok) throw new Error('Failed to load daily series');
  return res.json();
}

export async function fetchTransactions(params: {
  page?: number;
  pageSize?: number;
  channel?: string;
  band?: string;
  search?: string;
  sortBy?: string;
  sortOrder?: 'asc' | 'desc';
}): Promise<PagedTransactions> {
  const query = new URLSearchParams();
  if (params.page) query.set('page', params.page.toString());
  if (params.pageSize) query.set('page_size', params.pageSize.toString());
  if (params.channel && params.channel !== 'ALL') query.set('channel', params.channel);
  if (params.band && params.band !== 'ALL') query.set('band', params.band);
  if (params.search) query.set('search', params.search);
  if (params.sortBy) query.set('sort_by', params.sortBy);
  if (params.sortOrder) query.set('sort_order', params.sortOrder);

  const res = await fetch(`${API_BASE}/transactions?${query.toString()}`);
  if (!res.ok) throw new Error('Failed to load transactions');
  return res.json();
}

export async function fetchCases(params?: {
  status?: string;
  channel?: string;
  band?: string;
  limit?: number;
}): Promise<{ cases: CaseItem[]; count: number }> {
  const query = new URLSearchParams();
  if (params?.status && params.status !== 'ALL') query.set('status', params.status);
  if (params?.channel && params.channel !== 'ALL') query.set('channel', params.channel);
  if (params?.band && params.band !== 'ALL') query.set('band', params.band);
  if (params?.limit) query.set('limit', params.limit.toString());

  const res = await fetch(`${API_BASE}/cases?${query.toString()}`);
  if (!res.ok) throw new Error('Failed to load cases');
  return res.json();
}

export async function fetchCaseDetail(caseId: string): Promise<{ case: CaseItem; audit_logs: CaseAuditLog[] }> {
  const res = await fetch(`${API_BASE}/cases/${caseId}`);
  if (!res.ok) throw new Error(`Failed to load case ${caseId}`);
  return res.json();
}

export async function updateCaseStatus(
  caseId: string,
  payload: { status: string; notes: string; analyst_name: string }
): Promise<{ case: CaseItem; message: string }> {
  const res = await fetch(`${API_BASE}/cases/${caseId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`Failed to update case ${caseId}`);
  return res.json();
}

export async function fetchModelRegistry(): Promise<ModelRegistryData> {
  const res = await fetch(`${API_BASE}/models`);
  if (!res.ok) throw new Error('Failed to load model registry');
  return res.json();
}

export async function scoreTransaction(payload: {
  channel: string;
  payload: Record<string, unknown>;
  top_k?: number;
  update_history?: boolean;
}): Promise<ScoreResponse> {
  const res = await fetch(`${API_BASE}/score`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error('Scoring request failed');
  return res.json();
}

export async function fetchAttackPresets(): Promise<{ presets: AttackPreset[] }> {
  const res = await fetch(`${API_BASE}/attack-lab/presets`);
  if (!res.ok) throw new Error('Failed to load attack presets');
  return res.json();
}

export async function simulateAttack(payload: {
  channel: string;
  base_payload: Record<string, unknown>;
  attack_type: string;
  params: Record<string, unknown>;
}): Promise<{ simulation: AttackSimulationResult }> {
  const res = await fetch(`${API_BASE}/attack-lab/simulate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error('Attack simulation failed');
  return res.json();
}

export async function previewStepUpMessage(payload: {
  channel: string;
  payload: Record<string, unknown>;
  otp_code?: string;
}): Promise<{ messaging: StepUpMessageData }> {
  const res = await fetch(`${API_BASE}/step-up/preview`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error('Step-up preview failed');
  return res.json();
}

export async function fetchGlossary(): Promise<{ glossary: Record<string, string> }> {
  const res = await fetch(`${API_BASE}/glossary`);
  if (!res.ok) throw new Error('Failed to load glossary');
  return res.json();
}

export async function fetchStreamReplay(limit: number = 20): Promise<{
  status: string;
  mode: string;
  items: Array<{
    id: string;
    timestamp: string;
    time_label: string;
    step: number;
    channel: string;
    amount: number;
    risk_score: number;
    raw_score: number;
    band: 'Low' | 'Medium' | 'High';
    action: string;
    top_reasons: string[];
    latency_ms: number;
    is_fraud: number;
  }>;
  stats: {
    total_events: number;
    events_in_buffer: number;
    events_per_sec: number;
    loss_averted_inr: number;
    total_volume_inr: number;
    alert_rate_pct: number;
    latency_p50_ms: number;
    latency_p95_ms: number;
  };
}> {
  const res = await fetch(`${API_BASE}/stream?limit=${limit}`);
  if (!res.ok) throw new Error('Failed to load stream replay');
  return res.json();
}
