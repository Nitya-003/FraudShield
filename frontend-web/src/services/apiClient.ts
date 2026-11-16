export interface HealthResponse {
  status: 'healthy' | 'degraded';
  timestamp: string;
  uptime_seconds: number;
  platform_version: string;
  channels_available: string[];
  stream_replay: {
    active: boolean;
    events_in_buffer: number;
    total_events_replayed: number;
  };
  history_store_entities: number;
  sla_target_ms: number;
  banking_coverage_note?: string;
}

export interface VersionResponse {
  status: string;
  platform_version: string;
  api_version: string;
  git_commit: string;
  environment: string;
  champions: Record<string, {
    version: string;
    model_name: string;
    algorithm: string;
    trained_date: string;
    pr_auc: number;
  }>;
}

export interface ShapFactor {
  feature: string;
  shap_value: number;
  direction: 'pushes risk up' | 'pushes risk down';
  human_readable: string;
}

export interface ScoreRequest {
  channel: string;
  payload: Record<string, unknown>;
  top_k?: number;
  update_history?: boolean;
  use_group_fairness?: boolean;
}

export interface ScoreResponse {
  status: string;
  channel: string;
  data_source: string;
  risk_score: number;
  raw_score: number;
  band: 'Low' | 'Medium' | 'High';
  action: string;
  fairness_policy?: string;
  top_reasons: string[];
  factors: ShapFactor[];
  counterfactual?: Record<string, unknown> | null;
  latency_ms: number;
  thresholds_applied: {
    low_threshold: number;
    high_threshold: number;
  };
  timestamp: string;
}

export interface BatchScoreRequest {
  channel: string;
  transactions: Record<string, unknown>[];
  top_k?: number;
}

export interface BatchScoreResponse {
  status: string;
  channel: string;
  total_scored: number;
  high_risk_count: number;
  medium_risk_count: number;
  low_risk_count: number;
  mean_risk_score: number;
  total_duration_ms: number;
  results: Array<{
    risk_score: number;
    band: string;
    action: string;
    top_reasons: string[];
  }>;
}

export interface TransactionRecord {
  id?: string;
  step?: number;
  type?: string;
  amount: number;
  nameOrig?: string;
  nameDest?: string;
  risk_score: number;
  risk_band?: string;
  band?: string;
  action: string;
  channel: string;
  isFraud?: number;
  is_fraud?: number;
  timestamp?: string;
  time_label?: string;
  latency_ms?: number;
  top_reasons?: string[];
  payload_summary?: Record<string, unknown>;
}

export interface TransactionsLedgerResponse {
  status: string;
  source?: string;
  items: TransactionRecord[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface CaseRecord {
  case_id: string;
  created_at: string;
  channel: string;
  amount: number;
  risk_score: number;
  risk_band: 'High' | 'Medium';
  recommended_action: string;
  status: 'OPEN' | 'CONFIRMED_FRAUD' | 'FALSE_POSITIVE';
  user_id: string;
  counterparty_id: string;
  reasons_json?: string;
  raw_payload_json?: string;
}

export interface CaseAuditLog {
  log_id: number;
  case_id: string;
  timestamp: string;
  analyst_name: string;
  old_status: string;
  new_status: string;
  notes: string;
}

export interface StreamStats {
  total_events: number;
  events_in_buffer: number;
  events_per_sec: number;
  loss_averted_inr: number;
  total_volume_inr: number;
  alert_rate_pct: number;
  latency_p50_ms: number;
  latency_p95_ms: number;
}

export interface StreamReplayResponse {
  status: string;
  mode: string;
  items: TransactionRecord[];
  stats: StreamStats;
}

export interface FairnessBafResponse {
  status: string;
  dataset: string;
  demographic_attribute: string;
  groups: {
    younger: string;
    older: string;
  };
  disparity_at_1pct_target_fpr: {
    younger_fpr_pct: number;
    older_fpr_pct: number;
    disparity_ratio: number;
    younger_false_alarms_per_100k: number;
    older_false_alarms_per_100k: number;
    younger_recall_pct: number;
    older_recall_pct: number;
  };
  disparity_at_5pct_target_fpr: {
    younger_fpr_pct: number;
    older_fpr_pct: number;
    disparity_ratio: number;
    younger_false_alarms_per_100k: number;
    older_false_alarms_per_100k: number;
    overall_recall_pct: number;
  };
  equalized_fpr_mitigation: {
    intervention: string;
    younger_threshold: number;
    older_threshold: number;
    mitigated_younger_fpr_pct: number;
    mitigated_older_fpr_pct: number;
    mitigated_disparity_ratio: number;
    mitigated_overall_recall_pct: number;
    recall_tradeoff_delta_pct: number;
    cost_documentation: string;
  };
}

export interface ReasonCode {
  code: string;
  title: string;
  description: string;
  category: string;
}


class ApiClient {
  private baseUrl: string;
  private apiKey: string | null = null;

  constructor(baseUrl: string = '') {
    this.baseUrl = baseUrl;
    if (typeof window !== 'undefined') {
      this.apiKey = localStorage.getItem('fraudshield-api-key') || null;
    }
  }

  public setApiKey(key: string | null) {
    this.apiKey = key;
    if (typeof window !== 'undefined') {
      if (key) {
        localStorage.setItem('fraudshield-api-key', key);
      } else {
        localStorage.removeItem('fraudshield-api-key');
      }
    }
  }

  private async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...((options.headers as Record<string, string>) || {}),
    };

    if (this.apiKey) {
      headers['X-API-Key'] = this.apiKey;
    }

    const res = await fetch(`${this.baseUrl}${endpoint}`, {
      ...options,
      headers,
    });

    if (!res.ok) {
      let errorMsg = `API request failed with status ${res.status}`;
      try {
        const errorData = await res.json();
        if (errorData.message) {
          errorMsg = errorData.message;
        } else if (errorData.detail) {
          errorMsg = typeof errorData.detail === 'string' ? errorData.detail : JSON.stringify(errorData.detail);
        }
      } catch {
        // fallback
      }
      throw new Error(errorMsg);
    }

    return res.json() as Promise<T>;
  }

  public health(): Promise<HealthResponse> {
    return this.request<HealthResponse>('/health');
  }

  public version(): Promise<VersionResponse> {
    return this.request<VersionResponse>('/version');
  }

  public getOverview(): Promise<Record<string, unknown>> {
    return this.request<Record<string, unknown>>('/api/overview');
  }

  public getTransactions(params: {
    page?: number;
    page_size?: number;
    channel?: string;
    band?: string;
    search?: string;
    sort_by?: string;
    sort_order?: string;
    source?: 'sample' | 'replay';
  } = {}): Promise<TransactionsLedgerResponse> {
    const query = new URLSearchParams();
    if (params.page) query.set('page', params.page.toString());
    if (params.page_size) query.set('page_size', params.page_size.toString());
    if (params.channel) query.set('channel', params.channel);
    if (params.band) query.set('band', params.band);
    if (params.search) query.set('search', params.search);
    if (params.sort_by) query.set('sort_by', params.sort_by);
    if (params.sort_order) query.set('sort_order', params.sort_order);
    if (params.source) query.set('source', params.source);

    return this.request<TransactionsLedgerResponse>(`/api/transactions?${query.toString()}`);
  }

  public score(req: ScoreRequest): Promise<ScoreResponse> {
    return this.request<ScoreResponse>('/api/score', {
      method: 'POST',
      body: JSON.stringify(req),
    });
  }

  public batchScore(req: BatchScoreRequest): Promise<BatchScoreResponse> {
    return this.request<BatchScoreResponse>('/api/batch-score', {
      method: 'POST',
      body: JSON.stringify(req),
    });
  }

  public getCases(params: { status?: string; channel?: string; band?: string; limit?: number } = {}): Promise<{ status: string; count: number; cases: CaseRecord[] }> {
    const query = new URLSearchParams();
    if (params.status) query.set('status', params.status);
    if (params.channel) query.set('channel', params.channel);
    if (params.band) query.set('band', params.band);
    if (params.limit) query.set('limit', params.limit.toString());
    return this.request<{ status: string; count: number; cases: CaseRecord[] }>(`/api/cases?${query.toString()}`);
  }

  public getCase(caseId: string): Promise<{ status: string; case: CaseRecord; audit_trail: CaseAuditLog[] }> {
    return this.request<{ status: string; case: CaseRecord; audit_trail: CaseAuditLog[] }>(`/api/cases/${caseId}`);
  }

  public updateCaseStatus(caseId: string, status: string, notes: string = '', analystName: string = 'Analyst'): Promise<{ status: string; message: string }> {
    return this.request<{ status: string; message: string }>(`/api/cases/${caseId}/status`, {
      method: 'PUT',
      body: JSON.stringify({ status, notes, analyst_name: analystName }),
    });
  }

  public getStream(params: { limit?: number; channel?: string; band?: string } = {}): Promise<StreamReplayResponse> {
    const query = new URLSearchParams();
    if (params.limit) query.set('limit', params.limit.toString());
    if (params.channel) query.set('channel', params.channel);
    if (params.band) query.set('band', params.band);
    return this.request<StreamReplayResponse>(`/api/stream?${query.toString()}`);
  }

  public streamTick(): Promise<{ status: string; decision: TransactionRecord }> {
    return this.request<{ status: string; decision: TransactionRecord }>('/api/stream/tick', {
      method: 'POST',
    });
  }

  public getStreamStats(): Promise<{ status: string; stats: StreamStats }> {
    return this.request<{ status: string; stats: StreamStats }>('/api/stream/stats');
  }

  public getFairnessBaf(): Promise<FairnessBafResponse> {
    return this.request<FairnessBafResponse>('/api/fairness/baf');
  }

  public getReasonCodes(): Promise<{ status: string; catalog: ReasonCode[] }> {
    return this.request<{ status: string; catalog: ReasonCode[] }>('/api/reason-codes');
  }
}

export const apiClient = new ApiClient();

