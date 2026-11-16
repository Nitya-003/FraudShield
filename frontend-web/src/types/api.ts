export interface MetricCardData {
  label: string;
  value: number;
  format: 'currency' | 'pct' | 'decimal' | 'integer';
  currency_symbol?: string;
  currency_code?: string;
  unit?: string;
  sub_label?: string;
  definition?: string;
}

export interface ActiveChannelInfo {
  is_all: boolean;
  channel_key: string;
  channel_name: string;
  benchmark: string;
  data_source: string;
  currency_symbol: string;
  currency_code: string;
  currency_name: string;
  test_rows: number;
}

export interface ChannelOverviewRow {
  channel_key: string;
  channel_name: string;
  benchmark: string;
  data_source: string;
  pr_auc: number;
  recall_pct: number;
  false_alarms_per_100k: number;
  alerts_per_day: number;
  test_rows: number;
  currency_code: string;
  currency_symbol: string;
  currency_name: string;
  fraud_caught_value: number;
  fraud_caught_count: number;
  threshold_low: number;
  threshold_high: number;
}

export interface OverviewKPIs {
  status: string;
  total_volume_inr: number;
  total_transactions: number;
  total_alerts: number;
  alert_rate_pct: number;
  loss_averted_inr: number;
  fraud_recall_pct?: number;
  false_alarms_per_100k?: number;
  daily_alerts?: number;
  is_natural_prevalence?: boolean;
  channels_count: number;
  channels: string[];
  latency_p50_ms: number;
  latency_p95_ms: number;
  timestamp: string;
  headline_metric?: MetricCardData;
  supporting_metrics?: MetricCardData[];
  active_channel_info?: ActiveChannelInfo;
  per_channel_table?: ChannelOverviewRow[];
}

export interface RiskBandsSummary {
  status: string;
  counts: Record<string, number>;
  actions?: Record<string, string>;
  thresholds_per_channel: Record<string, { low: number; high: number }>;
}

export interface DailySeriesResponse {
  status: string;
  channel?: string;
  is_sample?: boolean;
  data_source?: string;
  series: DailySeriesPoint[];
}

export interface DailySeriesPoint {
  step: number;
  hour_label: string;
  tx_count: number;
  total_volume: number;
  high_risk_count: number;
  medium_risk_count: number;
}

export interface TransactionItem {
  step: number;
  type: string;
  amount: number;
  nameOrig?: string;
  nameDest?: string;
  risk_score: number;
  risk_band: 'Low' | 'Medium' | 'High';
  action: string;
  isFraud?: number;
  channel: string;
  reason?: string;
  top_reasons?: string | string[];
}

export interface PagedTransactions {
  status: string;
  items: TransactionItem[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface CaseItem {
  case_id: string;
  channel: string;
  entity_id: string;
  counterparty_id?: string;
  amount: number;
  risk_score: number;
  risk_band: string;
  timestamp: string;
  status: 'OPEN' | 'CONFIRMED_FRAUD' | 'FALSE_POSITIVE';
  analyst_notes?: string;
  updated_at?: string;
}

export interface CaseAuditLog {
  log_id: number;
  case_id: string;
  action: string;
  analyst_name: string;
  notes?: string;
  timestamp: string;
}

export interface ChampionModel {
  channel: string;
  display_name: string;
  model_name: string;
  algorithm: string;
  version: string;
  trained_date: string;
  pr_auc: number;
  roc_auc: number;
  recall: number;
  precision?: number;
  brier_score?: number;
  model_file?: string;
  dataset_name?: string;
  false_alarms_per_100k?: number;
  key_findings?: Record<string, string>;
  bootstrap_ci_95?: {
    pr_auc?: [number, number];
  };
  thresholds?: {
    low_risk_approve: number;
    high_risk_decline: number;
  };
}

export interface ModelRegistryData {
  status: string;
  registry_version: string;
  last_updated: string;
  champions: Record<string, ChampionModel>;
  cross_channel_comparison: Record<string, unknown>;
}

export interface ScoreResponse {
  status: string;
  channel: string;
  data_source: string;
  risk_score: number;
  raw_score: number;
  band: 'Low' | 'Medium' | 'High';
  action: string;
  top_reasons: string[];
  factors: Array<{ factor: string; value: string; shap_value: number; direction: string }>;
  counterfactual?: string;
  latency_ms: number;
  thresholds_applied: { low_threshold: number; high_threshold: number };
  timestamp: string;
}

export interface AttackPreset {
  id: string;
  name: string;
  channel: string;
  description: string;
  payload: Record<string, unknown>;
}

export interface AttackSimulationResult {
  channel: string;
  attack_type: string;
  params: Record<string, unknown>;
  base_score: number;
  base_band: string;
  base_action: string;
  peak_attack_score: number;
  mean_attack_score: number;
  verdict: 'BLOCKED' | 'STEP_UP_CHALLENGED' | 'EVADED';
  verdict_color: string;
  verdict_summary: string;
  mechanics_explanation: string;
  transfers: Array<{
    transfer_index: number;
    step: number;
    amount: number;
    amount_fraction: number;
    remaining_balance: number;
    risk_score: number;
    band: string;
    action: string;
    primary_signal: string;
  }>;
  thresholds: { low: number; high: number };
}

export interface StepUpMessageData {
  channel: string;
  otp_code: string;
  amount: number;
  amount_formatted: string;
  counterparty: string;
  is_collect_request: boolean;
  has_scam_warning: boolean;
  reason_plain_english: string;
  reason_plain_hindi: string;
  english_message: string;
  hindi_message: string;
  timestamp: string;
}

export interface StreamItem {
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
  payload_summary?: Record<string, unknown>;
}

export interface StreamReplayResponse {
  status: string;
  mode: string;
  items: StreamItem[];
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
}
