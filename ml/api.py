from __future__ import annotations

import functools
import json
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("fraudshield.api")

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from ml.attack_lab import (
    get_attack_history,
    get_attack_presets,
    simulate_attack_scenario,
)
from ml.case_management import (
    export_feedback_dataset,
    feed_confirmed_labels_to_feedback_loop,
    get_analyst_precision_metrics,
    get_case_audit_logs,
    get_case_by_id,
    get_cases,
    init_db,
    update_case_status,
)
from ml.case_report import generate_case_report_html
from ml.config import (
    BANKING_COVERAGE_NOTE,
    CHANNELS,
    METRICS_FILE,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
)
from ml.entity_view import (
    DEFAULT_MULE_HUB,
    PRESET_ENTITIES,
    build_entity_network_graph,
    build_flow_breakdown_chart,
    get_entity_profile,
    load_network_dataset,
)
from ml.history_store import get_history_store, reset_history_store
from ml.metrics_helper import (
    _resolve_key,
    get_calibrated_thresholds,
    get_champion_metrics,
    get_economics_summary,
)
from ml.reason_codes import get_reason_catalog, map_shap_to_reason_codes
from ml.router import get_channel_info, get_channel_registry, predict, resolve_channel
from ml.step_up import generate_step_up_message
from ml.stream_replay import get_stream_replayer

SERVER_START_TIME = time.time()

app = FastAPI(
    title="FraudShield Multi-Channel Fraud Detection API",
    description="Real-Time Decisioning, Graph Intelligence, Case Management & Adversarial Simulation API",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VALID_API_KEYS = {os.environ.get("FRAUDSHIELD_API_KEY", "fs_live_test_key_2026")}


def verify_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> Optional[str]:
   
    token = x_api_key
    if not token and authorization and authorization.startswith("Bearer "):
        token = authorization.split("Bearer ", 1)[1].strip()

    if token:
        if token not in VALID_API_KEYS:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API Key. Please provide a valid 'X-API-Key' header.",
            )
        return token

    if os.environ.get("FRAUDSHIELD_REQUIRE_AUTH", "false").lower() == "true":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please supply a valid 'X-API-Key' header.",
        )
    return None


_RATE_LIMIT_BUCKETS: dict[str, list[float]] = {}
RATE_LIMIT_MAX_PER_MINUTE = int(os.environ.get("FRAUDSHIELD_RATE_LIMIT", "1200"))


@app.middleware("http")
async def security_and_logging_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "127.0.0.1"
    now = time.time()
    timestamps = _RATE_LIMIT_BUCKETS.get(client_ip, [])
    timestamps = [t for t in timestamps if now - t < 60.0]

    is_exempt = request.url.path in ("/health", "/docs", "/openapi.json", "/redoc")
    if not is_exempt and len(timestamps) >= RATE_LIMIT_MAX_PER_MINUTE:
        _RATE_LIMIT_BUCKETS[client_ip] = timestamps
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "status": "error",
                "error_type": "rate_limit_exceeded",
                "message": f"Rate limit exceeded. Maximum {RATE_LIMIT_MAX_PER_MINUTE} requests per minute allowed.",
                "retry_after_sec": 60,
            },
            headers={"Retry-After": "60"},
        )

    if not is_exempt:
        timestamps.append(now)
        _RATE_LIMIT_BUCKETS[client_ip] = timestamps

    t0 = time.perf_counter()
    response = await call_next(request)
    proc_ms = round((time.perf_counter() - t0) * 1000, 2)

    response.headers["X-Process-Time-Ms"] = str(proc_ms)
    response.headers["X-Model-Version"] = "2.0.0"

    logger.info("%s %s -> %d (%.2f ms)", request.method, request.url.path, response.status_code, proc_ms)
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for err in exc.errors():
        field_path = " -> ".join(str(loc) for loc in err.get("loc", []))
        msg = err.get("msg", "Invalid value")
        errors.append({
            "field": field_path,
            "issue": msg,
            "type": err.get("type", "value_error"),
        })
    readable = "; ".join(f"Field '{e['field']}': {e['issue']}" for e in errors)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "error",
            "error_type": "validation_error",
            "message": f"Input validation failed: {readable}",
            "details": errors,
        },
    )


_CACHE_STORE: dict[str, tuple[float, Any]] = {}


def ttl_cached(seconds: float = 60.0):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
            now = time.time()
            if key in _CACHE_STORE:
                ts, val = _CACHE_STORE[key]
                if now - ts < seconds:
                    return val
            result = func(*args, **kwargs)
            _CACHE_STORE[key] = (now, result)
            return result
        return wrapper
    return decorator


def load_metrics_json() -> dict[str, Any]:
    if METRICS_FILE.exists():
        with open(METRICS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_champion_registry_json() -> dict[str, Any]:
    reg_path = BASE_DIR / "models" / "champion_registry.json"
    if reg_path.exists():
        with open(reg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

class ScoreRequest(BaseModel):
    channel: str = Field(..., description="Payment channel: card, transfer_wallet, account, upi")
    payload: dict[str, Any] = Field(..., description="Channel-specific transaction attributes")
    top_k: int = Field(3, ge=1, le=10, description="Top TreeSHAP explainability drivers")
    update_history: bool = Field(True, description="Whether to update stateful account history")
    use_group_fairness: bool = Field(False, description="Whether to apply equalized false alarm rate thresholds for demographic groups (BAF account channel)")

    @field_validator("channel")
    @classmethod
    def validate_channel(cls, v: str) -> str:
        try:
            return resolve_channel(v)
        except ValueError as err:
            raise ValueError(str(err))

    @field_validator("payload")
    @classmethod
    def validate_payload(cls, v: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(v, dict) or not v:
            raise ValueError("Payload must be a non-empty dictionary.")
        for amt_k in ["amount", "Amount"]:
            if amt_k in v:
                try:
                    val = float(v[amt_k])
                    if val < 0.0:
                        raise ValueError(f"{amt_k} cannot be negative: {val}")
                except (TypeError, ValueError) as e:
                    raise ValueError(f"Invalid numeric value for {amt_k}: {e}")
        return v


class ScoreResponse(BaseModel):
    status: str = "success"
    channel: str
    data_source: str
    risk_score: float
    raw_score: float
    band: str
    action: str
    top_reasons: list[str]
    factors: list[dict[str, Any]]
    counterfactual: Optional[str] = None
    latency_ms: float
    thresholds_applied: dict[str, float]
    timestamp: str


class BatchScoreRequest(BaseModel):
    channel: str
    transactions: list[dict[str, Any]]
    top_k: int = Field(2, ge=1, le=5)


class BatchScoreResponse(BaseModel):
    status: str = "success"
    channel: str
    total_scored: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    mean_risk_score: float
    total_duration_ms: float
    results: list[dict[str, Any]]


class CaseStatusUpdateRequest(BaseModel):
    status: str = Field(..., description="New status: OPEN, CONFIRMED_FRAUD, FALSE_POSITIVE")
    notes: str = Field("", description="Investigator notes and rationale")
    analyst_name: str = Field("Lead Analyst", description="Name of investigator")

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        allowed = {"OPEN", "CONFIRMED_FRAUD", "FALSE_POSITIVE"}
        if v not in allowed:
            raise ValueError(f"Status must be one of {allowed}, got '{v}'")
        return v


class AttackSimulateRequest(BaseModel):
    channel: str
    base_payload: dict[str, Any]
    attack_type: str = Field("split_transfers", description="split_transfers, change_timing, reduce_amount, add_hop, combined")
    params: dict[str, Any] = Field(default_factory=dict)


class StepUpPreviewRequest(BaseModel):
    channel: str
    payload: dict[str, Any]
    otp_code: Optional[str] = None

@app.get("/health", tags=["System"])
def health_check() -> dict[str, Any]:
    replayer = get_stream_replayer()
    store = get_history_store()
    uptime = round(time.time() - SERVER_START_TIME, 1)

    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "uptime_seconds": uptime,
        "platform_version": "2.0.0",
        "channels_available": list(CHANNELS.keys()),
        "stream_replay": {
            "active": replayer._running,
            "events_in_buffer": len(replayer.store._buffer),
            "total_events_replayed": replayer.store._total_added,
        },
        "history_store_entities": len(store.orig_tx_log),
        "sla_target_ms": 15.0,
        "banking_coverage_note": BANKING_COVERAGE_NOTE,
    }


@app.get("/version", tags=["System"])
def get_version_info() -> dict[str, Any]:
    registry = load_champion_registry_json()
    champs = registry.get("champions", {})
    versions = {}
    for ch, c in champs.items():
        versions[ch] = {
            "version": c.get("version", "2.0.0"),
            "model_name": c.get("model_name", ""),
            "algorithm": c.get("algorithm", ""),
            "trained_date": c.get("trained_date", ""),
            "pr_auc": c.get("pr_auc", 0.0),
        }

    return {
        "status": "success",
        "platform_version": "2.0.0",
        "api_version": "2.0.0",
        "git_commit": "main",
        "environment": "production",
        "champions": versions,
    }


@app.get("/api/auth/verify", tags=["System"])
def verify_auth_token(api_key: Optional[str] = Depends(verify_api_key)) -> dict[str, Any]:
    return {
        "status": "success",
        "authenticated": api_key is not None,
        "key_provided": bool(api_key),
        "message": "API key is valid" if api_key else "Public access mode (unauthenticated)",
    }


@app.get("/thresholds", tags=["Configuration"])
def get_thresholds_endpoint() -> dict[str, Any]:
    thresholds = {}
    for ch in CHANNELS.keys():
        low, high = get_calibrated_thresholds(ch)
        thresholds[ch] = {
            "low_risk_auto_approve": low,
            "high_risk_decline": high,
            "medium_risk_step_up": [low, high],
        }
    return {
        "status": "success",
        "thresholds": thresholds,
    }


@app.post("/history/reset", tags=["System"])
def reset_history(api_key: Optional[str] = Depends(verify_api_key)) -> dict[str, str]:
    reset_history_store()
    return {"status": "success", "message": "Account history store cleared."}

@app.get("/api/stream", tags=["Real Stream Replay"])
async def stream_decisions(
    request: Request,
    mode: str = Query("poll", description="'poll' for JSON snapshot, 'sse' for Server-Sent Events"),
    limit: int = Query(50, ge=1, le=500),
    channel: Optional[str] = None,
    band: Optional[str] = None,
):
    
    replayer = get_stream_replayer()
    accept_header = request.headers.get("accept", "")

    if mode.lower() == "sse" or "text/event-stream" in accept_header:
        return StreamingResponse(
            replayer.sse_event_generator(interval_sec=1.5),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    items = replayer.store.get_recent(limit=limit, channel=channel, band=band)
    stats = replayer.store.get_stats()
    return {
        "status": "success",
        "mode": "test_set_replay",
        "items": items,
        "stats": stats,
    }


@app.post("/api/stream/tick", tags=["Real Stream Replay"])
def stream_tick():
    replayer = get_stream_replayer()
    decision = replayer.tick()
    return {"status": "success", "decision": decision}


@app.get("/api/stream/stats", tags=["Real Stream Replay"])
def stream_stats():
    replayer = get_stream_replayer()
    return {"status": "success", "stats": replayer.store.get_stats()}


@app.get("/api/fairness/baf", tags=["Fairness & Demographics"])
def get_baf_fairness_analysis() -> dict[str, Any]:
 
    return {
        "status": "success",
        "dataset": "Bank Account Fraud (BAF - NeurIPS 2022)",
        "demographic_attribute": "customer_age",
        "groups": {
            "younger": "Age < 50",
            "older": "Age >= 50",
        },
        "disparity_at_1pct_target_fpr": {
            "younger_fpr_pct": 0.79,
            "older_fpr_pct": 2.42,
            "disparity_ratio": 3.06,
            "younger_false_alarms_per_100k": 781.7,
            "older_false_alarms_per_100k": 2344.7,
            "younger_recall_pct": 21.98,
            "older_recall_pct": 35.17,
        },
        "disparity_at_5pct_target_fpr": {
            "younger_fpr_pct": 4.21,
            "older_fpr_pct": 10.39,
            "disparity_ratio": 2.47,
            "younger_false_alarms_per_100k": 4155.6,
            "older_false_alarms_per_100k": 10049.7,
            "overall_recall_pct": 56.72,
        },
        "equalized_fpr_mitigation": {
            "intervention": "Per-Group Calibrated Thresholds",
            "younger_threshold": 0.0376,
            "older_threshold": 0.0863,
            "mitigated_younger_fpr_pct": 5.0,
            "mitigated_older_fpr_pct": 5.0,
            "mitigated_disparity_ratio": 1.0,
            "mitigated_overall_recall_pct": 53.29,
            "recall_tradeoff_delta_pct": -3.43,
            "cost_documentation": (
                "Equalizing false-alarm rates eliminates disparate impact (disparity ratio 2.47x -> 1.0x), "
                "at the cost of -3.43% overall fraud recall."
            ),
        },
    }


@app.get("/api/reason-codes", tags=["Configuration"])
def get_reason_codes_endpoint() -> dict[str, Any]:
    return {
        "status": "success",
        "catalog": get_reason_catalog(),
    }


CHANNEL_METADATA: dict[str, dict[str, Any]] = {
    "card": {
        "name": "Card Payments",
        "benchmark": "ULB European Credit Card Fraud",
        "data_source": "Real, anonymised",
        "currency_code": "EUR",
        "currency_symbol": "€",
        "currency_name": "EUR (€)",
        "test_rows": 42721,
        "alerts_per_day": 22.5,
    },
    "transfer_wallet": {
        "name": "Online Transfers & Wallets",
        "benchmark": "PaySim Mobile Money",
        "data_source": "Simulated (agent-based)",
        "currency_code": "INR",
        "currency_symbol": "₹",
        "currency_name": "INR (₹)",
        "test_rows": 918617,
        "alerts_per_day": 264.8,
    },
    "account": {
        "name": "Bank Accounts",
        "benchmark": "Bank Account Fraud (BAF NeurIPS 2022)",
        "data_source": "Privacy-preserving synthetic",
        "currency_code": "GBP",
        "currency_symbol": "£",
        "currency_name": "GBP (£ limit)",
        "test_rows": 96843,
        "alerts_per_day": 12.2,
    },
    "upi": {
        "name": "UPI Transactions",
        "benchmark": "Synthetic Indian UPI Simulator",
        "data_source": "Simulated by us",
        "currency_code": "INR",
        "currency_symbol": "₹",
        "currency_name": "INR (₹)",
        "test_rows": 145716,
        "alerts_per_day": 55.8,
    },
}


@app.get("/api/overview", tags=["Analytics"])
def get_overview_kpis(channel: Optional[str] = None) -> dict[str, Any]:
    metrics = load_metrics_json()
    api_lat = metrics.get("api_latency", {})
    econ = metrics.get("economics_natural_test", {})

    total_tx = econ.get("total_test_transactions", 918617)
    bands_data = econ.get("bands", {})
    low_inr = bands_data.get("Low (< 0.45)", {}).get("total_inr", 148781039616.0)
    med_inr = bands_data.get("Medium (0.45-0.90)", {}).get("total_inr", 892562.88)
    high_inr = bands_data.get("High (> 0.90)", {}).get("total_inr", 6327984128.0)
    total_vol = low_inr + med_inr + high_inr

    med_tx = bands_data.get("Medium (0.45-0.90)", {}).get("transactions", 9)
    high_tx = bands_data.get("High (> 0.90)", {}).get("transactions", 4007)
    alerts_count = med_tx + high_tx
    alert_rate = round((alerts_count / max(total_tx, 1)) * 100.0, 3)

    loss_averted = econ.get("fraud_inr_caught", 6322647552.0)
    recall_pct = econ.get("fraud_recall_pct", 99.83)
    false_alarms_per_100k = econ.get("false_alarms_per_100k", 1.2)
    daily_alerts = bands_data.get("High (> 0.90)", {}).get("daily_volume", 264.2) + bands_data.get("Medium (0.45-0.90)", {}).get("daily_volume", 0.6)

    per_channel_table = []
    total_all_test_rows = 0
    total_all_frauds_caught = 0

    for ch_key in ["card", "transfer_wallet", "account", "upi"]:
        info = CHANNEL_METADATA.get(ch_key, {})
        champ = get_champion_metrics(ch_key)
        ch_econ = get_economics_summary(ch_key)

        pr = float(champ.get("pr_auc", 0.0))
        rec = round(float(champ.get("recall", 0.0)) * 100.0, 2)
        far = round(float(champ.get("false_alarms_per_100k", 0.0)), 1)
        apd = float(info.get("alerts_per_day", 0.0))
        rows_cnt = int(info.get("test_rows", 0))

        caught_val = float(ch_econ.get("fraud_inr_caught", ch_econ.get("fraud_credit_limit_caught_inr", 0.0)))
        caught_cnt = int(champ.get("tp", ch_econ.get("fraud_count_caught", 0)))

        total_all_test_rows += rows_cnt
        total_all_frauds_caught += caught_cnt

        low_t, high_t = get_calibrated_thresholds(ch_key)

        per_channel_table.append({
            "channel_key": ch_key,
            "channel_name": info.get("name", ch_key),
            "benchmark": info.get("benchmark", ""),
            "data_source": info.get("data_source", ""),
            "pr_auc": pr,
            "recall_pct": rec,
            "false_alarms_per_100k": far,
            "alerts_per_day": apd,
            "test_rows": rows_cnt,
            "currency_code": info.get("currency_code", "INR"),
            "currency_symbol": info.get("currency_symbol", "₹"),
            "currency_name": info.get("currency_name", "INR (₹)"),
            "fraud_caught_value": caught_val,
            "fraud_caught_count": caught_cnt,
            "threshold_low": low_t,
            "threshold_high": high_t,
        })

    norm_ch = None
    if channel and channel.upper() != "ALL":
        norm_ch = _resolve_key(channel)

    if norm_ch and norm_ch in CHANNEL_METADATA:
        meta = CHANNEL_METADATA[norm_ch]
        champ = get_champion_metrics(norm_ch)
        ch_econ = get_economics_summary(norm_ch)
        low_t, high_t = get_calibrated_thresholds(norm_ch)

        ch_val = float(ch_econ.get("fraud_inr_caught", ch_econ.get("fraud_credit_limit_caught_inr", 0.0)))
        ch_recall = round(float(champ.get("recall", 0.0)) * 100.0, 1)
        ch_far = round(float(champ.get("false_alarms_per_100k", 0.0)), 1)
        ch_alerts_day = float(meta.get("alerts_per_day", 0.0))

        headline_metric = {
            "label": "Fraud Value Caught",
            "value": ch_val,
            "format": "currency",
            "currency_symbol": meta["currency_symbol"],
            "currency_code": meta["currency_code"],
            "unit": meta["currency_symbol"],
            "sub_label": f"Monetary value in {meta['currency_name']} on held-out test split.",
            "definition": f"Estimated monetary fraud value averted through hard-decline threshold interventions on the {meta['name']} held-out test split.",
        }

        supporting_metrics = [
            {
                "key": "recall",
                "label": "Fraud Detection Recall",
                "value": ch_recall,
                "format": "pct",
                "sub_label": f"Threshold score > {high_t:.2f}",
                "definition": f"Proportion of actual fraudulent payments intercepted at calibrated operating threshold ({high_t:.2f}).",
            },
            {
                "key": "false_alarms",
                "label": "False Alarms / 100k",
                "value": ch_far,
                "format": "decimal",
                "sub_label": f"Operating FPR: {(ch_far / 1000.0):.3f}%",
                "definition": "Number of legitimate transactions incorrectly flagged or challenged per 100,000 processed transactions.",
            },
            {
                "key": "daily_alerts",
                "label": "High-Risk Alerts / Day",
                "value": ch_alerts_day,
                "format": "decimal",
                "sub_label": "Projected natural velocity",
                "definition": "Projected daily volume of high-risk escalations requiring fraud specialist review under natural traffic volume.",
            },
        ]

        active_channel_info = {
            "is_all": False,
            "channel_key": norm_ch,
            "channel_name": meta["name"],
            "benchmark": meta["benchmark"],
            "data_source": meta["data_source"],
            "currency_symbol": meta["currency_symbol"],
            "currency_code": meta["currency_code"],
            "currency_name": meta["currency_name"],
            "test_rows": meta["test_rows"],
        }
    else:
        headline_metric = {
            "label": "Total Frauds Caught",
            "value": total_all_frauds_caught,
            "format": "integer",
            "currency_symbol": "",
            "currency_code": "INCIDENTS",
            "unit": "incidents stopped",
            "sub_label": "Currencies differ per channel (EUR, INR, GBP limit); displaying total incidents stopped.",
            "definition": "Total fraudulent payments, unauthorized transfers, fraudulent account applications, and scam bursts intercepted across all 4 benchmark test splits.",
        }

        supporting_metrics = [
            {
                "key": "eval_tx",
                "label": "Evaluated Transactions",
                "value": total_all_test_rows,
                "format": "integer",
                "sub_label": "4 held-out benchmarks",
                "definition": "Total chronological held-out test split transactions evaluated across ULB Card, PaySim, BAF, and UPI benchmarks.",
            },
            {
                "key": "models_count",
                "label": "Active Champion Models",
                "value": 4,
                "format": "integer",
                "sub_label": "Independently calibrated",
                "definition": "Production champions (LightGBM, CatBoost, Stacked Ensemble) independently trained and Platt-calibrated for their respective payment rail.",
            },
            {
                "key": "routing_modes",
                "label": "Decision Routing Stages",
                "value": 3,
                "format": "integer",
                "sub_label": "Approve / Step-up / Decline",
                "definition": "Deterministic 3-tier policy partition: Auto-Approve (Low Risk), Step-Up Challenge (Medium Risk), and Hard Decline (High Risk).",
            },
        ]

        active_channel_info = {
            "is_all": True,
            "channel_key": "ALL",
            "channel_name": "All Channels",
            "benchmark": "Composite Multi-Channel Evaluation",
            "data_source": "Card (ULB), PaySim Wallets, BAF Accounts & Synthetic UPI",
            "currency_symbol": "MULTI",
            "currency_code": "MULTI",
            "currency_name": "Multi-Currency",
            "test_rows": total_all_test_rows,
        }

    return {
        "status": "success",
        "total_volume_inr": total_vol,
        "total_transactions": total_tx,
        "total_alerts": alerts_count,
        "alert_rate_pct": alert_rate,
        "loss_averted_inr": loss_averted,
        "fraud_recall_pct": recall_pct,
        "false_alarms_per_100k": false_alarms_per_100k,
        "daily_alerts": round(daily_alerts, 1),
        "is_natural_prevalence": True,
        "channels_count": len(CHANNELS),
        "channels": list(CHANNELS.keys()),
        "latency_p50_ms": api_lat.get("p50_ms", 7.92),
        "latency_p95_ms": api_lat.get("p95_ms", 14.36),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "headline_metric": headline_metric,
        "supporting_metrics": supporting_metrics,
        "active_channel_info": active_channel_info,
        "per_channel_table": per_channel_table,
    }


@app.get("/api/risk-bands", tags=["Analytics"])
def get_risk_bands_summary(channel: Optional[str] = None) -> dict[str, Any]:
    metrics = load_metrics_json()
    econ = metrics.get("economics_natural_test", {})
    bands_data = econ.get("bands", {})

    channel_thresholds = {}
    for ch in CHANNELS.keys():
        l_th, h_th = get_calibrated_thresholds(ch)
        channel_thresholds[ch] = {"low": l_th, "high": h_th}

    norm_ch = None
    if channel and channel.upper() != "ALL":
        norm_ch = _resolve_key(channel)

    if norm_ch and norm_ch in CHANNELS:
        ch_econ = get_economics_summary(norm_ch)
        ch_bands = ch_econ.get("bands", {})

        low_cnt = 0
        med_cnt = 0
        high_cnt = 0
        low_action = "Auto-Approve"
        med_action = "Step-Up Challenge"
        high_action = "Block & Alert"

        for b_name, b_val in ch_bands.items():
            cnt = b_val.get("transactions", b_val.get("applications", 0))
            act = b_val.get("action", "")
            if "low" in b_name.lower():
                low_cnt = cnt
                if act:
                    low_action = act
            elif "medium" in b_name.lower():
                med_cnt = cnt
                if act:
                    med_action = act
            elif "high" in b_name.lower():
                high_cnt = cnt
                if act:
                    high_action = act

        counts = {
            "Low": low_cnt,
            "Medium": med_cnt,
            "High": high_cnt,
        }
        actions = {
            "Low": low_action,
            "Medium": med_action,
            "High": high_action,
        }
    else:
        low_cnt = bands_data.get("Low (< 0.45)", {}).get("transactions", 914601)
        med_cnt = bands_data.get("Medium (0.45-0.90)", {}).get("transactions", 9)
        high_cnt = bands_data.get("High (> 0.90)", {}).get("transactions", 4007)

        counts = {
            "Low": low_cnt,
            "Medium": med_cnt,
            "High": high_cnt,
        }
        actions = {
            "Low": "Auto-Approve",
            "Medium": "Step-Up Challenge",
            "High": "Block & Alert",
        }

    return {
        "status": "success",
        "counts": counts,
        "actions": actions,
        "is_natural_prevalence": True,
        "thresholds_per_channel": channel_thresholds,
    }


@app.get("/api/daily-series", tags=["Analytics"])
def get_daily_series(channel: Optional[str] = None) -> dict[str, Any]:
    norm_ch = None
    if channel and channel.upper() != "ALL":
        norm_ch = _resolve_key(channel)

    if norm_ch == "card":
        sample_path = PROCESSED_DATA_DIR / "card_test_sample.parquet"
        if sample_path.exists():
            df = pd.read_parquet(sample_path)
            df["hour"] = (df["Time"] // 3600).astype(int)
            df["day"] = (df["Time"] // 86400).astype(int) + 1
            df["high_risk"] = ((df["V14"] < -4.0) | (df.get("Class", 0) == 1)).astype(int)
            grouped = df.groupby(["day", "hour"]).agg(
                tx_count=("Amount", "count"),
                total_volume=("Amount", "sum"),
                high_risk_count=("high_risk", "sum"),
            ).reset_index().sort_values(["day", "hour"])
            series_data = []
            for _, row in grouped.tail(30).iterrows():
                series_data.append({
                    "step": int(row["hour"]),
                    "hour_label": f"Day {int(row['day'])}, {int(row['hour']) % 24:02d}:00",
                    "tx_count": int(row["tx_count"]),
                    "total_volume": round(float(row["total_volume"]), 2),
                    "high_risk_count": int(row["high_risk_count"]),
                    "medium_risk_count": 0,
                })
            return {
                "status": "success",
                "channel": "card",
                "is_sample": True,
                "data_source": "ULB European Cardholders Benchmark (sampled 5,000 txns, 2-day continuous evaluation)",
                "series": series_data,
            }

    if norm_ch == "upi":
        sample_path = PROCESSED_DATA_DIR / "upi_test_sample.parquet"
        if sample_path.exists():
            df = pd.read_parquet(sample_path)
            grouped = df.groupby(["day", "hour"]).agg(
                tx_count=("amount", "count"),
                total_volume=("amount", "sum"),
                high_risk_count=("is_first_time_payee", lambda s: int((s == 1).sum())),
            ).reset_index().sort_values(["day", "hour"])
            series_data = []
            for _, row in grouped.tail(30).iterrows():
                series_data.append({
                    "step": int(row["hour"]),
                    "hour_label": f"Day {int(row['day'])}, {int(row['hour']):02d}:00",
                    "tx_count": int(row["tx_count"]),
                    "total_volume": round(float(row["total_volume"]), 2),
                    "high_risk_count": int(row["high_risk_count"]),
                    "medium_risk_count": 0,
                })
            return {
                "status": "success",
                "channel": "upi",
                "is_sample": True,
                "data_source": "Synthetic Indian UPI Simulator (sampled 5,000 txns, Days 53–60)",
                "series": series_data,
            }

    sample_path = PROCESSED_DATA_DIR / "test_scored_sample.parquet"
    if not sample_path.exists():
        return {
            "status": "success",
            "channel": norm_ch or "transfer_wallet",
            "is_sample": True,
            "data_source": "Empty sample",
            "series": [],
        }

    df = pd.read_parquet(sample_path)
    if "step" not in df.columns:
        return {
            "status": "success",
            "channel": norm_ch or "transfer_wallet",
            "is_sample": True,
            "data_source": "Unsegmented sample",
            "series": [],
        }

    grouped = df.groupby("step").agg(
        tx_count=("amount", "count"),
        total_volume=("amount", "sum"),
        high_risk_count=("risk_band", lambda s: (s == "High").sum()),
        medium_risk_count=("risk_band", lambda s: (s == "Medium").sum()),
    ).reset_index().sort_values("step")

    series_data = []
    for _, row in grouped.tail(30).iterrows():
        s = int(row["step"])
        day = (s // 24) + 1
        hr = s % 24
        series_data.append({
            "step": s,
            "hour_label": f"Day {day}, {hr:02d}:00",
            "tx_count": int(row["tx_count"]),
            "total_volume": round(float(row["total_volume"]), 2),
            "high_risk_count": int(row["high_risk_count"]),
            "medium_risk_count": int(row["medium_risk_count"]),
        })

    return {
        "status": "success",
        "channel": norm_ch or "transfer_wallet",
        "is_sample": True,
        "data_source": "PaySim Mobile Money held-out test split (sampled 5,000 txns, Steps 700–743, Day 30)",
        "series": series_data,
    }

_CACHED_LEDGER_DF: Optional[pd.DataFrame] = None


def _get_combined_ledger_df() -> pd.DataFrame:
    global _CACHED_LEDGER_DF
    if _CACHED_LEDGER_DF is not None:
        return _CACHED_LEDGER_DF

    frames = []

    p_upi = PROCESSED_DATA_DIR / "upi_test_sample.parquet"
    if p_upi.exists():
        try:
            d = pd.read_parquet(p_upi)
            d["channel"] = "upi"
            d["risk_band"] = d["band"] if "band" in d.columns else "Low"
            d["step"] = (d["day"] * 24 + d["hour"]).astype(int) if "day" in d.columns and "hour" in d.columns else 720
            d["type"] = d["txn_type"] if "txn_type" in d.columns else "PAY"
            d["nameOrig"] = d["payer_vpa"] if "payer_vpa" in d.columns else "payer@bank"
            d["nameDest"] = d["payee_vpa"] if "payee_vpa" in d.columns else "payee@bank"
            keep_cols = [c for c in ["step", "type", "amount", "nameOrig", "nameDest", "risk_score", "risk_band", "action", "top_reasons", "channel"] if c in d.columns]
            frames.append(d[keep_cols])
        except Exception as e:
            logger.warning(f"Error loading upi sample: {e}")

    p_tw = PROCESSED_DATA_DIR / "test_scored_sample.parquet"
    if p_tw.exists():
        try:
            d = pd.read_parquet(p_tw)
            d["channel"] = "transfer_wallet"
            if "risk_band" not in d.columns and "band" in d.columns:
                d["risk_band"] = d["band"]
            keep_cols = [c for c in ["step", "type", "amount", "nameOrig", "nameDest", "risk_score", "risk_band", "action", "top_reasons", "channel"] if c in d.columns]
            frames.append(d[keep_cols])
        except Exception as e:
            logger.warning(f"Error loading transfer_wallet sample: {e}")

    p_card = PROCESSED_DATA_DIR / "card_test_sample.parquet"
    if p_card.exists():
        try:
            d = pd.read_parquet(p_card)
            d["channel"] = "card"
            if "nameOrig" not in d.columns:
                d["nameOrig"] = [f"Card •••• {1000 + i % 9000}" for i in range(len(d))]
            if "nameDest" not in d.columns:
                d["nameDest"] = [f"POS Terminal #{200 + i % 800}" for i in range(len(d))]
            if "risk_band" not in d.columns and "band" in d.columns:
                d["risk_band"] = d["band"]
            keep_cols = [c for c in ["step", "type", "amount", "nameOrig", "nameDest", "risk_score", "risk_band", "action", "top_reasons", "channel"] if c in d.columns]
            frames.append(d[keep_cols])
        except Exception as e:
            logger.warning(f"Error loading card sample: {e}")

    p_acc = PROCESSED_DATA_DIR / "account_test_sample.parquet"
    if p_acc.exists():
        try:
            d = pd.read_parquet(p_acc)
            d["channel"] = "account"
            d["step"] = d["days_since_request"].fillna(1).astype(int) if "days_since_request" in d.columns else 1
            d["type"] = "OPENING"
            d["amount"] = d["intended_balcon_amount"] if "intended_balcon_amount" in d.columns else 0.0
            d["nameOrig"] = [f"Applicant-{10000 + i}" for i in range(len(d))]
            d["nameDest"] = "BAF Onboarding"
            if "risk_band" not in d.columns and "band" in d.columns:
                d["risk_band"] = d["band"]
            keep_cols = [c for c in ["step", "type", "amount", "nameOrig", "nameDest", "risk_score", "risk_band", "action", "top_reasons", "channel"] if c in d.columns]
            frames.append(d[keep_cols])
        except Exception as e:
            logger.warning(f"Error loading account sample: {e}")

    if frames:
        combined = pd.concat(frames, ignore_index=True)
    else:
        combined = pd.DataFrame(columns=["step", "type", "amount", "nameOrig", "nameDest", "risk_score", "risk_band", "action", "top_reasons", "channel"])

    _CACHED_LEDGER_DF = combined
    return _CACHED_LEDGER_DF


@app.get("/api/transactions", tags=["Transactions"])
def get_transactions_ledger(
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=100),
    channel: Optional[str] = None,
    band: Optional[str] = None,
    search: Optional[str] = None,
    sort_by: str = Query("step", description="amount, risk_score, step"),
    sort_order: str = Query("desc", description="asc, desc"),
    source: str = Query("sample", description="'sample' for stored parquet sample, 'replay' for live stream replay"),
) -> dict[str, Any]:
    if source.lower() == "replay":
        replayer = get_stream_replayer()
        items = replayer.store.get_recent(limit=page_size, channel=channel, band=band)
        return {
            "status": "success",
            "source": "stream_replay",
            "items": items,
            "total": len(replayer.store._buffer),
            "page": 1,
            "page_size": page_size,
            "total_pages": int(np.ceil(len(replayer.store._buffer) / page_size)),
        }

    full_df = _get_combined_ledger_df()
    df = full_df.copy()

    if channel and channel.upper() != "ALL":
        norm_ch = resolve_channel(channel) or channel.lower()
        df = df[df["channel"] == norm_ch]

    if band and band.upper() != "ALL":
        df = df[df["risk_band"].astype(str).str.upper() == band.upper()]

    if search:
        s_term = search.strip().lower()
        mask = pd.Series(False, index=df.index)
        for c in ["nameOrig", "nameDest", "type", "channel"]:
            if c in df.columns:
                mask = mask | df[c].astype(str).str.lower().str.contains(s_term, regex=False)
        df = df[mask]

    if sort_by in df.columns:
        ascending = (sort_order.lower() == "asc")
        df = df.sort_values(by=sort_by, ascending=ascending)

    total_records = len(df)
    total_pages = int(np.ceil(total_records / page_size)) if total_records > 0 else 0

    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paged_df = df.iloc[start_idx:end_idx]

    cols_to_emit = [c for c in ["step", "type", "amount", "nameOrig", "nameDest", "risk_score", "risk_band", "action", "channel", "top_reasons"] if c in paged_df.columns]
    records = []
    for _, r in paged_df[cols_to_emit].iterrows():
        rec = r.to_dict()
        top_r = rec.get("top_reasons")
        if isinstance(top_r, str) and top_r.strip():
            rec["reason"] = top_r.split(";")[0].strip()
        elif isinstance(top_r, list) and len(top_r) > 0:
            rec["reason"] = str(top_r[0]).strip()
        else:
            rec["reason"] = "Standard payment profile"
        if "risk_score" in rec and rec["risk_score"] is not None:
            rec["risk_score"] = round(float(rec["risk_score"]), 4)
        if "amount" in rec and rec["amount"] is not None:
            rec["amount"] = round(float(rec["amount"]), 2)
        records.append(rec)

    return {
        "status": "success",
        "items": records,
        "total": total_records,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
    }

@app.get("/api/cases", tags=["Case Management"])
def get_cases_endpoint(
    status: Optional[str] = None,
    channel: Optional[str] = None,
    band: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200),
) -> dict[str, Any]:
    df_cases = get_cases(status=status, channel=channel, band=band, limit=limit)
    return {
        "status": "success",
        "count": len(df_cases),
        "cases": df_cases.to_dict(orient="records"),
    }


@app.get("/api/cases/precision", tags=["Case Management"])
def get_case_precision_metrics_endpoint() -> dict[str, Any]:
    return get_analyst_precision_metrics()


@app.get("/api/cases/{case_id}", tags=["Case Management"])
def get_case_detail_endpoint(case_id: str) -> dict[str, Any]:
    c = get_case_by_id(case_id)
    if not c:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found.")

    audit_logs = get_case_audit_logs(case_id)
    return {
        "status": "success",
        "case": c,
        "audit_logs": audit_logs,
    }


@app.patch("/api/cases/{case_id}", tags=["Case Management"])
def update_case_endpoint(case_id: str, req: CaseStatusUpdateRequest) -> dict[str, Any]:
    c = get_case_by_id(case_id)
    if not c:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found.")

    update_case_status(
        case_id=case_id,
        new_status=req.status,
        notes=req.notes,
        analyst_name=req.analyst_name,
    )

    updated = get_case_by_id(case_id)
    audit_logs = get_case_audit_logs(case_id)
    return {
        "status": "success",
        "message": f"Case '{case_id}' updated to '{req.status}'.",
        "case": updated,
        "latest_audit": audit_logs[0] if audit_logs else None,
    }


@app.get("/api/cases-feedback-export", tags=["Case Management"])
def export_feedback_endpoint() -> dict[str, Any]:
    df_feedback = export_feedback_dataset()
    return {
        "status": "success",
        "count": len(df_feedback),
        "records": df_feedback.to_dict(orient="records"),
    }

@app.get("/api/entity/{entity_id}", tags=["Graph Intelligence"])
def get_entity_graph_endpoint(
    entity_id: str,
    channel: str = Query("transfer_wallet"),
    hops: int = Query(1, ge=1, le=2),
) -> dict[str, Any]:
    profile = get_entity_profile(entity_id, channel=channel)
    target = profile["entity_id"]
    df = load_network_dataset()

    nodes: list[dict[str, Any]] = [
        {"id": target, "role": "Target Entity", "hop": 0, "is_target": True, "risk": profile["risk_score"]}
    ]
    edges: list[dict[str, Any]] = []

    if not df.empty:
        hop1_tx = df[(df["nameOrig"].astype(str) == target) | (df["nameDest"].astype(str) == target)].head(50)
        seen_nodes = {target}

        for _, row in hop1_tx.iterrows():
            u = str(row["nameOrig"])
            v = str(row["nameDest"])
            amt = float(row.get("amount", 1000.0))
            is_fraud = int(row.get("isFraud", 0))

            if u not in seen_nodes:
                nodes.append({"id": u, "role": "Inbound Sender", "hop": 1, "is_target": False, "risk": 0.5})
                seen_nodes.add(u)
            if v not in seen_nodes:
                nodes.append({"id": v, "role": "Outbound Receiver", "hop": 1, "is_target": False, "risk": 0.5})
                seen_nodes.add(v)

            edges.append({"source": u, "target": v, "amount": amt, "is_fraud": is_fraud})

    return {
        "status": "success",
        "profile": {
            "entity_id": target,
            "risk_score": profile["risk_score"],
            "risk_band": profile["risk_band"],
            "total_sent_inr": profile["total_sent_inr"],
            "total_received_inr": profile["total_received_inr"],
            "net_flow_inr": profile["net_flow_inr"],
            "transaction_count": profile["transaction_count"],
            "linked_accounts_count": profile["linked_accounts_count"],
            "is_sparse": profile["is_sparse"],
            "sparse_message": profile["sparse_message"],
        },
        "graph": {
            "nodes": nodes,
            "edges": edges,
            "hops": hops,
        },
    }

@app.get("/api/models", tags=["Model Registry"])
@ttl_cached(seconds=60)
def get_models_registry_endpoint() -> dict[str, Any]:
    champions = load_champion_registry_json()
    metrics = load_metrics_json()
    return {
        "status": "success",
        "champions": champions.get("champions", {}),
        "registry_version": champions.get("registry_version", "2.0.0"),
        "last_updated": champions.get("last_updated", ""),
        "cross_channel_comparison": metrics.get("channels", {}),
    }


@app.get("/api/benchmarks/{category}", tags=["Model Registry"])
@ttl_cached(seconds=60)
def get_benchmarks_endpoint(category: str) -> dict[str, Any]:
    cat = category.strip().lower()

    if cat == "evasion":
        p = REPORTS_DIR / "paysim_evasion_benchmark.json"
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return {"status": "success", "category": "evasion", "data": json.load(f)}
    elif cat == "fairness":
        p = REPORTS_DIR / "baf_operating_points_fairness.json"
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return {"status": "success", "category": "fairness", "data": json.load(f)}
    elif cat == "drift":
        p = REPORTS_DIR / "concept_drift_benchmark.json"
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return {"status": "success", "category": "drift", "data": json.load(f)}
    elif cat == "uncertainty":
        p = REPORTS_DIR / "uncertainty_routing.json"
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return {"status": "success", "category": "uncertainty", "data": json.load(f)}
    elif cat == "ablation":
        metrics = load_metrics_json()
        return {"status": "success", "category": "ablation", "data": metrics.get("network_ablation", {})}

    raise HTTPException(status_code=404, detail=f"Benchmark category '{category}' not found.")


@app.get("/api/monitoring", tags=["Monitoring"])
@ttl_cached(seconds=60)
def get_monitoring_endpoint() -> dict[str, Any]:
    metrics = load_metrics_json()
    return {
        "status": "success",
        "psi_drift": metrics.get("monitoring", {}).get("psi_drift", {}),
        "score_distribution": metrics.get("monitoring", {}).get("score_distribution", {}),
        "concept_drift": metrics.get("concept_drift", {}),
    }

@app.post("/score", response_model=ScoreResponse, tags=["Scoring"])
@app.post("/api/score", response_model=ScoreResponse, tags=["Scoring"])
def score_transaction_endpoint(req: ScoreRequest) -> ScoreResponse:
    t_start = time.perf_counter()
    canonical_ch = req.channel
    store = get_history_store()

    try:
        res = predict(
            channel=canonical_ch,
            payload=req.payload,
            top_k=req.top_k,
            history_store=store,
            commit_to_store=req.update_history,
            use_group_fairness=req.use_group_fairness,
        )
    except (ValueError, KeyError, TypeError) as e:
        logger.warning("Scoring validation error on %s: %s", canonical_ch, e)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Scoring request parameter error: {str(e)}",
        )
    except (RuntimeError, IndexError, AttributeError, FileNotFoundError, OSError) as e:
        logger.exception("Scoring pipeline failure on %s: %s", canonical_ch, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scoring pipeline execution error: {str(e)}",
        )

    t_total = (time.perf_counter() - t_start) * 1000.0
    low_th, high_th = get_calibrated_thresholds(canonical_ch)

    return ScoreResponse(
        status="success",
        channel=res["channel"],
        data_source=res["data_source"],
        risk_score=res["risk_score"],
        raw_score=res.get("raw_score", res.get("raw_model_score", 0.0)),
        band=res["band"],
        action=res["action"],
        top_reasons=res["top_reasons"],
        factors=res.get("factors", []),
        counterfactual=res.get("counterfactual", None),
        latency_ms=round(t_total, 2),
        thresholds_applied={"low_threshold": low_th, "high_threshold": high_th},
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


@app.post("/api/batch-score", response_model=BatchScoreResponse, tags=["Scoring"])
def batch_score_endpoint(req: BatchScoreRequest) -> BatchScoreResponse:
    canonical_ch = resolve_channel(req.channel)
    store = get_history_store()
    t_start = time.perf_counter()

    results = []
    high_count = 0
    med_count = 0
    low_count = 0

    for i, tx in enumerate(req.transactions, start=1):
        try:
            res = predict(channel=canonical_ch, payload=tx, top_k=req.top_k, history_store=store, commit_to_store=False)
            bd = res["band"]
            if bd == "High":
                high_count += 1
            elif bd == "Medium":
                med_count += 1
            else:
                low_count += 1

            results.append({
                "index": i,
                "amount": float(tx.get("amount") or tx.get("Amount") or 0.0),
                "risk_score": res["risk_score"],
                "band": bd,
                "action": res["action"],
                "top_reasons": res["top_reasons"],
            })
        except Exception as e:
            results.append({
                "index": i,
                "amount": float(tx.get("amount") or tx.get("Amount") or 0.0),
                "risk_score": 0.5,
                "band": "Medium",
                "action": "Review",
                "error": str(e),
            })

    total_duration = (time.perf_counter() - t_start) * 1000.0
    scores = [r["risk_score"] for r in results]
    mean_score = sum(scores) / len(scores) if scores else 0.0

    return BatchScoreResponse(
        status="success",
        channel=canonical_ch,
        total_scored=len(results),
        high_risk_count=high_count,
        medium_risk_count=med_count,
        low_risk_count=low_count,
        mean_risk_score=round(mean_score, 4),
        total_duration_ms=round(total_duration, 2),
        results=results,
    )

@app.get("/api/case-report/{case_id}", tags=["Case Management"])
def get_case_report_endpoint(case_id: str, format: str = Query("json", description="json or html")) -> Any:
    c = get_case_by_id(case_id)
    if not c:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found.")

    payload = {}
    if c.get("payload_json"):
        try:
            payload = json.loads(c["payload_json"])
        except Exception:
            pass
    if not payload:
        payload = {"amount": c["amount"], "nameOrig": c["entity_id"], "nameDest": c["counterparty_id"]}

    low_th, high_th = get_calibrated_thresholds(c["channel"])
    res = {
        "risk_score": c["risk_score"],
        "band": c["risk_band"],
        "action": "Decline & Alert" if c["risk_band"] == "High" else "Review",
        "timestamp": c["timestamp"],
        "data_source": "Case Queue",
        "top_reasons": ["High-value transaction pattern", "Elevated network risk"],
    }

    html_str = generate_case_report_html(
        channel=c["channel"],
        payload=payload,
        result=res,
        low_th=low_th,
        high_th=high_th,
        report_id=case_id,
    )

    if format.lower() == "html":
        return HTMLResponse(content=html_str)

    return {
        "status": "success",
        "case_id": case_id,
        "channel": c["channel"],
        "risk_score": c["risk_score"],
        "band": c["risk_band"],
        "html_content": html_str,
    }

@app.get("/api/attack-lab/presets", tags=["Adversarial Attack Lab"])
def get_attack_presets_endpoint() -> dict[str, Any]:
    return {"status": "success", "presets": get_attack_presets()}


@app.post("/api/attack-lab/simulate", tags=["Adversarial Attack Lab"])
def simulate_attack_endpoint(req: AttackSimulateRequest) -> dict[str, Any]:
    try:
        res = simulate_attack_scenario(
            channel=req.channel,
            base_payload=req.base_payload,
            attack_type=req.attack_type,
            params=req.params,
        )
        return {"status": "success", "simulation": res}
    except Exception as e:
        logger.exception("Attack simulation failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Simulation error: {str(e)}")


@app.get("/api/attack-lab/history", tags=["Adversarial Attack Lab"])
def get_attack_history_endpoint() -> dict[str, Any]:
    df_h = get_attack_history()
    return {"status": "success", "history": df_h.to_dict(orient="records")}

@app.post("/api/step-up/preview", tags=["Customer Messaging"])
def preview_step_up_endpoint(req: StepUpPreviewRequest) -> dict[str, Any]:
    canonical_ch = resolve_channel(req.channel)
    msg = generate_step_up_message(
        channel=canonical_ch,
        payload=req.payload,
        otp_code=req.otp_code,
    )
    return {"status": "success", "messaging": msg}

@app.get("/api/glossary", tags=["Configuration"])
def get_glossary_endpoint() -> dict[str, Any]:
    glossary = {
        "PR_AUC": "Area Under the Precision-Recall Curve. Evaluates performance under extreme class imbalance without being inflated by true negatives.",
        "ROC_AUC": "Area Under the Receiver Operating Characteristic Curve. Evaluates ranking discrimination across all classification thresholds.",
        "FPR": "False Positive Rate. The fraction of legitimate transactions or applicants mistakenly flagged as fraud.",
        "False_Alarms_Per_100k": "Number of legitimate users challenged or blocked for every 100,000 legitimate transactions processed.",
        "TreeSHAP": "Shapley Additive Explanations for tree-based models. Quantifies each feature's contribution in log-odds space.",
        "Platt_Calibration": "Logistic regression post-processing mapping uncalibrated model scores to empirical fraud probabilities.",
        "PSI": "Population Stability Index. Measures concept and feature drift between baseline training and live scoring populations.",
        "Step_Up_Challenge": "Two-factor authentication (SMS OTP or in-app biometric challenge) issued for medium-risk transactions.",
        "Mule_Account": "An account used to receive illicit funds from compromised accounts and rapidly forward them downstream.",
        "Collect_Request_Scam": "A UPI social-engineering scam where a fraudster sends a payment request masquerading as a refund or prize.",
    }
    return {"status": "success", "glossary": glossary}

from starlette.exceptions import HTTPException as StarletteHTTPException

class SinglePageAppStaticFiles(StaticFiles):

    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except (HTTPException, StarletteHTTPException) as exc:
            if exc.status_code == 404 and not path.startswith("api/"):
                return await super().get_response("index.html", scope)
            raise exc


dist_dir = BASE_DIR / "frontend-web" / "dist"
if dist_dir.exists():
    app.mount("/", SinglePageAppStaticFiles(directory=str(dist_dir), html=True), name="static-frontend")
