"""
API Contract Validation Tests

Validates that FastAPI endpoints strictly adhere to the schema and types
defined in frontend-web/src/types/api.ts.
"""

from __future__ import annotations

import math
from starlette.testclient import TestClient
from api.app import app

client = TestClient(app)


def test_contract_overview_kpis() -> None:
    resp = client.get("/api/overview")
    assert resp.status_code == 200
    data = resp.json()

    # Required fields matching OverviewKPIs in api.ts
    required_keys = [
        "status",
        "total_volume_inr",
        "total_transactions",
        "total_alerts",
        "alert_rate_pct",
        "loss_averted_inr",
        "channels_count",
        "channels",
        "latency_p50_ms",
        "latency_p95_ms",
        "timestamp",
    ]
    for key in required_keys:
        assert key in data, f"Missing key {key} in /api/overview"

    assert isinstance(data["status"], str)
    assert isinstance(data["total_volume_inr"], (int, float)) and math.isfinite(data["total_volume_inr"])
    assert isinstance(data["total_transactions"], int) and data["total_transactions"] >= 0
    assert isinstance(data["total_alerts"], int) and data["total_alerts"] >= 0
    assert isinstance(data["alert_rate_pct"], (int, float)) and 0.0 <= data["alert_rate_pct"] <= 100.0
    assert isinstance(data["loss_averted_inr"], (int, float)) and math.isfinite(data["loss_averted_inr"])
    assert isinstance(data["channels_count"], int) and data["channels_count"] == 4
    assert isinstance(data["channels"], list) and len(data["channels"]) == 4
    assert isinstance(data["latency_p50_ms"], (int, float)) and data["latency_p50_ms"] > 0
    assert isinstance(data["latency_p95_ms"], (int, float)) and data["latency_p95_ms"] > 0


def test_contract_risk_bands() -> None:
    resp = client.get("/api/risk-bands")
    assert resp.status_code == 200
    data = resp.json()

    assert "status" in data
    assert "counts" in data
    assert "thresholds_per_channel" in data

    counts = data["counts"]
    for band in ["Low", "Medium", "High"]:
        assert band in counts
        assert isinstance(counts[band], int) and counts[band] >= 0

    thresholds = data["thresholds_per_channel"]
    for ch in ["transfer_wallet", "upi", "card", "account"]:
        assert ch in thresholds
        assert "low" in thresholds[ch]
        assert "high" in thresholds[ch]
        assert thresholds[ch]["low"] <= thresholds[ch]["high"]


def test_contract_daily_series() -> None:
    resp = client.get("/api/daily-series")
    assert resp.status_code == 200
    data = resp.json()

    assert "status" in data
    assert "series" in data
    assert isinstance(data["series"], list)
    assert len(data["series"]) > 0

    for pt in data["series"][:5]:
        assert "step" in pt and isinstance(pt["step"], int)
        assert "hour_label" in pt and isinstance(pt["hour_label"], str)
        assert "tx_count" in pt and isinstance(pt["tx_count"], int)
        assert "total_volume" in pt and isinstance(pt["total_volume"], (int, float))
        assert "high_risk_count" in pt and isinstance(pt["high_risk_count"], int)
        assert "medium_risk_count" in pt and isinstance(pt["medium_risk_count"], int)


def test_contract_paged_transactions() -> None:
    resp = client.get("/api/transactions?page=1&page_size=10")
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "success"
    assert "items" in data
    assert "total" in data
    assert "page" in data
    assert "page_size" in data
    assert "total_pages" in data

    assert isinstance(data["items"], list)
    assert len(data["items"]) <= 10
    if data["items"]:
        tx = data["items"][0]
        assert "step" in tx
        assert "type" in tx
        assert "amount" in tx and math.isfinite(tx["amount"])
        assert "risk_score" in tx and 0.0 <= tx["risk_score"] <= 1.0
        assert tx["risk_band"] in ["Low", "Medium", "High"]
        assert "action" in tx
        assert "channel" in tx


def test_contract_cases_and_audit() -> None:
    resp = client.get("/api/cases?limit=5")
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "success"
    assert "cases" in data
    assert isinstance(data["cases"], list)
    assert len(data["cases"]) > 0

    c = data["cases"][0]
    assert "case_id" in c
    assert "channel" in c
    assert "entity_id" in c
    assert "amount" in c
    assert "risk_score" in c and 0.0 <= c["risk_score"] <= 1.0
    assert c["status"] in ["OPEN", "CONFIRMED_FRAUD", "FALSE_POSITIVE"]


def test_contract_model_registry() -> None:
    resp = client.get("/api/models")
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "success"
    assert "registry_version" in data
    assert "champions" in data

    champions = data["champions"]
    assert len(champions) == 4
    for ch_name, champ in champions.items():
        assert "channel" in champ
        assert "display_name" in champ
        assert "algorithm" in champ
        assert "pr_auc" in champ and 0.0 <= champ["pr_auc"] <= 1.0
        assert "roc_auc" in champ and 0.0 <= champ["roc_auc"] <= 1.0
        assert "recall" in champ and 0.0 <= champ["recall"] <= 1.0
        assert "bootstrap_ci_95" in champ
        ci = champ["bootstrap_ci_95"].get("pr_auc")
        if ci:
            assert ci[0] <= champ["pr_auc"] <= ci[1], f"Point estimate {champ['pr_auc']} outside CI {ci}"


def test_contract_single_scoring() -> None:
    payload = {
        "channel": "transfer_wallet",
        "payload": {
            "type": "TRANSFER",
            "amount": 250000.0,
            "oldbalanceOrg": 250000.0,
            "newbalanceOrig": 0.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "step": 1,
        },
    }
    resp = client.post("/score", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] in ["success", "scored"]
    assert data["channel"] == "transfer_wallet"
    assert 0.0 <= data["risk_score"] <= 1.0
    assert data["band"] in ["Low", "Medium", "High"]
    assert "action" in data
    assert isinstance(data["top_reasons"], list)
    assert isinstance(data["factors"], list)
    assert "latency_ms" in data and data["latency_ms"] > 0
    assert "thresholds_applied" in data


def test_contract_attack_simulation() -> None:
    req = {
        "channel": "transfer_wallet",
        "base_payload": {
            "type": "TRANSFER",
            "amount": 200000.0,
            "oldbalanceOrg": 200000.0,
            "newbalanceOrig": 0.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "step": 1,
        },
        "attack_type": "split_transfers",
        "params": {"k_splits": 3, "split_mode": "equal", "time_gap_hours": 1},
    }
    resp = client.post("/api/attack-lab/simulate", json=req)
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "success"
    sim = data["simulation"]
    assert sim["channel"] == "transfer_wallet"
    assert sim["attack_type"] == "split_transfers"
    assert 0.0 <= sim["base_score"] <= 1.0
    assert 0.0 <= sim["peak_attack_score"] <= 1.0
    assert sim["verdict"] in ["BLOCKED", "STEP_UP_CHALLENGED", "EVADED"]
    assert len(sim["transfers"]) == 3
    for tx in sim["transfers"]:
        assert "risk_score" in tx
        assert "band" in tx
        assert "action" in tx


def test_contract_step_up_message() -> None:
    req = {
        "channel": "upi",
        "payload": {
            "amount": 15000.0,
            "payee_vpa": "electricity_officer@paygov",
            "type": "COLLECT",
            "is_collect_request": True,
        },
        "otp_code": "842-195",
    }
    resp = client.post("/api/step-up/preview", json=req)
    assert resp.status_code == 200
    msg = resp.json()["messaging"]

    assert "english_message" in msg
    assert "hindi_message" in msg
    assert "otp_code" in msg
    assert "amount_formatted" in msg
    assert msg["otp_code"] == "842-195"
    assert "₹15,000" in msg["amount_formatted"]
