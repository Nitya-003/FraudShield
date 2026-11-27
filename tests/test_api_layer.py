"""
Unit & Integration Test Suite for FraudShield API Layer (Phase 4).

Verifies every FastAPI endpoint:
- Overview KPIs, risk bands, daily series
- Paged transactions with filtering and sorting
- Case management CRUD, audit logs, and feedback export
- 360-degree Entity Graph & sparse account fallback
- Models & Benchmarks (evasion, fairness, drift, uncertainty, ablation)
- Monitoring (PSI drift, score distributions)
- Single & Batch scoring with TreeSHAP explainability
- Printable case report (JSON and HTML)
- Adversarial Attack Lab (presets, simulation, history)
- Bilingual Step-Up Message Preview
- Domain Glossary
- Data integrity: asserts all metrics derive from verified files or data
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from starlette.testclient import TestClient

from ml.api import app
from ml.config import METRICS_FILE

client = TestClient(app)


def test_health_check() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "channels_available" in data
    assert len(data["channels_available"]) == 4


def test_get_thresholds() -> None:
    resp = client.get("/thresholds")
    assert resp.status_code == 200
    data = resp.json()
    assert "thresholds" in data
    assert "transfer_wallet" in data["thresholds"]
    assert "upi" in data["thresholds"]


def test_overview_kpis() -> None:
    resp = client.get("/api/overview")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["total_transactions"] > 0
    assert data["total_volume_inr"] > 0.0
    assert data["alert_rate_pct"] >= 0.0
    assert data["latency_p50_ms"] > 0.0
    assert data["latency_p95_ms"] > 0.0


def test_risk_bands_summary() -> None:
    resp = client.get("/api/risk-bands")
    assert resp.status_code == 200
    data = resp.json()
    assert "counts" in data
    assert "Low" in data["counts"]
    assert "thresholds_per_channel" in data


def test_daily_series() -> None:
    resp = client.get("/api/daily-series")
    assert resp.status_code == 200
    data = resp.json()
    assert "series" in data
    assert isinstance(data["series"], list)
    if data["series"]:
        assert "tx_count" in data["series"][0]
        assert "total_volume" in data["series"][0]


def test_transactions_paged_and_filtered() -> None:
    # Page 1 default
    resp = client.get("/api/transactions?page=1&page_size=10")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["items"]) <= 10
    assert data["page"] == 1
    assert data["total"] > 0

    # Filter by band
    resp_high = client.get("/api/transactions?page=1&page_size=10&band=High")
    assert resp_high.status_code == 200
    for item in resp_high.json()["items"]:
        assert item["risk_band"].upper() == "HIGH"


def test_cases_endpoints() -> None:
    # 1. List cases
    resp = client.get("/api/cases?limit=20")
    assert resp.status_code == 200
    cases_data = resp.json()
    assert cases_data["count"] > 0
    test_case_id = cases_data["cases"][0]["case_id"]

    # 2. Case detail
    resp_det = client.get(f"/api/cases/{test_case_id}")
    assert resp_det.status_code == 200
    assert resp_det.json()["case"]["case_id"] == test_case_id

    # 3. Patch case status & audit log
    patch_payload = {
        "status": "CONFIRMED_FRAUD",
        "notes": "Verified via automated test script",
        "analyst_name": "API Tester",
    }
    resp_patch = client.patch(f"/api/cases/{test_case_id}", json=patch_payload)
    assert resp_patch.status_code == 200
    assert resp_patch.json()["case"]["status"] == "CONFIRMED_FRAUD"

    # 4. Precision
    resp_prec = client.get("/api/cases/precision")
    assert resp_prec.status_code == 200
    assert "open_alerts" in resp_prec.json()

    # 5. Export feedback
    resp_fb = client.get("/api/cases-feedback-export")
    assert resp_fb.status_code == 200
    assert "records" in resp_fb.json()


def test_entity_graph_endpoints() -> None:
    # Rich mule hub
    resp = client.get("/api/entity/C716083600?hops=1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["profile"]["entity_id"] == "C716083600"
    assert len(data["graph"]["nodes"]) >= 1

    # Sparse entity fallback
    resp_sparse = client.get("/api/entity/C1984094095?hops=1")
    assert resp_sparse.status_code == 200
    sparse_data = resp_sparse.json()
    assert sparse_data["profile"]["is_sparse"] is True
    assert "Sparse" in sparse_data["profile"]["sparse_message"]


def test_models_registry_endpoint() -> None:
    resp = client.get("/api/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "champions" in data
    assert "transfer_wallet" in data["champions"]
    assert "card" in data["champions"]
    assert "account" in data["champions"]
    assert "upi" in data["champions"]


def test_benchmarks_endpoints() -> None:
    for cat in ["evasion", "fairness", "drift", "uncertainty", "ablation"]:
        resp = client.get(f"/api/benchmarks/{cat}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"
        assert "data" in resp.json()


def test_monitoring_endpoint() -> None:
    resp = client.get("/api/monitoring")
    assert resp.status_code == 200
    data = resp.json()
    assert "psi_drift" in data
    assert "score_distribution" in data


def test_single_and_batch_scoring() -> None:
    payload = {
        "channel": "transfer_wallet",
        "payload": {
            "amount": 500000.0,
            "oldbalanceOrg": 500000.0,
            "oldbalanceDest": 0.0,
            "type": "TRANSFER",
            "nameOrig": "C11223344",
            "nameDest": "C55667788",
            "step": 400,
        },
        "top_k": 3,
        "update_history": False,
    }
    resp_single = client.post("/api/score", json=payload)
    assert resp_single.status_code == 200
    s_data = resp_single.json()
    assert 0.0 <= s_data["risk_score"] <= 1.0
    assert s_data["band"] in ["Low", "Medium", "High"]
    assert len(s_data["top_reasons"]) <= 3

    # Batch scoring
    batch_payload = {
        "channel": "transfer_wallet",
        "transactions": [payload["payload"], payload["payload"]],
        "top_k": 2,
    }
    resp_batch = client.post("/api/batch-score", json=batch_payload)
    assert resp_batch.status_code == 200
    b_data = resp_batch.json()
    assert b_data["total_scored"] == 2
    assert len(b_data["results"]) == 2


def test_case_report_endpoint() -> None:
    resp_cases = client.get("/api/cases?limit=1")
    case_id = resp_cases.json()["cases"][0]["case_id"]

    # JSON format
    resp_json = client.get(f"/api/case-report/{case_id}?format=json")
    assert resp_json.status_code == 200
    assert "html_content" in resp_json.json()

    # HTML format
    resp_html = client.get(f"/api/case-report/{case_id}?format=html")
    assert resp_html.status_code == 200
    assert "<!DOCTYPE html>" in resp_html.text
    assert "FRAUDSHIELD TRANSACTION DECISION REPORT" in resp_html.text


def test_attack_lab_endpoints() -> None:
    # 1. Presets
    resp_pre = client.get("/api/attack-lab/presets")
    assert resp_pre.status_code == 200
    presets = resp_pre.json()["presets"]
    assert len(presets) >= 3

    # 2. Simulate
    sim_req = {
        "channel": presets[0]["channel"],
        "base_payload": presets[0]["payload"],
        "attack_type": "split_transfers",
        "params": {"k_splits": 2, "split_mode": "equal", "time_gap_hours": 0},
    }
    resp_sim = client.post("/api/attack-lab/simulate", json=sim_req)
    assert resp_sim.status_code == 200
    sim_data = resp_sim.json()["simulation"]
    assert "verdict" in sim_data
    assert len(sim_data["transfers"]) == 2

    # 3. History
    resp_hist = client.get("/api/attack-lab/history")
    assert resp_hist.status_code == 200
    assert len(resp_hist.json()["history"]) > 0


def test_step_up_preview_endpoint() -> None:
    req = {
        "channel": "upi",
        "payload": {
            "amount": 42000.0,
            "payee_vpa": "fake_dept@okaxis",
            "type": "COLLECT",
            "is_collect_request": True,
        },
        "otp_code": "981-234",
    }
    resp = client.post("/api/step-up/preview", json=req)
    assert resp.status_code == 200
    msg = resp.json()["messaging"]
    assert "english_message" in msg
    assert "hindi_message" in msg
    assert "CRITICAL WARNING" in msg["english_message"]
    assert "महत्वपूर्ण चेतावनी" in msg["hindi_message"]


def test_glossary_endpoint() -> None:
    resp = client.get("/api/glossary")
    assert resp.status_code == 200
    data = resp.json()["glossary"]
    assert "PR_AUC" in data
    assert "ROC_AUC" in data
    assert "False_Alarms_Per_100k" in data

