"""
FraudShield Dedicated Quality Assurance & Component Test Suite.
Validates:
1. Batch scoring engine (ml/batch_scoring.py & CSV pipeline)
2. Case management SQLite CRUD & feedback loop (ml/case_management.py)
3. API latency recorder & FastAPI scoring microservice (ml/api.py & scripts/measure_api_latency.py)
4. Printable case report generator (ml/case_report.py)
5. Decision router aliases, payload normalization, and exception handling (ml/router.py)
"""

from __future__ import annotations

import io
import json
import sqlite3
import pytest
import pandas as pd
from fastapi.testclient import TestClient

from ml.api import app
from ml.batch_scoring import CHANNEL_REQUIRED_COLUMNS, get_sample_template_csv, score_batch
from ml.case_management import (
    get_analyst_precision_metrics,
    get_case_by_id,
    get_cases,
    get_db_connection,
    init_db,
    update_case_status,
    feed_confirmed_labels_to_feedback_loop,
)
from ml.case_report import generate_case_report_html
from ml.router import CHANNEL_ALIASES, FraudRouter, get_channel_info, get_channel_registry, predict, resolve_channel
from scripts.measure_api_latency import run_latency_benchmark


# =============================================================================
# 1. BATCH SCORING TESTS
# =============================================================================
class TestBatchScoring:
    def test_sample_template_generation_all_channels(self):
        """Verify sample CSV template generation for each supported channel."""
        for ch in ["transfer_wallet", "card", "account", "upi"]:
            csv_str = get_sample_template_csv(ch)
            assert isinstance(csv_str, str)
            df = pd.read_csv(io.StringIO(csv_str))
            assert len(df) > 0
            for req in CHANNEL_REQUIRED_COLUMNS[ch]:
                assert req in df.columns, f"Channel {ch} template missing column {req}"

    def test_score_batch_transfer_wallet(self):
        """Verify batch scoring on transfer_wallet transactions."""
        csv_str = get_sample_template_csv("transfer_wallet")
        df_in = pd.read_csv(io.StringIO(csv_str))

        scored_df, summary = score_batch(df_in, channel="transfer_wallet")

        assert len(scored_df) == len(df_in)
        expected_cols = ["risk_score", "risk_band", "action", "primary_reason", "counterfactual"]
        for c in expected_cols:
            assert c in scored_df.columns

        assert summary["total_transactions"] == len(df_in)
        assert summary["high_risk_declines"] + summary["medium_risk_challenges"] + summary["low_risk_approvals"] == len(df_in)
        assert "thresholds_applied" in summary
        assert summary["avg_latency_ms"] >= 0.0

    def test_score_batch_card(self):
        """Verify batch scoring on card transactions."""
        csv_str = get_sample_template_csv("card")
        df_in = pd.read_csv(io.StringIO(csv_str))

        scored_df, summary = score_batch(df_in, channel="card")
        assert len(scored_df) == len(df_in)
        assert "risk_score" in scored_df.columns
        assert summary["channel"] == "card"

    def test_score_batch_missing_required_columns(self):
        """Verify batch scoring raises ValueError on malformed schema."""
        bad_df = pd.DataFrame([{"foo": 1, "bar": 2}])
        with pytest.raises(ValueError, match="missing required columns"):
            score_batch(bad_df, channel="transfer_wallet")


# =============================================================================
# 2. CASE MANAGEMENT SQLITE CRUD TESTS
# =============================================================================
class TestCaseManagementStorage:
    @pytest.fixture(autouse=True)
    def setup_db(self, monkeypatch, tmp_path):
        """Use a clean temporary SQLite database for test isolation."""
        test_db = tmp_path / "test_cases.db"
        monkeypatch.setattr("ml.case_management.DB_PATH", test_db)
        init_db(seed_if_empty=False)

    def test_init_db_schema(self):
        """Verify database schema creation."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            assert "cases" in tables
            assert "analyst_logs" in tables

    def test_case_lifecycle_crud(self):
        """Verify inserting, retrieving, updating status, and logging decisions."""
        test_case_id = "TEST-CASE-001"
        with get_db_connection() as conn:
            conn.execute("""
                INSERT INTO cases (
                    case_id, channel, entity_id, counterparty_id, amount,
                    risk_score, risk_band, timestamp, status, analyst_notes, updated_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                test_case_id, "transfer_wallet", "C_USER_1", "M_MERCH_1", 150000.0,
                0.96, "High", "2026-11-10 12:00:00", "OPEN", "", "2026-11-10 12:00:00", "{}"
            ))
            conn.commit()

        # Retrieve
        case = get_case_by_id(test_case_id)
        assert case is not None
        assert case["case_id"] == test_case_id
        assert case["status"] == "OPEN"
        assert case["amount"] == 150000.0

        # Update status to CONFIRMED_FRAUD
        update_case_status(test_case_id, "CONFIRMED_FRAUD", notes="Confirmed account takeover", analyst_name="Senior Officer")

        updated = get_case_by_id(test_case_id)
        assert updated["status"] == "CONFIRMED_FRAUD"
        assert updated["analyst_notes"] == "Confirmed account takeover"

        # Check analyst_logs
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM analyst_logs WHERE case_id = ?", (test_case_id,))
            logs = cursor.fetchall()
            assert len(logs) == 1
            assert logs[0]["action"] == "STATUS_CHANGE_CONFIRMED_FRAUD"
            assert logs[0]["analyst_name"] == "Senior Officer"

    def test_analyst_precision_calculation(self):
        """Verify precision-of-alerts formula on analyst decisions requires >= 20 decisions."""
        with get_db_connection() as conn:
            # 3 confirmed frauds, 1 false positive, 2 open
            cases_data = [
                ("C1", "transfer_wallet", "E1", None, 100.0, 0.9, "High", "2026-11-10", "CONFIRMED_FRAUD", "", "2026-11-10", "{}"),
                ("C2", "card", "E2", None, 200.0, 0.92, "High", "2026-11-10", "CONFIRMED_FRAUD", "", "2026-11-10", "{}"),
                ("C3", "upi", "E3", None, 300.0, 0.95, "High", "2026-11-10", "CONFIRMED_FRAUD", "", "2026-11-10", "{}"),
                ("C4", "card", "E4", None, 50.0, 0.85, "High", "2026-11-10", "FALSE_POSITIVE", "", "2026-11-10", "{}"),
                ("C5", "account", "E5", None, 500.0, 0.88, "High", "2026-11-10", "OPEN", "", "2026-11-10", "{}"),
                ("C6", "upi", "E6", None, 150.0, 0.89, "High", "2026-11-10", "OPEN", "", "2026-11-10", "{}"),
            ]
            conn.executemany("""
                INSERT INTO cases (case_id, channel, entity_id, counterparty_id, amount,
                                   risk_score, risk_band, timestamp, status, analyst_notes, updated_at, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, cases_data)
            conn.commit()

        metrics = get_analyst_precision_metrics()
        assert metrics["open_alerts"] == 2
        assert metrics["confirmed_fraud"] == 3
        assert metrics["false_positives"] == 1
        assert metrics["total_decided"] == 4
        # Under 20 reviews: precision should be None and status "not enough reviews (<20)"
        assert metrics["precision_of_alerts_pct"] is None
        assert metrics["status"] == "not enough reviews (<20)"

        # Now add 16 more decisions (12 confirmed fraud, 4 false positive -> total 15 confirmed, 5 FP = 20 decided)
        extra_cases = [
            (f"CF_{i}", "upi", f"E{10+i}", None, 100.0, 0.9, "High", "2026-11-10", "CONFIRMED_FRAUD", "", "2026-11-10", "{}")
            for i in range(12)
        ] + [
            (f"FP_{i}", "card", f"E{30+i}", None, 50.0, 0.85, "High", "2026-11-10", "FALSE_POSITIVE", "", "2026-11-10", "{}")
            for i in range(4)
        ]
        with get_db_connection() as conn:
            conn.executemany("""
                INSERT INTO cases (case_id, channel, entity_id, counterparty_id, amount,
                                   risk_score, risk_band, timestamp, status, analyst_notes, updated_at, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, extra_cases)
            conn.commit()

        metrics_sufficient = get_analyst_precision_metrics()
        assert metrics_sufficient["total_decided"] == 20
        assert metrics_sufficient["confirmed_fraud"] == 15
        assert metrics_sufficient["false_positives"] == 5
        # Precision: 15 / 20 = 75.0%
        assert metrics_sufficient["precision_of_alerts_pct"] == 75.0
        assert metrics_sufficient["precision_label"] == "75.0%"

    def test_feedback_loop_integration(self):
        """Verify feeding confirmed fraud labels triggers UPI model feedback calculation."""
        res_empty = feed_confirmed_labels_to_feedback_loop()
        assert res_empty["status"] == "no_confirmed_labels"

        with get_db_connection() as conn:
            conn.execute("""
                INSERT INTO cases (case_id, channel, entity_id, amount, risk_score, risk_band, timestamp, status, updated_at, payload_json)
                VALUES ('UPI-01', 'upi', 'user@okhdfcbank', 50000.0, 0.92, 'High', '2026-11-10', 'CONFIRMED_FRAUD', '2026-11-10', '{}')
            """)
            conn.commit()

        res_active = feed_confirmed_labels_to_feedback_loop()
        assert res_active["status"] == "success"
        assert res_active["confirmed_cases_ingested"] >= 1
        assert res_active["updated_unseen_recall_pct"] > 0.94


# =============================================================================
# 3. API LATENCY RECORDER & FASTAPI SCORING MICROSERVICE TESTS
# =============================================================================
class TestAPILatencyAndMicroservice:
    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_health_check_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "transfer_wallet" in data["channels_available"]

    def test_thresholds_endpoint(self, client):
        resp = client.get("/thresholds")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        from ml.metrics_helper import get_calibrated_thresholds
        for ch in ["transfer_wallet", "card", "account", "upi"]:
            assert ch in data["thresholds"]
            exp_low, exp_high = get_calibrated_thresholds(ch)
            assert data["thresholds"][ch]["low_risk_auto_approve"] == exp_low
            assert data["thresholds"][ch]["high_risk_decline"] == exp_high

    def test_score_endpoint_success(self, client):
        payload = {
            "channel": "transfer_wallet",
            "payload": {
                "step": 500,
                "type": "TRANSFER",
                "amount": 250000.0,
                "oldbalanceOrg": 250000.0,
                "oldbalanceDest": 0.0,
                "nameOrig": "C_TEST_01",
                "nameDest": "M_TEST_01",
            },
            "top_k": 3,
            "update_history": False,
        }
        resp = client.post("/score", json=payload)
        assert resp.status_code == 200
        res = resp.json()
        assert res["status"] == "success"
        assert 0.0 <= res["risk_score"] <= 1.0
        assert res["band"] in ["Low", "Medium", "High"]
        assert len(res["top_reasons"]) > 0
        assert res["latency_ms"] >= 0.0

    def test_score_endpoint_negative_amount_rejection(self, client):
        payload = {
            "channel": "transfer_wallet",
            "payload": {"amount": -100.0, "oldbalanceOrg": 100.0},
        }
        resp = client.post("/score", json=payload)
        assert resp.status_code == 422

    def test_score_endpoint_invalid_channel_rejection(self, client):
        payload = {
            "channel": "unsupported_channel_xyz",
            "payload": {"amount": 100.0},
        }
        resp = client.post("/score", json=payload)
        assert resp.status_code == 422

    def test_latency_benchmark_sampler(self):
        """Verify the measure_api_latency runner captures valid distributions."""
        bench = run_latency_benchmark(num_calls=12, save_to_file=False)
        assert bench["sample_size"] == 12
        assert bench["p50_ms"] >= 0.0
        assert bench["p95_ms"] >= bench["p50_ms"]
        assert "roundtrip_p50_ms" in bench
        assert "by_channel_p50" in bench
        for ch in ["transfer_wallet", "card", "account", "upi"]:
            assert ch in bench["by_channel_p50"]


# =============================================================================
# 4. PRINTABLE CASE REPORT GENERATOR TESTS
# =============================================================================
class TestCaseReportGenerator:
    def test_case_report_html_content(self):
        """Verify generation of printable HTML case report."""
        payload = {"amount": 450000.0, "type": "TRANSFER", "nameOrig": "C123"}
        result = {
            "risk_score": 0.985,
            "band": "High",
            "action": "Immediate Block & Account Freeze",
            "data_source": "Simulated (agent-based)",
            "top_reasons": ["Origin balance drained to ₹0.00", "Amount exceeds 99th percentile"],
            "counterfactual": "Reduce transfer amount below ₹50,000 to lower risk score.",
        }

        html_content = generate_case_report_html(
            channel="transfer_wallet",
            payload=payload,
            result=result,
            low_th=0.45,
            high_th=0.90,
            report_id="FS-TEST-9999",
        )

        assert "<!DOCTYPE html>" in html_content
        assert "FRAUDSHIELD TRANSACTION DECISION REPORT" in html_content
        assert "FS-TEST-9999" in html_content
        assert "₹450,000.00" in html_content
        assert "0.9850" in html_content
        assert "HIGH" in html_content
        assert "Origin balance drained to ₹0.00" in html_content
        assert "Reduce transfer amount below ₹50,000" in html_content
        assert "Audited Machine Learning Pipeline · No Handcrafted Overrides" in html_content
        assert "@media print" in html_content

    def test_case_report_card_defaults(self):
        """Verify case report handles card channel and missing reasons gracefully."""
        payload = {"Amount": 12.50}
        result = {
            "risk_score": 0.02,
            "band": "Low",
            "action": "Approve",
            "data_source": "Real, anonymised",
        }

        html_content = generate_case_report_html(
            channel="card",
            payload=payload,
            result=result,
        )

        assert "₹12.50" in html_content
        assert "LOW" in html_content
        assert "Normal authorization pattern" in html_content


# =============================================================================
# 5. MULTI-CHANNEL ROUTER & ALIASES TESTS
# =============================================================================
class TestDecisionRouter:
    def test_all_registered_aliases(self):
        """Verify all aliases in CHANNEL_ALIASES resolve to one of 4 canonical channels."""
        canonical_channels = {"card", "transfer_wallet", "account", "upi"}
        for alias, canonical in CHANNEL_ALIASES.items():
            assert canonical in canonical_channels
            assert resolve_channel(alias) == canonical

    def test_case_and_whitespace_normalization(self):
        """Verify router normalizes dashes, spaces, and mixed case."""
        assert resolve_channel("  Credit-Card  ") == "card"
        assert resolve_channel("TRANSFER_WALLET") == "transfer_wallet"
        assert resolve_channel("Bank-Account-Fraud") == "account"
        assert resolve_channel("  uPi_pAyMeNt  ") == "upi"

    def test_unknown_channel_error(self):
        """Verify unknown channel raises descriptive ValueError."""
        with pytest.raises(ValueError, match="Unknown channel: 'crypto_rail'"):
            resolve_channel("crypto_rail")

    def test_get_channel_registry_and_info(self):
        """Verify registry inspection functions."""
        reg = get_channel_registry()
        assert len(reg) == 4
        info = get_channel_info("upi")
        assert "name" in info
        assert "data_source" in info
        assert info["data_source"] == "Simulated by us"

    def test_predict_across_all_canonical_channels(self):
        """Verify end-to-end predict() execution for each channel."""
        samples = {
            "transfer_wallet": {
                "step": 1, "type": "TRANSFER", "amount": 1000.0,
                "oldbalanceOrg": 1000.0, "newbalanceOrig": 0.0,
                "nameOrig": "C10", "nameDest": "M20",
                "oldbalanceDest": 0.0, "newbalanceDest": 0.0
            },
            "card": {
                "Time": 1000.0, "Amount": 55.0, "V1": 0.1, "V2": -0.2, "V14": -0.5
            },
            "account": {
                "income": 0.6, "credit_risk_score": 120, "proposed_credit_limit": 3000.0
            },
            "upi": {
                "amount": 2500.0, "hour": 15, "txn_type": "PAY",
                "payer_vpa": "a@upi", "payee_vpa": "b@upi", "is_first_time_payee": 0
            },
        }

        for ch, payload in samples.items():
            res = FraudRouter.predict(channel=ch, payload=payload, top_k=2)
            assert res["channel"] == ch
            assert 0.0 <= res["risk_score"] <= 1.0
            assert res["band"] in ["Low", "Medium", "High"]
            assert isinstance(res["action"], str) and len(res["action"]) > 0
            assert "top_reasons" in res
            assert "data_source" in res
            assert res["latency_ms"] >= 0.0
