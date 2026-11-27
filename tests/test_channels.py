"""
Comprehensive Unit & Integration Tests for FraudShield Multi-Channel Architecture.
Tests:
- Router alias resolution and error handling
- End-to-end real-time scoring across all 4 channels
- Scoring latency verification across channels
- Unified metrics helper and cross-channel summary table consistency
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pytest

from ml.config import BASE_DIR, CHANNELS, REPORTS_DIR
from ml.metrics_helper import (
    get_calibrated_thresholds,
    get_champion_metrics,
    get_cross_channel_summary,
    get_economics_summary,
)
from ml.router import get_channel_info, get_channel_registry, predict, resolve_channel


def test_channel_registry_completeness():
    """Verify all 4 canonical channels are configured with required properties."""
    registry = get_channel_registry()
    expected_channels = {"card", "transfer_wallet", "account", "upi"}
    assert set(registry.keys()) == expected_channels

    for ch_id, ch_info in registry.items():
        assert ch_info["id"] == ch_id
        assert "name" in ch_info
        assert "data_source" in ch_info
        assert "dataset_path" in ch_info
        assert "cost_params" in ch_info
        assert "otp_challenge_inr" in ch_info["cost_params"]
        assert "false_decline_inr" in ch_info["cost_params"]


@pytest.mark.parametrize(
    "alias,expected",
    [
        ("card", "card"),
        ("credit_card", "card"),
        ("creditcard", "card"),
        ("debit_card", "card"),
        ("cards", "card"),
        ("transfer_wallet", "transfer_wallet"),
        ("transfer", "transfer_wallet"),
        ("wallet", "transfer_wallet"),
        ("paysim", "transfer_wallet"),
        ("bank_transfer", "transfer_wallet"),
        ("account", "account"),
        ("bank_account", "account"),
        ("baf", "account"),
        ("account_opening", "account"),
        ("upi", "upi"),
        ("bhim", "upi"),
        ("upi_payment", "upi"),
    ],
)
def test_router_alias_resolution(alias: str, expected: str):
    """Verify that colloquial channel aliases resolve to canonical identifiers."""
    assert resolve_channel(alias) == expected


def test_router_unknown_channel():
    """Verify that invalid channel strings raise an informative ValueError."""
    with pytest.raises(ValueError) as excinfo:
        resolve_channel("cryptocurrency_rail")
    assert "Unknown channel: 'cryptocurrency_rail'" in str(excinfo.value)
    assert "transfer_wallet" in str(excinfo.value)


def test_card_channel_predict():
    """Test Card payment scoring end-to-end via router."""
    payload = {
        "Amount": 149.50,
        "Time": 45000.0,
        "V1": -1.2,
        "V2": 0.5,
        "V3": -0.8,
        "V4": 1.1,
        "V14": -5.2,
    }
    # Warm-up calls
    _ = predict("credit_card", payload, top_k=3)
    _ = predict("credit_card", payload, top_k=3)

    t0 = time.perf_counter()
    res = predict("credit_card", payload, top_k=3)
    latency_ms = (time.perf_counter() - t0) * 1000

    assert res["channel"] == "card"
    assert res["data_source"] == "Real, anonymised"
    assert 0.0 <= res["risk_score"] <= 1.0
    assert res["band"] in ["Low", "Medium", "High"]
    assert res["action"] in ["Approve", "Step-Up OTP", "Decline & Alert"]
    assert len(res["top_reasons"]) > 0
    assert len(res["factors"]) > 0
    assert latency_ms < 100.0, f"Card scoring latency exceeded operational SLA: {latency_ms:.2f}ms"


def test_transfer_wallet_channel_predict():
    """Test Transfer & Wallet (PaySim) scoring end-to-end via router."""
    payload = {
        "type": "TRANSFER",
        "amount": 250000.0,
        "oldbalanceOrg": 250000.0,
        "step": 400,
        "nameDest": "C123456789",
    }
    # Warm-up calls
    _ = predict("bank_transfer", payload, top_k=3)
    _ = predict("bank_transfer", payload, top_k=3)

    t0 = time.perf_counter()
    res = predict("bank_transfer", payload, top_k=3)
    latency_ms = (time.perf_counter() - t0) * 1000

    assert res["channel"] == "transfer_wallet"
    assert res["data_source"] == "Simulated (agent-based)"
    assert 0.0 <= res["risk_score"] <= 1.0
    assert res["band"] in ["Low", "Medium", "High"]
    assert len(res["top_reasons"]) > 0
    assert latency_ms < 100.0, f"Transfer scoring latency exceeded operational SLA: {latency_ms:.2f}ms"


def test_account_channel_predict():
    """Test Bank Account opening fraud scoring end-to-end via router."""
    payload = {
        "income": 0.4,
        "name_email_similarity": 0.15,
        "prev_address_months_count": -1,
        "current_address_months_count": 12,
        "days_since_request": 0.01,
        "phone_home_valid": 0,
        "email_is_free": 1,
        "device_fraud_count": 1,
        "proposed_credit_limit": 1500.0,
        "payment_type": "AB",
        "employment_status": "CA",
        "housing_status": "BC",
        "source": "INTERNET",
        "device_os": "windows",
    }
    # Warm-up calls
    _ = predict("baf", payload, top_k=3)
    _ = predict("baf", payload, top_k=3)

    t0 = time.perf_counter()
    res = predict("baf", payload, top_k=3)
    latency_ms = (time.perf_counter() - t0) * 1000

    assert res["channel"] == "account"
    assert res["data_source"] == "Privacy-preserving synthetic"
    assert 0.0 <= res["risk_score"] <= 1.0
    assert res["band"] in ["Low", "Medium", "High"]
    assert res["action"] in ["Auto-Approve", "Enhanced KYC", "Reject & KYC Flag"]
    assert len(res["top_reasons"]) > 0
    assert len(res["factors"]) > 0
    assert latency_ms < 100.0, f"Account scoring latency exceeded operational SLA: {latency_ms:.2f}ms"


def test_upi_channel_predict():
    """Test UPI real-time scoring end-to-end via router."""
    payload = {
        "amount": 4999.0,
        "txn_type": "COLLECT_REQUEST",
        "hour": 15.0,
        "payer_vpa_age_days": 180,
        "payee_vpa_age_days": 2,
        "is_first_time_payee": 1,
        "is_new_device": 0,
        "sim_change_last_48h": 0,
        "remote_access_app_flag": 0,
        "velocity_1h": 1,
        "velocity_24h": 2,
        "payer_handle": "okhdfcbank",
        "payee_handle": "paytm",
    }
    # Warm-up calls
    _ = predict("bhim", payload, top_k=3)
    _ = predict("bhim", payload, top_k=3)

    t0 = time.perf_counter()
    res = predict("bhim", payload, top_k=3)
    latency_ms = (time.perf_counter() - t0) * 1000

    assert res["channel"] == "upi"
    assert res["data_source"] == "Simulated by us"
    assert 0.0 <= res["risk_score"] <= 1.0
    assert res["band"] in ["Low", "Medium", "High"]
    assert res["action"] in ["Auto-Approve", "Step-Up MPIN", "Block & Alert"]
    assert len(res["top_reasons"]) > 0
    assert len(res["factors"]) > 0
    assert latency_ms < 100.0, f"UPI scoring latency exceeded operational SLA: {latency_ms:.2f}ms"


def test_metrics_helper_cross_channel_summary():
    """Verify that get_cross_channel_summary generates valid summary table and CSV."""
    df = get_cross_channel_summary(save_csv=True)

    expected_cols = [
        "Channel",
        "Data source",
        "Benchmark name",
        "Test rows",
        "Test prevalence",
        "Champion PR-AUC",
        "95% CI",
        "Recall",
        "False alarms / 100k",
        "INR value protected %",
    ]
    assert list(df.columns) == expected_cols
    assert len(df) == 4

    channels = df["Channel"].tolist()
    assert "Card Payments" in channels
    assert "Online Transfers & Wallets" in channels
    assert "Bank Accounts" in channels
    assert "UPI Transactions" in channels

    # Check that CSV was written to disk
    csv_path = REPORTS_DIR / "cross_channel_summary.csv"
    assert csv_path.exists()
    assert csv_path.stat().st_size > 0


def test_metrics_helper_per_channel_access():
    """Verify that metrics_helper functions accept channel parameters."""
    for ch in ["card", "transfer_wallet", "account", "upi"]:
        low, high = get_calibrated_thresholds(ch)
        assert 0.0 < low < high <= 1.0

        champ = get_champion_metrics(ch)
        assert "pr_auc" in champ
        assert float(champ["pr_auc"]) > 0.0

        econ = get_economics_summary(ch)
        assert len(econ) > 0

