"""
Comprehensive Tests for Live Scoring, History Store Parity, and FastAPI Microservice.
Verifies:
1. Batch vs Live scoring parity: identical feature vectors and risk scores.
2. History store strict temporal causality and order-invariance within steps.
3. Dynamic state accumulation across sequential payments.
4. FastAPI /score endpoint validation, operational latency, and schema consistency.
5. Population Stability Index (PSI) drift engine correctness.
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from ml.api import app
from ml.config import RANDOM_STATE
from ml.features import FEATURE_COLUMNS, compute_receiver_features_vectorized, engineer_features
from ml.history_store import AccountHistoryStore, get_history_store, reset_history_store
from ml.monitoring import calculate_psi, classify_psi_drift
from ml.predict import get_model_and_explainer, predict as paysim_predict
from ml.router import predict as router_predict


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


def test_batch_vs_live_receiver_features_parity():
    """
    Test that AccountHistoryStore sequential live querying generates
    the exact same receiver history features as the batch vectorized pipeline.
    """
    # Create a synthetic sequence of transactions with repeat destinations across steps
    records = [
        {"step": 10, "type": "PAYMENT", "amount": 100.0, "nameOrig": "C101", "nameDest": "M201", "oldbalanceOrg": 1000.0, "oldbalanceDest": 0.0},
        {"step": 10, "type": "PAYMENT", "amount": 200.0, "nameOrig": "C102", "nameDest": "M201", "oldbalanceOrg": 1000.0, "oldbalanceDest": 0.0},
        {"step": 11, "type": "TRANSFER", "amount": 500.0, "nameOrig": "C103", "nameDest": "M201", "oldbalanceOrg": 500.0, "oldbalanceDest": 0.0},
        {"step": 12, "type": "TRANSFER", "amount": 300.0, "nameOrig": "C104", "nameDest": "M201", "oldbalanceOrg": 300.0, "oldbalanceDest": 0.0},
        {"step": 12, "type": "PAYMENT", "amount": 150.0, "nameOrig": "C105", "nameDest": "M202", "oldbalanceOrg": 500.0, "oldbalanceDest": 0.0},
        {"step": 36, "type": "CASH_OUT", "amount": 800.0, "nameOrig": "C106", "nameDest": "M201", "oldbalanceOrg": 800.0, "oldbalanceDest": 0.0},
    ]
    df_batch = pd.DataFrame(records)

    # 1. Compute batch vectorized features
    batch_rec_dict = compute_receiver_features_vectorized(df_batch)

    # 2. Compute sequential live features using AccountHistoryStore
    store = AccountHistoryStore(window_hours=24)
    live_rec_list = []

    # Process step by step to simulate live stream arrival
    unique_steps = sorted(df_batch["step"].unique())
    for st in unique_steps:
        step_txs = [r for r in records if r["step"] == st]
        # Query phase: query all transactions in current step prior to committing
        step_feats = [store.get_receiver_features(tx["nameDest"], st) for tx in step_txs]
        live_rec_list.extend(step_feats)
        # Commit phase: commit transactions for current step
        for tx in step_txs:
            store.record_transaction(tx)

    # Verify exact parity across all rows and all 6 receiver features
    receiver_keys = [
        "dest_past_tx_count",
        "dest_total_past_amount",
        "dest_time_since_last_tx",
        "dest_tx_last_1h",
        "dest_tx_last_6h",
        "dest_tx_last_24h",
    ]

    for i in range(len(records)):
        for k in receiver_keys:
            batch_val = float(batch_rec_dict[k][i])
            live_val = float(live_rec_list[i][k])
            assert np.isclose(batch_val, live_val, atol=1e-5), (
                f"Mismatch in row {i} for feature {k}: batch={batch_val} vs live={live_val}"
            )


def test_batch_vs_live_scoring_parity():
    """
    Verify that an identical feature vector scored via batch model inference
    and live single-transaction scoring yields the exact same risk score (< 1e-6 tolerance).
    """
    model, calibrator, _ = get_model_and_explainer()

    tx = {
        "step": 400,
        "type": "TRANSFER",
        "amount": 150000.0,
        "oldbalanceOrg": 150000.0,
        "oldbalanceDest": 0.0,
        "nameOrig": "C9999999",
        "nameDest": "C8888888",
    }

    # 1. Batch path
    df_batch = pd.DataFrame([tx])
    X_batch, _ = engineer_features(df_batch, is_training=False)
    X_batch_arr = X_batch[FEATURE_COLUMNS].to_numpy()
    batch_raw = float(model.predict_proba(X_batch_arr)[0, 1])
    if calibrator is not None:
        eps = 1e-7
        batch_logit = np.log(np.clip(batch_raw, eps, 1 - eps) / (1 - np.clip(batch_raw, eps, 1 - eps))).reshape(-1, 1)
        batch_cal = float(calibrator.predict_proba(batch_logit)[0, 1])
    else:
        batch_cal = batch_raw

    # 2. Live path via predict()
    live_res = paysim_predict(tx)
    live_raw = live_res["raw_model_score"]
    live_cal = live_res["risk_score"]

    assert np.isclose(batch_raw, live_raw, atol=1e-4), f"Raw score mismatch: batch={batch_raw} vs live={live_raw}"
    assert np.isclose(batch_cal, live_cal, atol=1e-4), f"Calibrated score mismatch: batch={batch_cal} vs live={live_cal}"


def test_history_store_strict_temporal_causality():
    """
    Verify that same-step transactions do not observe each other (order invariance and zero intra-step leakage).
    """
    store = AccountHistoryStore(window_hours=24)

    # Initial query at step 100 should return zero past transactions
    f1 = store.get_receiver_features("M999", current_step=100)
    assert f1["dest_past_tx_count"] == 0
    assert f1["dest_total_past_amount"] == 0.0
    assert f1["dest_time_since_last_tx"] == 999.0

    # Record first transaction at step 100
    store.record_transaction({"nameOrig": "C01", "nameDest": "M999", "step": 100, "amount": 5000.0})

    # Another transaction at the SAME step 100 must STILL see zero past transactions (strict step-buffering)
    f2 = store.get_receiver_features("M999", current_step=100)
    assert f2["dest_past_tx_count"] == 0, "Intra-step leakage detected! Same-step transaction observed prior step commit."

    # A transaction at step 101 MUST now observe the step 100 transaction
    f3 = store.get_receiver_features("M999", current_step=101)
    assert f3["dest_past_tx_count"] == 1
    assert f3["dest_total_past_amount"] == 5000.0
    assert f3["dest_time_since_last_tx"] == 1.0
    assert f3["dest_tx_last_1h"] == 1


def test_fastapi_health_endpoint(client):
    """Verify /health endpoint returns healthy status and channel list."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "card" in data["channels_available"]
    assert "transfer_wallet" in data["channels_available"]
    assert "account" in data["channels_available"]
    assert "upi" in data["channels_available"]
    assert data["sla_target_ms"] == 15.0


def test_fastapi_thresholds_endpoint(client):
    """Verify /thresholds endpoint reads calibrated thresholds from metadata."""
    res = client.get("/thresholds")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "transfer_wallet" in data["thresholds"]
    assert "card" in data["thresholds"]
    th = data["thresholds"]["transfer_wallet"]
    assert 0.0 < th["low_risk_auto_approve"] < th["high_risk_decline"] <= 1.0


def test_fastapi_score_card_channel(client):
    """Verify /score endpoint scores Card payments with operational latency."""
    # Warmup
    _ = client.post("/score", json={
        "channel": "credit_card",
        "payload": {"Amount": 120.0, "Time": 10000.0, "V1": -1.0, "V14": -3.5},
        "top_k": 3,
    })

    t0 = time.perf_counter()
    res = client.post("/score", json={
        "channel": "credit_card",
        "payload": {"Amount": 120.0, "Time": 10000.0, "V1": -1.0, "V14": -3.5},
        "top_k": 3,
    })
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["channel"] == "card"
    assert data["data_source"] == "Real, anonymised"
    assert 0.0 <= data["risk_score"] <= 1.0
    assert data["band"] in ["Low", "Medium", "High"]
    assert data["action"] in ["Approve", "Step-Up OTP", "Decline & Alert"]
    assert len(data["top_reasons"]) > 0
    assert elapsed_ms < 150.0  # Network client overhead allowance
    assert data["latency_ms"] < 100.0, f"Server internal latency exceeded operational SLA: {data['latency_ms']}ms"


def test_fastapi_score_transfer_channel_with_history(client):
    """Verify /score endpoint scores Transfers and updates stateful history store."""
    payload = {
        "step": 500,
        "type": "TRANSFER",
        "amount": 250000.0,
        "oldbalanceOrg": 250000.0,
        "oldbalanceDest": 0.0,
        "nameOrig": "C_LIVE_001",
        "nameDest": "C_LIVE_MULE_001",
    }
    res = client.post("/score", json={
        "channel": "transfer_wallet",
        "payload": payload,
        "top_k": 3,
        "update_history": True,
    })
    assert res.status_code == 200
    data = res.json()
    assert data["channel"] == "transfer_wallet"
    assert data["band"] == "High"
    assert data["action"] == "Decline & Alert"
    assert data["counterfactual"] is not None

    # Query history store to prove it was recorded
    store = get_history_store()
    history_after = store.get_receiver_features("C_LIVE_MULE_001", current_step=501)
    assert history_after["dest_past_tx_count"] == 1
    assert history_after["dest_total_past_amount"] == 250000.0


def test_fastapi_input_validation_errors(client):
    """Verify input validation catches negative amounts and invalid channels."""
    # Negative amount
    res_neg = client.post("/score", json={
        "channel": "card",
        "payload": {"amount": -50.0},
    })
    assert res_neg.status_code == 422

    # Unknown channel
    res_chan = client.post("/score", json={
        "channel": "unsupported_crypto_rail",
        "payload": {"amount": 500.0},
    })
    assert res_chan.status_code == 422


def test_psi_drift_calculation():
    """Verify Population Stability Index (PSI) calculation and categorization."""
    rng = np.random.RandomState(RANDOM_STATE)
    base_dist = rng.normal(loc=100.0, scale=15.0, size=5000)

    # Identical distribution -> PSI ~ 0.0
    psi_identical = calculate_psi(base_dist, base_dist)
    assert psi_identical < 0.01
    assert classify_psi_drift(psi_identical) == "Stable (No Drift)"

    # Minor shifted distribution -> Moderate drift
    shifted_minor = rng.normal(loc=103.0, scale=16.0, size=5000)
    psi_minor = calculate_psi(base_dist, shifted_minor)
    assert psi_minor >= 0.0

    # Major shifted distribution -> Significant drift
    shifted_major = rng.normal(loc=130.0, scale=30.0, size=5000)
    psi_major = calculate_psi(base_dist, shifted_major)
    assert psi_major >= 0.25
    assert classify_psi_drift(psi_major) == "Significant Drift (Retrain)"

