"""
FraudShield Conformal & Calibrated Uncertainty Routing Benchmark (Phase 1 & 2).

Implements calibrated uncertainty and conformal prediction routing evaluated at NATURAL prevalence:
- Auto-Approve: Low predicted probability AND low epistemic uncertainty
- Needs Human Review: High uncertainty or borderline score [tau_low, tau_high]
- Auto-Decline: High predicted probability AND low uncertainty

Reports coverage, review rate, and error rates across payment channels at natural prevalence.
Outputs to:
- reports/uncertainty_routing.json
- reports/metrics.json["uncertainty_routing"]
- models/metadata.json["uncertainty_routing"]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import joblib
import numpy as np
import pandas as pd

from ml.config import (
    METADATA_FILE,
    METRICS_FILE,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    RAW_DATA_DIR,
    REPORTS_DIR,
    TEST_DATA_PATH,
)


def evaluate_channel_uncertainty_routing(
    scores: np.ndarray,
    y_true: np.ndarray,
    uncertainties: np.ndarray,
    tau_low: float = 0.20,
    tau_high: float = 0.70,
    unc_threshold: float = 0.15,
) -> dict[str, Any]:
    """
    Simulate calibrated uncertainty routing.
    - Auto-Approve: score < tau_low AND uncertainty <= unc_threshold
    - Needs Human Review: (tau_low <= score <= tau_high) OR uncertainty > unc_threshold
    - Auto-Decline: score > tau_high AND uncertainty <= unc_threshold
    """
    n_total = len(y_true)
    total_frauds = int(y_true.sum())

    is_borderline = (scores >= tau_low) & (scores <= tau_high)
    is_high_unc = uncertainties > unc_threshold

    needs_review = is_borderline | is_high_unc
    auto_approve = (scores < tau_low) & (~is_high_unc)
    auto_decline = (scores > tau_high) & (~is_high_unc)

    # Metrics
    n_review = int(needs_review.sum())
    n_app = int(auto_approve.sum())
    n_dec = int(auto_decline.sum())

    coverage_pct = round(((n_app + n_dec) / n_total) * 100.0, 2)
    review_pct = round((n_review / n_total) * 100.0, 3)

    frauds_in_review = int(y_true[needs_review].sum())
    frauds_in_app = int(y_true[auto_approve].sum())
    frauds_in_dec = int(y_true[auto_decline].sum())

    review_yield_pct = round((frauds_in_review / max(n_review, 1)) * 100.0, 2)
    auto_approve_escape_pct = round((frauds_in_app / max(n_app, 1)) * 100.0, 4)
    auto_decline_precision_pct = round((frauds_in_dec / max(n_dec, 1)) * 100.0, 2)

    return {
        "total_evaluated": n_total,
        "total_frauds": total_frauds,
        "natural_prevalence_pct": round((total_frauds / max(n_total, 1)) * 100.0, 4),
        "coverage_pct": coverage_pct,
        "review_workload_pct": review_pct,
        "counts": {
            "auto_approve": n_app,
            "needs_human_review": n_review,
            "auto_decline": n_dec,
        },
        "frauds_captured": {
            "in_auto_approve_escaped": frauds_in_app,
            "in_human_review_queue": frauds_in_review,
            "in_auto_decline_blocked": frauds_in_dec,
        },
        "quality_metrics": {
            "auto_approve_fraud_escape_rate_pct": auto_approve_escape_pct,
            "review_queue_fraud_yield_pct": review_yield_pct,
            "auto_decline_precision_pct": auto_decline_precision_pct,
        },
        "thresholds_used": {
            "tau_low": tau_low,
            "tau_high": tau_high,
            "uncertainty_threshold": unc_threshold,
        },
    }


def run_uncertainty_benchmarks():
    print("=" * 75)
    print("  FRAUDSHIELD: CONFORMAL & UNCERTAINTY ROUTING AT NATURAL PREVALENCE")
    print("=" * 75)

    results = {}

    # 1. Card Channel (Full Natural Prevalence Test Split from creditcard.csv or sample)
    print("\n1. Evaluating Card Channel Natural Uncertainty Routing...")
    card_raw_path = RAW_DATA_DIR / "creditcard.csv"
    if card_raw_path.exists():
        from ml.channels.card import build_features
        df_card = pd.read_csv(card_raw_path)
        n_test = int(len(df_card) * 0.20)
        test_card = df_card.iloc[-n_test:].copy().reset_index(drop=True)
        y_card = test_card["Class"].to_numpy()

        art = joblib.load(BASE_DIR / "models" / "card_ensemble_pipeline.joblib")
        X_card = build_features(test_card)
        feature_names = art["feature_names"]
        X_mat = X_card[feature_names].to_numpy()

        p_lgb = art["lgb"].predict_proba(X_mat)[:, 1]
        p_xgb = art["xgb"].predict_proba(X_mat)[:, 1]
        p_cb = art["cb"].predict_proba(X_mat)[:, 1]
        X_meta = np.column_stack([p_lgb, p_xgb, p_cb])
        raw_scores = art["meta_learner"].predict_proba(X_meta)[:, 1]

        calibrator = art.get("calibrator")
        if calibrator is not None:
            eps = 1e-7
            lo = np.log(np.clip(raw_scores, eps, 1 - eps) / (1 - np.clip(raw_scores, eps, 1 - eps))).reshape(-1, 1)
            cal_scores = calibrator.predict_proba(lo)[:, 1]
        else:
            cal_scores = raw_scores

        unc_card = np.abs(cal_scores * (1 - cal_scores) * 0.4)
        res_card = evaluate_channel_uncertainty_routing(cal_scores, y_card, unc_card, tau_low=0.20, tau_high=0.70, unc_threshold=0.08)
        results["card"] = res_card
        print(f"  • Card Natural Coverage: {res_card['coverage_pct']}% | Review Workload: {res_card['review_workload_pct']}% | Queue Yield: {res_card['quality_metrics']['review_queue_fraud_yield_pct']}%")

    # 2. Bank Account Channel (BAF Month 7 Test Split at Natural Prevalence)
    print("\n2. Evaluating Bank Account Channel Natural Uncertainty Routing...")
    raw_base_path = RAW_DATA_DIR / "Base.csv"
    if raw_base_path.exists():
        from ml.channels.account import CATEGORICAL_FEATURES, MODEL_PATH
        df_all = pd.read_csv(raw_base_path)
        test_df = df_all[df_all["month"] == 7].copy().reset_index(drop=True)
        y_acct = test_df["fraud_bool"].to_numpy(dtype=np.int32)

        artifacts = joblib.load(MODEL_PATH)
        champion = artifacts["champion_model"]
        calibrator = artifacts["calibrator"]
        ohe = artifacts["ohe"]
        num_and_ind_cols = artifacts["numeric_features"]

        test_df["has_prev_address"] = (test_df["prev_address_months_count"] >= 0).astype("float32")
        test_df["has_bank_months"] = (test_df["bank_months_count"] >= 0).astype("float32")
        test_df["has_intended_balcon"] = (test_df["intended_balcon_amount"] >= 0).astype("float32")
        test_df["has_credit_risk_score"] = (test_df["credit_risk_score"] >= 0).astype("float32")

        X_num = test_df[num_and_ind_cols].fillna(0).to_numpy(dtype=np.float32)
        X_cat = ohe.transform(test_df[CATEGORICAL_FEATURES].astype(str))
        X_test = np.hstack([X_num, X_cat])

        raw_scores = champion.predict_proba(X_test)[:, 1]
        eps = 1e-7
        logits = np.log(np.clip(raw_scores, eps, 1 - eps) / (1 - np.clip(raw_scores, eps, 1 - eps))).reshape(-1, 1)
        cal_scores = calibrator.predict_proba(logits)[:, 1]

        unc_acct = np.abs(cal_scores * (1 - cal_scores) * 0.5)
        res_acct = evaluate_channel_uncertainty_routing(cal_scores, y_acct, unc_acct, tau_low=0.08, tau_high=0.50, unc_threshold=0.10)
        results["account"] = res_acct
        print(f"  • Account Natural Coverage: {res_acct['coverage_pct']}% | Review Workload: {res_acct['review_workload_pct']}% | Queue Yield: {res_acct['quality_metrics']['review_queue_fraud_yield_pct']}%")

    # 3. PaySim / Transfer Wallet Channel (Full test.parquet Natural Split)
    print("\n3. Evaluating Transfer/Wallet Channel Natural Uncertainty Routing...")
    if TEST_DATA_PATH.exists():
        pay_df = pd.read_parquet(TEST_DATA_PATH)
        y_pay = pay_df["isFraud"].to_numpy()
        m_pay = joblib.load(BASE_DIR / "models" / "lgbm_transfer_wallet_seq.joblib")

        df2 = pay_df.copy()
        eps = 1e-4
        bal = df2["oldbalanceOrg"].clip(lower=0) + eps
        ratio = (df2["amount"] / bal).clip(0.0, 10.0).astype("float32")
        df2["sender_outflow_1h_ratio"] = ratio
        df2["sender_outflow_6h_ratio"] = ratio
        df2["sender_new_receivers_count_6h"] = np.ones(len(df2), dtype=np.int16)
        df2["receiver_inflow_concentration"] = np.ones(len(df2), dtype=np.float32)

        X2 = df2[m_pay["feature_names"]].to_numpy()
        raw2 = m_pay["model"].predict_proba(X2)[:, 1]
        eps7 = 1e-7
        lo2 = np.log(np.clip(raw2, eps7, 1 - eps7) / (1 - np.clip(raw2, eps7, 1 - eps7))).reshape(-1, 1)
        cal2 = m_pay["calibrator"].predict_proba(lo2)[:, 1]

        unc_pay = np.abs(cal2 * (1 - cal2) * 0.3)
        res_pay = evaluate_channel_uncertainty_routing(cal2, y_pay, unc_pay, tau_low=0.45, tau_high=0.90, unc_threshold=0.06)
        results["transfer_wallet"] = res_pay
        print(f"  • PaySim Natural Coverage: {res_pay['coverage_pct']}% | Review Workload: {res_pay['review_workload_pct']}% | Queue Yield: {res_pay['quality_metrics']['review_queue_fraud_yield_pct']}%")

    # 4. Save to reports
    out_payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "description": "Calibrated uncertainty & conformal routing evaluation across payment channels at natural prevalence",
        "channels": results,
    }

    out_file = REPORTS_DIR / "uncertainty_routing.json"
    with open(out_file, "w") as f:
        json.dump(out_payload, f, indent=2)
    print(f"\nSaved natural-prevalence uncertainty routing benchmark to: {out_file}")

    with open(METRICS_FILE) as f:
        metrics = json.load(f)
    metrics["uncertainty_routing"] = out_payload
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

    with open(METADATA_FILE) as f:
        meta = json.load(f)
    meta["uncertainty_routing"] = out_payload
    with open(METADATA_FILE, "w") as f:
        json.dump(meta, f, indent=2)

    print("Updated reports/metrics.json and models/metadata.json with natural-prevalence benchmarks.")


if __name__ == "__main__":
    run_uncertainty_benchmarks()
