from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd

from ml.config import DEFAULT_HIGH_THRESHOLD, DEFAULT_LOW_THRESHOLD, METADATA_FILE
from ml.explain import explain_transaction, get_model_and_explainer
from ml.features import FEATURE_COLUMNS, engineer_features

_THRESHOLDS = None


def get_thresholds() -> tuple[float, float]:
    global _THRESHOLDS
    if _THRESHOLDS is None:
        if METADATA_FILE.exists():
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                meta = json.load(f)
                th = meta.get("thresholds", {})
                low = float(th.get("low_risk_approve", DEFAULT_LOW_THRESHOLD))
                high = float(th.get("high_risk_decline", DEFAULT_HIGH_THRESHOLD))
                _THRESHOLDS = (low, high)
        else:
            _THRESHOLDS = (DEFAULT_LOW_THRESHOLD, DEFAULT_HIGH_THRESHOLD)
    return _THRESHOLDS


def predict(
    transaction_dict: dict,
    top_k: int = 3,
    history_store: Any | None = None,
    commit_to_store: bool = False,
) -> dict:

    t_start = time.perf_counter()
    model, calibrator, _ = get_model_and_explainer()
    t_low, t_high = get_thresholds()

    X_df, _ = engineer_features(transaction_dict, is_training=False, history_store=history_store)
    X_arr = X_df[FEATURE_COLUMNS].to_numpy()

    raw_prob = float(model.predict_proba(X_arr)[0, 1])
    if calibrator is not None:
        eps = 1e-7
        log_odds = np.log(np.clip(raw_prob, eps, 1 - eps) / (1 - np.clip(raw_prob, eps, 1 - eps))).reshape(-1, 1)
        cal_prob = float(calibrator.predict_proba(log_odds)[0, 1])
    else:
        cal_prob = raw_prob

    if cal_prob < t_low:
        band = "Low"
        action = "Approve"
    elif cal_prob <= t_high:
        band = "Medium"
        action = "Step-Up (OTP)"
    else:
        band = "High"
        action = "Decline & Alert"

    top_reasons, factor_details = explain_transaction(
        transaction_dict,
        top_k=top_k,
        precomputed_X=X_arr,
    )

    cf = generate_counterfactual(
        transaction_dict,
        target_band="Low",
        current_cal_prob=cal_prob,
        current_band=band,
        X_arr=X_arr,
    )

    if commit_to_store and history_store is not None:
        history_store.record_transaction(transaction_dict)

    t_elapsed = (time.perf_counter() - t_start) * 1000.0

    return {
        "risk_score": round(cal_prob, 4),
        "raw_model_score": round(raw_prob, 4),
        "raw_score": round(raw_prob, 4),
        "band": band,
        "action": action,
        "top_reasons": top_reasons,
        "factors": factor_details,
        "factor_details": factor_details,
        "counterfactual": cf["explanation"],
        "counterfactual_details": cf,
        "latency_ms": round(t_elapsed, 2),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


def generate_counterfactual(
    transaction_dict: dict,
    target_band: str = "Low",
    current_cal_prob: float | None = None,
    current_band: str | None = None,
    X_arr: np.ndarray | None = None,
) -> dict:

    model, calibrator, _ = get_model_and_explainer()
    t_low, t_high = get_thresholds()
    target_thresh = t_low if target_band == "Low" else t_high

    orig_amt = float(transaction_dict.get("amount", 0.0))
    old_bal = float(transaction_dict.get("oldbalanceOrg", 0.0))

    if X_arr is None or current_cal_prob is None or current_band is None:
        X_df, _ = engineer_features(transaction_dict, is_training=False)
        X_arr = X_df[FEATURE_COLUMNS].to_numpy()
        raw_prob = float(model.predict_proba(X_arr)[0, 1])
        eps = 1e-7
        log_odds = np.log(np.clip(raw_prob, eps, 1 - eps) / (1 - np.clip(raw_prob, eps, 1 - eps))).reshape(-1, 1)
        current_cal_prob = float(calibrator.predict_proba(log_odds)[0, 1])
        current_band = "Low" if current_cal_prob < t_low else ("Medium" if current_cal_prob <= t_high else "High")

    if current_cal_prob < target_thresh:
        return {
            "original_amount": orig_amt,
            "original_band": current_band,
            "original_risk_score": round(current_cal_prob, 4),
            "target_band": target_band,
            "counterfactual_amount": orig_amt,
            "amount_reduction_required": 0.0,
            "explanation": f"Transaction is already in the {current_band} risk band (no amount reduction required).",
        }

    amt_idx = 0
    log_amt_idx = 1
    emptied_idx = 13
    ratio_idx = 14

    fractions = np.array([0.05, 0.10, 0.25, 0.50, 0.70, 0.80, 0.85, 0.90, 0.95, 0.98, 0.99, 0.999])
    candidates = fractions * orig_amt
    eps = 1e-7

    X_batch = np.tile(X_arr, (len(candidates), 1))
    X_batch[:, amt_idx] = candidates
    X_batch[:, log_amt_idx] = np.log1p(candidates)
    X_batch[:, emptied_idx] = np.where((old_bal > 0.0) & (candidates >= old_bal), 1.0, 0.0)
    X_batch[:, ratio_idx] = candidates / (old_bal + 1.0)

    raw_p = model.predict_proba(X_batch)[:, 1]
    if calibrator is not None:
        lo = np.log(np.clip(raw_p, eps, 1 - eps) / (1 - np.clip(raw_p, eps, 1 - eps))).reshape(-1, 1)
        cal_p = calibrator.predict_proba(lo)[:, 1]
    else:
        cal_p = raw_p

    valid_idx = np.where(cal_p < target_thresh)[0]
    if len(valid_idx) > 0:
        best_i = valid_idx[-1]
        best_amt = float(candidates[best_i])
        best_prob = float(cal_p[best_i])
    else:
        best_amt = float(candidates[0])
        best_prob = float(cal_p[0])

    reduction = max(0.0, orig_amt - best_amt)
    explanation = (
        f"If the transaction amount had been ₹{reduction:,.2f} lower "
        f"(₹{orig_amt:,.2f} -> ₹{best_amt:,.2f}), the risk score would drop from "
        f"{current_cal_prob:.4f} to {best_prob:.4f} ({target_band} risk - Approved)."
    )

    return {
        "original_amount": round(orig_amt, 2),
        "original_band": current_band,
        "original_risk_score": round(current_cal_prob, 4),
        "target_band": target_band,
        "counterfactual_amount": round(best_amt, 2),
        "amount_reduction_required": round(reduction, 2),
        "explanation": explanation,
    }


if __name__ == "__main__":
    sample_normal = {
        "step": 14,
        "type": "PAYMENT",
        "amount": 250.0,
        "oldbalanceOrg": 12500.0,
        "nameDest": "M12345678",
        "oldbalanceDest": 0.0,
    }
    res_normal = predict(sample_normal)
    print("Normal Transaction:")
    print(json.dumps(res_normal, indent=2))

    sample_fraud = {
        "step": 3,
        "type": "TRANSFER",
        "amount": 1000000.0,
        "oldbalanceOrg": 1000000.0,
        "nameDest": "C98765432",
        "oldbalanceDest": 0.0,
    }
    res_fraud = predict(sample_fraud)
    print("\nFraudulent Transaction:")
    print(json.dumps(res_fraud, indent=2))
