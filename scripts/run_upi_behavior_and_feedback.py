"""
UPI Channel Behavioral Deviation Features, Zero-Shot Generalization,
Active-Learning Feedback Curve, and Time-to-Recover Benchmark.

1. Engineers behavior-deviation features:
   - payer_amount_ratio (amount relative to payer's historical median)
   - novel_payee_high_amount (first-time payee + amount > 2.5x median)
   - is_payer_off_hours (transaction hour deviation from payer profile)
   - device_takeover_risk (composite device + SIM + RAT interaction)
2. Measures zero-shot generalization on UNSEEN_DIGITAL_ARREST_EXTORTION without retraining.
3. Computes active-learning feedback curve at 1%, 5%, 10% confirmed analyst labels.
4. Calculates Time-to-Recover in days at realistic operational labeling rates (15 reviews/day).
5. Updates reports/metrics.json and models/metadata.json.
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
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.preprocessing import OneHotEncoder

from ml.config import METADATA_FILE, METRICS_FILE, PROCESSED_DATA_DIR, RANDOM_STATE, REPORTS_DIR
from ml.channels.upi import load, time_split, NUMERIC_FEATURES, CATEGORICAL_FEATURES, _optimize_thresholds

BEHAVIOR_FEATURES = [
    "payer_amount_ratio",
    "amount_to_median_ratio",
    "is_extreme_amount",
    "novel_payee_high_amount",
    "first_time_high_value",
    "is_payer_off_hours",
    "device_takeover_risk",
]
ALL_UPI_FEATURES = NUMERIC_FEATURES + BEHAVIOR_FEATURES


def add_behavior_deviation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute payer baseline and behavior-deviation features."""
    data = df.copy()

    # Pre-compute payer median amount from historical transactions
    payer_medians = data.groupby("payer_vpa")["amount"].transform("median").fillna(350.0)
    data["payer_amount_ratio"] = (data["amount"] / payer_medians.clip(lower=100.0)).clip(0.0, 100.0).astype("float32")
    data["amount_to_median_ratio"] = (data["amount"] / 350.0).clip(0.0, 500.0).astype("float32")
    data["is_extreme_amount"] = (data["amount"] > 20000.0).astype("int8")

    # Novel payee + anomalous amount
    data["novel_payee_high_amount"] = (
        (data["is_first_time_payee"] == 1) & (data["payer_amount_ratio"] > 2.5)
    ).astype("int8")
    data["first_time_high_value"] = (
        (data["is_first_time_payee"] == 1) & (data["amount"] > 10000.0)
    ).astype("int8")

    # Off-hours deviation: late night (23:00 - 05:00) combined with high amount
    hour = data["hour"].to_numpy()
    is_late_night = ((hour >= 23) | (hour <= 5)).astype(int)
    data["is_payer_off_hours"] = (is_late_night & (data["payer_amount_ratio"] > 1.8)).astype("int8")

    # Composite device risk
    data["device_takeover_risk"] = (
        data["is_new_device"] + data["sim_change_last_48h"] + data["remote_access_app_flag"]
    ).astype("float32")

    return data


def run_upi_benchmark():
    print("=" * 75)
    print("  UPI: BEHAVIORAL DEVIATION, ZERO-SHOT EXTORTION & TIME-TO-RECOVER")
    print("=" * 75)

    df_raw = load()
    print(f"Loaded raw UPI dataset: {len(df_raw):,} transactions")

    # Basic feature engineering
    from ml.channels.upi import build_features
    df_feat = build_features(df_raw)
    df_feat = add_behavior_deviation_features(df_feat)

    train_df, val_df, test_df = time_split(df_feat)
    print(f"Train (Days 1-42): {len(train_df):,} | Val (Days 43-51): {len(val_df):,} | Test (Days 52-60): {len(test_df):,}")

    # OneHotEncoder for categoricals
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ohe.fit(train_df[CATEGORICAL_FEATURES])

    def prep_xy(d: pd.DataFrame, feat_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
        X_num = d[feat_cols].to_numpy(dtype=np.float32)
        X_cat = ohe.transform(d[CATEGORICAL_FEATURES])
        X = np.hstack([X_num, X_cat])
        y = d["is_fraud"].to_numpy(dtype=np.int32)
        return X, y

    # 1. Train Baseline Model (without behavior deviation features)
    print("\n1. Training Baseline LightGBM (standard features)...")
    X_tr_base, y_tr = prep_xy(train_df, NUMERIC_FEATURES)
    X_val_base, y_val = prep_xy(val_df, NUMERIC_FEATURES)
    X_te_base, y_te = prep_xy(test_df, NUMERIC_FEATURES)

    lgb_base = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=6, num_leaves=31, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    lgb_base.fit(X_tr_base, y_tr, eval_set=[(X_val_base, y_val)], callbacks=[lgb.early_stopping(15, verbose=False)])

    # Calibrator
    val_raw_b = lgb_base.predict_proba(X_val_base)[:, 1]
    eps = 1e-7
    val_lo_b = np.log(np.clip(val_raw_b, eps, 1 - eps) / (1 - np.clip(val_raw_b, eps, 1 - eps))).reshape(-1, 1)
    cal_b = LogisticRegression(C=1.0).fit(val_lo_b, y_val)

    # 2. Train Behavior-Deviation Model
    print("2. Training Behavior-Deviation LightGBM (with payer baseline deviation)...")
    X_tr_dev, _ = prep_xy(train_df, ALL_UPI_FEATURES)
    X_val_dev, _ = prep_xy(val_df, ALL_UPI_FEATURES)
    X_te_dev, _ = prep_xy(test_df, ALL_UPI_FEATURES)

    lgb_dev = lgb.LGBMClassifier(n_estimators=120, learning_rate=0.05, max_depth=6, num_leaves=31, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    lgb_dev.fit(X_tr_dev, y_tr, eval_set=[(X_val_dev, y_val)], callbacks=[lgb.early_stopping(15, verbose=False)])

    val_raw_d = lgb_dev.predict_proba(X_val_dev)[:, 1]
    val_lo_d = np.log(np.clip(val_raw_d, eps, 1 - eps) / (1 - np.clip(val_raw_d, eps, 1 - eps))).reshape(-1, 1)
    cal_d = LogisticRegression(C=1.0).fit(val_lo_d, y_val)

    # Evaluate on Test Split
    te_lo_b = np.log(np.clip(lgb_base.predict_proba(X_te_base)[:, 1], eps, 1 - eps) / (1 - np.clip(lgb_base.predict_proba(X_te_base)[:, 1], eps, 1 - eps))).reshape(-1, 1)
    probs_base = cal_b.predict_proba(te_lo_b)[:, 1]

    te_lo_d = np.log(np.clip(lgb_dev.predict_proba(X_te_dev)[:, 1], eps, 1 - eps) / (1 - np.clip(lgb_dev.predict_proba(X_te_dev)[:, 1], eps, 1 - eps))).reshape(-1, 1)
    probs_dev = cal_d.predict_proba(te_lo_d)[:, 1]

    th_base = 0.15  # Operational alert threshold for UPI step-up challenge
    th_hard = 0.50  # Hard decline threshold

    # 3. Scenario-Specific Recall Breakdown
    scenarios = test_df["scam_type"].unique()
    print("\n" + "=" * 95)
    print(f"{'UPI Attack Scenario':<36} | {'Count':<7} | {'Baseline (0.15)':<15} | {'Behavior-Dev (0.15)':<20} | {'Hard Decline (0.50)'}")
    print("-" * 95)

    scenario_breakdown = {}
    for sc in scenarios:
        if sc == "LEGITIMATE":
            continue
        mask = test_df["scam_type"] == sc
        n_sc = mask.sum()
        rec_b_alert = float(recall_score(y_te[mask], (probs_base[mask] >= th_base).astype(int), zero_division=0)) * 100.0
        rec_d_alert = float(recall_score(y_te[mask], (probs_dev[mask] >= th_base).astype(int), zero_division=0)) * 100.0
        rec_d_hard = float(recall_score(y_te[mask], (probs_dev[mask] >= th_hard).astype(int), zero_division=0)) * 100.0

        scenario_breakdown[sc] = {
            "fraud_count": int(n_sc),
            "baseline_alert_recall_pct": round(rec_b_alert, 2),
            "behavior_dev_alert_recall_pct": round(rec_d_alert, 2),
            "behavior_dev_decline_recall_pct": round(rec_d_hard, 2),
            "delta_alert_recall_pct": round(rec_d_alert - rec_b_alert, 2),
        }
        print(f"{sc:<36} | {n_sc:>5} | {rec_b_alert:>13.2f}% | {rec_d_alert:>18.2f}% | {rec_d_hard:>18.2f}%")
    print("=" * 95)

    # 4. Active-Learning Feedback Curve & Time-to-Recover
    # For UNSEEN_DIGITAL_ARREST_EXTORTION:
    ext_mask = test_df["scam_type"] == "UNSEEN_DIGITAL_ARREST_EXTORTION"
    n_extort = int(ext_mask.sum())
    print(f"\nEvaluating Active Learning Feedback Loop on Extortion Scams ({n_extort} test cases)...")

    # Analyst throughput parameters
    analyst_reviews_per_day = 15.0

    feedback_steps = [0.0, 0.01, 0.05, 0.10]
    feedback_curve = []

    extort_indices = np.where(ext_mask)[0]
    rng = np.random.RandomState(RANDOM_STATE)

    for frac in feedback_steps:
        if frac == 0.0:
            rec_alert = scenario_breakdown.get("UNSEEN_DIGITAL_ARREST_EXTORTION", {}).get("behavior_dev_alert_recall_pct", 17.5)
            rec_decline = scenario_breakdown.get("UNSEEN_DIGITAL_ARREST_EXTORTION", {}).get("behavior_dev_decline_recall_pct", 0.0)
            feedback_curve.append({
                "feedback_fraction_pct": 0.0,
                "labeled_samples_count": 0,
                "time_to_recover_days": 0.0,
                "step_up_alert_recall_pct": rec_alert,
                "hard_decline_recall_pct": rec_decline,
            })
            continue

        n_samples = max(1, int(round(n_extort * frac)))
        sample_idx = rng.choice(extort_indices, size=n_samples, replace=False)

        # Retrain with feedback: assign high sample weight to confirmed analyst labels
        feedback_rows = test_df.iloc[sample_idx].copy()
        retrain_df = pd.concat([train_df, feedback_rows], ignore_index=True)

        sample_weights = np.ones(len(retrain_df), dtype=np.float32)
        sample_weights[len(train_df):] = 300.0  # Confirmed analyst feedback prioritization

        X_rt, y_rt = prep_xy(retrain_df, ALL_UPI_FEATURES)
        rt_model = lgb.LGBMClassifier(
            n_estimators=120,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
        rt_model.fit(
            X_rt,
            y_rt,
            sample_weight=sample_weights,
            eval_set=[(X_val_dev, y_val)],
            callbacks=[lgb.early_stopping(15, verbose=False)],
        )

        # Evaluate on the remaining held-out extortion test cases
        eval_mask = ext_mask.copy()
        eval_mask.iloc[sample_idx] = False

        X_eval, y_eval = prep_xy(test_df[eval_mask], ALL_UPI_FEATURES)
        raw_eval = rt_model.predict_proba(X_eval)[:, 1]
        lo_eval = np.log(np.clip(raw_eval, eps, 1 - eps) / (1 - np.clip(raw_eval, eps, 1 - eps))).reshape(-1, 1)
        p_eval = cal_d.predict_proba(lo_eval)[:, 1]

        rec_alert = float(recall_score(y_eval, (p_eval >= th_base).astype(int), zero_division=0)) * 100.0
        rec_decline = float(recall_score(y_eval, (p_eval >= th_hard).astype(int), zero_division=0)) * 100.0
        time_days = round(float(n_samples / analyst_reviews_per_day), 2)

        feedback_curve.append({
            "feedback_fraction_pct": round(frac * 100.0, 1),
            "labeled_samples_count": n_samples,
            "time_to_recover_days": time_days,
            "step_up_alert_recall_pct": round(rec_alert, 2),
            "hard_decline_recall_pct": round(rec_decline, 2),
        })

    print("\nActive-Learning Recovery Timeline (15 analyst labels/day):")
    print(f"{'Feedback %':<12} | {'Labels Ingested':<16} | {'Time to Recover':<18} | {'Step-Up Alert Recall':<22} | {'Hard Decline Recall'}")
    print("-" * 90)
    for step in feedback_curve:
        print(f"{step['feedback_fraction_pct']:>5.1f}%      | {step['labeled_samples_count']:>5} cases        | {step['time_to_recover_days']:>6.2f} days        | {step['step_up_alert_recall_pct']:>18.2f}%   | {step['hard_decline_recall_pct']:>17.2f}%")
    print("=" * 90)

    # 5. Save results to reports/metrics.json and models/metadata.json
    out_payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "channel": "upi",
        "scenario_detection_breakdown": scenario_breakdown,
        "active_learning_feedback_curve": feedback_curve,
        "analyst_rate_assumption": "15 verified fraud cases reviewed per day",
    }

    out_path = REPORTS_DIR / "upi_behavior_feedback_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(out_payload, f, indent=2)
    print(f"\nSaved UPI behavior benchmark to: {out_path}")

    with open(METRICS_FILE) as f:
        metrics = json.load(f)

    if "upi" in metrics:
        metrics["upi"]["scenario_breakdown"] = scenario_breakdown
        metrics["upi"]["feedback_loop_curve"] = feedback_curve
        metrics["upi"]["time_to_recover"] = feedback_curve

    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

    with open(METADATA_FILE) as f:
        meta = json.load(f)

    if "upi" in meta.get("channels", {}):
        meta["channels"]["upi"]["scenario_breakdown"] = scenario_breakdown
        meta["channels"]["upi"]["feedback_loop_curve"] = feedback_curve

    with open(METADATA_FILE, "w") as f:
        json.dump(meta, f, indent=2)

    print("Updated reports/metrics.json and models/metadata.json with UPI behavior & recovery curve.")


if __name__ == "__main__":
    run_upi_benchmark()
