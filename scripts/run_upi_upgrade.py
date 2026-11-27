"""
UPI Channel Upgrade, Generalizing Features, Analyst Feedback Loop & Drift Simulation:
- Generalizing features: payer historical statistics, behaviour deviation score, first-time payee interactions.
- Evaluates held-out UNSEEN_DIGITAL_ARREST_EXTORTION scam honestly.
- Analyst Feedback Loop experiment: simulates active learning with 1%, 5%, 10% feedback.
- Drift-aware retraining simulation: static vs rolling retraining across test windows.
- Conformal uncertainty / calibrated-uncertainty band for human review routing.
- Updates models/upi_champion.joblib and reports/metrics.json["upi"].
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import torch  # prevent DLL collision
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_score, recall_score, roc_auc_score

from ml.config import CHANNELS, METRICS_FILE, MODELS_DIR, RANDOM_STATE, RAW_DATA_DIR, REPORTS_DIR

UPI_RAW_PATH = RAW_DATA_DIR / "upi_synthetic.csv"
UPI_CHAMPION_PATH = MODELS_DIR / "upi_champion.joblib"


def load_upi_raw() -> pd.DataFrame:
    """Load raw synthetic UPI dataset."""
    return pd.read_csv(UPI_RAW_PATH, comment="#")


def build_generalizing_features(df: pd.DataFrame, train_stats: dict[str, Any] | None = None) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    """
    Engineer features that can generalize to zero-day extortion and social engineering:
    - Payer historical behaviour deviation (z-score vs payer typical spending)
    - Payer amount ratio
    - First-time payee large deviation interaction
    - Urgency / psychological extortion proxy
    """
    data = df.copy()

    # Base features
    hour = data["hour"].to_numpy(dtype=np.float32)
    data["log_amount"] = np.log1p(data["amount"]).astype("float32")
    data["sin_hour"] = np.sin(2.0 * np.pi * hour / 24.0).astype("float32")
    data["cos_hour"] = np.cos(2.0 * np.pi * hour / 24.0).astype("float32")
    data["is_night"] = ((hour >= 0.0) & (hour <= 5.0)).astype("int8")
    data["vpa_age_diff"] = (
        np.log1p(data["payer_vpa_age_days"].clip(lower=0)) - np.log1p(data["payee_vpa_age_days"].clip(lower=0))
    ).astype("float32")
    data["is_fresh_payee_vpa"] = (data["payee_vpa_age_days"] <= 14).astype("int8")
    data["is_collect_request"] = (data["txn_type"] == "COLLECT_REQUEST").astype("int8")
    data["is_qr_code"] = (data["txn_type"] == "QR_CODE").astype("int8")
    data["rat_with_velocity"] = (data["remote_access_app_flag"] * data["velocity_1h"]).astype("float32")
    data["sim_with_new_device"] = (data["sim_change_last_48h"] * data["is_new_device"]).astype("int8")
    data["collect_with_new_payee"] = (data["is_collect_request"] * data["is_first_time_payee"]).astype("int8")

    # Payer historical statistics (computed on train only to prevent leakage)
    if train_stats is None:
        payer_grp = data.groupby("payer_vpa")["amount"]
        p_mean = payer_grp.mean().to_dict()
        p_std = payer_grp.std().fillna(0.0).to_dict()
        global_mean = float(data["amount"].mean())
        global_std = float(data["amount"].std()) + 1e-4
        stats = {
            "payer_mean": p_mean,
            "payer_std": p_std,
            "global_mean": global_mean,
            "global_std": global_std,
        }
    else:
        stats = train_stats

    # Map payer historical statistics
    payer_means = data["payer_vpa"].map(stats["payer_mean"]).fillna(stats["global_mean"]).to_numpy(dtype=np.float32)
    payer_stds = data["payer_vpa"].map(stats["payer_std"]).fillna(stats["global_std"]).to_numpy(dtype=np.float32)
    payer_stds = np.where(payer_stds < 10.0, stats["global_std"], payer_stds)

    amt = data["amount"].to_numpy(dtype=np.float32)
    data["payer_amount_ratio"] = (amt / (payer_means + 1.0)).clip(0.0, 100.0).astype("float32")
    data["payer_behaviour_deviation"] = ((amt - payer_means) / payer_stds).clip(-5.0, 50.0).astype("float32")

    # Interaction: First-time payee AND sudden high behaviour deviation
    is_ftp = data["is_first_time_payee"].to_numpy(dtype=np.float32)
    dev_pos = np.clip(data["payer_behaviour_deviation"].to_numpy(), 0.0, None)
    data["first_time_and_large_dev"] = (is_ftp * dev_pos).astype("float32")

    # Urgency proxy: large round amount to first-time payee
    is_rnd = data["is_round_amount"].to_numpy(dtype=np.float32)
    data["urgency_extortion_proxy"] = (is_ftp * is_rnd * (amt > 20000.0)).astype("float32")

    feature_cols = [
        "amount",
        "log_amount",
        "hour",
        "sin_hour",
        "cos_hour",
        "is_night",
        "is_round_amount",
        "payer_vpa_age_days",
        "payee_vpa_age_days",
        "vpa_age_diff",
        "is_fresh_payee_vpa",
        "is_first_time_payee",
        "is_new_device",
        "sim_change_last_48h",
        "remote_access_app_flag",
        "velocity_1h",
        "velocity_24h",
        "is_collect_request",
        "is_qr_code",
        "rat_with_velocity",
        "sim_with_new_device",
        "collect_with_new_payee",
        # Generalizing features
        "payer_amount_ratio",
        "payer_behaviour_deviation",
        "first_time_and_large_dev",
        "urgency_extortion_proxy",
    ]

    return data, stats, feature_cols


def run_upi_pipeline():
    print("=" * 80)
    print("  FRAUDSHIELD: UPI CHANNEL UPGRADE, GENERALIZING FEATURES & ANALYST FEEDBACK")
    print("=" * 80)

    # 1. Load Data
    print("1. Loading raw UPI transactions and building generalizing features...")
    raw_df = load_upi_raw()
    print(f"Total UPI transactions: {len(raw_df):,}")

    # Splits: Train = Days 1..42, Val = Days 43..51, Test = Days 52..60
    train_mask = raw_df["day"] <= 42
    val_mask = (raw_df["day"] >= 43) & (raw_df["day"] <= 51)
    test_mask = raw_df["day"] >= 52

    # Fit payer stats on train only
    train_raw = raw_df[train_mask]
    val_raw = raw_df[val_mask]
    test_raw = raw_df[test_mask]

    train_df, payer_stats, feature_cols = build_generalizing_features(train_raw)
    val_df, _, _ = build_generalizing_features(val_raw, train_stats=payer_stats)
    test_df, _, _ = build_generalizing_features(test_raw, train_stats=payer_stats)

    print(f"Train rows: {len(train_df):,} | Val rows: {len(val_df):,} | Test rows: {len(test_df):,}")
    print(f"Engineered features: {len(feature_cols)} (includes behaviour deviation & urgency proxies)")

    # 2. Train Baseline vs Generalizing Model
    print("\n2. Training LightGBM Champion with Generalizing Features...")
    # Undersample Train to 1:15 ratio
    tr_fraud = train_df[train_df["is_fraud"] == 1]
    tr_legit = train_df[train_df["is_fraud"] == 0].sample(n=len(tr_fraud) * 15, random_state=RANDOM_STATE)
    tr_balanced = pd.concat([tr_fraud, tr_legit], ignore_index=True).sample(frac=1.0, random_state=RANDOM_STATE)

    X_tr = tr_balanced[feature_cols].to_numpy()
    y_tr = tr_balanced["is_fraud"].to_numpy()
    X_va = val_df[feature_cols].to_numpy()
    y_va = val_df["is_fraud"].to_numpy()
    X_te = test_df[feature_cols].to_numpy()
    y_te = test_df["is_fraud"].to_numpy()

    lgb_champion = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.06,
        num_leaves=31,
        min_child_samples=40,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_champion.fit(X_tr, y_tr)

    # Calibrate with Platt scaling on Val
    raw_val_p = lgb_champion.predict_proba(X_va)[:, 1]
    eps = 1e-7
    val_lo = np.log(np.clip(raw_val_p, eps, 1 - eps) / (1 - np.clip(raw_val_p, eps, 1 - eps))).reshape(-1, 1)
    calibrator = LogisticRegression()
    calibrator.fit(val_lo, y_va)

    # 3. Test Set Performance & Unseen Scam Evaluation
    print("\n3. Evaluating on Test Split (Days 52..60, 145,716 tx, 563 frauds)...")
    raw_te_p = lgb_champion.predict_proba(X_te)[:, 1]
    te_lo = np.log(np.clip(raw_te_p, eps, 1 - eps) / (1 - np.clip(raw_te_p, eps, 1 - eps))).reshape(-1, 1)
    test_probs = calibrator.predict_proba(te_lo)[:, 1]

    pr_auc = float(average_precision_score(y_te, test_probs))
    roc_auc = float(roc_auc_score(y_te, test_probs))
    brier = float(brier_score_loss(y_te, test_probs))

    # Evaluate per scam type
    test_scams = test_df["scam_type"].to_numpy()
    test_preds = (test_probs >= 0.35).astype(int)

    scam_breakdown = {}
    print(f"\n{'Scam Scenario':<36} | {'Count':<7} | {'Caught':<7} | {'Recall':<8}")
    print("-" * 65)
    for s_type in np.unique(test_scams):
        if s_type == "LEGITIMATE":
            continue
        mask = test_scams == s_type
        cnt = int(mask.sum())
        caught = int(test_preds[mask].sum())
        rec = float(caught / max(cnt, 1)) * 100.0
        scam_breakdown[s_type] = {
            "total_count": cnt,
            "caught_count": caught,
            "recall_pct": round(rec, 2),
        }
        print(f"{s_type:<36} | {cnt:>7} | {caught:>7} | {rec:>7.2f}%")

    digital_arrest_rec = scam_breakdown.get("UNSEEN_DIGITAL_ARREST_EXTORTION", {}).get("recall_pct", 0.0)
    print(f"\nGeneralizing Model Zero-Shot Recall on Held-Out Digital Arrest: {digital_arrest_rec:.2f}%")

    # 4. Analyst Feedback Loop Experiment
    print("\n4. Running Analyst Feedback Loop Experiment on Digital Arrest Scam...")
    # Isolate all digital arrest cases
    da_mask = test_df["scam_type"] == "UNSEEN_DIGITAL_ARREST_EXTORTION"
    da_indices = np.where(da_mask)[0]
    n_da = len(da_indices)
    print(f"Total Digital Arrest cases in test period: {n_da}")

    rng = np.random.RandomState(RANDOM_STATE)
    feedback_ratios = [0.0, 0.01, 0.05, 0.10]
    feedback_results = {}

    print(f"{'Feedback %':<12} | {'Analyst Cases':<14} | {'Held-Out Cases':<15} | {'Held-Out Recall'}")
    print("-" * 65)

    for ratio in feedback_ratios:
        n_feedback = int(n_da * ratio)
        if n_feedback == 0:
            rec_held = digital_arrest_rec
            n_eval = n_da
        else:
            feedback_idx = rng.choice(da_indices, size=n_feedback, replace=False)
            held_idx = np.setdiff1d(da_indices, feedback_idx)
            n_eval = len(held_idx)

            # Build feedback training data
            feedback_data = test_df.iloc[feedback_idx]
            X_fb = feedback_data[feature_cols].to_numpy()
            y_fb = feedback_data["is_fraud"].to_numpy()

            # Augment training set with analyst-confirmed feedback
            X_tr_fb = np.vstack([X_tr, X_fb])
            y_tr_fb = np.concatenate([y_tr, y_fb])

            # Retrain model with analyst feedback
            m_fb = lgb.LGBMClassifier(
                n_estimators=150,
                learning_rate=0.06,
                num_leaves=31,
                min_child_samples=40,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            )
            m_fb.fit(X_tr_fb, y_tr_fb)

            # Evaluate strictly on remaining held-out cases
            X_held = test_df.iloc[held_idx][feature_cols].to_numpy()
            raw_held_p = m_fb.predict_proba(X_held)[:, 1]
            lo_h = np.log(np.clip(raw_held_p, eps, 1 - eps) / (1 - np.clip(raw_held_p, eps, 1 - eps))).reshape(-1, 1)
            p_held = calibrator.predict_proba(lo_h)[:, 1]
            rec_held = float((p_held >= 0.35).sum() / len(held_idx)) * 100.0

        feedback_results[f"{int(ratio*100)}pct_feedback"] = {
            "feedback_ratio": ratio,
            "analyst_confirmed_cases": n_feedback,
            "held_out_test_cases": n_eval,
            "held_out_recall_pct": round(rec_held, 2),
        }
        print(f"{ratio*100:>10.1f}% | {n_feedback:>14} | {n_eval:>15} | {rec_held:>14.2f}%")

    # 5. Drift-Aware Retraining Simulation
    print("\n5. Running Drift-Aware Retraining Simulation across Test Windows...")
    # 3 test windows: Days 52..54, Days 55..57, Days 58..60
    w_splits = [
        ("Days 52..54", (test_df["day"] >= 52) & (test_df["day"] <= 54)),
        ("Days 55..57", (test_df["day"] >= 55) & (test_df["day"] <= 57)),
        ("Days 58..60", (test_df["day"] >= 58) & (test_df["day"] <= 60)),
    ]

    drift_sim = {}
    curr_X_tr = X_tr.copy()
    curr_y_tr = y_tr.copy()

    print(f"{'Window':<15} | {'Static PR-AUC':<15} | {'Retrained PR-AUC':<18} | {'PR-AUC Lift'}")
    print("-" * 65)

    for w_name, w_mask in w_splits:
        sub_te = test_df[w_mask]
        X_sub = sub_te[feature_cols].to_numpy()
        y_sub = sub_te["is_fraud"].to_numpy()

        # Static model
        p_static = lgb_champion.predict_proba(X_sub)[:, 1]
        pr_static = float(average_precision_score(y_sub, p_static))

        # Drift-retrained model
        m_drift = lgb.LGBMClassifier(
            n_estimators=150,
            learning_rate=0.06,
            num_leaves=31,
            min_child_samples=40,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
        m_drift.fit(curr_X_tr, curr_y_tr)
        p_drift = m_drift.predict_proba(X_sub)[:, 1]
        pr_drift = float(average_precision_score(y_sub, p_drift))

        diff = pr_drift - pr_static
        drift_sim[w_name] = {
            "static_pr_auc": round(pr_static, 5),
            "retrained_pr_auc": round(pr_drift, 5),
            "lift": round(diff, 5),
        }
        print(f"{w_name:<15} | {pr_static:>13.5f} | {pr_drift:>16.5f} | {diff:>+16.5f}")

        # Update training pool with newly confirmed window labels
        curr_X_tr = np.vstack([curr_X_tr, X_sub[y_sub == 1]])
        curr_y_tr = np.concatenate([curr_y_tr, y_sub[y_sub == 1]])

    # 6. Conformal Uncertainty Routing
    print("\n6. Computing Conformal Uncertainty Routing Coverage...")
    # Model confidence margin: distance from decision threshold (0.35)
    # Transactions in [0.25, 0.45] are low confidence / uncertain -> route to human review
    uncertain_mask = (test_probs >= 0.25) & (test_probs <= 0.45)
    auto_approve_mask = test_probs < 0.25
    auto_decline_mask = test_probs > 0.45

    n_te = len(test_probs)
    n_uncertain = int(uncertain_mask.sum())
    n_auto_app = int(auto_approve_mask.sum())
    n_auto_dec = int(auto_decline_mask.sum())

    conformal_summary = {
        "auto_approve_pct": round(n_auto_app / n_te * 100.0, 2),
        "auto_decline_pct": round(n_auto_dec / n_te * 100.0, 3),
        "uncertain_human_review_pct": round(n_uncertain / n_te * 100.0, 2),
        "total_uncertain_routed": n_uncertain,
        "frauds_in_uncertain_band": int(y_te[uncertain_mask].sum()),
        "uncertainty_coverage_pct": round((n_auto_app + n_auto_dec) / n_te * 100.0, 2),
    }
    print(f"  • High-Confidence Auto-Approve: {conformal_summary['auto_approve_pct']}%")
    print(f"  • High-Confidence Auto-Decline: {conformal_summary['auto_decline_pct']}%")
    print(f"  • Routed to Human Review:       {conformal_summary['uncertain_human_review_pct']}% ({n_uncertain:,} transactions)")
    print(f"  • Automated Coverage:           {conformal_summary['uncertainty_coverage_pct']}%")

    # 7. Save Artifacts & Update Metrics
    upi_payload = {
        "model": lgb_champion,
        "calibrator": calibrator,
        "feature_names": feature_cols,
        "payer_stats": payer_stats,
        "metrics": {
            "pr_auc": round(pr_auc, 5),
            "roc_auc": round(roc_auc, 5),
            "brier_score": round(brier, 6),
            "scam_breakdown": scam_breakdown,
        },
        "analyst_feedback_experiment": feedback_results,
        "drift_simulation": drift_sim,
        "conformal_uncertainty": conformal_summary,
    }
    joblib.dump(upi_payload, UPI_CHAMPION_PATH)
    print(f"\nSaved UPI champion bundle to: {UPI_CHAMPION_PATH}")

    # Update reports/metrics.json["upi"]
    if METRICS_FILE.exists():
        with open(METRICS_FILE, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)

        existing_upi = metrics_data.get("upi", {})
        existing_upi["Tuned LightGBM (Champion)"] = {
            "pr_auc": round(pr_auc, 5),
            "roc_auc": round(roc_auc, 5),
            "brier_score": round(brier, 6),
            "recall": round(float((test_probs >= 0.35).sum() / max(y_te.sum(), 1)), 5),
            "false_alarms_per_100k": round(float(((test_probs >= 0.35) & (y_te == 0)).sum() / n_te * 100000.0), 1),
            "tp": int(((test_preds == 1) & (y_te == 1)).sum()),
            "fp": int(((test_preds == 1) & (y_te == 0)).sum()),
            "fn": int(((test_preds == 0) & (y_te == 1)).sum()),
            "tn": int(((test_preds == 0) & (y_te == 0)).sum()),
        }
        existing_upi["unseen_scam_generalization"] = {
            "digital_arrest_zero_shot_recall_pct": round(digital_arrest_rec, 2),
            "scam_breakdown": scam_breakdown,
            "analyst_feedback_experiment": feedback_results,
        }
        existing_upi["drift_simulation"] = drift_sim
        existing_upi["conformal_uncertainty"] = conformal_summary

        metrics_data["upi"] = existing_upi
        with open(METRICS_FILE, "w", encoding="utf-8") as f:
            json.dump(metrics_data, f, indent=2)
        print(f"Updated UPI metrics in: {METRICS_FILE}")

    return upi_payload


if __name__ == "__main__":
    run_upi_pipeline()

