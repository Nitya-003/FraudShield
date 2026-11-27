"""
PaySim Evasion Robustness, Adversarial Augmentation, and Value-Weighted Training.

1. Evaluates evasion robustness: attackers transfer 90%, 70%, 50% of victim balance,
   or split transactions into 3 transfers (non-emptying attacks).
   Reports recall for Rule 1, Rule 2, Rule 3, Decision Tree (Depth 3), and Champion LightGBM.
2. Trains an Adversarially-Augmented LightGBM (training-set only augmentation)
   and measures evasion recovery honestly.
3. Implements Value-Weighted Training (sample weights proportional to amount)
   and compares INR financial losses against standard count-weighted LightGBM.
4. Updates reports/metrics.json and reports/paysim_evasion_benchmark.json.
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
from sklearn.metrics import average_precision_score, recall_score, precision_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

from ml.config import (
    FIGURES_DIR,
    METRICS_FILE,
    MODEL_FILE,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET_COL,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
)
from ml.features import FEATURE_COLUMNS


def create_evasion_test_variant(
    test_df: pd.DataFrame,
    mode: str = "pct_balance",
    pct: float = 0.50,
) -> pd.DataFrame:
    """
    Simulates evasive fraud tactics by modifying test frauds:
    - mode='pct_balance': attacker transfers pct of sender's balance (no account drainage).
    - mode='split_3': attacker splits transfer into 3 smaller fractions (33% each).
    """
    evasive_df = test_df.copy()
    fraud_mask = evasive_df[TARGET_COL] == 1

    if mode == "pct_balance":
        # New amount = pct * oldbalanceOrg for senders with positive balance
        has_bal = fraud_mask & (evasive_df["oldbalanceOrg"] > 0)
        evasive_df.loc[has_bal, "amount"] = evasive_df.loc[has_bal, "oldbalanceOrg"] * pct
        # Senders with 0 balance remain at current amount or small probing fee
        evasive_df.loc[fraud_mask & (~has_bal), "amount"] = 5000.0

    elif mode == "split_3":
        # Split into 3 micro transfers (~33% of balance)
        has_bal = fraud_mask & (evasive_df["oldbalanceOrg"] > 0)
        evasive_df.loc[has_bal, "amount"] = (evasive_df.loc[has_bal, "oldbalanceOrg"] * 0.33)
        evasive_df.loc[fraud_mask & (~has_bal), "amount"] = 2000.0

    # Recompute derived pre-transaction features
    evasive_df.loc[fraud_mask, "log_amount"] = np.log1p(evasive_df.loc[fraud_mask, "amount"].clip(lower=0)).astype("float32")
    evasive_df.loc[fraud_mask, "orig_emptied"] = (
        (evasive_df.loc[fraud_mask, "oldbalanceOrg"] > 0.0)
        & (evasive_df.loc[fraud_mask, "amount"] >= evasive_df.loc[fraud_mask, "oldbalanceOrg"])
    ).astype("int8")
    
    eps = 1e-4
    denom = evasive_df.loc[fraud_mask, "oldbalanceOrg"] + eps
    evasive_df.loc[fraud_mask, "amount_to_balance_ratio"] = (
        evasive_df.loc[fraud_mask, "amount"] / denom
    ).clip(0.0, 100.0).astype("float32")

    return evasive_df


def evaluate_models_on_data(
    df: pd.DataFrame,
    lgb_model: Any,
    dt_model: Any,
    calibrator: Any,
    threshold: float = 0.45,
) -> dict[str, Any]:
    """Evaluate heuristic rules and ML models on a specified test dataset."""
    y = df[TARGET_COL].to_numpy()
    X = df[FEATURE_COLUMNS].to_numpy()
    amounts = df["amount"].to_numpy()

    # Rule 1: Amount > 50,000
    r1_preds = (df["amount"] > 50000.0).astype(int).to_numpy()
    r1_rec = float(recall_score(y, r1_preds, zero_division=0))

    # Rule 2: Channel TRANSFER/CASH_OUT & orig_emptied == 1
    r2_preds = (
        ((df["type_TRANSFER"] == 1) | (df["type_CASH_OUT"] == 1))
        & (df["orig_emptied"] == 1)
    ).astype(int).to_numpy()
    r2_rec = float(recall_score(y, r2_preds, zero_division=0))

    # Rule 3: Channel & orig_emptied & dest_balance_zero
    r3_preds = (
        ((df["type_TRANSFER"] == 1) | (df["type_CASH_OUT"] == 1))
        & (df["orig_emptied"] == 1)
        & (df["dest_balance_zero"] == 1)
    ).astype(int).to_numpy()
    r3_rec = float(recall_score(y, r3_preds, zero_division=0))

    # Decision Tree (Depth 3)
    dt_probs = dt_model.predict_proba(X)[:, 1]
    dt_preds = (dt_probs >= threshold).astype(int)
    dt_rec = float(recall_score(y, dt_preds, zero_division=0))

    # LightGBM (Platt Calibrated)
    raw_p = lgb_model.predict_proba(X)[:, 1]
    eps = 1e-7
    lo = np.log(np.clip(raw_p, eps, 1 - eps) / (1 - np.clip(raw_p, eps, 1 - eps))).reshape(-1, 1)
    cal_p = calibrator.predict_proba(lo)[:, 1]
    lgb_preds = (cal_p >= threshold).astype(int)
    lgb_rec = float(recall_score(y, lgb_preds, zero_division=0))
    lgb_pr = float(average_precision_score(y, cal_p))
    
    # Financial metrics
    missed_mask = (y == 1) & (lgb_preds == 0)
    inr_missed = float(amounts[missed_mask].sum())

    return {
        "rule1_recall": round(r1_rec * 100, 2),
        "rule2_recall": round(r2_rec * 100, 2),
        "rule3_recall": round(r3_rec * 100, 2),
        "decision_tree_recall": round(dt_rec * 100, 2),
        "lightgbm_recall": round(lgb_rec * 100, 2),
        "lightgbm_pr_auc": round(lgb_pr, 5),
        "lightgbm_missed_inr": round(inr_missed, 2),
    }


def run_evasion_and_value_weighting_pipeline():
    print("=" * 75)
    print("  FRAUDSHIELD: PAYSIM EVASION ROBUSTNESS, ADVERSARIAL AUGMENTATION & VALUE-WEIGHTING")
    print("=" * 75)

    # 1. Load Data
    print("\n1. Loading datasets and baseline models...")
    train_us = pd.read_parquet(PROCESSED_DATA_DIR / "train_undersampled.parquet")
    val_df = pd.read_parquet(VAL_DATA_PATH)
    test_df = pd.read_parquet(TEST_DATA_PATH)

    payload = joblib.load(MODEL_FILE)
    base_lgb = payload["model"]
    calibrator = payload["calibrator"]

    # Fit standard Decision Tree Depth 3 for baseline comparison
    X_train_us = train_us[FEATURE_COLUMNS].to_numpy()
    y_train_us = train_us[TARGET_COL].to_numpy()
    dt3 = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=RANDOM_STATE)
    dt3.fit(X_train_us, y_train_us)

    # 2. Build Evasion Scenarios on Test Set
    print("\n2. Evaluating standard models on evasive non-emptying fraud variants...")
    scenarios = {
        "Natural Test (Baseline - 97.5% Drained)": test_df,
        "Evasion 90% Balance (No Drain)": create_evasion_test_variant(test_df, mode="pct_balance", pct=0.90),
        "Evasion 70% Balance (No Drain)": create_evasion_test_variant(test_df, mode="pct_balance", pct=0.70),
        "Evasion 50% Balance (Partial Drain)": create_evasion_test_variant(test_df, mode="pct_balance", pct=0.50),
        "Evasion Split 3 Micro-Transfers (33% each)": create_evasion_test_variant(test_df, mode="split_3"),
    }

    baseline_evasion_results = {}
    print(f"\n{'Scenario':<42} | {'R1 (50k)':<8} | {'R2 (Empty)':<10} | {'DT (Depth3)':<11} | {'LGBM Champ':<10} | {'Missed INR'}")
    print("-" * 105)
    for name, s_df in scenarios.items():
        res = evaluate_models_on_data(s_df, base_lgb, dt3, calibrator, threshold=0.45)
        baseline_evasion_results[name] = res
        print(f"{name:<42} | {res['rule1_recall']:>7.2f}% | {res['rule2_recall']:>9.2f}% | {res['decision_tree_recall']:>10.2f}% | {res['lightgbm_recall']:>9.2f}% | ₹{res['lightgbm_missed_inr']:,.0f}")

    # 3. Retrain with Adversarial Augmentation (Training Only)
    print("\n3. Training Adversarially Augmented LightGBM (Augmented Frauds in Training Only)...")
    # Augment training set with 50%, 70%, 90% evasive variants of training frauds
    train_frauds = train_us[train_us[TARGET_COL] == 1].copy()
    
    aug_variants = []
    for pct in [0.50, 0.70, 0.90]:
        v = train_frauds.copy()
        has_bal = v["oldbalanceOrg"] > 0
        v.loc[has_bal, "amount"] = v.loc[has_bal, "oldbalanceOrg"] * pct
        v.loc[~has_bal, "amount"] = 5000.0
        v["log_amount"] = np.log1p(v["amount"].clip(lower=0)).astype("float32")
        v["orig_emptied"] = ((v["oldbalanceOrg"] > 0.0) & (v["amount"] >= v["oldbalanceOrg"])).astype("int8")
        eps = 1e-4
        v["amount_to_balance_ratio"] = (v["amount"] / (v["oldbalanceOrg"] + eps)).clip(0.0, 100.0).astype("float32")
        aug_variants.append(v)

    train_augmented = pd.concat([train_us] + aug_variants, ignore_index=True)
    X_train_aug = train_augmented[FEATURE_COLUMNS].to_numpy()
    y_train_aug = train_augmented[TARGET_COL].to_numpy()

    adv_lgb = lgb.LGBMClassifier(
        n_estimators=120,
        learning_rate=0.07,
        num_leaves=31,
        min_child_samples=30,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    adv_lgb.fit(X_train_aug, y_train_aug)

    # Calibrate on validation set
    X_val = val_df[FEATURE_COLUMNS].to_numpy()
    y_val = val_df[TARGET_COL].to_numpy()
    raw_val_p = adv_lgb.predict_proba(X_val)[:, 1]
    eps = 1e-7
    val_lo = np.log(np.clip(raw_val_p, eps, 1 - eps) / (1 - np.clip(raw_val_p, eps, 1 - eps))).reshape(-1, 1)
    from sklearn.linear_model import LogisticRegression
    adv_calibrator = LogisticRegression()
    adv_calibrator.fit(val_lo, y_val)

    # Evaluate Augmented Model on Evasion Scenarios
    print("\n  Evaluating Adversarially-Augmented Champion on Evasion Scenarios:")
    augmented_evasion_results = {}
    print(f"\n{'Scenario':<42} | {'Standard Champion':<17} | {'Adv-Augmented Champion':<22} | {'Recall Recovery'}")
    print("-" * 105)
    for name, s_df in scenarios.items():
        base_res = baseline_evasion_results[name]
        adv_res = evaluate_models_on_data(s_df, adv_lgb, dt3, adv_calibrator, threshold=0.45)
        augmented_evasion_results[name] = adv_res
        diff = adv_res["lightgbm_recall"] - base_res["lightgbm_recall"]
        print(f"{name:<42} | {base_res['lightgbm_recall']:>15.2f}% | {adv_res['lightgbm_recall']:>20.2f}% | {diff:>+14.2f}%")

    # 4. Value-Weighted Training (Loss weighted by transaction amount)
    print("\n4. Training Value-Weighted LightGBM (Sample Weights ~ INR Amount)...")
    # Normalize sample weights: median amount weight = 1.0, clipped between 0.1 and 50.0
    med_amt = float(train_us["amount"].median())
    amt_weights = np.clip(train_us["amount"].to_numpy() / max(med_amt, 1.0), 0.1, 50.0)
    # Give fraud cases an extra financial loss penalty
    amt_weights[y_train_us == 1] *= 2.0

    vw_lgb = lgb.LGBMClassifier(
        n_estimators=120,
        learning_rate=0.07,
        num_leaves=31,
        min_child_samples=30,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    vw_lgb.fit(X_train_us, y_train_us, sample_weight=amt_weights)

    vw_val_p = vw_lgb.predict_proba(X_val)[:, 1]
    vw_lo = np.log(np.clip(vw_val_p, eps, 1 - eps) / (1 - np.clip(vw_val_p, eps, 1 - eps))).reshape(-1, 1)
    vw_calibrator = LogisticRegression()
    vw_calibrator.fit(vw_lo, y_val)

    # Compare Natural Test INR Financial Losses
    X_test = test_df[FEATURE_COLUMNS].to_numpy()
    y_test = test_df[TARGET_COL].to_numpy()
    amounts_test = test_df["amount"].to_numpy()

    # Base Champion loss
    base_raw = base_lgb.predict_proba(X_test)[:, 1]
    base_lo = np.log(np.clip(base_raw, eps, 1 - eps) / (1 - np.clip(base_raw, eps, 1 - eps))).reshape(-1, 1)
    base_probs = calibrator.predict_proba(base_lo)[:, 1]
    base_missed = (y_test == 1) & (base_probs < 0.45)
    base_loss_inr = float(amounts_test[base_missed].sum())
    base_fp = int(((base_probs > 0.90) & (y_test == 0)).sum())

    # Value-Weighted Champion loss
    vw_raw = vw_lgb.predict_proba(X_test)[:, 1]
    vw_lo_te = np.log(np.clip(vw_raw, eps, 1 - eps) / (1 - np.clip(vw_raw, eps, 1 - eps))).reshape(-1, 1)
    vw_probs = vw_calibrator.predict_proba(vw_lo_te)[:, 1]
    vw_missed = (y_test == 1) & (vw_probs < 0.45)
    vw_loss_inr = float(amounts_test[vw_missed].sum())
    vw_fp = int(((vw_probs > 0.90) & (y_test == 0)).sum())
    vw_pr = float(average_precision_score(y_test, vw_probs))

    print("\nValue-Weighted Training Comparison on Natural Test Split (918,617 tx):")
    print(f"  • Standard Champion:    Missed Frauds: {int(base_missed.sum()):,} | Loss: ₹{base_loss_inr:,.2f} | False Declines: {base_fp}")
    print(f"  • Value-Weighted Model: Missed Frauds: {int(vw_missed.sum()):,} | Loss: ₹{vw_loss_inr:,.2f} | False Declines: {vw_fp} | PR-AUC: {vw_pr:.5f}")
    diff_inr = base_loss_inr - vw_loss_inr
    print(f"  • Net Financial Savings from Value-Weighted Training: ₹{diff_inr:,.2f}")

    # Compile Results
    evasion_payload = {
        "baseline_scenarios": baseline_evasion_results,
        "adversarially_augmented_scenarios": augmented_evasion_results,
        "value_weighted_comparison": {
            "standard_model": {
                "missed_frauds": int(base_missed.sum()),
                "financial_loss_inr": round(base_loss_inr, 2),
                "false_declines": base_fp,
            },
            "value_weighted_model": {
                "missed_frauds": int(vw_missed.sum()),
                "financial_loss_inr": round(vw_loss_inr, 2),
                "false_declines": vw_fp,
                "test_pr_auc": round(vw_pr, 5),
            },
            "financial_savings_inr": round(diff_inr, 2),
        },
    }

    out_file = REPORTS_DIR / "paysim_evasion_benchmark.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(evasion_payload, f, indent=2)
    print(f"\nSaved evasion findings to: {out_file}")

    # Update reports/metrics.json
    if METRICS_FILE.exists():
        with open(METRICS_FILE, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
        metrics_data["paysim_evasion_robustness"] = evasion_payload
        with open(METRICS_FILE, "w", encoding="utf-8") as f:
            json.dump(metrics_data, f, indent=2)
        print(f"Updated 'paysim_evasion_robustness' in {METRICS_FILE}")

    return evasion_payload


if __name__ == "__main__":
    run_evasion_and_value_weighting_pipeline()

