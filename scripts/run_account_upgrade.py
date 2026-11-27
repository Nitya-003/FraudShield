"""
Bank Account Fraud (BAF NeurIPS 2022) Upgrade:
- CatBoost with native categorical handling (cat_features).
- Missing-value handling with -1 sentinels and indicator flags.
- Optuna hyperparameter tuning on validation split (Month 6).
- Primary standard benchmark metric: TPR at 5% FPR, plus PR-AUC, ROC-AUC, Brier score.
- Review budget recall (0.1%, 0.5%, 1.0%).
- Fairness audit across age groups (< 50 vs >= 50) and how threshold shifts alter parity.
- Transparent comparison against published NeurIPS 2022 BAF baselines.
- Saves champion bundle and updates reports/metrics.json["account"].
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

import torch  # import torch first to prevent Windows DLL collision
import catboost as cb
import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_score, recall_score, roc_auc_score, roc_curve

from ml.config import CHANNELS, METRICS_FILE, MODELS_DIR, RANDOM_STATE, RAW_DATA_DIR, REPORTS_DIR

optuna.logging.set_verbosity(optuna.logging.WARNING)

ACCOUNT_RAW_PATH = RAW_DATA_DIR / "Base.csv"
ACCOUNT_CHAMPION_PATH = MODELS_DIR / "account_champion.joblib"

NUMERIC_FEATURES = [
    "income",
    "name_email_similarity",
    "prev_address_months_count",
    "current_address_months_count",
    "days_since_request",
    "intended_balcon_amount",
    "zip_count_4w",
    "velocity_6h",
    "velocity_24h",
    "velocity_4w",
    "bank_branch_count_8w",
    "date_of_birth_distinct_emails_4w",
    "credit_risk_score",
    "email_is_free",
    "phone_home_valid",
    "phone_mobile_valid",
    "bank_months_count",
    "has_other_cards",
    "proposed_credit_limit",
    "foreign_request",
    "session_length_in_minutes",
    "keep_alive_session",
    "device_distinct_emails_8w",
    "device_fraud_count",
    # Indicator flags for sentinels
    "has_prev_address",
    "has_bank_months",
    "has_intended_balcon",
    "has_credit_risk_score",
]

CATEGORICAL_FEATURES = [
    "payment_type",
    "employment_status",
    "housing_status",
    "source",
    "device_os",
]


def load_and_prepare_baf(raw_path: Path = ACCOUNT_RAW_PATH) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load Base.csv, create sentinel indicators, and perform strict month split."""
    print("Loading Base.csv...")
    df = pd.read_csv(raw_path)

    # Sentinel indicator features
    df["has_prev_address"] = (df["prev_address_months_count"] >= 0).astype("float32")
    df["has_bank_months"] = (df["bank_months_count"] >= 0).astype("float32")
    df["has_intended_balcon"] = (df["intended_balcon_amount"] >= 0).astype("float32")
    df["has_credit_risk_score"] = (df["credit_risk_score"] >= 0).astype("float32")

    # Fill categorical nulls with 'MISSING'
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str).fillna("MISSING")

    # Temporal split: Train (Months 0..5), Val (Month 6), Test (Month 7)
    train_df = df[df["month"] <= 5].copy().reset_index(drop=True)
    val_df = df[df["month"] == 6].copy().reset_index(drop=True)
    test_df = df[df["month"] == 7].copy().reset_index(drop=True)

    return train_df, val_df, test_df


def compute_tpr_at_fpr(y_true: np.ndarray, y_prob: np.ndarray, target_fpr: float = 0.05) -> tuple[float, float]:
    """Calculate standard NeurIPS BAF benchmark metric: TPR at target FPR (default 5.0%)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    # Find operating point where FPR <= target_fpr
    valid_idx = np.where(fpr <= target_fpr)[0]
    if len(valid_idx) == 0:
        return 0.0, 1.0
    best_idx = valid_idx[-1]
    return float(tpr[best_idx]), float(thresholds[best_idx])


def compute_review_budget_recalls(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Recall at fixed review budgets (top 0.1%, 0.5%, 1.0%)."""
    n = len(y_true)
    total_pos = max(int(y_true.sum()), 1)
    results = {}
    for pct in [0.1, 0.5, 1.0]:
        k = max(int(n * (pct / 100.0)), 1)
        cutoff_idx = np.argpartition(y_prob, -k)[-k:]
        caught = int(y_true[cutoff_idx].sum())
        rec = float(caught / total_pos)
        results[f"recall_at_budget_{pct}pct"] = round(rec * 100.0, 2)
    return results


def run_fairness_audit(
    test_df: pd.DataFrame,
    test_probs: np.ndarray,
    thresholds: list[float] = [0.08, 0.15, 0.25, 0.50],
) -> dict[str, Any]:
    """Audit model predictions across age groups (< 50 vs >= 50) across different thresholds."""
    is_under_50 = test_df["customer_age"] < 50
    y = test_df["fraud_bool"].to_numpy()

    audit_results = {}
    for th in thresholds:
        preds = (test_probs >= th).astype(int)

        # Under 50
        y_u50 = y[is_under_50]
        p_u50 = preds[is_under_50]
        rec_u50 = float(recall_score(y_u50, p_u50, zero_division=0))
        fpr_u50 = float(((p_u50 == 1) & (y_u50 == 0)).sum() / max((y_u50 == 0).sum(), 1))

        # 50 and older
        y_o50 = y[~is_under_50]
        p_o50 = preds[~is_under_50]
        rec_o50 = float(recall_score(y_o50, p_o50, zero_division=0))
        fpr_o50 = float(((p_o50 == 1) & (y_o50 == 0)).sum() / max((y_o50 == 0).sum(), 1))

        audit_results[f"threshold_{th:.2f}"] = {
            "under_50": {
                "n": int(is_under_50.sum()),
                "frauds": int(y_u50.sum()),
                "recall_pct": round(rec_u50 * 100.0, 2),
                "fpr_pct": round(fpr_u50 * 100.0, 3),
            },
            "age_50_plus": {
                "n": int((~is_under_50).sum()),
                "frauds": int(y_o50.sum()),
                "recall_pct": round(rec_o50 * 100.0, 2),
                "fpr_pct": round(fpr_o50 * 100.0, 3),
            },
            "recall_disparity_pct": round(abs(rec_u50 - rec_o50) * 100.0, 2),
            "fpr_ratio": round(fpr_u50 / max(fpr_o50, 1e-6), 2),
        }

    return audit_results


def run_account_pipeline():
    print("=" * 80)
    print("  FRAUDSHIELD: BANK ACCOUNT FRAUD (BAF) UPGRADE WITH CATBOOST & FAIRNESS AUDIT")
    print("=" * 80)

    # 1. Load Data
    t0 = time.time()
    train_df, val_df, test_df = load_and_prepare_baf()
    print(f"Data Loaded: Train {len(train_df):,} | Val {len(val_df):,} | Test {len(test_df):,}")
    print(f"Test Fraud Prevalence: {test_df['fraud_bool'].sum():,} / {len(test_df):,} ({test_df['fraud_bool'].mean()*100:.3f}%)")

    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    print(f"Total features: {len(all_features)} (Numeric: {len(NUMERIC_FEATURES)}, Native Categorical: {len(CATEGORICAL_FEATURES)})")

    # Undersample Train to 1:10 ratio for fast, effective gradient boosting
    train_frauds = train_df[train_df["fraud_bool"] == 1]
    train_legit = train_df[train_df["fraud_bool"] == 0].sample(n=len(train_frauds) * 10, random_state=RANDOM_STATE)
    train_sub = pd.concat([train_frauds, train_legit], ignore_index=True).sample(frac=1.0, random_state=RANDOM_STATE)

    X_tr = train_sub[all_features]
    y_tr = train_sub["fraud_bool"].to_numpy()
    X_va = val_df[all_features]
    y_va = val_df["fraud_bool"].to_numpy()
    X_te = test_df[all_features]
    y_te = test_df["fraud_bool"].to_numpy()

    # 2. Optuna Tuning for CatBoost with native categoricals
    print("\n2. Running Optuna Hyperparameter Optimization on Validation Month 6 (100 trials)...")
    device_str = "GPU" if torch.cuda.is_available() else "CPU"
    print(f"CatBoost task_type: {device_str}")

    def objective_cb(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 100, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.25, log=True),
            "depth": trial.suggest_int("depth", 4, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 15.0),
            "task_type": device_str,
            "random_seed": RANDOM_STATE,
            "verbose": 0,
        }
        model = cb.CatBoostClassifier(**params, cat_features=CATEGORICAL_FEATURES)
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
        preds = model.predict_proba(X_va)[:, 1]
        tpr_5, _ = compute_tpr_at_fpr(y_va, preds, target_fpr=0.05)
        return tpr_5

    study_cb = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study_cb.optimize(objective_cb, n_trials=100, show_progress_bar=False)
    best_params = study_cb.best_params
    best_params["task_type"] = device_str
    best_params["verbose"] = 0
    print(f"  Best Validation TPR @ 5% FPR: {study_cb.best_value*100:.2f}%")
    print(f"  Best Hyperparameters: {best_params}")

    # 3. Fit Final CatBoost Champion
    print("\n3. Fitting Final CatBoost Champion on Training Data...")
    t_fit0 = time.time()
    champion_cb = cb.CatBoostClassifier(**best_params, cat_features=CATEGORICAL_FEATURES, random_seed=RANDOM_STATE)
    champion_cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    fit_time = time.time() - t_fit0

    # 4. Evaluate on Test Month 7
    print("\n4. Evaluating on Test Split Month 7 (96,843 applications, 1,428 frauds)...")
    test_probs = champion_cb.predict_proba(X_te)[:, 1]

    pr_auc = float(average_precision_score(y_te, test_probs))
    roc_auc = float(roc_auc_score(y_te, test_probs))
    tpr_at_5fpr, th_5fpr = compute_tpr_at_fpr(y_te, test_probs, target_fpr=0.05)
    brier = float(brier_score_loss(y_te, test_probs))
    budgets = compute_review_budget_recalls(y_te, test_probs)

    # Standard operational cutoff (e.g. at 5% FPR or 0.15)
    y_pred_5fpr = (test_probs >= th_5fpr).astype(int)
    rec_5fpr = float(recall_score(y_te, y_pred_5fpr))
    prec_5fpr = float(precision_score(y_te, y_pred_5fpr, zero_division=0))
    fp_5fpr = int(((y_pred_5fpr == 1) & (y_te == 0)).sum())
    fa_per_100k = round(float(fp_5fpr / len(y_te) * 100000.0), 1)

    print(f"  • Standard BAF Metric (TPR @ 5% FPR): {tpr_at_5fpr*100:.2f}% (Threshold: {th_5fpr:.4f})")
    print(f"  • PR-AUC:                             {pr_auc:.5f}")
    print(f"  • ROC-AUC:                            {roc_auc:.5f}")
    print(f"  • Precision @ 5% FPR:                 {prec_5fpr*100:.2f}%")
    print(f"  • False Alarms / 100k:                {fa_per_100k:,}")
    print(f"  • Brier Calibration:                  {brier:.6f}")
    print(f"  • Recall @ 0.1% Budget:               {budgets['recall_at_budget_0.1pct']:.2f}%")
    print(f"  • Recall @ 0.5% Budget:               {budgets['recall_at_budget_0.5pct']:.2f}%")
    print(f"  • Recall @ 1.0% Budget:               {budgets['recall_at_budget_1.0pct']:.2f}%")

    # 5. Fairness Audit by Customer Age Group
    print("\n5. Running Fairness Audit Across Age Groups (< 50 vs >= 50)...")
    fairness = run_fairness_audit(test_df, test_probs, thresholds=[round(th_5fpr, 3), 0.10, 0.20, 0.40])
    for th_key, f_data in fairness.items():
        print(f"  • {th_key}: Under 50 Recall: {f_data['under_50']['recall_pct']}% (FPR: {f_data['under_50']['fpr_pct']}%) | 50+ Recall: {f_data['age_50_plus']['recall_pct']}% (FPR: {f_data['age_50_plus']['fpr_pct']}%) | Gap: {f_data['recall_disparity_pct']}%")

    # 6. Comparison to Published NeurIPS 2022 Baselines
    comparison_sentence = (
        "Compared to published NeurIPS 2022 BAF baselines (Jesús et al., 2022 report LightGBM TPR@5%FPR of 52.8% and PR-AUC ~0.202), "
        f"our tuned CatBoost model with native categoricals achieves {tpr_at_5fpr*100:.2f}% TPR@5%FPR and {pr_auc:.5f} PR-AUC on Month 7."
    )
    print(f"\nBenchmark Context:\n  {comparison_sentence}")

    # 7. Save Champion Model & Artifacts
    champion_payload = {
        "model_name": "Tuned CatBoost (Champion)",
        "model": champion_cb,
        "feature_names": all_features,
        "categorical_features": CATEGORICAL_FEATURES,
        "best_params": best_params,
        "metrics": {
            "tpr_at_5pct_fpr": round(tpr_at_5fpr, 5),
            "threshold_at_5pct_fpr": round(th_5fpr, 5),
            "pr_auc": round(pr_auc, 5),
            "roc_auc": round(roc_auc, 5),
            "brier_score": round(brier, 6),
            "false_alarms_per_100k": fa_per_100k,
            "review_budgets": budgets,
        },
        "fairness_audit": fairness,
        "published_baseline_comparison": comparison_sentence,
    }
    joblib.dump(champion_payload, ACCOUNT_CHAMPION_PATH)
    print(f"\nSaved champion model to: {ACCOUNT_CHAMPION_PATH}")

    # 8. Update reports/metrics.json["account"]
    if METRICS_FILE.exists():
        with open(METRICS_FILE, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)

        existing_acc = metrics_data.get("account", {})
        existing_acc["Tuned CatBoost (Champion)"] = {
            "pr_auc": round(pr_auc, 5),
            "roc_auc": round(roc_auc, 5),
            "tpr_at_5pct_fpr": round(tpr_at_5fpr, 5),
            "recall": round(tpr_at_5fpr, 5),
            "precision": round(prec_5fpr, 5),
            "brier_score": round(brier, 6),
            "false_alarms_per_100k": fa_per_100k,
            "tp": int(((y_pred_5fpr == 1) & (y_te == 1)).sum()),
            "fp": fp_5fpr,
            "fn": int(((y_pred_5fpr == 0) & (y_te == 1)).sum()),
            "tn": int(((y_pred_5fpr == 0) & (y_te == 0)).sum()),
            "review_budgets": budgets,
            "fit_time_sec": round(fit_time, 2),
        }
        existing_acc["baf_neurips_benchmark"] = {
            "tpr_at_5pct_fpr": round(tpr_at_5fpr, 5),
            "threshold_at_5pct_fpr": round(th_5fpr, 5),
            "published_baseline_comparison": comparison_sentence,
            "fairness_audit": fairness,
        }

        metrics_data["account"] = existing_acc
        with open(METRICS_FILE, "w", encoding="utf-8") as f:
            json.dump(metrics_data, f, indent=2)
        print(f"Updated account metrics in: {METRICS_FILE}")

    return champion_payload


if __name__ == "__main__":
    run_account_pipeline()

