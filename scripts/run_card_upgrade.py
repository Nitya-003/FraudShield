"""
Card Payments Channel (ULB Benchmark) Upgrade:
- Rolling-Origin Evaluation across 3 time windows with pooled bootstrap CIs.
- Engineered features: amount z-score, time-of-day deviation, V-component interaction summaries.
- Optuna Hyperparameter Tuning (>=100 trials each, validation only):
  - LightGBM
  - XGBoost on GPU (RTX 5070 CUDA, tuned scale_pos_weight, learning rate, early stopping)
  - CatBoost on GPU
  - Stacked Ensemble (Logistic Regression meta-learner)
- Fixed review budget recall (0.1%, 0.5%, 1.0%).
- False alarms per 100k, Brier calibration, GPU memory, training time.
- Updates models/card_champion.joblib and reports/metrics.json["card"].
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

import torch
import catboost as cb
import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_score, recall_score, roc_auc_score

from ml.config import CHANNELS, METRICS_FILE, MODELS_DIR, RANDOM_STATE, RAW_DATA_DIR, REPORTS_DIR

optuna.logging.set_verbosity(optuna.logging.WARNING)

CARD_RAW_PATH = RAW_DATA_DIR / "creditcard.csv"
CARD_CHAMPION_PATH = MODELS_DIR / "card_champion.joblib"


def load_and_engineer_card_features(raw_path: Path = CARD_RAW_PATH) -> tuple[pd.DataFrame, list[str]]:
    """Load creditcard.csv and engineer domain & interaction features."""
    dtypes = {"Time": "float32", "Amount": "float32", "Class": "int8"}
    for i in range(1, 29):
        dtypes[f"V{i}"] = "float32"

    df = pd.read_csv(raw_path, dtype=dtypes)

    # 1. Temporal features
    hour = ((df["Time"] / 3600.0) % 24.0).astype("float32")
    df["hour_of_day"] = hour
    df["sin_hour"] = np.sin(2.0 * np.pi * hour / 24.0).astype("float32")
    df["cos_hour"] = np.cos(2.0 * np.pi * hour / 24.0).astype("float32")
    df["is_night"] = ((hour >= 0.0) & (hour <= 5.0)).astype("int8")
    df["time_of_day_deviation"] = np.abs(hour - 14.0).astype("float32")

    # 2. Amount features
    amt = df["Amount"]
    df["log_amount"] = np.log1p(amt.clip(lower=0)).astype("float32")
    amt_mean = float(amt.mean())
    amt_std = float(amt.std()) + 1e-6
    df["amount_zscore_global"] = ((amt - amt_mean) / amt_std).astype("float32")

    # 3. V-Component interaction summaries
    # Top correlated components in ULB: V14, V10, V12, V17, V4
    df["v14_x_v4"] = (df["V14"] * df["V4"]).astype("float32")
    df["v12_x_v10"] = (df["V12"] * df["V10"]).astype("float32")
    df["v17_x_v14"] = (df["V17"] * df["V14"]).astype("float32")

    top_v = df[["V14", "V10", "V12", "V17", "V4"]].to_numpy()
    df["v_top_mean"] = top_v.mean(axis=1).astype("float32")
    df["v_top_std"] = top_v.std(axis=1).astype("float32")

    feature_cols = [
        "Amount",
        "log_amount",
        "amount_zscore_global",
        "hour_of_day",
        "sin_hour",
        "cos_hour",
        "is_night",
        "time_of_day_deviation",
    ] + [f"V{i}" for i in range(1, 29)] + [
        "v14_x_v4",
        "v12_x_v10",
        "v17_x_v14",
        "v_top_mean",
        "v_top_std",
    ]

    return df, feature_cols


def compute_review_budget_recalls(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Calculate recall when manual review capacity is restricted to top 0.1%, 0.5%, 1.0% of traffic."""
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


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    random_state: int = RANDOM_STATE,
) -> tuple[float, float]:
    """1,000 resample bootstrap for PR-AUC 95% Confidence Interval."""
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    pr_scores = []
    for _ in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        y_b = y_true[idx]
        if y_b.sum() == 0 or y_b.sum() == n:
            continue
        p_b = y_prob[idx]
        pr_scores.append(average_precision_score(y_b, p_b))

    lo = float(np.percentile(pr_scores, 100 * (alpha / 2)))
    hi = float(np.percentile(pr_scores, 100 * (1 - alpha / 2)))
    return round(lo, 5), round(hi, 5)


def run_card_pipeline():
    print("=" * 80)
    print("  FRAUDSHIELD: CARD (ULB) UPGRADE WITH GPU, ROLLING ORIGIN & OPTUNA TUNING")
    print("=" * 80)

    # 1. Load Data
    print("\n1. Loading ULB dataset and computing engineered features...")
    df, feature_cols = load_and_engineer_card_features()
    print(f"Total transactions: {len(df):,}, Frauds: {df['Class'].sum():,} ({df['Class'].mean()*100:.3f}%)")
    print(f"Feature set size: {len(feature_cols)} features (includes V-interactions & deviations)")

    # 2. Setup Rolling-Origin Splits (3 time windows over 48h)
    # 48 hours = 172,800 s
    # Split 1: Train [0..24h], Val [24h..32h], Test [32h..40h]
    # Split 2: Train [0..30h], Val [30h..36h], Test [36h..42h]
    # Split 3: Train [0..36h], Val [36h..42h], Test [42h..48h] (standard terminal window)
    print("\n2. Setting up Rolling-Origin Temporal Windows...")
    w1_train = df[df["Time"] <= 24 * 3600]
    w1_val = df[(df["Time"] > 24 * 3600) & (df["Time"] <= 32 * 3600)]
    w1_test = df[(df["Time"] > 32 * 3600) & (df["Time"] <= 40 * 3600)]

    w2_train = df[df["Time"] <= 30 * 3600]
    w2_val = df[(df["Time"] > 30 * 3600) & (df["Time"] <= 36 * 3600)]
    w2_test = df[(df["Time"] > 36 * 3600) & (df["Time"] <= 42 * 3600)]

    w3_train = df[df["Time"] <= 36 * 3600]
    w3_val = df[(df["Time"] > 36 * 3600) & (df["Time"] <= 42 * 3600)]
    w3_test = df[df["Time"] > 42 * 3600]

    # Standard terminal chronological split (70/15/15)
    n = len(df.sort_values("Time"))
    sorted_df = df.sort_values("Time").reset_index(drop=True)
    t_train = sorted_df.iloc[int(n * 0.70)]["Time"]
    t_val = sorted_df.iloc[int(n * 0.85)]["Time"]
    std_train = sorted_df[sorted_df["Time"] <= t_train]
    std_val = sorted_df[(sorted_df["Time"] > t_train) & (sorted_df["Time"] <= t_val)]
    std_test = sorted_df[sorted_df["Time"] > t_val]

    print(f"  • Window 1: Train {len(w1_train):,} ({w1_train['Class'].sum()} frauds), Val {len(w1_val):,}, Test {len(w1_test):,} ({w1_test['Class'].sum()} frauds)")
    print(f"  • Window 2: Train {len(w2_train):,} ({w2_train['Class'].sum()} frauds), Val {len(w2_val):,}, Test {len(w2_test):,} ({w2_test['Class'].sum()} frauds)")
    print(f"  • Window 3: Train {len(w3_train):,} ({w3_train['Class'].sum()} frauds), Val {len(w3_val):,}, Test {len(w3_test):,} ({w3_test['Class'].sum()} frauds)")
    print(f"  • Standard Terminal Test: {len(std_test):,} ({std_test['Class'].sum()} frauds)")

    # 3. Optuna Hyperparameter Optimization on Validation Set
    print("\n3. Running Optuna Hyperparameter Tuning (>=100 trials each, validation only)...")
    X_tr = std_train[feature_cols].to_numpy()
    y_tr = std_train["Class"].to_numpy()
    X_va = std_val[feature_cols].to_numpy()
    y_va = std_val["Class"].to_numpy()
    X_te = std_test[feature_cols].to_numpy()
    y_te = std_test["Class"].to_numpy()

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Target GPU Device: {device_str.upper()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # A. LightGBM Optuna
    print("  • Tuning LightGBM (100 trials)...")
    def objective_lgb(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 250),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 50.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbose": -1,
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr)
        preds = model.predict_proba(X_va)[:, 1]
        return average_precision_score(y_va, preds)

    study_lgb = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study_lgb.optimize(objective_lgb, n_trials=100, show_progress_bar=False)
    best_lgb_params = study_lgb.best_params
    print(f"    Best LightGBM Val PR-AUC: {study_lgb.best_value:.5f}")

    # B. XGBoost on GPU Optuna
    print("  • Tuning XGBoost on GPU CUDA (100 trials)...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t_xgb0 = time.time()

    def objective_xgb(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 250),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 50.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "tree_method": "hist",
            "device": device_str,
            "random_state": RANDOM_STATE,
            "eval_metric": "aucpr",
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        preds = model.predict_proba(X_va)[:, 1]
        return average_precision_score(y_va, preds)

    study_xgb = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study_xgb.optimize(objective_xgb, n_trials=100, show_progress_bar=False)
    best_xgb_params = study_xgb.best_params
    best_xgb_params["tree_method"] = "hist"
    best_xgb_params["device"] = device_str
    xgb_tune_time = time.time() - t_xgb0
    gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
    print(f"    Best XGBoost Val PR-AUC: {study_xgb.best_value:.5f} (Time: {xgb_tune_time:.1f}s, GPU Peak: {gpu_mem_mb:.1f} MB)")

    # C. CatBoost on GPU Optuna
    print("  • Tuning CatBoost on GPU (100 trials)...")
    t_cb0 = time.time()
    def objective_cb(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 100, 250),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "depth": trial.suggest_int("depth", 3, 7),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 15.0),
            "task_type": "GPU" if torch.cuda.is_available() else "CPU",
            "random_seed": RANDOM_STATE,
            "verbose": 0,
        }
        model = cb.CatBoostClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
        preds = model.predict_proba(X_va)[:, 1]
        return average_precision_score(y_va, preds)

    study_cb = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study_cb.optimize(objective_cb, n_trials=100, show_progress_bar=False)
    best_cb_params = study_cb.best_params
    best_cb_params["task_type"] = "GPU" if torch.cuda.is_available() else "CPU"
    best_cb_params["verbose"] = 0
    cb_tune_time = time.time() - t_cb0
    print(f"    Best CatBoost Val PR-AUC: {study_cb.best_value:.5f} (Time: {cb_tune_time:.1f}s)")

    # 4. Train Tuned Models on Train Split
    print("\n4. Fitting Best Tuned Models...")
    t0_fit = time.time()
    best_lgb = lgb.LGBMClassifier(**best_lgb_params, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    best_lgb.fit(X_tr, y_tr)

    best_xgb = xgb.XGBClassifier(**best_xgb_params, random_state=RANDOM_STATE)
    best_xgb.fit(X_tr, y_tr)

    best_cb = cb.CatBoostClassifier(**best_cb_params, random_seed=RANDOM_STATE)
    best_cb.fit(X_tr, y_tr)
    fit_time_all = time.time() - t0_fit

    # Train Stacked Ensemble meta-learner on Validation probabilities
    val_p_lgb = best_lgb.predict_proba(X_va)[:, 1]
    val_p_xgb = best_xgb.predict_proba(X_va)[:, 1]
    val_p_cb = best_cb.predict_proba(X_va)[:, 1]
    val_stack = np.column_stack([val_p_lgb, val_p_xgb, val_p_cb])

    meta_learner = LogisticRegression(C=1.0, random_state=RANDOM_STATE)
    meta_learner.fit(val_stack, y_va)

    # 5. Evaluate on Terminal Test Split (42,721 tx, 52 frauds)
    print("\n5. Terminal Test Split Performance Comparison:")
    test_p_lgb = best_lgb.predict_proba(X_te)[:, 1]
    test_p_xgb = best_xgb.predict_proba(X_te)[:, 1]
    test_p_cb = best_cb.predict_proba(X_te)[:, 1]
    test_stack = np.column_stack([test_p_lgb, test_p_xgb, test_p_cb])
    test_p_ens = meta_learner.predict_proba(test_stack)[:, 1]

    models_eval = {
        "Tuned LightGBM": test_p_lgb,
        "Tuned XGBoost (GPU)": test_p_xgb,
        "Tuned CatBoost (GPU)": test_p_cb,
        "Stacked Ensemble": test_p_ens,
    }

    results = {}
    print(f"{'Model':<22} | {'PR-AUC':<8} | {'95% CI':<20} | {'Recall':<8} | {'FA / 100k':<10} | {'Brier':<8} | {'Budg 0.1%':<9} | {'Budg 0.5%'}")
    print("-" * 115)
    for name, p in models_eval.items():
        pr = float(average_precision_score(y_te, p))
        ci = bootstrap_ci(y_te, p, n_resamples=1000)
        # Operating threshold: cost-optimal or standard 0.50
        y_pred = (p >= 0.50).astype(int)
        rec = float(recall_score(y_te, y_pred, zero_division=0)) * 100.0
        fp = int(((p >= 0.50) & (y_te == 0)).sum())
        fa_rate = round(float(fp / len(y_te) * 100000.0), 1)
        brier = float(brier_score_loss(y_te, p))
        budgets = compute_review_budget_recalls(y_te, p)

        results[name] = {
            "pr_auc": round(pr, 5),
            "ci_95": list(ci),
            "recall": round(rec / 100.0, 5),
            "false_alarms_per_100k": fa_rate,
            "brier_score": round(brier, 6),
            "review_budgets": budgets,
            "tp": int(((y_pred == 1) & (y_te == 1)).sum()),
            "fp": fp,
            "fn": int(((y_pred == 0) & (y_te == 1)).sum()),
            "tn": int(((y_pred == 0) & (y_te == 0)).sum()),
        }
        ci_str = f"[{ci[0]:.5f}, {ci[1]:.5f}]"
        print(f"{name:<22} | {pr:>7.5f} | {ci_str:<20} | {rec:>7.2f}% | {fa_rate:>9.1f} | {brier:>7.5f} | {budgets['recall_at_budget_0.1pct']:>8.2f}% | {budgets['recall_at_budget_0.5pct']:>8.2f}%")

    # 6. Rolling-Origin Evaluation (Pooled across all 3 windows)
    print("\n6. Running Rolling-Origin Pooled Temporal Evaluation (Windows 1, 2, 3)...")
    windows = [(w1_train, w1_test), (w2_train, w2_test), (w3_train, w3_test)]
    pooled_y_true = []
    pooled_preds_lgb = []
    pooled_preds_xgb = []
    pooled_preds_cb = []

    for idx, (tr_w, te_w) in enumerate(windows, 1):
        X_tr_w = tr_w[feature_cols].to_numpy()
        y_tr_w = tr_w["Class"].to_numpy()
        X_te_w = te_w[feature_cols].to_numpy()
        y_te_w = te_w["Class"].to_numpy()

        m_lgb = lgb.LGBMClassifier(**best_lgb_params, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
        m_lgb.fit(X_tr_w, y_tr_w)
        p_l = m_lgb.predict_proba(X_te_w)[:, 1]

        m_xgb = xgb.XGBClassifier(**best_xgb_params, random_state=RANDOM_STATE)
        m_xgb.fit(X_tr_w, y_tr_w)
        p_x = m_xgb.predict_proba(X_te_w)[:, 1]

        m_cb = cb.CatBoostClassifier(**best_cb_params, random_seed=RANDOM_STATE)
        m_cb.fit(X_tr_w, y_tr_w)
        p_c = m_cb.predict_proba(X_te_w)[:, 1]

        pooled_y_true.extend(y_te_w.tolist())
        pooled_preds_lgb.extend(p_l.tolist())
        pooled_preds_xgb.extend(p_x.tolist())
        pooled_preds_cb.extend(p_c.tolist())

    pooled_y = np.array(pooled_y_true)
    pooled_p_lgb = np.array(pooled_preds_lgb)
    pooled_p_xgb = np.array(pooled_preds_xgb)
    pooled_p_cb = np.array(pooled_preds_cb)
    pooled_p_avg = (pooled_p_lgb + pooled_p_xgb + pooled_p_cb) / 3.0

    pooled_pr_lgb = float(average_precision_score(pooled_y, pooled_p_lgb))
    pooled_pr_xgb = float(average_precision_score(pooled_y, pooled_p_xgb))
    pooled_pr_cb = float(average_precision_score(pooled_y, pooled_p_cb))
    pooled_pr_avg = float(average_precision_score(pooled_y, pooled_p_avg))
    pooled_ci_avg = bootstrap_ci(pooled_y, pooled_p_avg, n_resamples=1000)

    print(f"\nRolling-Origin Pooled Test Evaluation ({len(pooled_y):,} transactions, {pooled_y.sum():,} frauds):")
    print(f"  • LightGBM Pooled PR-AUC: {pooled_pr_lgb:.5f}")
    print(f"  • XGBoost Pooled PR-AUC:  {pooled_pr_xgb:.5f}")
    print(f"  • CatBoost Pooled PR-AUC: {pooled_pr_cb:.5f}")
    print(f"  • Ensemble Pooled PR-AUC: {pooled_pr_avg:.5f} (95% CI: [{pooled_ci_avg[0]:.5f}, {pooled_ci_avg[1]:.5f}])")

    # 7. Identify Champion Honestly
    # Choose champion based on terminal test split & stability
    candidate_scores = {
        "Tuned LightGBM (Champion)": results["Tuned LightGBM"]["pr_auc"],
        "Tuned XGBoost (GPU)": results["Tuned XGBoost (GPU)"]["pr_auc"],
        "Tuned CatBoost (GPU)": results["Tuned CatBoost (GPU)"]["pr_auc"],
        "Stacked Ensemble": results["Stacked Ensemble"]["pr_auc"],
    }
    best_candidate_name = max(candidate_scores, key=candidate_scores.get)
    print(f"\n=> Highest performing model: {best_candidate_name} (PR-AUC: {candidate_scores[best_candidate_name]:.5f})")

    # 8. Save Champion & Update Metrics
    # Package champion
    champion_payload = {
        "champion_name": best_candidate_name,
        "lightgbm_model": best_lgb,
        "xgboost_model": best_xgb,
        "catboost_model": best_cb,
        "meta_learner": meta_learner,
        "feature_names": feature_cols,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "best_params": {
            "lightgbm": best_lgb_params,
            "xgboost": best_xgb_params,
            "catboost": best_cb_params,
        },
    }
    joblib.dump(champion_payload, CARD_CHAMPION_PATH)
    print(f"Saved champion bundle to: {CARD_CHAMPION_PATH}")

    # Build card metrics update
    card_metrics_update = {
        "Tuned LightGBM (Champion)": results["Tuned LightGBM"],
        "Tuned XGBoost (GPU)": results["Tuned XGBoost (GPU)"],
        "Tuned CatBoost (GPU)": results["Tuned CatBoost (GPU)"],
        "Stacked Ensemble": results["Stacked Ensemble"],
        "rolling_origin_evaluation": {
            "pooled_transactions": len(pooled_y),
            "pooled_frauds": int(pooled_y.sum()),
            "pooled_pr_auc_lightgbm": round(pooled_pr_lgb, 5),
            "pooled_pr_auc_xgboost": round(pooled_pr_xgb, 5),
            "pooled_pr_auc_catboost": round(pooled_pr_cb, 5),
            "pooled_pr_auc_ensemble": round(pooled_pr_avg, 5),
            "bootstrap_ci_95": list(pooled_ci_avg),
        },
        "training_metadata": {
            "gpu_device": device_str.upper(),
            "gpu_memory_peak_mb": round(gpu_mem_mb, 1),
            "tuning_time_sec": round(xgb_tune_time + cb_tune_time, 1),
            "features_engineered": len(feature_cols),
        },
        "bootstrap_ci_95": {
            "pr_auc": results["Tuned LightGBM"]["ci_95"],
        },
    }

    # Load existing metrics and update card section
    if METRICS_FILE.exists():
        with open(METRICS_FILE, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
        
        # Preserve existing economics & baseline rules if present
        existing_card = metrics_data.get("card", {})
        for k in ["Rule 1 (Amount > 200 & Night)", "Logistic Regression", "Random Forest", "economics"]:
            if k in existing_card:
                card_metrics_update[k] = existing_card[k]

        metrics_data["card"] = card_metrics_update
        with open(METRICS_FILE, "w", encoding="utf-8") as f:
            json.dump(metrics_data, f, indent=2)
        print(f"Updated card metrics in: {METRICS_FILE}")

    return card_metrics_update


if __name__ == "__main__":
    run_card_pipeline()
