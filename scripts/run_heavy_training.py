"""
Heavy Model Training, Optuna GPU Hyperparameter Tuning, and Stacked Ensemble Benchmark.

Executes:
1. Expanding-window temporal cross-validation (3 folds) on train timeline.
2. Optuna hyperparameter optimization (100 trials each) for LightGBM (CPU), XGBoost (GPU), CatBoost (GPU).
3. Fair baseline comparison with Logistic Regression and Random Forest.
4. Stacked ensemble (blender / meta-learner on validation probabilities).
5. Comprehensive evaluation on the untouched natural-prevalence test set (918,617 transactions):
   - PR-AUC, ROC-AUC, Recall, Precision, F1, Brier score, False Alarms per 100k, Confusion Matrix.
   - 1,000-resample bootstrap 95% confidence intervals for PR-AUC and False Alarms per 100k.
   - INR expected cost curves across thresholds.
   - Inference latency benchmarking (ms per 10,000 transactions).
6. Artifact generation:
   - reports/heavy_training_benchmarks.csv
   - reports/figures/model_comparison.png
   - reports/figures/cost_curves.png
   - reports/metrics.json update
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

import catboost as cb
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from ml.config import (
    COST_FALSE_DECLINE_INR,
    COST_OTP_CHALLENGE_INR,
    FIGURES_DIR,
    METRICS_FILE,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    STEP_TRAIN_MAX,
    STEP_VAL_MAX,
    TARGET_COL,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
)
from ml.features import ALL_FEATURE_COLUMNS, FEATURE_COLUMNS

optuna.logging.set_verbosity(optuna.logging.WARNING)


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute primary fraud metrics at natural prevalence."""
    y_pred = (y_prob >= threshold).astype(int)
    pr_auc = float(average_precision_score(y_true, y_prob))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    brier = float(brier_score_loss(y_true, y_prob))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    n_total = len(y_true)
    fa_per_100k = float((fp / n_total) * 100000)

    return {
        "pr_auc": round(pr_auc, 5),
        "roc_auc": round(roc_auc, 5),
        "recall": round(rec, 5),
        "precision": round(prec, 5),
        "f1": round(f1, 5),
        "brier_score": round(brier, 6),
        "false_alarms_per_100k": round(fa_per_100k, 1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    n_resamples: int = 1000,
    random_seed: int = 42,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Compute 95% bootstrap confidence intervals for PR-AUC and False Alarms per 100k
    using fast Poisson resampling on sorted rank arrays.
    """
    rng = np.random.default_rng(random_seed)
    n = len(y_true)
    y_pred = (y_prob >= threshold).astype(np.int8)

    # 1. Fast PR-AUC bootstrap via pre-sorted rank permutation
    sort_idx = np.argsort(-y_prob)
    y_sorted = y_true[sort_idx].astype(np.float64)

    pr_auc_samples = np.empty(n_resamples, dtype=np.float64)
    fa_samples = np.empty(n_resamples, dtype=np.float64)

    is_fp = ((y_pred == 1) & (y_true == 0)).astype(np.float64)

    for b in range(n_resamples):
        w = rng.poisson(1.0, size=n)
        
        # PR-AUC
        tp_cum = np.cumsum(w * y_sorted)
        fp_cum = np.cumsum(w * (1.0 - y_sorted))
        total_p = tp_cum[-1]
        if total_p > 0:
            rec = tp_cum / total_p
            prec = tp_cum / np.maximum(tp_cum + fp_cum, 1.0)
            diff_r = np.diff(np.concatenate(([0.0], rec)))
            pr_auc_samples[b] = np.sum(diff_r * prec)
        else:
            pr_auc_samples[b] = 0.0

        # False Alarms / 100k
        boot_fp = np.sum(w * is_fp)
        boot_total = np.sum(w)
        fa_samples[b] = (boot_fp / max(boot_total, 1)) * 100000.0

    ci_pr_low, ci_pr_high = np.percentile(pr_auc_samples, [2.5, 97.5])
    ci_fa_low, ci_fa_high = np.percentile(fa_samples, [2.5, 97.5])

    return (round(float(ci_pr_low), 5), round(float(ci_pr_high), 5)), (
        round(float(ci_fa_low), 1),
        round(float(ci_fa_high), 1),
    )


def compute_cost_curve(
    y_true: np.ndarray,
    amounts: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Compute financial loss curve in INR across thresholds."""
    costs = np.empty(len(thresholds), dtype=np.float64)
    for idx, th in enumerate(thresholds):
        fn_mask = (y_prob < th) & (y_true == 1)
        fp_mask = (y_prob >= th) & (y_true == 0)
        missed_fraud_inr = np.sum(amounts[fn_mask])
        friction_inr = np.sum(fp_mask) * COST_FALSE_DECLINE_INR
        costs[idx] = missed_fraud_inr + friction_inr

    best_idx = int(np.argmin(costs))
    return costs, float(thresholds[best_idx]), float(costs[best_idx])


def main():
    print("=" * 75)
    print("  FRAUDSHIELD: HEAVY MODEL TRAINING (GPU & CPU) + OPTUNA + ENSEMBLE")
    print("=" * 75)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("\n1. Loading preprocessed splits...")
    train_us_df = pd.read_parquet(PROCESSED_DATA_DIR / "train_undersampled.parquet")
    val_df = pd.read_parquet(VAL_DATA_PATH)
    test_df = pd.read_parquet(TEST_DATA_PATH)

    features = ALL_FEATURE_COLUMNS
    X_train_us = train_us_df[features].to_numpy()
    y_train_us = train_us_df[TARGET_COL].to_numpy()

    X_val = val_df[features].to_numpy()
    y_val = val_df[TARGET_COL].to_numpy()

    X_test = test_df[features].to_numpy()
    y_test = test_df[TARGET_COL].to_numpy()
    amounts_test = test_df["amount"].to_numpy()

    print(f"  Undersampled Train: {len(X_train_us):,} rows (Features: {len(features)})")
    print(f"  Validation (Natural):{len(X_val):,} rows (Frauds: {y_val.sum():,} - {y_val.mean()*100:.4f}%)")
    print(f"  Test (Natural):      {len(X_test):,} rows (Frauds: {y_test.sum():,} - {y_test.mean()*100:.4f}%)")

    # -----------------------------------------------------------------
    # STEP 1: Expanding-Window Cross-Validation (3 Folds)
    # -----------------------------------------------------------------
    print("\n2. Running Expanding-Window Temporal Cross-Validation on Train (3 Folds)...")
    train_steps = train_us_df["step"].to_numpy()
    cv_folds = [(150, 200), (200, 260), (260, STEP_TRAIN_MAX)]
    for fold_idx, (tr_end, val_end) in enumerate(cv_folds, 1):
        tr_m = train_steps <= tr_end
        va_m = (train_steps > tr_end) & (train_steps <= val_end)
        if tr_m.sum() == 0 or va_m.sum() == 0:
            continue
        lgb_fold = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
        lgb_fold.fit(X_train_us[tr_m], y_train_us[tr_m])
        p_fold = lgb_fold.predict_proba(X_train_us[va_m])[:, 1]
        score_fold = average_precision_score(y_train_us[va_m], p_fold)
        print(f"  Fold {fold_idx}: Train steps 1..{tr_end} ({tr_m.sum():,} rows) -> Val steps {tr_end+1}..{val_end} ({va_m.sum():,} rows) | PR-AUC: {score_fold:.5f}")

    # -----------------------------------------------------------------
    # STEP 2: Optuna GPU & CPU Hyperparameter Optimization (100 trials each)
    # -----------------------------------------------------------------
    print("\n3. Running Optuna Hyperparameter Optimization (100 trials per boosted model)...")

    # LightGBM (CPU)
    print("  [Optuna] Tuning LightGBM (CPU) over 100 trials...")
    t_opt_lgb0 = time.time()
    def obj_lgb(trial):
        lr = trial.suggest_float("learning_rate", 0.02, 0.20, log=True)
        max_depth = trial.suggest_int("max_depth", 4, 8)
        num_leaves = trial.suggest_int("num_leaves", 15, 63)
        subsample = trial.suggest_float("subsample", 0.65, 1.0)
        colsample = trial.suggest_float("colsample_bytree", 0.65, 1.0)
        min_child = trial.suggest_int("min_child_samples", 10, 50)
        clf = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=lr,
            max_depth=max_depth,
            num_leaves=num_leaves,
            subsample=subsample,
            colsample_bytree=colsample,
            min_child_samples=min_child,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
        clf.fit(X_train_us, y_train_us)
        preds = clf.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, preds)

    study_lgb = optuna.create_study(direction="maximize")
    study_lgb.optimize(obj_lgb, n_trials=100)
    best_lgb_params = study_lgb.best_params
    print(f"  [Optuna] LightGBM Best Val PR-AUC: {study_lgb.best_value:.5f} ({time.time()-t_opt_lgb0:.1f}s)")

    # XGBoost (GPU)
    print("  [Optuna] Tuning XGBoost (GPU - CUDA) over 100 trials...")
    t_opt_xgb0 = time.time()
    def obj_xgb(trial):
        lr = trial.suggest_float("learning_rate", 0.02, 0.20, log=True)
        max_depth = trial.suggest_int("max_depth", 4, 8)
        subsample = trial.suggest_float("subsample", 0.65, 1.0)
        colsample = trial.suggest_float("colsample_bytree", 0.65, 1.0)
        min_child = trial.suggest_int("min_child_weight", 1, 10)
        clf = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=lr,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample,
            min_child_weight=min_child,
            device="cuda",
            tree_method="hist",
            random_state=RANDOM_STATE,
        )
        clf.fit(X_train_us, y_train_us)
        preds = clf.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, preds)

    study_xgb = optuna.create_study(direction="maximize")
    study_xgb.optimize(obj_xgb, n_trials=100)
    best_xgb_params = study_xgb.best_params
    print(f"  [Optuna] XGBoost Best Val PR-AUC: {study_xgb.best_value:.5f} ({time.time()-t_opt_xgb0:.1f}s)")

    # CatBoost (GPU)
    print("  [Optuna] Tuning CatBoost (GPU - CUDA) over 100 trials...")
    t_opt_cat0 = time.time()
    def obj_cat(trial):
        lr = trial.suggest_float("learning_rate", 0.03, 0.20, log=True)
        depth = trial.suggest_int("depth", 4, 7)
        l2_reg = trial.suggest_float("l2_leaf_reg", 1.0, 10.0)
        clf = cb.CatBoostClassifier(
            iterations=150,
            learning_rate=lr,
            depth=depth,
            l2_leaf_reg=l2_reg,
            task_type="GPU",
            random_seed=RANDOM_STATE,
            verbose=0,
        )
        clf.fit(X_train_us, y_train_us)
        preds = clf.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, preds)

    study_cat = optuna.create_study(direction="maximize")
    study_cat.optimize(obj_cat, n_trials=100)
    best_cat_params = study_cat.best_params
    print(f"  [Optuna] CatBoost Best Val PR-AUC: {study_cat.best_value:.5f} ({time.time()-t_opt_cat0:.1f}s)")

    # -----------------------------------------------------------------
    # STEP 3: Train All Final Candidate Models & Measure Latency
    # -----------------------------------------------------------------
    print("\n4. Training Final Models & Benchmarking Latencies...")

    models = {}
    train_times = {}
    inference_latencies = {}  # ms for 10k predictions
    sample_10k = X_test[:10000]

    # Model 1: Logistic Regression
    print("  Training Logistic Regression...")
    t0 = time.time()
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train_us)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    sample_10k_scaled = scaler.transform(sample_10k)

    lr_model = LogisticRegression(C=10.0, max_iter=1000, random_state=RANDOM_STATE)
    lr_model.fit(X_tr_scaled, y_train_us)
    train_times["Logistic Regression"] = time.time() - t0

    t_inf = time.time()
    lr_model.predict_proba(sample_10k_scaled)
    inference_latencies["Logistic Regression"] = (time.time() - t_inf) * 1000.0
    models["Logistic Regression"] = (lr_model, X_val_scaled, X_test_scaled)

    # Model 2: Random Forest
    print("  Training Random Forest (100 trees)...")
    t0 = time.time()
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1)
    rf_model.fit(X_train_us, y_train_us)
    train_times["Random Forest"] = time.time() - t0

    t_inf = time.time()
    rf_model.predict_proba(sample_10k)
    inference_latencies["Random Forest"] = (time.time() - t_inf) * 1000.0
    models["Random Forest"] = (rf_model, X_val, X_test)

    # Model 3: Tuned LightGBM (CPU)
    print("  Training Tuned LightGBM (CPU)...")
    t0 = time.time()
    lgb_champion = lgb.LGBMClassifier(
        n_estimators=300,
        **best_lgb_params,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_champion.fit(X_train_us, y_train_us)
    train_times["Tuned LightGBM (CPU)"] = time.time() - t0

    t_inf = time.time()
    lgb_champion.predict_proba(sample_10k)
    inference_latencies["Tuned LightGBM (CPU)"] = (time.time() - t_inf) * 1000.0
    models["Tuned LightGBM (CPU)"] = (lgb_champion, X_val, X_test)

    # Model 4: Tuned XGBoost (GPU)
    print("  Training Tuned XGBoost (GPU)...")
    t0 = time.time()
    xgb_champion = xgb.XGBClassifier(
        n_estimators=300,
        **best_xgb_params,
        device="cuda",
        tree_method="hist",
        random_state=RANDOM_STATE,
    )
    xgb_champion.fit(X_train_us, y_train_us)
    train_times["Tuned XGBoost (GPU)"] = time.time() - t0

    t_inf = time.time()
    xgb_champion.predict_proba(sample_10k)
    inference_latencies["Tuned XGBoost (GPU)"] = (time.time() - t_inf) * 1000.0
    models["Tuned XGBoost (GPU)"] = (xgb_champion, X_val, X_test)

    # Model 5: Tuned CatBoost (GPU)
    print("  Training Tuned CatBoost (GPU)...")
    t0 = time.time()
    cat_champion = cb.CatBoostClassifier(
        iterations=300,
        **best_cat_params,
        task_type="GPU",
        random_seed=RANDOM_STATE,
        verbose=0,
    )
    cat_champion.fit(X_train_us, y_train_us)
    train_times["Tuned CatBoost (GPU)"] = time.time() - t0

    t_inf = time.time()
    cat_champion.predict_proba(sample_10k)
    inference_latencies["Tuned CatBoost (GPU)"] = (time.time() - t_inf) * 1000.0
    models["Tuned CatBoost (GPU)"] = (cat_champion, X_val, X_test)

    # -----------------------------------------------------------------
    # STEP 4: Stacked Ensemble (Meta-Learner)
    # -----------------------------------------------------------------
    print("\n5. Fitting Stacked Ensemble (Meta-Learner on Validation Probs)...")
    val_probs_matrix = np.column_stack([
        m.predict_proba(x_va)[:, 1] for m, x_va, _ in models.values()
    ])
    test_probs_matrix = np.column_stack([
        m.predict_proba(x_te)[:, 1] for m, _, x_te in models.values()
    ])

    t0 = time.time()
    meta_learner = LogisticRegression(C=1.0, random_state=RANDOM_STATE)
    meta_learner.fit(val_probs_matrix, y_val)
    train_times["Stacked Ensemble (Blender)"] = time.time() - t0 + sum(train_times.values())

    t_inf = time.time()
    meta_learner.predict_proba(test_probs_matrix[:10000])
    inference_latencies["Stacked Ensemble (Blender)"] = sum(inference_latencies.values()) + ((time.time() - t_inf) * 1000.0)

    val_ensemble_preds = meta_learner.predict_proba(val_probs_matrix)[:, 1]
    test_ensemble_preds = meta_learner.predict_proba(test_probs_matrix)[:, 1]

    # Collect all test predictions
    test_predictions = {name: m.predict_proba(x_te)[:, 1] for name, (m, _, x_te) in models.items()}
    test_predictions["Stacked Ensemble (Blender)"] = test_ensemble_preds

    # -----------------------------------------------------------------
    # STEP 5: Comprehensive Test Evaluation & 1,000 Bootstrap CIs
    # -----------------------------------------------------------------
    print("\n6. Evaluating on Untouched Test Set at Natural Prevalence (with 1,000 Bootstrap Resamples)...")
    results = []
    threshold_grid = np.linspace(0.01, 0.99, 99)
    cost_curves_data = {}

    for name, p_test in test_predictions.items():
        print(f"  Evaluating {name}...")
        metrics = compute_comprehensive_metrics(y_test, p_test, threshold=0.5)
        (pr_ci_low, pr_ci_high), (fa_ci_low, fa_ci_high) = compute_bootstrap_ci(
            y_test, p_test, threshold=0.5, n_resamples=1000, random_seed=RANDOM_STATE
        )
        costs, best_th, min_cost = compute_cost_curve(y_test, amounts_test, p_test, threshold_grid)
        cost_curves_data[name] = (costs, best_th, min_cost)

        rec = {
            "model_name": name,
            "pr_auc": metrics["pr_auc"],
            "pr_auc_95_ci": f"[{pr_ci_low:.5f}, {pr_ci_high:.5f}]",
            "pr_ci_low": pr_ci_low,
            "pr_ci_high": pr_ci_high,
            "roc_auc": metrics["roc_auc"],
            "recall": metrics["recall"],
            "precision": metrics["precision"],
            "f1": metrics["f1"],
            "brier_score": metrics["brier_score"],
            "false_alarms_per_100k": metrics["false_alarms_per_100k"],
            "fa_95_ci": f"[{fa_ci_low:.1f}, {fa_ci_high:.1f}]",
            "fa_ci_low": fa_ci_low,
            "fa_ci_high": fa_ci_high,
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "tn": metrics["tn"],
            "train_time_sec": round(train_times.get(name, 0.0), 2),
            "infer_latency_ms_10k": round(inference_latencies.get(name, 0.0), 2),
            "cost_min_threshold": round(best_th, 3),
            "min_expected_cost_inr": round(min_cost, 2),
        }
        results.append(rec)
        print(f"    PR-AUC: {metrics['pr_auc']:.5f} (95% CI {rec['pr_auc_95_ci']}) | FP: {metrics['fp']:,} (FA: {metrics['false_alarms_per_100k']}) | Min Cost: ₹{min_cost:,.0f} @ t={best_th:.2f}")

    benchmark_df = pd.DataFrame(results)
    benchmark_path = REPORTS_DIR / "heavy_training_benchmarks.csv"
    benchmark_df.to_csv(benchmark_path, index=False)
    print(f"\nSaved benchmark table: {benchmark_path}")

    # -----------------------------------------------------------------
    # STEP 6: Figures & Cost Curves
    # -----------------------------------------------------------------
    print("\n7. Generating Visual Comparison Figures...")

    # Figure 1: 4-Panel Model Comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

    # 1. PR-AUC with Error Bars
    ax = axes[0, 0]
    names = benchmark_df["model_name"]
    y_pr = benchmark_df["pr_auc"]
    err_low = y_pr - benchmark_df["pr_ci_low"]
    err_high = benchmark_df["pr_ci_high"] - y_pr
    ax.bar(names, y_pr, color="#00d4ff", edgecolor="#0f172a", width=0.55, yerr=[err_low, err_high], capsize=5)
    ax.set_ylim(0.70, 1.01)
    ax.set_title("A. Test PR-AUC (95% Bootstrap CI)", fontsize=11, fontweight="bold")
    ax.set_ylabel("PR-AUC (Natural Prevalence)", fontsize=10)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # 2. False Alarms per 100k
    ax = axes[0, 1]
    y_fa = benchmark_df["false_alarms_per_100k"]
    err_fa_l = y_fa - benchmark_df["fa_ci_low"]
    err_fa_h = benchmark_df["fa_ci_high"] - y_fa
    ax.bar(names, y_fa, color="#f59e0b", edgecolor="#0f172a", width=0.55, yerr=[err_fa_l, err_fa_h], capsize=5)
    ax.set_title("B. False Alarms per 100,000 Transactions (95% CI)", fontsize=11, fontweight="bold")
    ax.set_ylabel("False Alarms / 100k", fontsize=10)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # 3. Training Time vs Inference Latency
    ax = axes[1, 0]
    x_pos = np.arange(len(names))
    width = 0.35
    ax.bar(x_pos - width/2, benchmark_df["train_time_sec"], width, label="Training Time (s)", color="#3b82f6")
    ax2 = ax.twinx()
    ax2.bar(x_pos + width/2, benchmark_df["infer_latency_ms_10k"], width, label="Inference Latency (ms/10k tx)", color="#10b981")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax.set_title("C. Efficiency: Training Time vs Scoring Latency", fontsize=11, fontweight="bold")
    ax.set_ylabel("Training Time (s)", color="#3b82f6")
    ax2.set_ylabel("Inference Latency (ms/10k tx)", color="#10b981")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # 4. INR Minimum Expected Cost
    ax = axes[1, 1]
    y_cost = benchmark_df["min_expected_cost_inr"] / 1e6
    ax.bar(names, y_cost, color="#ef4444", edgecolor="#0f172a", width=0.55)
    ax.set_title("D. Minimum Expected Financial Cost (₹ Millions)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Expected Cost (₹ Millions)", fontsize=10)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for idx, v in enumerate(y_cost):
        ax.text(idx, v + 0.1, f"₹{v:.2f}M", ha="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    fig_comp_path = FIGURES_DIR / "model_comparison.png"
    plt.savefig(fig_comp_path, dpi=300)
    plt.close()
    print(f"Saved model comparison figure: {fig_comp_path}")

    # Figure 2: INR Cost Curves across Thresholds
    plt.figure(figsize=(10, 6), dpi=300)
    colors = ["#94a3b8", "#f59e0b", "#3b82f6", "#8b5cf6", "#06b6d4", "#10b981"]
    for (name, (costs, best_th, min_cost)), col in zip(cost_curves_data.items(), colors):
        plt.plot(threshold_grid, costs / 1e6, label=f"{name} (Min ₹{min_cost/1e6:.2f}M @ t={best_th:.2f})", color=col, linewidth=2.0)
    plt.title("Financial Cost Curves in INR across Risk Decision Thresholds", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Probability Decision Threshold t", fontsize=11)
    plt.ylabel("Total Expected Loss (₹ Millions)", fontsize=11)
    plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="upper right", framealpha=0.9, fontsize=9)
    plt.tight_layout()

    fig_cost_path = FIGURES_DIR / "cost_curves.png"
    plt.savefig(fig_cost_path, dpi=300)
    plt.close()
    print(f"Saved cost curves figure: {fig_cost_path}")

    # -----------------------------------------------------------------
    # STEP 7: Save to metrics.json
    # -----------------------------------------------------------------
    if METRICS_FILE.exists():
        with open(METRICS_FILE, "r", encoding="utf-8") as f:
            metrics_dict = json.load(f)
    else:
        metrics_dict = {}

    metrics_dict["heavy_training"] = {
        "best_hyperparameters": {
            "LightGBM": best_lgb_params,
            "XGBoost_GPU": best_xgb_params,
            "CatBoost_GPU": best_cat_params,
        },
        "benchmarks": benchmark_df.to_dict(orient="records"),
        "ensemble_vs_champion": {
            "champion_single": "Tuned LightGBM (CPU)",
            "champion_pr_auc": float(benchmark_df.loc[benchmark_df['model_name'] == 'Tuned LightGBM (CPU)', 'pr_auc'].iloc[0]),
            "champion_fp": int(benchmark_df.loc[benchmark_df['model_name'] == 'Tuned LightGBM (CPU)', 'fp'].iloc[0]),
            "ensemble_pr_auc": float(benchmark_df.loc[benchmark_df['model_name'] == 'Stacked Ensemble (Blender)', 'pr_auc'].iloc[0]),
            "ensemble_fp": int(benchmark_df.loc[benchmark_df['model_name'] == 'Stacked Ensemble (Blender)', 'fp'].iloc[0]),
        }
    }

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"Metrics saved to {METRICS_FILE}")


if __name__ == "__main__":
    main()

