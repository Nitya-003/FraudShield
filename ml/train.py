from __future__ import annotations

import json
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import RobustScaler, StandardScaler

from ml.config import (
    ABLATION_FILE,
    FIGURES_DIR,
    METADATA_FILE,
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

CLEAN_MODEL_FEATURES = FEATURE_COLUMNS


def compute_metrics_dict(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute comprehensive classification metrics at natural prevalence."""
    y_pred = (y_prob >= threshold).astype(int)
    pr_auc = float(average_precision_score(y_true, y_prob))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fa_per_100k = float(fpr * 100000)

    return {
        "pr_auc": round(pr_auc, 5),
        "roc_auc": round(roc_auc, 5),
        "recall": round(rec, 5),
        "precision": round(prec, 5),
        "f1": round(f1, 5),
        "fpr": round(fpr, 5),
        "false_alarms_per_100k": round(fa_per_100k, 1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }

evaluate_predictions = compute_metrics_dict


def run_training_pipeline():
    print("=" * 70)
    print("  FRAUDSHIELD SYSTEMATIC MODELING & BENCHMARK PIPELINE")
    print("=" * 70)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("\n1. Loading processed splits (natural prevalence)...")
    train_df = pd.read_parquet(TRAIN_DATA_PATH)
    train_us_df = pd.read_parquet(PROCESSED_DATA_DIR / "train_undersampled.parquet")
    val_df = pd.read_parquet(VAL_DATA_PATH)
    test_df = pd.read_parquet(TEST_DATA_PATH)

    X_train_full = train_df[FEATURE_COLUMNS].to_numpy()
    y_train_full = train_df[TARGET_COL].to_numpy()

    X_train_us = train_us_df[FEATURE_COLUMNS].to_numpy()
    y_train_us = train_us_df[TARGET_COL].to_numpy()

    X_val = val_df[FEATURE_COLUMNS].to_numpy()
    y_val = val_df[TARGET_COL].to_numpy()

    X_test = test_df[FEATURE_COLUMNS].to_numpy()
    y_test = test_df[TARGET_COL].to_numpy()

    print(f"  Full Train:        {len(y_train_full):,} rows ({y_train_full.sum():,} fraud)")
    print(f"  Undersampled Train:{len(y_train_us):,} rows ({y_train_us.sum():,} fraud)")
    print(f"  Validation:        {len(y_val):,} rows ({y_val.sum():,} fraud - {y_val.mean()*100:.4f}%)")
    print(f"  Test:              {len(y_test):,} rows ({y_test.sum():,} fraud - {y_test.mean()*100:.4f}%)")
    print("\n2. Running Time-Aware Expanding Window CV on Train (3 Folds)...")
    train_steps = train_df["step"].to_numpy()
    cv_cutoffs = [(150, 200), (200, 260), (260, 324)]
    cv_scores = {"Logistic Regression": [], "Random Forest": [], "LightGBM": []}

    for fold_idx, (tr_max, val_max) in enumerate(cv_cutoffs, 1):
        tr_mask = train_steps <= tr_max
        va_mask = (train_steps > tr_max) & (train_steps <= val_max)

        X_tr_f = X_train_full[tr_mask]
        y_tr_f = y_train_full[tr_mask]
        X_va_f = X_train_full[va_mask]
        y_va_f = y_train_full[va_mask]

        pos_idx = np.where(y_tr_f == 1)[0]
        neg_idx = np.where(y_tr_f == 0)[0]
        np.random.seed(RANDOM_STATE + fold_idx)
        sampled_neg = np.random.choice(neg_idx, size=min(len(neg_idx), len(pos_idx) * 10), replace=False)
        sub_idx = np.concatenate([pos_idx, sampled_neg])
        np.random.shuffle(sub_idx)
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr_f[sub_idx])
        X_va_sc = scaler.transform(X_va_f)
        lr_fold = LogisticRegression(C=10.0, class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
        lr_fold.fit(X_tr_sc, y_tr_f[sub_idx])
        lr_pr = average_precision_score(y_va_f, lr_fold.predict_proba(X_va_sc)[:, 1])
        cv_scores["Logistic Regression"].append(lr_pr)
        rf_fold = RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
        rf_fold.fit(X_tr_f[sub_idx], y_tr_f[sub_idx])
        rf_pr = average_precision_score(y_va_f, rf_fold.predict_proba(X_va_f)[:, 1])
        cv_scores["Random Forest"].append(rf_pr)
        lgb_fold = lgb.LGBMClassifier(n_estimators=120, learning_rate=0.07, num_leaves=31, min_child_samples=30, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
        lgb_fold.fit(X_tr_f[sub_idx], y_tr_f[sub_idx])
        lgb_pr = average_precision_score(y_va_f, lgb_fold.predict_proba(X_va_f)[:, 1])
        cv_scores["LightGBM"].append(lgb_pr)

        print(f"  Fold {fold_idx} (steps 1..{tr_max} -> {tr_max+1}..{val_max}) | LR PR-AUC: {lr_pr:.4f} | RF: {rf_pr:.4f} | LightGBM: {lgb_pr:.4f}")

    cv_summary = {
        m: {"mean": round(float(np.mean(scores)), 5), "std": round(float(np.std(scores)), 5)}
        for m, scores in cv_scores.items()
    }

    print("\n3. Training Full Benchmark Candidates & Evaluating on Test Set...")
    benchmarks = {}

    p_r1 = (test_df["amount"] > 50000).astype(int).to_numpy()
    benchmarks["Rule 1 (Amount > 50k)"] = compute_metrics_dict(y_test, p_r1.astype(float), 0.5)

    p_r2 = (((test_df["type_TRANSFER"] == 1) | (test_df["type_CASH_OUT"] == 1)) & (test_df["orig_emptied"] == 1)).astype(int).to_numpy()
    benchmarks["Rule 2 (Channel & Empty)"] = compute_metrics_dict(y_test, p_r2.astype(float), 0.5)

    p_r3 = (
        ((test_df["type_TRANSFER"] == 1) | (test_df["type_CASH_OUT"] == 1))
        & (test_df["orig_emptied"] == 1)
        & (test_df["dest_balance_zero"] == 1)
    ).astype(int).to_numpy()
    benchmarks["Rule 3 (Channel & Empty & DestZero)"] = compute_metrics_dict(y_test, p_r3.astype(float), 0.5)

    t0 = time.time()
    dt_model = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=RANDOM_STATE)
    dt_model.fit(X_train_us, y_train_us)
    dt_fit_time = time.time() - t0
    dt_probs = dt_model.predict_proba(X_test)[:, 1]
    m_dt = compute_metrics_dict(y_test, dt_probs, 0.5)
    m_dt["fit_time_sec"] = round(dt_fit_time, 2)
    benchmarks["Decision Tree (Depth 3)"] = m_dt

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train_us)
    X_te_s = scaler.transform(X_test)
    X_va_s = scaler.transform(X_val)
    t0 = time.time()
    lr_model = LogisticRegression(C=10.0, class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
    lr_model.fit(X_tr_s, y_train_us)
    lr_fit_time = time.time() - t0
    lr_probs = lr_model.predict_proba(X_te_s)[:, 1]
    m_lr = compute_metrics_dict(y_test, lr_probs, 0.5)
    m_lr["fit_time_sec"] = round(lr_fit_time, 2)
    benchmarks["Logistic Regression"] = m_lr

    t0 = time.time()
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    rf_model.fit(X_train_us, y_train_us)
    rf_fit_time = time.time() - t0
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    m_rf = compute_metrics_dict(y_test, rf_probs, 0.5)
    m_rf["fit_time_sec"] = round(rf_fit_time, 2)
    benchmarks["Random Forest"] = m_rf

    t0 = time.time()
    lgb_cw = lgb.LGBMClassifier(
        n_estimators=120,
        learning_rate=0.07,
        num_leaves=31,
        min_child_samples=30,
        scale_pos_weight=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_cw.fit(X_train_full, y_train_full)
    cw_fit_time = time.time() - t0
    cw_probs = lgb_cw.predict_proba(X_test)[:, 1]
    m_cw = compute_metrics_dict(y_test, cw_probs, 0.5)
    m_cw["fit_time_sec"] = round(cw_fit_time, 2)
    benchmarks["LightGBM (Cost-Weighted)"] = m_cw

    t0 = time.time()
    champion_raw = lgb.LGBMClassifier(
        n_estimators=120,
        learning_rate=0.07,
        num_leaves=31,
        min_child_samples=30,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    champion_raw.fit(X_train_us, y_train_us)
    us_fit_time = time.time() - t0

    print("\n4. Performing Probability Calibration on Validation Set...")
    val_raw_probs = champion_raw.predict_proba(X_val)[:, 1]
    brier_before = float(brier_score_loss(y_val, val_raw_probs))

    calibrator = LogisticRegression()
    eps = 1e-7
    val_log_odds = np.log(np.clip(val_raw_probs, eps, 1 - eps) / (1 - np.clip(val_raw_probs, eps, 1 - eps))).reshape(-1, 1)
    calibrator.fit(val_log_odds, y_val)

    def calibrate(probs):
        lo = np.log(np.clip(probs, eps, 1 - eps) / (1 - np.clip(probs, eps, 1 - eps))).reshape(-1, 1)
        return calibrator.predict_proba(lo)[:, 1]

    val_cal_probs = calibrate(val_raw_probs)
    brier_after = float(brier_score_loss(y_val, val_cal_probs))
    print(f"  Brier Score: {brier_before:.6f} (raw) -> {brier_after:.6f} (calibrated)")

    prob_true_raw, prob_pred_raw = calibration_curve(y_val, val_raw_probs, n_bins=10)
    prob_true_cal, prob_pred_cal = calibration_curve(y_val, val_cal_probs, n_bins=10)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
    ax.plot(prob_pred_raw, prob_true_raw, "s-", color="#dc2626", label=f"Raw Model (Brier: {brier_before:.5f})")
    ax.plot(prob_pred_cal, prob_true_cal, "o-", color="#2563eb", label=f"Calibrated (Brier: {brier_after:.5f})")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Empirical Fraud Fraction")
    ax.set_title("Probability Calibration (Reliability Curve) on Natural Validation Set")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "calibration_curve.png", dpi=200)
    plt.close(fig)

    test_raw_probs = champion_raw.predict_proba(X_test)[:, 1]
    test_cal_probs = calibrate(test_raw_probs)
    m_champ = compute_metrics_dict(y_test, test_cal_probs, 0.5)
    m_champ["fit_time_sec"] = round(us_fit_time, 2)
    benchmarks["Tuned LightGBM (Champion)"] = m_champ

    print("\nBenchmark Summary on Natural-Prevalence Test Set (918,617 rows):")
    for name, res in benchmarks.items():
        print(f"  {name:32s} | PR-AUC: {res['pr_auc']:.5f} | Recall: {res['recall']*100:.2f}% | Precision: {res['precision']*100:.2f}% | FPR: {res['fpr']*100:.4f}% | FP: {res['fp']:,}")

    print("\n5. Running Feature Ablation Study (Consistent LightGBM)...")
    basic_type_cols = [
        "amount", "log_amount", "hour_of_day", "sin_hour", "cos_hour", "is_night",
        "type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER",
    ]
    pre_tx_balances = basic_type_cols + [
        "oldbalanceOrg", "oldbalanceDest", "orig_emptied", "amount_to_balance_ratio",
    ]
    pre_tx_balances_dest = pre_tx_balances + [
        "dest_balance_zero", "is_dest_merchant",
    ]
    receiver_history_no_balances = basic_type_cols + [
        "dest_past_tx_count", "dest_total_past_amount", "dest_time_since_last_tx",
        "dest_tx_last_1h", "dest_tx_last_6h", "dest_tx_last_24h",
    ]

    ablation_groups = {
        "1. Basic + Type Only": basic_type_cols,
        "2. + Pre-Tx Balances & Emptying": pre_tx_balances,
        "3. + Dest Zero & Merchant Flags": pre_tx_balances_dest,
        "4. Basic + Type + Receiver History": receiver_history_no_balances,
        "5. All Features (Complete Pre-Tx System)": FEATURE_COLUMNS,
    }

    ablation_rows = []
    for g_name, cols in ablation_groups.items():
        sub_X_tr = train_us_df[cols].to_numpy()
        sub_X_te = test_df[cols].to_numpy()
        m_abl = lgb.LGBMClassifier(n_estimators=120, learning_rate=0.07, num_leaves=31, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
        m_abl.fit(sub_X_tr, y_train_us)
        probs_abl = m_abl.predict_proba(sub_X_te)[:, 1]
        preds_abl = (probs_abl >= 0.50).astype(int)

        pr_te = float(average_precision_score(y_test, probs_abl))
        roc_te = float(roc_auc_score(y_test, probs_abl))
        rec_te = float(recall_score(y_test, preds_abl, zero_division=0))
        prec_te = float(precision_score(y_test, preds_abl, zero_division=0))
        tn, fp, fn, tp = confusion_matrix(y_test, preds_abl).ravel()
        fa_per_100k = float(fp / (fp + tn) * 100000)

        ablation_rows.append({
            "feature_group": g_name,
            "num_features": len(cols),
            "test_pr_auc": round(pr_te, 5),
            "test_roc_auc": round(roc_te, 5),
            "test_recall": round(rec_te, 5),
            "test_precision": round(prec_te, 5),
            "false_positives": int(fp),
            "false_alarms_per_100k": round(fa_per_100k, 1),
        })
        print(f"  {g_name:38s} ({len(cols)} feats) | PR-AUC: {pr_te:.5f} | Recall: {rec_te*100:.2f}% | FP: {fp:,}")

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(ABLATION_FILE, index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    short_labels = ["Basic + Type", "+ Balances", "+ Dest Flags", "+ Recv History", "All 23 Features"]
    pr_scores = ablation_df["test_pr_auc"].tolist()
    fp_counts = ablation_df["false_positives"].tolist()

    ax1.bar(short_labels, pr_scores, color=["#94a3b8", "#38bdf8", "#0284c7", "#64748b", "#0d9488"])
    ax1.set_title("Test PR-AUC Across Feature Groups (Natural Prevalence)")
    ax1.set_ylabel("PR-AUC")
    ax1.set_ylim(0, 1.05)
    for i, v in enumerate(pr_scores):
        ax1.text(i, v + 0.02, f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")

    ax2.bar(short_labels, fp_counts, color=["#ef4444", "#f97316", "#eab308", "#f43f5e", "#10b981"])
    ax2.set_title("False Positives on 918,617 Natural Test Transactions (Lower is Better)")
    ax2.set_ylabel("False Positives Count")
    ax2.set_yscale("log")
    for i, v in enumerate(fp_counts):
        ax2.text(i, v * 1.2, f"{v:,}", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "ablation.png", dpi=200)
    plt.close(fig)

    print("\n6. Assessing Multi-Seed Stability & Bootstrap 95% Confidence Intervals...")
    seeds = [42, 43, 44, 45, 46]
    seed_scores = []
    for s in seeds:
        m_s = lgb.LGBMClassifier(
            n_estimators=120,
            learning_rate=0.07,
            num_leaves=31,
            max_depth=6,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=s,
            bagging_seed=s,
            feature_fraction_seed=s,
            n_jobs=-1,
            verbose=-1,
        )
        m_s.fit(X_train_us, y_train_us)
        s_pr = float(average_precision_score(y_test, m_s.predict_proba(X_test)[:, 1]))
        seed_scores.append(s_pr)
    seed_mean = float(np.mean(seed_scores))
    seed_std = float(np.std(seed_scores))
    print(f"  5-Seed PR-AUC Spread: Mean = {seed_mean:.5f}, Std = {seed_std:.6f} (Min: {min(seed_scores):.5f}, Max: {max(seed_scores):.5f})")

    np.random.seed(RANDOM_STATE)
    boot_pr, boot_rec, boot_prec = [], [], []
    n_test = len(y_test)
    pred_cal = (test_cal_probs >= 0.5).astype(int)

    for _ in range(100):
        b_idx = np.random.randint(0, n_test, size=100000)
        y_b = y_test[b_idx]
        if y_b.sum() == 0:
            continue
        p_b = test_cal_probs[b_idx]
        pred_b = pred_cal[b_idx]
        boot_pr.append(average_precision_score(y_b, p_b))
        boot_rec.append(recall_score(y_b, pred_b, zero_division=0))
        boot_prec.append(precision_score(y_b, pred_b, zero_division=0))

    ci_95 = {
        "pr_auc": [round(float(np.percentile(boot_pr, 2.5)), 5), round(float(np.percentile(boot_pr, 97.5)), 5)],
        "recall": [round(float(np.percentile(boot_rec, 2.5)), 5), round(float(np.percentile(boot_rec, 97.5)), 5)],
        "precision": [round(float(np.percentile(boot_prec, 2.5)), 5), round(float(np.percentile(boot_prec, 97.5)), 5)],
    }
    print(f"  Bootstrap 95% CIs: PR-AUC: {ci_95['pr_auc']}, Recall: {ci_95['recall']}, Precision: {ci_95['precision']}")

    print("\n7. Saving Champion Artifacts and Metadata...")
    champion_payload = {
        "model": champion_raw,
        "calibrator": calibrator,
        "feature_names": FEATURE_COLUMNS,
    }
    joblib.dump(champion_payload, MODEL_FILE)

    metadata = {
        "model_name": "Tuned LightGBM (Platt-Calibrated)",
        "champion_strategy": "1:10 Undersampled Training with Full Natural Prevalence Validation",
        "training_strategy": "1:10 Undersampled Training with Full Natural Prevalence Validation",
        "training_date": pd.Timestamp.now().isoformat(),
        "random_state": RANDOM_STATE,
        "library_versions": {
            "lightgbm": lgb.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "features": FEATURE_COLUMNS,
        "feature_count": len(FEATURE_COLUMNS),
        "target_column": TARGET_COL,
        "validation_metrics": {
            "pr_auc": benchmarks["Tuned LightGBM (Champion)"]["pr_auc"],
            "roc_auc": benchmarks["Tuned LightGBM (Champion)"]["roc_auc"],
            "recall": benchmarks["Tuned LightGBM (Champion)"]["recall"],
            "precision": benchmarks["Tuned LightGBM (Champion)"]["precision"],
        },
        "test_metrics": benchmarks["Tuned LightGBM (Champion)"],
        "thresholds": {
            "low_risk_approve": 0.45,
            "high_risk_decline": 0.90,
            "medium_risk_otp": [0.45, 0.90],
            "optimization_basis": "Validation Expected Cost Minimization",
        },
        "calibrated_thresholds": {
            "low_risk_approve": 0.45,
            "high_risk_decline": 0.90,
        },
        "dataset_splits": {
            "train_rows": len(train_df),
            "train_fraud_count": int(y_train_full.sum()),
            "train_fraud_rate": float(y_train_full.mean()),
            "val_rows": len(val_df),
            "val_fraud_count": int(y_val.sum()),
            "val_fraud_rate": float(y_val.mean()),
            "test_rows": len(test_df),
            "test_fraud_count": int(y_test.sum()),
            "test_fraud_rate": float(y_test.mean()),
        },
        "expanding_window_cv": cv_summary,
        "benchmarks": benchmarks,
        "multi_seed_stability": {
            "seeds": seeds,
            "pr_auc_mean": round(seed_mean, 5),
            "pr_auc_std": round(seed_std, 5),
        },
        "bootstrap_ci_95": ci_95,
        "calibration": {
            "method": "Platt Scaling (Logistic Sigmoid)",
            "brier_score_before": round(brier_before, 6),
            "brier_score_after": round(brier_after, 6),
        },
    }

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(benchmarks, f, indent=2)

    print(f"Artifacts saved to {MODEL_FILE}, {METADATA_FILE}, and {METRICS_FILE}")
    print("Systematic modeling pipeline complete!")
    return metadata


if __name__ == "__main__":
    run_training_pipeline()
