"""
Empirical Feature Ablation and Feature Importance Benchmark:
Baseline Features (23) vs Network Mule & Ring Features (31).
Evaluated at natural test prevalence.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml.config import (
    FIGURES_DIR,
    METRICS_FILE,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET_COL,
    TEST_DATA_PATH,
    VAL_DATA_PATH,
)
from ml.features import ALL_FEATURE_COLUMNS, FEATURE_COLUMNS
from ml.network import NETWORK_FEATURE_COLUMNS


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
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


def main():
    print("=" * 70)
    print("  FRAUDSHIELD: NETWORK (MULE & RING) ABLATION & IMPORTANCE BENCHMARK")
    print("=" * 70)

    print("\n1. Loading datasets...")
    train_us_df = pd.read_parquet(PROCESSED_DATA_DIR / "train_undersampled.parquet")
    val_df = pd.read_parquet(VAL_DATA_PATH)
    test_df = pd.read_parquet(TEST_DATA_PATH)

    y_train = train_us_df[TARGET_COL].to_numpy()
    y_val = val_df[TARGET_COL].to_numpy()
    y_test = test_df[TARGET_COL].to_numpy()

    lgb_params = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbose": -1,
    }

    # Model A: Baseline 23 Features
    print("\n2. Training Model A: Baseline (23 Features)...")
    X_train_base = train_us_df[FEATURE_COLUMNS].to_numpy()
    X_val_base = val_df[FEATURE_COLUMNS].to_numpy()
    X_test_base = test_df[FEATURE_COLUMNS].to_numpy()

    model_base = lgb.LGBMClassifier(**lgb_params)
    model_base.fit(X_train_base, y_train)

    val_preds_base = model_base.predict_proba(X_val_base)[:, 1]
    test_preds_base = model_base.predict_proba(X_test_base)[:, 1]

    val_metrics_base = compute_metrics(y_val, val_preds_base)
    test_metrics_base = compute_metrics(y_test, test_preds_base)
    print(f"  Model A (Baseline 23) -> Val PR-AUC: {val_metrics_base['pr_auc']:.5f} | Test PR-AUC: {test_metrics_base['pr_auc']:.5f} | Test FP: {test_metrics_base['fp']:,}")

    # Model B: Extended 31 Features (+ 8 Network Features)
    print("\n3. Training Model B: Extended (+ 8 Network Features = 31 Features)...")
    X_train_net = train_us_df[ALL_FEATURE_COLUMNS].to_numpy()
    X_val_net = val_df[ALL_FEATURE_COLUMNS].to_numpy()
    X_test_net = test_df[ALL_FEATURE_COLUMNS].to_numpy()

    model_net = lgb.LGBMClassifier(**lgb_params)
    model_net.fit(X_train_net, y_train)

    val_preds_net = model_net.predict_proba(X_val_net)[:, 1]
    test_preds_net = model_net.predict_proba(X_test_net)[:, 1]

    val_metrics_net = compute_metrics(y_val, val_preds_net)
    test_metrics_net = compute_metrics(y_test, test_preds_net)
    print(f"  Model B (+ Network 31)-> Val PR-AUC: {val_metrics_net['pr_auc']:.5f} | Test PR-AUC: {test_metrics_net['pr_auc']:.5f} | Test FP: {test_metrics_net['fp']:,}")

    # Feature Importance Extraction
    importances_gain = model_net.booster_.feature_importance(importance_type="gain")
    importances_split = model_net.booster_.feature_importance(importance_type="split")

    feat_imp_df = pd.DataFrame({
        "feature": ALL_FEATURE_COLUMNS,
        "gain": importances_gain,
        "split": importances_split,
        "is_network": [f in NETWORK_FEATURE_COLUMNS for f in ALL_FEATURE_COLUMNS],
    }).sort_values("gain", ascending=False).reset_index(drop=True)

    print("\nTop 10 Features by Gain:")
    for idx, row in feat_imp_df.head(10).iterrows():
        tag = "[NETWORK]" if row["is_network"] else "[BASELINE]"
        print(f"  {idx+1:2d}. {row['feature']:<30} {tag} Gain: {row['gain']:,.1f}")

    print("\nNetwork Features Performance & Gain:")
    for idx, row in feat_imp_df[feat_imp_df["is_network"]].iterrows():
        print(f"  Rank #{idx+1:2d}: {row['feature']:<30} Gain: {row['gain']:,.1f} | Splits: {row['split']}")

    # Save Ablation CSV
    ablation_df = pd.DataFrame([
        {
            "configuration": "Baseline (23 Features)",
            "features_count": 23,
            "val_pr_auc": val_metrics_base["pr_auc"],
            "val_recall": val_metrics_base["recall"],
            "val_precision": val_metrics_base["precision"],
            "test_pr_auc": test_metrics_base["pr_auc"],
            "test_recall": test_metrics_base["recall"],
            "test_precision": test_metrics_base["precision"],
            "test_fp": test_metrics_base["fp"],
            "test_false_alarms_per_100k": test_metrics_base["false_alarms_per_100k"],
        },
        {
            "configuration": "Extended (+ 8 Network Graph Features)",
            "features_count": 31,
            "val_pr_auc": val_metrics_net["pr_auc"],
            "val_recall": val_metrics_net["recall"],
            "val_precision": val_metrics_net["precision"],
            "test_pr_auc": test_metrics_net["pr_auc"],
            "test_recall": test_metrics_net["recall"],
            "test_precision": test_metrics_net["precision"],
            "test_fp": test_metrics_net["fp"],
            "test_false_alarms_per_100k": test_metrics_net["false_alarms_per_100k"],
        },
    ])
    ablation_path = REPORTS_DIR / "network_ablation.csv"
    ablation_df.to_csv(ablation_path, index=False)
    print(f"\nAblation table saved: {ablation_path}")

    # Plot 1: Feature Importance Chart
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    top_n = feat_imp_df.head(15).iloc[::-1]  # Top 15 ascending for horizontal bar chart
    colors = ["#00d4ff" if is_net else "#3b82f6" for is_net in top_n["is_network"]]

    bars = ax.barh(top_n["feature"], top_n["gain"], color=colors, edgecolor="#1e293b", height=0.65)
    ax.set_title("LightGBM Feature Importance (Gain): Baseline vs Network Graph Features", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Total Information Gain", fontsize=11, fontweight="semibold")
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    # Custom legend
    import matplotlib.patches as mpatches
    baseline_patch = mpatches.Patch(color="#3b82f6", label="Baseline Features (23)")
    network_patch = mpatches.Patch(color="#00d4ff", label="Network Graph Features (8)")
    ax.legend(handles=[baseline_patch, network_patch], loc="lower right", framealpha=0.9)
    plt.tight_layout()

    fig_imp_path = FIGURES_DIR / "network_feature_importance.png"
    plt.savefig(fig_imp_path, dpi=300)
    plt.close()
    print(f"Feature importance figure saved: {fig_imp_path}")

    # Plot 2: Before vs After Ablation Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)

    # PR-AUC comparison
    models = ["Baseline\n(23 Feats)", "+ Network\n(31 Feats)"]
    pr_aucs = [test_metrics_base["pr_auc"], test_metrics_net["pr_auc"]]
    ax1.bar(models, pr_aucs, color=["#64748b", "#00d4ff"], width=0.45, edgecolor="#1e293b")
    ax1.set_ylim(0.99, 1.0005)
    ax1.set_ylabel("Natural Test PR-AUC", fontsize=10, fontweight="bold")
    ax1.set_title("Test PR-AUC Comparison", fontsize=11, fontweight="bold")
    for idx, v in enumerate(pr_aucs):
        ax1.text(idx, v + 0.0003, f"{v:.5f}", ha="center", fontsize=10, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    # False Positives comparison
    fps = [test_metrics_base["fp"], test_metrics_net["fp"]]
    ax2.bar(models, fps, color=["#f59e0b", "#10b981"], width=0.45, edgecolor="#1e293b")
    ax2.set_ylabel("False Positives (918k Test Stream)", fontsize=10, fontweight="bold")
    ax2.set_title("Operational False Positives", fontsize=11, fontweight="bold")
    for idx, v in enumerate(fps):
        ax2.text(idx, v + 0.5, f"{v}", ha="center", fontsize=10, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    fig_ablation_path = FIGURES_DIR / "network_ablation.png"
    plt.savefig(fig_ablation_path, dpi=300)
    plt.close()
    print(f"Ablation figure saved: {fig_ablation_path}")

    # Append to reports/metrics.json
    if METRICS_FILE.exists():
        with open(METRICS_FILE, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
    else:
        metrics_data = {}

    metrics_data["network_ablation"] = {
        "baseline_23": test_metrics_base,
        "extended_network_31": test_metrics_net,
        "top_features_by_gain": feat_imp_df.head(10)[["feature", "gain", "is_network"]].to_dict(orient="records"),
        "network_features_breakdown": feat_imp_df[feat_imp_df["is_network"]][["feature", "gain", "split"]].to_dict(orient="records"),
    }

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Updated metrics saved to: {METRICS_FILE}")


if __name__ == "__main__":
    main()

