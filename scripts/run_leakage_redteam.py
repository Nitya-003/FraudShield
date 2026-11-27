"""
Leakage Red-Team and Forensic Audit Script (Phase 1).
Executes:
1. Feature-group ablation with identical hyperparameters on natural test set (saves reports/ablation.csv and figure).
2. Leave-one-feature-out analysis on top 6 features.
3. Shuffled-label sanity check (verifying PR-AUC collapses to natural base rate ~0.0044).
4. Simplicity check: depth-2 and depth-3 Decision Tree baselines to prove the scripted nature of PaySim fraud.
5. True multi-seed variance check with bagging and feature fraction enabled.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
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
from sklearn.tree import DecisionTreeClassifier, export_text

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


def run_redteam():
    print("=" * 70)
    print("  FRAUDSHIELD PHASE 1: LEAKAGE RED-TEAM & SCRIPTED PATTERN AUDIT")
    print("=" * 70)

    # 1. Load Data
    print("\n1. Loading natural prevalence parquet datasets...")
    train_us = pd.read_parquet(PROCESSED_DATA_DIR / "train_undersampled.parquet")
    val_df = pd.read_parquet(VAL_DATA_PATH)
    test_df = pd.read_parquet(TEST_DATA_PATH)

    X_train_us = train_us[FEATURE_COLUMNS]
    y_train_us = train_us[TARGET_COL].to_numpy()

    X_val = val_df[FEATURE_COLUMNS]
    y_val = val_df[TARGET_COL].to_numpy()

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COL].to_numpy()

    natural_prevalence = float(y_test.mean())
    print(f"  Test rows: {len(y_test):,} | Test frauds: {int(y_test.sum()):,} | Prevalence: {natural_prevalence*100:.4f}%")

    # Common LightGBM params for fair comparison
    lgb_params = {
        "n_estimators": 120,
        "learning_rate": 0.07,
        "num_leaves": 31,
        "max_depth": 6,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbose": -1,
    }

    # -------------------------------------------------------------
    # 2. Subtask 1b: Feature-Group Ablation Study
    # -------------------------------------------------------------
    print("\n2. Executing Strict Feature-Group Ablation Study on Natural Test Set...")

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
    all_features = FEATURE_COLUMNS

    ablation_groups = [
        ("1. Basic + Type Only", basic_type_cols),
        ("2. + Pre-Tx Balances & Emptying", pre_tx_balances),
        ("3. + Dest Zero & Merchant Flags", pre_tx_balances_dest),
        ("4. Basic + Type + Receiver History", receiver_history_no_balances),
        ("5. All Features (Complete Pre-Tx System)", all_features),
    ]

    ablation_results = []
    for grp_name, cols in ablation_groups:
        clf = lgb.LGBMClassifier(**lgb_params)
        clf.fit(train_us[cols], y_train_us)
        probs_test = clf.predict_proba(test_df[cols])[:, 1]
        preds_test = (probs_test >= 0.50).astype(int)

        pr_auc = float(average_precision_score(y_test, probs_test))
        rec = float(recall_score(y_test, preds_test, zero_division=0))
        prec = float(precision_score(y_test, preds_test, zero_division=0))
        tn, fp, fn, tp = confusion_matrix(y_test, preds_test).ravel()
        fa_per_100k = float(fp / (fp + tn) * 100000)

        ablation_results.append({
            "Feature Group": grp_name,
            "Feature Count": len(cols),
            "PR-AUC": round(pr_auc, 5),
            "Recall (%)": round(rec * 100, 2),
            "Precision (%)": round(prec * 100, 2),
            "False Positives": int(fp),
            "False Alarms / 100k": round(fa_per_100k, 1),
        })
        print(f"  {grp_name:38s} | Feats: {len(cols):2d} | PR-AUC: {pr_auc:.5f} | Recall: {rec*100:5.2f}% | FP: {fp:5d}")

    ablation_df = pd.DataFrame(ablation_results)
    ablation_df.to_csv(ABLATION_FILE, index=False)
    print(f"  Saved ablation results to {ABLATION_FILE}")

    # Plot ablation figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    short_labels = ["Basic + Type", "+ Balances", "+ Dest Flags", "+ Recv History", "All 23 Features"]
    pr_scores = [r["PR-AUC"] for r in ablation_results]
    fp_counts = [r["False Positives"] for r in ablation_results]

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
    fig_ablation_path = FIGURES_DIR / "ablation.png"
    plt.savefig(fig_ablation_path, dpi=200)
    plt.close()
    print(f"  Saved ablation figure to {fig_ablation_path}")

    # -------------------------------------------------------------
    # 3. Subtask 1c: Leave-One-Feature-Out & Shuffled Label Sanity Check
    # -------------------------------------------------------------
    print("\n3. Executing Leave-One-Feature-Out on Top 6 Features...")
    # Train full champion to get top feature importances
    full_clf = lgb.LGBMClassifier(**lgb_params)
    full_clf.fit(X_train_us, y_train_us)
    importances = pd.Series(full_clf.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
    top6_features = importances.head(6).index.tolist()
    print(f"  Top 6 Features by Split Importance: {top6_features}")

    lofo_results = []
    base_probs = full_clf.predict_proba(X_test)[:, 1]
    base_pr_auc = float(average_precision_score(y_test, base_probs))
    lofo_results.append({
        "Omitted Feature": "None (Full 23 Features)",
        "PR-AUC": round(base_pr_auc, 5),
        "PR-AUC Delta": 0.0,
    })

    for feat in top6_features:
        rem_cols = [c for c in FEATURE_COLUMNS if c != feat]
        clf_lofo = lgb.LGBMClassifier(**lgb_params)
        clf_lofo.fit(train_us[rem_cols], y_train_us)
        probs_lofo = clf_lofo.predict_proba(test_df[rem_cols])[:, 1]
        pr_lofo = float(average_precision_score(y_test, probs_lofo))
        delta = round(pr_lofo - base_pr_auc, 5)
        lofo_results.append({
            "Omitted Feature": f"Excluding '{feat}'",
            "PR-AUC": round(pr_lofo, 5),
            "PR-AUC Delta": delta,
        })
        print(f"  Excluding {feat:25s} | PR-AUC: {pr_lofo:.5f} | Delta: {delta:+.5f}")

    # Shuffled Label Test
    print("\n  Running Shuffled-Label Sanity Check (Permuted y_train)...")
    np.random.seed(RANDOM_STATE)
    y_shuffled = np.random.permutation(y_train_us)
    clf_shuff = lgb.LGBMClassifier(**lgb_params)
    clf_shuff.fit(X_train_us, y_shuffled)
    probs_shuff = clf_shuff.predict_proba(X_test)[:, 1]
    pr_shuff = float(average_precision_score(y_test, probs_shuff))
    print(f"  Shuffled-Label Test PR-AUC: {pr_shuff:.5f} (Natural Base Rate: {natural_prevalence:.5f})")
    assert pr_shuff < 0.02, f"Target leakage alert: shuffled label model scored {pr_shuff:.5f}!"
    print("  Sanity Check Passed: Shuffled-label model collapses to baseline random guessing.")

    # -------------------------------------------------------------
    # 4. Subtask 1d: Simplicity Check (Depth-2 and Depth-3 Decision Trees with balanced class weights)
    # -------------------------------------------------------------
    print("\n4. Executing Simplicity Check (Shallow Decision Trees)...")
    dt2 = DecisionTreeClassifier(max_depth=2, class_weight="balanced", random_state=RANDOM_STATE)
    dt2.fit(X_train_us, y_train_us)
    dt2_probs = dt2.predict_proba(X_test)[:, 1]
    dt2_preds = (dt2_probs >= 0.50).astype(int)
    dt2_pr = float(average_precision_score(y_test, dt2_probs))
    dt2_rec = float(recall_score(y_test, dt2_preds, zero_division=0))
    dt2_prec = float(precision_score(y_test, dt2_preds, zero_division=0))
    _, dt2_fp, _, _ = confusion_matrix(y_test, dt2_preds).ravel()

    dt3 = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=RANDOM_STATE)
    dt3.fit(X_train_us, y_train_us)
    dt3_probs = dt3.predict_proba(X_test)[:, 1]
    dt3_preds = (dt3_probs >= 0.50).astype(int)
    dt3_pr = float(average_precision_score(y_test, dt3_probs))
    dt3_rec = float(recall_score(y_test, dt3_preds, zero_division=0))
    dt3_prec = float(precision_score(y_test, dt3_preds, zero_division=0))
    _, dt3_fp, _, _ = confusion_matrix(y_test, dt3_preds).ravel()

    print(f"  Decision Tree (Depth 2): PR-AUC: {dt2_pr:.5f} | Recall: {dt2_rec*100:.2f}% | Precision: {dt2_prec*100:.2f}% | FP: {dt2_fp:,}")
    print(f"  Decision Tree (Depth 3): PR-AUC: {dt3_pr:.5f} | Recall: {dt3_rec*100:.2f}% | Precision: {dt3_prec*100:.2f}% | FP: {dt3_fp:,}")

    print("\n  Depth-3 Decision Tree Rules Text:")
    tree_rules = export_text(dt3, feature_names=FEATURE_COLUMNS)
    print(tree_rules)

    # -------------------------------------------------------------
    # 5. Subtask 1e: True Multi-Seed Variance Check (Bagging & Subsampling Active)
    # -------------------------------------------------------------
    print("\n5. Testing True Multi-Seed Variance (Bagging & Feature Fraction = 0.8)...")
    seeds = [42, 43, 44, 45, 46]
    seed_scores = []

    for s in seeds:
        clf_s = lgb.LGBMClassifier(
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
        clf_s.fit(X_train_us, y_train_us)
        s_prob = clf_s.predict_proba(X_test)[:, 1]
        s_pr = float(average_precision_score(y_test, s_prob))
        seed_scores.append(s_pr)
        print(f"  Seed {s:2d} | Test PR-AUC: {s_pr:.5f}")

    seed_mean = float(np.mean(seed_scores))
    seed_std = float(np.std(seed_scores))
    print(f"  Multi-Seed Spread (5 seeds with bagging): Mean = {seed_mean:.5f}, Std = {seed_std:.6f}")

    # Compile Redteam Findings Payload
    findings = {
        "ablation_results": ablation_results,
        "lofo_results": lofo_results,
        "shuffled_label_pr_auc": round(pr_shuff, 5),
        "natural_prevalence": round(natural_prevalence, 5),
        "tree_simplicity": {
            "depth_2": {"pr_auc": round(dt2_pr, 5), "recall": round(dt2_rec, 4), "precision": round(dt2_prec, 4), "fp": int(dt2_fp)},
            "depth_3": {"pr_auc": round(dt3_pr, 5), "recall": round(dt3_rec, 4), "precision": round(dt3_prec, 4), "fp": int(dt3_fp)},
        },
        "multi_seed_with_subsampling": {
            "seeds": seeds,
            "scores": [round(x, 5) for x in seed_scores],
            "mean": round(seed_mean, 5),
            "std": round(seed_std, 6),
        },
    }

    with open(REPORTS_DIR / "redteam_findings.json", "w", encoding="utf-8") as f:
        json.dump(findings, f, indent=2)

    print("\nPhase 1 Leakage Red-Team fully completed!")
    return findings


if __name__ == "__main__":
    run_redteam()

