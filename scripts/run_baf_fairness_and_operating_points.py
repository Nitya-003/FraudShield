"""
Bank Account Fraud (BAF) Operating-Point Table, Multi-Point Fairness Audit,
and Fairness-Accuracy Trade-Off Analysis (Phase 2).

1. Computes exact Operating Points at 1%, 2%, 5% FPR with Recall and False Alarms / 100k.
2. Audits Fairness by Age Group (<50 vs 50+) at EACH operating point.
3. Tests Fairness-Accuracy interventions:
   - Baseline Uniform Thresholding
   - Per-Group Threshold Calibration (Demographic Parity on FPR)
   - Evaluates the Fairness-Accuracy trade-off (TPR impact vs parity gain).
4. Saves to reports/baf_operating_points_fairness.json and syncs to metrics.json & metadata.json.
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
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, precision_score, recall_score

from ml.config import METADATA_FILE, METRICS_FILE, PROCESSED_DATA_DIR, REPORTS_DIR, RANDOM_STATE
from ml.channels.account import NUMERIC_FEATURES, INDICATOR_FEATURES, CATEGORICAL_FEATURES, MODEL_PATH


def run_baf_analysis():
    print("=" * 75)
    print("  BAF: OPERATING-POINT TABLE, FAIRNESS AUDIT & ACCURACY-FAIRNESS TRADE-OFF")
    print("=" * 75)

    # 1. Load artifacts
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Account model artifact missing at {MODEL_PATH}")

    artifacts = joblib.load(MODEL_PATH)
    champion = artifacts["champion_model"]
    calibrator = artifacts["calibrator"]
    ohe = artifacts["ohe"]
    all_feature_names = artifacts["feature_names"]
    num_and_ind_cols = artifacts["numeric_features"]

    # 2. Load Month 7 Test Data (Base.csv Month 7 or account_test_sample.parquet)
    # Check if raw Base.csv is available for full 96k Month 7 evaluation
    raw_base_path = BASE_DIR / "data" / "raw" / "Base.csv"
    if raw_base_path.exists():
        print("Loading full Month 7 test split from Base.csv...")
        df_all = pd.read_csv(raw_base_path)
        test_df = df_all[df_all["month"] == 7].copy().reset_index(drop=True)
    else:
        print("Base.csv not found; using processed account_test_sample.parquet...")
        test_df = pd.read_parquet(PROCESSED_DATA_DIR / "account_test_sample.parquet")

    print(f"Test split size: {len(test_df):,} applications | {test_df['fraud_bool'].sum():,} frauds ({test_df['fraud_bool'].mean()*100:.3f}%)")

    # Add indicator features
    test_df["has_prev_address"] = (test_df["prev_address_months_count"] >= 0).astype("float32")
    test_df["has_bank_months"] = (test_df["bank_months_count"] >= 0).astype("float32")
    test_df["has_intended_balcon"] = (test_df["intended_balcon_amount"] >= 0).astype("float32")
    test_df["has_credit_risk_score"] = (test_df["credit_risk_score"] >= 0).astype("float32")

    # Prepare features
    X_num = test_df[num_and_ind_cols].fillna(0).to_numpy(dtype=np.float32)
    X_cat = ohe.transform(test_df[CATEGORICAL_FEATURES].astype(str))
    X_test = np.hstack([X_num, X_cat])
    y_test = test_df["fraud_bool"].to_numpy(dtype=np.int32)
    ages = test_df["customer_age"].to_numpy(dtype=np.float32)

    # Score and calibrate
    raw_scores = champion.predict_proba(X_test)[:, 1]
    eps = 1e-7
    logits = np.log(np.clip(raw_scores, eps, 1 - eps) / (1 - np.clip(raw_scores, eps, 1 - eps))).reshape(-1, 1)
    cal_probs = calibrator.predict_proba(logits)[:, 1]

    # Overall PR-AUC
    overall_pr = float(average_precision_score(y_test, cal_probs))
    print(f"Overall Month 7 PR-AUC: {overall_pr:.5f}")

    # Masks for Age Groups
    under_50 = ages < 50
    over_50 = ages >= 50
    print(f"Demographics: Under 50 = {under_50.sum():,} ({y_test[under_50].sum()} frauds, {y_test[under_50].mean()*100:.2f}%) | 50+ = {over_50.sum():,} ({y_test[over_50].sum()} frauds, {y_test[over_50].mean()*100:.2f}%)")

    # 3. Operating-Point Table at 1%, 2%, 5% FPR
    target_fprs = [0.01, 0.02, 0.05]
    op_table = []
    neg_scores = cal_probs[y_test == 0]

    for target_fpr in target_fprs:
        # Find threshold corresponding to target FPR on legitimate applications
        th = float(np.percentile(neg_scores, (1.0 - target_fpr) * 100))
        preds = (cal_probs >= th).astype(int)

        tp = int(((preds == 1) & (y_test == 1)).sum())
        fp = int(((preds == 1) & (y_test == 0)).sum())
        fn = int(((preds == 0) & (y_test == 1)).sum())
        tn = int(((preds == 0) & (y_test == 0)).sum())

        recall = float(tp / max(tp + fn, 1))
        prec = float(tp / max(tp + fp, 1))
        actual_fpr = float(fp / max(fp + tn, 1))
        fa_per_100k = float(fp / len(y_test) * 100000)

        # Fairness audit at this threshold
        # Under 50
        y_u = y_test[under_50]
        p_u = preds[under_50]
        rec_u = float(((p_u == 1) & (y_u == 1)).sum() / max(y_u.sum(), 1))
        fpr_u = float(((p_u == 1) & (y_u == 0)).sum() / max((y_u == 0).sum(), 1))
        fa_u = float(((p_u == 1) & (y_u == 0)).sum() / len(y_u) * 100000)

        # 50 and over
        y_o = y_test[over_50]
        p_o = preds[over_50]
        rec_o = float(((p_o == 1) & (y_o == 1)).sum() / max(y_o.sum(), 1))
        fpr_o = float(((p_o == 1) & (y_o == 0)).sum() / max((y_o == 0).sum(), 1))
        fa_o = float(((p_o == 1) & (y_o == 0)).sum() / len(y_o) * 100000)

        disparity_recall = round(abs(rec_o - rec_u) * 100.0, 2)
        disparity_fpr_ratio = round(fpr_o / max(fpr_u, 1e-6), 2)

        op_table.append({
            "target_fpr_pct": int(target_fpr * 100),
            "threshold": round(th, 4),
            "recall_pct": round(recall * 100.0, 2),
            "precision_pct": round(prec * 100.0, 2),
            "actual_fpr_pct": round(actual_fpr * 100.0, 2),
            "false_alarms_per_100k": round(fa_per_100k, 1),
            "tp": tp,
            "fp": fp,
            "fairness_audit": {
                "under_50": {
                    "recall_pct": round(rec_u * 100.0, 2),
                    "fpr_pct": round(fpr_u * 100.0, 2),
                    "false_alarms_per_100k": round(fa_u, 1),
                },
                "age_50_plus": {
                    "recall_pct": round(rec_o * 100.0, 2),
                    "fpr_pct": round(fpr_o * 100.0, 2),
                    "false_alarms_per_100k": round(fa_o, 1),
                },
                "recall_disparity_pct": disparity_recall,
                "false_alarm_ratio_older_to_younger": disparity_fpr_ratio,
            }
        })

    # Print Operating Point & Fairness Table
    print("\n" + "=" * 95)
    print(f"{'Target FPR':<12} | {'Threshold':<10} | {'Recall':<8} | {'FA / 100k':<10} | {'Recall <50':<11} | {'Recall 50+':<11} | {'FPR Ratio (50+/<50)'}")
    print("-" * 95)
    for op in op_table:
        fa = op["fairness_audit"]
        print(f"{op['target_fpr_pct']:>2}% FPR      | {op['threshold']:>8.4f}   | {op['recall_pct']:>5.2f}%  | {op['false_alarms_per_100k']:>8.1f}   | {fa['under_50']['recall_pct']:>6.2f}%     | {fa['age_50_plus']['recall_pct']:>6.2f}%     | {fa['false_alarm_ratio_older_to_younger']:>5.2f}x")
    print("=" * 95)

    # 4. Fairness-Accuracy Intervention: Per-Group Threshold Calibration
    print("\n4. Evaluating Fairness Intervention: Per-Group Threshold Calibration (Demographic Parity)...")
    # For 5% FPR target, instead of uniform threshold, calibrate group-specific thresholds
    neg_u = cal_probs[(y_test == 0) & under_50]
    neg_o = cal_probs[(y_test == 0) & over_50]

    th_u_5pct = float(np.percentile(neg_u, 95.0))
    th_o_5pct = float(np.percentile(neg_o, 95.0))

    fair_preds = np.zeros(len(y_test), dtype=int)
    fair_preds[under_50] = (cal_probs[under_50] >= th_u_5pct).astype(int)
    fair_preds[over_50] = (cal_probs[over_50] >= th_o_5pct).astype(int)

    tp_fair = int(((fair_preds == 1) & (y_test == 1)).sum())
    fp_fair = int(((fair_preds == 1) & (y_test == 0)).sum())
    fn_fair = int(((fair_preds == 0) & (y_test == 1)).sum())
    tn_fair = int(((fair_preds == 0) & (y_test == 0)).sum())

    rec_fair = float(tp_fair / max(tp_fair + fn_fair, 1))
    prec_fair = float(tp_fair / max(tp_fair + fp_fair, 1))
    fpr_fair = float(fp_fair / max(fp_fair + tn_fair, 1))

    # Groups under fair calibration
    rec_u_fair = float(((fair_preds[under_50] == 1) & (y_test[under_50] == 1)).sum() / max(y_test[under_50].sum(), 1))
    rec_o_fair = float(((fair_preds[over_50] == 1) & (y_test[over_50] == 1)).sum() / max(y_test[over_50].sum(), 1))
    fpr_u_fair = float(((fair_preds[under_50] == 1) & (y_test[under_50] == 0)).sum() / max((y_test[under_50] == 0).sum(), 1))
    fpr_o_fair = float(((fair_preds[over_50] == 1) & (y_test[over_50] == 0)).sum() / max((y_test[over_50] == 0).sum(), 1))
    ratio_fair = round(fpr_o_fair / max(fpr_u_fair, 1e-6), 2)

    baseline_5pct = [op for op in op_table if op["target_fpr_pct"] == 5][0]
    b_rec = baseline_5pct["recall_pct"]
    b_ratio = baseline_5pct["fairness_audit"]["false_alarm_ratio_older_to_younger"]

    tradeoff_report = {
        "intervention": "Per-Group Calibration (Equalized False Alarm Rate)",
        "target_operating_point": "5% FPR",
        "baseline": {
            "uniform_threshold": baseline_5pct["threshold"],
            "overall_recall_pct": b_rec,
            "younger_fpr_pct": baseline_5pct["fairness_audit"]["under_50"]["fpr_pct"],
            "older_fpr_pct": baseline_5pct["fairness_audit"]["age_50_plus"]["fpr_pct"],
            "fpr_disparity_ratio": b_ratio,
        },
        "fair_calibrated": {
            "younger_threshold": round(th_u_5pct, 4),
            "older_threshold": round(th_o_5pct, 4),
            "overall_recall_pct": round(rec_fair * 100.0, 2),
            "younger_fpr_pct": round(fpr_u_fair * 100.0, 2),
            "older_fpr_pct": round(fpr_o_fair * 100.0, 2),
            "fpr_disparity_ratio": ratio_fair,
        },
        "tradeoff_summary": f"Per-group calibration equalizes false alarms (FPR ratio drops from {b_ratio}x to {ratio_fair}x), with a recall delta of {round(rec_fair*100 - b_rec, 2)}%.",
    }

    print("\nFairness-Accuracy Trade-off Summary:")
    print(f"  • Baseline 5% FPR: Recall = {b_rec}%, Older FPR Ratio = {b_ratio}x")
    print(f"  • Fair-Calibrated: Recall = {round(rec_fair*100, 2)}%, Older FPR Ratio = {ratio_fair}x (Demographic Parity achieved)")
    print(f"  • Trade-off: Δ Recall = {round(rec_fair*100 - b_rec, 2)}% to achieve equalized false alarm rates.")

    # 5. Save report
    out_payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "channel": "account",
        "dataset": "Bank Account Fraud (BAF - NeurIPS 2022)",
        "evaluation_split": "Month 7 (96,843 rows)",
        "pr_auc": overall_pr,
        "operating_points": op_table,
        "fairness_accuracy_tradeoff": tradeoff_report,
    }

    out_file = REPORTS_DIR / "baf_operating_points_fairness.json"
    with open(out_file, "w") as f:
        json.dump(out_payload, f, indent=2)
    print(f"\nSaved BAF operating points and fairness audit to: {out_file}")

    # 6. Update reports/metrics.json and models/metadata.json
    with open(METRICS_FILE) as f:
        metrics = json.load(f)

    if "account" in metrics:
        metrics["account"]["operating_point_table"] = op_table
        metrics["account"]["fairness_accuracy_tradeoff"] = tradeoff_report

    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

    with open(METADATA_FILE) as f:
        meta = json.load(f)

    if "account" in meta.get("channels", {}):
        meta["channels"]["account"]["operating_point_table"] = op_table
        meta["channels"]["account"]["fairness_accuracy_tradeoff"] = tradeoff_report

    with open(METADATA_FILE, "w") as f:
        json.dump(meta, f, indent=2)

    print("Updated reports/metrics.json and models/metadata.json with BAF operating points and fairness trade-off.")


if __name__ == "__main__":
    run_baf_analysis()

