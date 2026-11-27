"""
FraudShield Concept Drift Benchmark (Phase 2).
Compares Static vs Monthly/Periodically-Retrained Models over time on two channels:
1. Bank Account Fraud (BAF): Months 3 to 7
2. UPI Transactions: Cohorts across 60 days (Days 16-30, 31-45, 46-60)

Outputs:
- reports/concept_drift_benchmark.json
- reports/metrics.json["concept_drift"]
- models/metadata.json["concept_drift"]
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

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, recall_score
from sklearn.preprocessing import OneHotEncoder

from ml.config import METADATA_FILE, METRICS_FILE, RANDOM_STATE, REPORTS_DIR


def run_baf_drift_benchmark() -> dict[str, Any]:
    """Compare Static vs Monthly Retrained LightGBM models on BAF across Months 3..7."""
    print("\n--- Running BAF Concept Drift Benchmark (Months 3 to 7) ---")
    raw_path = BASE_DIR / "data" / "raw" / "Base.csv"
    if not raw_path.exists():
        print("  Base.csv not found; skipping full BAF drift.")
        return {}

    # Read necessary columns
    usecols = [
        "month", "fraud_bool", "income", "name_email_similarity", "prev_address_months_count",
        "current_address_months_count", "days_since_request", "intended_balcon_amount",
        "zip_count_4w", "velocity_6h", "velocity_24h", "velocity_4w", "bank_branch_count_8w",
        "date_of_birth_distinct_emails_4w", "credit_risk_score", "email_is_free",
        "phone_home_valid", "phone_mobile_valid", "bank_months_count", "has_other_cards",
        "proposed_credit_limit", "foreign_request", "session_length_in_minutes",
        "keep_alive_session", "device_distinct_emails_8w", "device_fraud_count",
        "payment_type", "employment_status", "housing_status", "source", "device_os",
    ]
    df = pd.read_csv(raw_path, usecols=usecols)

    num_cols = [c for c in usecols if c not in ["month", "fraud_bool", "payment_type", "employment_status", "housing_status", "source", "device_os"]]
    cat_cols = ["payment_type", "employment_status", "housing_status", "source", "device_os"]

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ohe.fit(df[cat_cols])

    def prep(d):
        X_num = d[num_cols].fillna(0).to_numpy(dtype=np.float32)
        X_cat = ohe.transform(d[cat_cols])
        X = np.hstack([X_num, X_cat])
        y = d["fraud_bool"].to_numpy(dtype=np.int32)
        return X, y

    # 1. Train Static Model on Months 0, 1, 2
    train_m012 = df[df["month"] <= 2]
    X_static_tr, y_static_tr = prep(train_m012)
    static_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    static_model.fit(X_static_tr, y_static_tr)

    eval_months = [3, 4, 5, 6, 7]
    drift_series = []

    for m in eval_months:
        test_m = df[df["month"] == m]
        X_test_m, y_test_m = prep(test_m)

        # Static model evaluation
        p_static = static_model.predict_proba(X_test_m)[:, 1]
        pr_static = float(average_precision_score(y_test_m, p_static))

        # TPR @ 5% FPR for static
        neg_p_s = p_static[y_test_m == 0]
        th_s_5pct = float(np.percentile(neg_p_s, 95.0))
        tpr_static = float(recall_score(y_test_m, (p_static >= th_s_5pct).astype(int), zero_division=0)) * 100.0

        # Monthly retrained model (trained on all months up to m-1)
        train_up_to_m = df[df["month"] < m]
        X_rt, y_rt = prep(train_up_to_m)
        rt_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
        rt_model.fit(X_rt, y_rt)

        p_rt = rt_model.predict_proba(X_test_m)[:, 1]
        pr_rt = float(average_precision_score(y_test_m, p_rt))

        neg_p_rt = p_rt[y_test_m == 0]
        th_rt_5pct = float(np.percentile(neg_p_rt, 95.0))
        tpr_rt = float(recall_score(y_test_m, (p_rt >= th_rt_5pct).astype(int), zero_division=0)) * 100.0

        drift_series.append({
            "month": m,
            "total_applications": len(test_m),
            "fraud_count": int(y_test_m.sum()),
            "static_model": {
                "pr_auc": round(pr_static, 5),
                "tpr_at_5pct_fpr": round(tpr_static, 2),
            },
            "monthly_retrained_model": {
                "pr_auc": round(pr_rt, 5),
                "tpr_at_5pct_fpr": round(tpr_rt, 2),
            },
            "pr_auc_delta": round(pr_rt - pr_static, 5),
            "tpr_delta_pct": round(tpr_rt - tpr_static, 2),
        })

        print(f"  Month {m}: Static PR-AUC = {pr_static:.4f} (TPR@5% = {tpr_static:.1f}%) | Retrained PR-AUC = {pr_rt:.4f} (TPR@5% = {tpr_rt:.1f}%) | Δ TPR = +{tpr_rt-tpr_static:.1f}%")

    return {
        "channel": "account",
        "description": "Comparison of static model trained on Months 0-2 vs monthly retrained models over Months 3 to 7",
        "series": drift_series,
    }


def run_upi_drift_benchmark() -> dict[str, Any]:
    """Compare Static vs Periodically Retrained LightGBM models on UPI cohorts."""
    print("\n--- Running UPI Concept Drift Benchmark (Cohorts 2, 3, 4) ---")
    raw_path = BASE_DIR / "data" / "raw" / "upi_synthetic.csv"
    if not raw_path.exists():
        return {}

    df = pd.read_csv(raw_path, comment="#")
    df["log_amount"] = np.log1p(df["amount"]).astype("float32")
    df["amount_to_median_ratio"] = (df["amount"] / 350.0).clip(0.0, 500.0).astype("float32")
    df["is_extreme_amount"] = (df["amount"] > 20000.0).astype("int8")
    df["first_time_high_value"] = ((df["is_first_time_payee"] == 1) & (df["amount"] > 10000.0)).astype("int8")

    features = [
        "amount", "log_amount", "amount_to_median_ratio", "is_extreme_amount", "first_time_high_value",
        "is_round_amount", "is_first_time_payee", "is_new_device", "sim_change_last_48h",
        "remote_access_app_flag", "velocity_1h", "velocity_24h",
    ]

    # Cohort 1: Days 1-15
    # Cohort 2: Days 16-30
    # Cohort 3: Days 31-45
    # Cohort 4: Days 46-60
    c1 = df[df["day"] <= 15]
    static_m = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    static_m.fit(c1[features], c1["is_fraud"])

    cohorts = [
        ("Cohort 2 (Days 16-30)", 16, 30),
        ("Cohort 3 (Days 31-45)", 31, 45),
        ("Cohort 4 (Days 46-60, Drift)", 46, 60),
    ]

    series = []
    for name, start_d, end_d in cohorts:
        eval_c = df[(df["day"] >= start_d) & (df["day"] <= end_d)]
        X_eval = eval_c[features]
        y_eval = eval_c["is_fraud"]

        # Static
        p_static = static_m.predict_proba(X_eval)[:, 1]
        pr_static = float(average_precision_score(y_eval, p_static))
        rec_static = float(recall_score(y_eval, (p_static >= 0.50).astype(int), zero_division=0)) * 100.0

        # Retrained up to start_d - 1
        tr_rt = df[df["day"] < start_d]
        rt_m = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
        rt_m.fit(tr_rt[features], tr_rt["is_fraud"])

        p_rt = rt_m.predict_proba(X_eval)[:, 1]
        pr_rt = float(average_precision_score(y_eval, p_rt))
        rec_rt = float(recall_score(y_eval, (p_rt >= 0.50).astype(int), zero_division=0)) * 100.0

        series.append({
            "cohort": name,
            "days": [start_d, end_d],
            "total_transactions": len(eval_c),
            "fraud_count": int(y_eval.sum()),
            "static_model": {
                "pr_auc": round(pr_static, 5),
                "recall_pct": round(rec_static, 2),
            },
            "retrained_model": {
                "pr_auc": round(pr_rt, 5),
                "recall_pct": round(rec_rt, 2),
            },
            "pr_auc_delta": round(pr_rt - pr_static, 5),
            "recall_delta_pct": round(rec_rt - rec_static, 2),
        })

        print(f"  {name}: Static PR-AUC = {pr_static:.4f} (Recall = {rec_static:.1f}%) | Retrained PR-AUC = {pr_rt:.4f} (Recall = {rec_rt:.1f}%)")

    return {
        "channel": "upi",
        "description": "Comparison of static model trained on Days 1-15 vs periodically retrained models over bi-weekly cohorts",
        "series": series,
    }


def main():
    print("=" * 75)
    print("  FRAUDSHIELD: MULTI-CHANNEL CONCEPT DRIFT BENCHMARK")
    print("=" * 75)

    baf_drift = run_baf_drift_benchmark()
    upi_drift = run_upi_drift_benchmark()

    drift_payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "baf_account_drift": baf_drift,
        "upi_drift": upi_drift,
    }

    out_file = REPORTS_DIR / "concept_drift_benchmark.json"
    with open(out_file, "w") as f:
        json.dump(drift_payload, f, indent=2)
    print(f"\nSaved concept drift benchmark to: {out_file}")

    with open(METRICS_FILE) as f:
        metrics = json.load(f)
    metrics["concept_drift"] = drift_payload
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

    with open(METADATA_FILE) as f:
        meta = json.load(f)
    meta["concept_drift"] = drift_payload
    with open(METADATA_FILE, "w") as f:
        json.dump(meta, f, indent=2)

    print("Updated reports/metrics.json and models/metadata.json with concept drift benchmark.")


if __name__ == "__main__":
    main()

