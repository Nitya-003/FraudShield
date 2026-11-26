from __future__ import annotations

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

from ml.config import (
    COST_FALSE_DECLINE_INR,
    COST_OTP_CHALLENGE_INR,
    DEFAULT_HIGH_THRESHOLD,
    DEFAULT_LOW_THRESHOLD,
    FIGURES_DIR,
    MAX_DAILY_REVIEW_CAPACITY,
    METADATA_FILE,
    METRICS_FILE,
    MODEL_FILE,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    TARGET_COL,
    TEST_DATA_PATH,
    TEST_SAMPLE_PATH,
    VAL_DATA_PATH,
)
from ml.features import FEATURE_COLUMNS


def find_cost_optimal_thresholds(
    y_val: np.ndarray,
    val_probs: np.ndarray,
    val_amounts: np.ndarray,
    val_steps: np.ndarray,
    cost_otp: float = COST_OTP_CHALLENGE_INR,
    cost_false_decline: float = COST_FALSE_DECLINE_INR,
    max_daily_review: int = MAX_DAILY_REVIEW_CAPACITY,
) -> tuple[float, float, dict]:
    n_days = max(1.0, (val_steps.max() - val_steps.min()) / 24.0)
    best_cost = float("inf")
    best_t_low = DEFAULT_LOW_THRESHOLD
    best_t_high = DEFAULT_HIGH_THRESHOLD
    best_breakdown = {}

    candidate_lows = np.linspace(0.05, 0.45, 9)
    candidate_highs = np.linspace(0.55, 0.90, 8)

    for t_l in candidate_lows:
        for t_h in candidate_highs:
            if t_l >= t_h:
                continue

            low_mask = val_probs < t_l
            med_mask = (val_probs >= t_l) & (val_probs <= t_h)
            high_mask = val_probs > t_h

            daily_med = med_mask.sum() / n_days
            if daily_med > max_daily_review:
                continue

            cost_missed = val_amounts[low_mask & (y_val == 1)].sum()
            cost_otp_total = med_mask.sum() * cost_otp
            cost_declines = (high_mask & (y_val == 0)).sum() * cost_false_decline
            total_cost = cost_missed + cost_otp_total + cost_declines

            if total_cost < best_cost:
                best_cost = total_cost
                best_t_low = round(float(t_l), 3)
                best_t_high = round(float(t_h), 3)
                best_breakdown = {
                    "total_cost_inr": round(float(total_cost), 2),
                    "missed_fraud_loss_inr": round(float(cost_missed), 2),
                    "otp_cost_inr": round(float(cost_otp_total), 2),
                    "false_decline_cost_inr": round(float(cost_declines), 2),
                    "daily_otp_challenges": round(float(daily_med), 1),
                }

    print(f"Validation Cost Optimization: Best Thresholds: Low < {best_t_low}, High > {best_t_high}")
    print(f"  Validation Expected Total Cost: ₹{best_breakdown.get('total_cost_inr', 0):,.2f}")
    return best_t_low, best_t_high, best_breakdown


def evaluate_test_economics(
    y_test: np.ndarray,
    test_probs: np.ndarray,
    test_amounts: np.ndarray,
    test_steps: np.ndarray,
    t_low: float,
    t_high: float,
) -> dict:
    n_days = max(1.0, (test_steps.max() - test_steps.min()) / 24.0)
    total_tx = len(y_test)
    total_fraud_count = int(y_test.sum())
    total_fraud_inr = float(test_amounts[y_test == 1].sum())

    bands = {
        "Low (< {:.2f})".format(t_low): test_probs < t_low,
        "Medium ({:.2f}-{:.2f})".format(t_low, t_high): (test_probs >= t_low) & (test_probs <= t_high),
        "High (> {:.2f})".format(t_high): test_probs > t_high,
    }

    band_report = {}
    for name, mask in bands.items():
        count = int(mask.sum())
        frauds_in_band = int((mask & (y_test == 1)).sum())
        total_inr = float(test_amounts[mask].sum())
        fraud_inr = float(test_amounts[mask & (y_test == 1)].sum())

        band_report[name] = {
            "transactions": count,
            "pct_traffic": round(float(count / total_tx * 100), 2),
            "frauds": frauds_in_band,
            "total_inr": round(total_inr, 2),
            "fraud_inr": round(fraud_inr, 2),
            "daily_volume": round(float(count / n_days), 1),
        }

    frauds_caught = int(((test_probs >= t_low) & (y_test == 1)).sum())
    fraud_inr_caught = float(test_amounts[(test_probs >= t_low) & (y_test == 1)].sum())
    frauds_missed = int(((test_probs < t_low) & (y_test == 1)).sum())
    fraud_inr_missed = float(test_amounts[(test_probs < t_low) & (y_test == 1)].sum())

    false_alarms = int(((test_probs >= t_high) & (y_test == 0)).sum())
    tn = int(((test_probs <= t_high) & (y_test == 0)).sum())
    fpr = float(false_alarms / (false_alarms + tn)) if (false_alarms + tn) > 0 else 0.0
    fa_per_100k = float(fpr * 100000)

    summary = {
        "total_test_transactions": total_tx,
        "total_test_fraud_count": total_fraud_count,
        "total_exposed_fraud_inr": round(total_fraud_inr, 2),
        "fraud_count_caught": frauds_caught,
        "fraud_count_missed": frauds_missed,
        "fraud_recall_pct": round(float(frauds_caught / total_fraud_count * 100), 2),
        "fraud_inr_caught": round(fraud_inr_caught, 2),
        "fraud_inr_missed": round(fraud_inr_missed, 2),
        "fraud_value_protection_pct": round(float(fraud_inr_caught / total_fraud_inr * 100), 4) if total_fraud_inr > 0 else 100.0,
        "false_declines": false_alarms,
        "false_positive_rate": round(fpr, 5),
        "false_alarms_per_100k": round(fa_per_100k, 1),
        "daily_otp_challenges": round(float(band_report["Medium ({:.2f}-{:.2f})".format(t_low, t_high)]["daily_volume"]), 1),
        "bands": band_report,
    }
    return summary


def run_evaluation():
    print("=" * 70)
    print("  FRAUDSHIELD COMPREHENSIVE TEST EVALUATION & ECONOMICS")
    print("=" * 70)

    val_df = pd.read_parquet(VAL_DATA_PATH)
    test_df = pd.read_parquet(TEST_DATA_PATH)
    payload = joblib.load(MODEL_FILE)
    model = payload["model"]
    calibrator = payload["calibrator"]

    eps = 1e-7
    def get_calibrated_probs(df):
        X = df[FEATURE_COLUMNS].to_numpy()
        raw_p = model.predict_proba(X)[:, 1]
        lo = np.log(np.clip(raw_p, eps, 1 - eps) / (1 - np.clip(raw_p, eps, 1 - eps))).reshape(-1, 1)
        return calibrator.predict_proba(lo)[:, 1]

    print("\n1. Scoring validation and test distributions...")
    val_probs = get_calibrated_probs(val_df)
    test_probs = get_calibrated_probs(test_df)

    y_val = val_df[TARGET_COL].to_numpy()
    y_test = test_df[TARGET_COL].to_numpy()
    val_amounts = val_df["amount"].to_numpy()
    test_amounts = test_df["amount"].to_numpy()
    val_steps = val_df["step"].to_numpy()
    test_steps = test_df["step"].to_numpy()

    print("\n2. Finding Cost-Optimal Thresholds on Validation Data...")
    opt_t_low, opt_t_high, val_breakdown = find_cost_optimal_thresholds(
        y_val, val_probs, val_amounts, val_steps
    )

    print("\n3. Evaluating Test Economics with Cost-Optimal Thresholds...")
    test_economics = evaluate_test_economics(
        y_test, test_probs, test_amounts, test_steps, opt_t_low, opt_t_high
    )

    print(f"\nTest Economic Impact Summary (918,617 Natural Test Transactions):")
    print(f"  Exposed Fraud Value:   ₹{test_economics['total_exposed_fraud_inr']:,.2f}")
    print(f"  Fraud Value Caught:    ₹{test_economics['fraud_inr_caught']:,.2f} ({test_economics['fraud_value_protection_pct']}%)")
    print(f"  Fraud Value Missed:    ₹{test_economics['fraud_inr_missed']:,.2f}")
    print(f"  Frauds Caught:         {test_economics['fraud_count_caught']:,} / {test_economics['total_test_fraud_count']:,} ({test_economics['fraud_recall_pct']}%)")
    print(f"  False Declines:        {test_economics['false_declines']:,} (FPR: {test_economics['false_positive_rate']*100:.4f}%, {test_economics['false_alarms_per_100k']} per 100k)")
    print(f"  Daily OTP Challenges:  {test_economics['daily_otp_challenges']} per day")

    print("\nRisk Band Breakdown:")
    for b_name, b_info in test_economics["bands"].items():
        print(f"  {b_name:24s} | Volume: {b_info['transactions']:,} ({b_info['pct_traffic']}%) | Fraud: {b_info['frauds']:,} | Fraud INR: ₹{b_info['fraud_inr']:,.2f}")

    print("\n4. Generating High-Resolution Evaluation Figures in reports/figures/...")
    prec, rec, _ = precision_recall_curve(y_test, test_probs)
    pr_auc = average_precision_score(y_test, test_probs)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rec, prec, color="#2563eb", lw=2, label=f"Calibrated LightGBM (PR-AUC = {pr_auc:.4f})")
    ax.axhline(y_test.mean(), color="gray", linestyle="--", label=f"Random Chance ({y_test.mean()*100:.3f}%)")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (Positive Predictive Value)")
    ax.set_title("Precision-Recall Curve on Full Natural Test Set (918,617 Rows)")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "06_test_precision_recall_curves.png", dpi=200)
    plt.close(fig)

    cm = confusion_matrix(y_test, (test_probs >= opt_t_high).astype(int))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Approved / OTP", "Declined (Fraud Alert)"])
    ax.set_yticklabels(["Legitimate", "Actual Fraud"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=12, weight="bold")
    ax.set_title(f"Confusion Matrix @ Hard Decline Threshold > {opt_t_high}")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "07_champion_confusion_matrix.png", dpi=200)
    plt.close(fig)

    fi_series = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    fi_series.tail(15).plot(kind="barh", ax=ax, color="#10b981", edgecolor="#059669")
    ax.set_xlabel("Split Importance (Gain Count)")
    ax.set_title("Top 15 Feature Importances (Champion Model)")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "09_feature_importance.png", dpi=200)
    plt.close(fig)

    print("\n5. Precomputing 5,000-Row Scored Sample for Streamlit Table...")
    np.random.seed(RANDOM_STATE)
    fraud_idx = np.where(y_test == 1)[0]
    legit_idx = np.where(y_test == 0)[0]
    sample_fraud = np.random.choice(fraud_idx, size=min(len(fraud_idx), 500), replace=False)
    sample_legit = np.random.choice(legit_idx, size=4500, replace=False)
    sample_indices = np.concatenate([sample_fraud, sample_legit])
    np.random.shuffle(sample_indices)

    sample_df = test_df.iloc[sample_indices].copy()
    sample_probs = test_probs[sample_indices]

    sample_df["risk_score"] = sample_probs.round(4)
    sample_df["risk_band"] = np.where(
        sample_probs < opt_t_low,
        "Low",
        np.where(sample_probs <= opt_t_high, "Medium", "High"),
    )
    sample_df["band"] = sample_df["risk_band"]
    sample_df["action"] = np.where(
        sample_df["risk_band"] == "Low",
        "Approve",
        np.where(sample_df["risk_band"] == "Medium", "Step-Up (OTP)", "Decline & Alert"),
    )

    def get_reasons(row):
        reasons = []
        if row["orig_emptied"] == 1:
            reasons.append("Account balance liquidated")
        if row["amount"] > 100000:
            reasons.append(f"High amount (₹{row['amount']:,.0f})")
        if row.get("type_TRANSFER", 0) == 1:
            reasons.append("Transfer drainage channel")
        elif row.get("type_CASH_OUT", 0) == 1:
            reasons.append("Cash-out drainage channel")
        if row.get("dest_tx_last_24h", 0) > 0:
            reasons.append(f"Receiver velocity burst ({row['dest_tx_last_24h']} txs)")
        if len(reasons) == 0:
            reasons.append("Standard payment parameters")
        return "; ".join(reasons[:3])

    sample_df["top_reasons"] = sample_df.apply(get_reasons, axis=1)
    sample_df.to_parquet(TEST_SAMPLE_PATH, compression="snappy", index=False)
    print(f"Scored sample saved to {TEST_SAMPLE_PATH}")

    print("\n6. Updating metadata.json and metrics.json...")
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)

    meta["thresholds"] = {
        "low_risk_approve": opt_t_low,
        "high_risk_decline": opt_t_high,
        "medium_risk_otp": [opt_t_low, opt_t_high],
        "optimization_basis": "Validation Expected Cost Minimization",
    }
    meta["calibrated_thresholds"] = {
        "low_risk_approve": opt_t_low,
        "high_risk_decline": opt_t_high,
    }
    meta["economics_natural_prevalence"] = test_economics
    meta["risk_band_economics_inr"] = test_economics

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    with open(METRICS_FILE, "r", encoding="utf-8") as f:
        metrics_dict = json.load(f)

    metrics_dict["economics_natural_test"] = test_economics
    metrics_dict["calibrated_thresholds"] = meta["thresholds"]

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)

    print("Phase 4 Evaluation & Economics successfully completed!")


if __name__ == "__main__":
    run_evaluation()
