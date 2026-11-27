"""
Leakage Diagnostic and Balance Feature Audit.
Empirically tests whether post-transaction balance features (newbalanceOrig, newbalanceDest,
orig_balance_diff_error, orig_emptied) cause target leakage or artificial perfection.
"""

from __future__ import annotations

from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from ml.config import RANDOM_STATE, TARGET_COL, TRAIN_DATA_PATH, VAL_DATA_PATH
from ml.features import FEATURE_COLUMNS


def run_leakage_experiment():
    print("Loading processed train and validation sets...")
    train_df = pd.read_parquet(TRAIN_DATA_PATH)
    val_df = pd.read_parquet(VAL_DATA_PATH)

    y_train = train_df[TARGET_COL].values
    y_val = val_df[TARGET_COL].values

    # Compute scale_pos_weight
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = float(n_neg / n_pos)

    # Define feature configurations to test
    post_balance_cols = [
        "newbalanceOrig",
        "newbalanceDest",
        "orig_balance_diff_error",
        "dest_balance_diff_error",
        "orig_emptied",
    ]
    pre_tx_only_cols = [c for c in FEATURE_COLUMNS if c not in post_balance_cols]

    experiments = {
        "Full Feature Set (All 23 features)": FEATURE_COLUMNS,
        "Pre-Transaction Only (No post balances/errors)": pre_tx_only_cols,
        "Pre-Tx + Account Emptied Flag": pre_tx_only_cols + ["orig_emptied"],
        "Pre-Tx + Balance Discrepancy Errors": pre_tx_only_cols + ["orig_balance_diff_error", "dest_balance_diff_error"],
    }

    results = []

    print("\n--- Running Balance Feature Leakage Audit ---")
    for name, cols in experiments.items():
        print(f"\nEvaluating: {name} ({len(cols)} features)...")
        X_train = train_df[cols]
        X_val = val_df[cols]

        model = lgb.LGBMClassifier(
            objective="binary",
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_estimators=100,
            learning_rate=0.08,
            max_depth=6,
            num_leaves=31,
            n_jobs=-1,
            verbose=-1,
        )

        model.fit(X_train, y_train)

        probs_val = model.predict_proba(X_val)[:, 1]
        preds_val = (probs_val >= 0.50).astype(int)

        pr_auc = average_precision_score(y_val, probs_val)
        roc_auc = roc_auc_score(y_val, probs_val)
        prec = precision_score(y_val, preds_val, zero_division=0)
        rec = recall_score(y_val, preds_val, zero_division=0)
        f1 = f1_score(y_val, preds_val, zero_division=0)

        results.append({
            "Configuration": name,
            "Feature Count": len(cols),
            "PR-AUC": round(pr_auc, 5),
            "ROC-AUC": round(roc_auc, 5),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1, 4),
        })

        # Feature importances
        imp_df = pd.DataFrame({
            "feature": cols,
            "importance": model.feature_importances_
        }).sort_values(by="importance", ascending=False)
        top3 = imp_df.head(3)["feature"].tolist()
        print(f"  PR-AUC: {pr_auc:.5f} | ROC-AUC: {roc_auc:.5f} | Recall: {rec*100:.2f}% | Top Features: {top3}")

    res_df = pd.DataFrame(results)
    print("\n=================== LEAKAGE AUDIT COMPARISON ===================")
    print(res_df.to_string(index=False))

    return res_df


def test_pre_transaction_features_contain_zero_leakage():
    """Verify that post-transaction features and discrepancies are completely excluded."""
    post_leakage_features = {
        "newbalanceOrig",
        "newbalanceDest",
        "orig_balance_diff_error",
        "dest_balance_diff_error",
    }
    for feat in post_leakage_features:
        assert feat not in FEATURE_COLUMNS, f"Leaked post-transaction feature '{feat}' present in model features!"


def test_receiver_features_order_independence_within_steps():
    """Verify that shuffling rows within a step produces strictly identical receiver feature values."""
    from ml.features import compute_receiver_features_vectorized

    # Construct test data with repeating receivers across multiple steps and multiple rows per step
    data = []
    np.random.seed(42)
    destinations = [f"MULE_{i}" for i in range(10)]
    row_id = 0
    for s in range(1, 15):
        # 20 transactions per step
        for _ in range(20):
            d = np.random.choice(destinations)
            amt = float(np.random.uniform(100, 50000))
            data.append({"row_id": row_id, "step": s, "nameDest": d, "amount": amt})
            row_id += 1

    df_orig = pd.DataFrame(data)

    # Compute on original order
    feat_orig = compute_receiver_features_vectorized(df_orig)
    df_with_orig = df_orig.copy()
    for col, arr in feat_orig.items():
        df_with_orig[col] = arr

    # Shuffle rows within each step
    shuffled_rows = []
    for s, grp in df_orig.groupby("step", sort=False):
        shuffled_grp = grp.sample(frac=1.0, random_state=123).copy()
        shuffled_rows.append(shuffled_grp)
    df_shuffled = pd.concat(shuffled_rows, ignore_index=True)

    # Compute on shuffled order
    feat_shuff = compute_receiver_features_vectorized(df_shuffled)
    df_with_shuff = df_shuffled.copy()
    for col, arr in feat_shuff.items():
        df_with_shuff[col] = arr

    # Sort back by original row_id to align rows
    df_with_shuff = df_with_shuff.sort_values("row_id").reset_index(drop=True)
    df_with_orig = df_with_orig.sort_values("row_id").reset_index(drop=True)

    # Check identical values for all receiver features (within float32 summation tolerance)
    for col in feat_orig.keys():
        np.testing.assert_allclose(
            df_with_orig[col].to_numpy(),
            df_with_shuff[col].to_numpy(),
            rtol=1e-5,
            atol=1e-3,
            err_msg=f"Receiver feature '{col}' differs after within-step row shuffling!",
        )


if __name__ == "__main__":
    run_leakage_experiment()

