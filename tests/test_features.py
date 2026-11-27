"""
Unit tests for feature engineering, time splitting, and strict leakage prevention.
Verifies Phase 2 requirements:
- Receiver past-only behavior strictly excludes the current row.
- Pre-transaction features only (no post-transaction balances or errors).
- Zero training/serving skew: identical feature columns between batch and single-transaction scoring.
- Chronological time-split integrity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.features import (
    FEATURE_COLUMNS,
    compute_receiver_features_vectorized,
    engineer_features,
    split_data_chronological,
)


def test_receiver_past_behavior_strictly_excludes_current_row():
    """
    Test that receiver history statistics strictly exclude the current row
    and only count transactions strictly prior to the current step.
    """
    df = pd.DataFrame({
        "step": [10, 11, 12, 15, 20],
        "type": ["TRANSFER", "TRANSFER", "CASH_OUT", "TRANSFER", "CASH_OUT"],
        "amount": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
        "nameOrig": ["User_1", "User_2", "User_3", "User_4", "User_5"],
        "oldbalanceOrg": [10000.0, 9000.0, 7000.0, 4000.0, 5000.0],
        "nameDest": ["Dest_1", "Dest_2", "Dest_1", "Dest_1", "Dest_3"],
        "oldbalanceDest": [0.0, 0.0, 1000.0, 4000.0, 0.0],
    })

    rec_feats = compute_receiver_features_vectorized(df)

    # Row 0: Dest_1 first time
    # Must be 0 count, 0 past volume, 999 time since
    assert rec_feats["dest_past_tx_count"][0] == 0
    assert rec_feats["dest_total_past_amount"][0] == 0.0
    assert rec_feats["dest_tx_last_1h"][0] == 0
    assert rec_feats["dest_tx_last_6h"][0] == 0
    assert rec_feats["dest_time_since_last_tx"][0] == 999.0

    # Row 1: Dest_2 first time
    assert rec_feats["dest_past_tx_count"][1] == 0
    assert rec_feats["dest_total_past_amount"][1] == 0.0

    # Row 2: Dest_1 second transaction at step 12 (previous at step 10, amount 1000.0)
    # Strictly past: 1 transaction of ₹1000.0. Current transaction of ₹3000.0 must NOT be included!
    assert rec_feats["dest_past_tx_count"][2] == 1
    assert np.isclose(rec_feats["dest_total_past_amount"][2], 1000.0)
    assert rec_feats["dest_time_since_last_tx"][2] == 2.0  # 12 - 10 = 2 hours
    assert rec_feats["dest_tx_last_1h"][2] == 0  # 2h ago is not <= 1h
    assert rec_feats["dest_tx_last_6h"][2] == 1  # 2h ago is <= 6h

    # Row 3: Dest_1 third transaction at step 15 (previous at 10 and 12, amounts 1000 and 3000)
    assert rec_feats["dest_past_tx_count"][3] == 2
    assert np.isclose(rec_feats["dest_total_past_amount"][3], 4000.0)
    assert rec_feats["dest_time_since_last_tx"][3] == 3.0  # 15 - 12 = 3 hours
    assert rec_feats["dest_tx_last_1h"][3] == 0  # last was 3h ago
    assert rec_feats["dest_tx_last_6h"][3] == 2  # steps 10 and 12 are both within 6h of 15 (diffs: 5 and 3)

    # Row 4: Dest_3 first time
    assert rec_feats["dest_past_tx_count"][4] == 0


def test_receiver_features_within_step_order_independent():
    """
    Test that shuffling rows within the same hourly step produces 100% identical feature values.
    Transactions within the same step must NEVER observe each other.
    """
    df = pd.DataFrame({
        "tx_id": [1, 2, 3, 4, 5, 6],
        "step": [10, 10, 10, 11, 11, 11],
        "type": ["TRANSFER"] * 6,
        "amount": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0],
        "nameDest": ["Dest_A", "Dest_A", "Dest_B", "Dest_A", "Dest_A", "Dest_B"],
    })

    # Forward order
    res1 = compute_receiver_features_vectorized(df)

    # Shuffle order within each step: step 10 -> [2, 0, 1], step 11 -> [4, 5, 3]
    df_shuffled = df.iloc[[2, 0, 1, 4, 5, 3]].reset_index(drop=True)
    res2 = compute_receiver_features_vectorized(df_shuffled)

    # Convert to DataFrames indexed by tx_id
    df_out1 = pd.DataFrame(res1, index=df["tx_id"])
    df_out2 = pd.DataFrame(res2, index=df_shuffled["tx_id"]).sort_index()

    for col in df_out1.columns:
        assert np.allclose(df_out1[col].sort_index(), df_out2[col].sort_index()), f"Column {col} differs after within-step shuffle!"

    # In step 10, tx 1 and 2 to Dest_A must BOTH have past_count = 0 (neither observes the other)
    assert res1["dest_past_tx_count"][0] == 0
    assert res1["dest_past_tx_count"][1] == 0

    # In step 11, tx 4 and 5 to Dest_A must BOTH see past_count = 2 (sum from step 10, neither sees the other in step 11)
    assert res1["dest_past_tx_count"][3] == 2
    assert res1["dest_past_tx_count"][4] == 2
    assert np.isclose(res1["dest_total_past_amount"][3], 3000.0)
    assert np.isclose(res1["dest_total_past_amount"][4], 3000.0)


def test_no_post_transaction_balance_leakage_in_features():
    """Verify post-transaction balance columns are completely removed from model feature columns."""
    forbidden = ["newbalanceOrig", "newbalanceDest", "orig_balance_diff_error", "dest_balance_diff_error"]
    for col in forbidden:
        assert col not in FEATURE_COLUMNS, f"Forbidden leakage column '{col}' found in FEATURE_COLUMNS"

    assert len(FEATURE_COLUMNS) == 23


def test_live_scoring_code_path_schema_matches_batch():
    """
    Test that single dictionary live scoring produces the exact same columns,
    order, and dtypes as the batch pipeline (zero training/serving mismatch).
    """
    single_tx = {
        "step": 42,
        "type": "TRANSFER",
        "amount": 250000.0,
        "oldbalanceOrg": 250000.0,
        "oldbalanceDest": 0.0,
        "nameDest": "C12345678",
    }

    live_df, _ = engineer_features(single_tx)

    # Feature columns must match exactly
    assert list(live_df.columns) == FEATURE_COLUMNS
    assert len(live_df) == 1
    assert live_df["orig_emptied"].iloc[0] == 1
    assert live_df["type_TRANSFER"].iloc[0] == 1
    assert live_df["type_CASH_OUT"].iloc[0] == 0
    assert live_df["dest_balance_zero"].iloc[0] == 1
    assert live_df["is_dest_merchant"].iloc[0] == 0

    # Explicit test: starting balance of 0 must NOT be marked as emptied
    zero_bal_tx = {
        "step": 42,
        "type": "TRANSFER",
        "amount": 250000.0,
        "oldbalanceOrg": 0.0,
        "oldbalanceDest": 0.0,
        "nameDest": "C12345678",
    }
    zero_df, _ = engineer_features(zero_bal_tx)
    assert zero_df["orig_emptied"].iloc[0] == 0, "orig_emptied must be 0 when oldbalanceOrg is 0.0"


def test_chronological_split_properties():
    """Test that chronological split preserves time ordering with no overlap."""
    df = pd.DataFrame({
        "step": list(range(1, 101)),
        "isFraud": [0] * 95 + [1] * 5,
        "amount": [100.0] * 100,
        "type": ["PAYMENT"] * 100,
        "nameOrig": [f"C{i}" for i in range(100)],
        "oldbalanceOrg": [100.0] * 100,
        "nameDest": [f"M{i}" for i in range(100)],
        "oldbalanceDest": [0.0] * 100,
    })

    train, val, test = split_data_chronological(df, step_train_cutoff=70, step_val_cutoff=85)
    assert train["step"].max() <= 70
    assert val["step"].min() > 70
    assert val["step"].max() <= 85
    assert test["step"].min() > 85
    assert len(train) + len(val) + len(test) == 100
