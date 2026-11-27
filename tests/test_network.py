"""
Unit tests for Network (Mule & Ring) Graph Features in FraudShield.

Validates Task 2 requirements:
- Strictly past-only causality: future steps cannot leak into past or present.
- Order-independence within hourly steps: within-step row shuffling produces identical features.
- Parity between offline batch pipeline and real-time single-transaction scoring.
- Exact detection of mule transit velocity (in-to-out gap) and 2-hop flows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.network import (
    NETWORK_FEATURE_COLUMNS,
    NetworkGraphTracker,
    compute_network_features,
)


def test_network_features_strictly_past_only():
    """Verify future transactions cannot leak into past or present steps."""
    df_base = pd.DataFrame({
        "step": [10, 11, 12],
        "nameOrig": ["User_A", "User_B", "User_C"],
        "nameDest": ["Mule_M", "Mule_M", "Cash_Out"],
    })
    res_base = compute_network_features(df_base)

    # Now add future transactions at step 20
    df_future = pd.DataFrame({
        "step": [10, 11, 12, 20, 20],
        "nameOrig": ["User_A", "User_B", "User_C", "Mule_M", "User_Z"],
        "nameDest": ["Mule_M", "Mule_M", "Cash_Out", "Cash_Out", "Mule_M"],
    })
    res_future = compute_network_features(df_future)

    # First 3 rows must be mathematically identical
    for col in NETWORK_FEATURE_COLUMNS:
        np.testing.assert_allclose(
            res_base[col],
            res_future[col][:3],
            err_msg=f"Column {col} was corrupted by future transactions!",
        )


def test_network_features_within_step_order_invariance():
    """Verify shuffling rows within the same step produces 100% identical feature values."""
    df = pd.DataFrame({
        "tx_id": [1, 2, 3, 4, 5, 6],
        "step": [10, 10, 10, 11, 11, 11],
        "nameOrig": ["User_A", "User_B", "User_C", "User_D", "User_E", "Mule_M"],
        "nameDest": ["Mule_M", "Mule_M", "Merchant_X", "Mule_M", "Merchant_X", "Cash_Out"],
    })

    # Forward evaluation
    res_orig = compute_network_features(df)

    # Shuffle order within each hourly step: step 10 -> [2, 0, 1], step 11 -> [4, 5, 3]
    df_shuffled = df.iloc[[2, 0, 1, 4, 5, 3]].reset_index(drop=True)
    res_shuffled = compute_network_features(df_shuffled)

    df_out1 = pd.DataFrame(res_orig, index=df["tx_id"]).sort_index()
    df_out2 = pd.DataFrame(res_shuffled, index=df_shuffled["tx_id"]).sort_index()

    for col in NETWORK_FEATURE_COLUMNS:
        np.testing.assert_allclose(
            df_out1[col].to_numpy(),
            df_out2[col].to_numpy(),
            err_msg=f"Within-step shuffling changed feature '{col}'!",
        )


def test_network_features_batch_vs_single_parity():
    """Verify exact zero-skew parity between batch computation and single-transaction scoring."""
    df = pd.DataFrame({
        "step": [5, 6, 8, 10, 12],
        "nameOrig": ["Alice", "Bob", "Alice", "Charlie", "Mule_B"],
        "nameDest": ["Mule_B", "Mule_B", "Mule_B", "Merchant", "Dan"],
    })

    # 1. Batch extraction
    batch_res = compute_network_features(df)

    # 2. Sequential tracker extraction (simulating live stream with step buffer)
    tracker = NetworkGraphTracker(window_hours=24)
    single_records = []

    # Process step by step matching step buffer
    steps = df["step"].tolist()
    i = 0
    n = len(df)
    while i < n:
        cur_step = steps[i]
        j = i
        while j < n and steps[j] == cur_step:
            j += 1

        # READ PHASE
        for k in range(i, j):
            row_dict = df.iloc[k].to_dict()
            feats = tracker.extract_single(row_dict, cur_step=cur_step)
            single_records.append(feats)

        # WRITE PHASE
        for k in range(i, j):
            row_dict = df.iloc[k].to_dict()
            tracker.update_single(row_dict, cur_step=cur_step)

        i = j

    df_single = pd.DataFrame(single_records)

    for col in NETWORK_FEATURE_COLUMNS:
        np.testing.assert_allclose(
            batch_res[col],
            df_single[col].to_numpy(),
            err_msg=f"Batch vs Single parity failed on feature '{col}'!",
        )


def test_mule_in_to_out_gap_and_flow():
    """Test mule pass-through velocity and 2-hop flow detection."""
    df = pd.DataFrame({
        "step": [10, 10, 13, 14],
        "nameOrig": ["Victim_1", "Victim_2", "Mule_X", "Normal_User"],
        "nameDest": ["Mule_X", "Mule_X", "Cash_Out_Target", "Merchant_A"],
    })

    res = compute_network_features(df)

    # Step 10: Mule_X receives from Victim_1 and Victim_2
    # In step 10, Mule_X has 0 past distinct senders (step-buffered)
    assert res["dest_distinct_senders_total"][0] == 0
    assert res["dest_distinct_senders_total"][1] == 0

    # Step 13: Mule_X sends money out to Cash_Out_Target
    # Mule_X received funds at step 10. Gap must be exactly 13 - 10 = 3.0 hours!
    assert res["orig_in_to_out_gap"][2] == 3.0
    # Mule_X received incoming funds in past 24h, so 2-hop flow must be 1
    assert res["is_2hop_flow"][2] == 1
    # Cash_Out_Target has never been seen
    assert res["dest_distinct_senders_total"][2] == 0

    # Step 14: Normal_User sending to Merchant_A
    # Normal_User never received funds: in_to_out_gap = 999.0
    assert res["orig_in_to_out_gap"][3] == 999.0


def test_repeat_pair_count():
    """Verify repeat sender-receiver pair counter increments across distinct steps."""
    df = pd.DataFrame({
        "step": [1, 2, 3, 3, 5],
        "nameOrig": ["User_A", "User_A", "User_A", "User_B", "User_A"],
        "nameDest": ["User_B", "User_B", "User_B", "User_A", "User_B"],
    })

    res = compute_network_features(df)

    # Row 0: Step 1 (A -> B, first time) -> 0
    assert res["repeat_pair_count"][0] == 0
    # Row 1: Step 2 (A -> B, second time) -> 1
    assert res["repeat_pair_count"][1] == 1
    # Row 2: Step 3 (A -> B, third time) -> 2
    assert res["repeat_pair_count"][2] == 2
    # Row 3: Step 3 (B -> A, first time in this direction) -> 0
    assert res["repeat_pair_count"][3] == 0
    # Row 4: Step 5 (A -> B, fourth time) -> 3
    assert res["repeat_pair_count"][4] == 3


def test_engineer_features_with_network_columns():
    """Verify engineer_features integrates network columns cleanly (23 vs 31 features)."""
    from ml.features import ALL_FEATURE_COLUMNS, FEATURE_COLUMNS, engineer_features

    tx = {
        "step": 42,
        "type": "TRANSFER",
        "amount": 250000.0,
        "oldbalanceOrg": 250000.0,
        "oldbalanceDest": 0.0,
        "nameOrig": "C111111",
        "nameDest": "C222222",
    }

    # Baseline 23 features
    df_base, _ = engineer_features(tx, include_network=False)
    assert len(df_base.columns) == 23
    assert list(df_base.columns) == FEATURE_COLUMNS

    # Extended 31 network features
    df_net, _ = engineer_features(tx, include_network=True)
    assert len(df_net.columns) == 31
    assert list(df_net.columns) == ALL_FEATURE_COLUMNS
    for col in NETWORK_FEATURE_COLUMNS:
        assert col in df_net.columns

