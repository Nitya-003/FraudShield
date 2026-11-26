from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd

from ml.config import (
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    STEP_TRAIN_MAX,
    STEP_VAL_MAX,
    TARGET_COL,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    TRANSACTION_TYPES,
    VAL_DATA_PATH,
)
from ml.network import (
    DEFAULT_NETWORK_VALUES,
    NETWORK_FEATURE_COLUMNS,
    NetworkGraphTracker,
    compute_network_features,
)

FEATURE_COLUMNS = [

    "amount",
    "log_amount",
    "hour_of_day",
    "sin_hour",
    "cos_hour",
    "is_night",
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
    "oldbalanceOrg",
    "oldbalanceDest",
    "orig_emptied",
    "amount_to_balance_ratio",
    "is_dest_merchant",
    "dest_balance_zero",
    "dest_past_tx_count",
    "dest_total_past_amount",
    "dest_time_since_last_tx",
    "dest_tx_last_1h",
    "dest_tx_last_6h",
    "dest_tx_last_24h",
]

ALL_FEATURE_COLUMNS = FEATURE_COLUMNS + NETWORK_FEATURE_COLUMNS


def compute_receiver_features_vectorized(df: pd.DataFrame) -> dict[str, np.ndarray]:
    n = len(df)
    steps = df["step"].to_numpy(dtype=np.int32)
    amounts = df["amount"].to_numpy(dtype=np.float32)
    dest_codes, _ = pd.factorize(df["nameDest"])

    past_count = np.zeros(n, dtype=np.int32)
    past_amount = np.zeros(n, dtype=np.float32)
    time_since = np.full(n, 999.0, dtype=np.float32)
    tx_1h = np.zeros(n, dtype=np.int16)
    tx_6h = np.zeros(n, dtype=np.int16)
    tx_24h = np.zeros(n, dtype=np.int16)

    cum_counts: dict[int, int] = defaultdict(int)
    cum_amounts: dict[int, float] = defaultdict(float)
    last_steps: dict[int, int] = {}
    recent_steps: dict[int, list[int]] = defaultdict(list)

    i = 0
    while i < n:
        cur_step = steps[i]
        j = i
        while j < n and steps[j] == cur_step:
            j += 1

        for k in range(i, j):
            d = dest_codes[k]
            if d in last_steps:
                past_count[k] = cum_counts[d]
                past_amount[k] = cum_amounts[d]
                time_since[k] = cur_step - last_steps[d]

                h = recent_steps[d]
                c1, c6, c24 = 0, 0, 0
                for past_st in h:
                    diff = cur_step - past_st
                    if diff <= 1:
                        c1 += 1
                    if diff <= 6:
                        c6 += 1
                    if diff <= 24:
                        c24 += 1
                tx_1h[k] = c1
                tx_6h[k] = c6
                tx_24h[k] = c24

        for k in range(i, j):
            d = dest_codes[k]
            amt = amounts[k]
            cum_counts[d] += 1
            cum_amounts[d] += amt
            last_steps[d] = cur_step
            recent_steps[d].append(cur_step)

        for k in range(i, j):
            d = dest_codes[k]
            h = recent_steps[d]
            cutoff = cur_step - 24
            idx = 0
            while idx < len(h) and h[idx] < cutoff:
                idx += 1
            if idx > 0:
                del h[:idx]

        i = j

    return {
        "dest_past_tx_count": past_count,
        "dest_total_past_amount": past_amount,
        "dest_time_since_last_tx": time_since,
        "dest_tx_last_1h": tx_1h,
        "dest_tx_last_6h": tx_6h,
        "dest_tx_last_24h": tx_24h,
    }


def engineer_features(
    data: pd.DataFrame | dict,
    is_training: bool = False,
    include_network: bool = False,
    network_tracker: NetworkGraphTracker | None = None,
    history_store: Any | None = None,
) -> tuple[pd.DataFrame, pd.Series | None]:

    if isinstance(data, dict):
        df = pd.DataFrame([data])
        if history_store is not None:
            live_feats = history_store.get_all_live_features(data)
            for k, v in live_feats.items():
                if k not in data:  
                    df[k] = v
    else:
        df = data.copy()

    amt = df["amount"].astype(np.float32)
    step = df["step"].astype(np.int32) if "step" in df.columns else pd.Series(12, index=df.index, dtype=np.int32)
    hour = (step % 24).astype(np.int8)

    df["amount"] = amt
    df["log_amount"] = np.log1p(amt).astype(np.float32)
    df["hour_of_day"] = hour
    df["sin_hour"] = np.sin(2.0 * np.pi * hour / 24.0).astype(np.float32)
    df["cos_hour"] = np.cos(2.0 * np.pi * hour / 24.0).astype(np.float32)
    df["is_night"] = ((hour >= 0) & (hour <= 5)).astype(np.int8)

    for t in TRANSACTION_TYPES:
        col = f"type_{t}"
        if "type" in df.columns:
            df[col] = (df["type"] == t).astype(np.int8)
        elif col not in df.columns:
            df[col] = np.int8(0)

    old_org = df["oldbalanceOrg"].astype(np.float32)
    old_dest = df["oldbalanceDest"].astype(np.float32) if "oldbalanceDest" in df.columns else pd.Series(0.0, index=df.index, dtype=np.float32)

    df["oldbalanceOrg"] = old_org
    df["oldbalanceDest"] = old_dest
    df["orig_emptied"] = ((old_org > 0.0) & (amt >= old_org)).astype(np.int8)
    df["amount_to_balance_ratio"] = (amt / np.maximum(old_org, 1.0)).astype(np.float32)
    
    name_dest = df["nameDest"].astype(str) if "nameDest" in df.columns else pd.Series("", index=df.index)
    df["is_dest_merchant"] = name_dest.str.startswith("M").astype(np.int8)
    df["dest_balance_zero"] = (old_dest == 0.0).astype(np.int8)

    if "dest_past_tx_count" not in df.columns:
        df["dest_past_tx_count"] = np.int32(0)
    else:
        df["dest_past_tx_count"] = df["dest_past_tx_count"].astype(np.int32)

    if "dest_total_past_amount" not in df.columns:
        df["dest_total_past_amount"] = np.float32(0.0)
    else:
        df["dest_total_past_amount"] = df["dest_total_past_amount"].astype(np.float32)

    if "dest_time_since_last_tx" not in df.columns:
        df["dest_time_since_last_tx"] = np.float32(999.0)
    else:
        df["dest_time_since_last_tx"] = df["dest_time_since_last_tx"].astype(np.float32)

    for w, dt in [("dest_tx_last_1h", np.int16), ("dest_tx_last_6h", np.int16), ("dest_tx_last_24h", np.int16)]:
        if w not in df.columns:
            df[w] = dt(0)
        else:
            df[w] = df[w].astype(dt)

    if include_network:
        for net_col, default_val in DEFAULT_NETWORK_VALUES.items():
            if net_col not in df.columns:
                if isinstance(data, dict) and network_tracker is not None:
                    net_dict = network_tracker.extract_single(data)
                    df[net_col] = net_dict[net_col]
                else:
                    df[net_col] = default_val

    chosen_cols = ALL_FEATURE_COLUMNS if include_network else FEATURE_COLUMNS
    X = df[chosen_cols].copy()

    y = None
    if TARGET_COL in df.columns:
        y = df[TARGET_COL].astype(np.int8)

    return X, y


def split_data_chronological(
    df: pd.DataFrame,
    step_train_cutoff: int = STEP_TRAIN_MAX,
    step_val_cutoff: int = STEP_VAL_MAX,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    train_df = df[df["step"] <= step_train_cutoff].copy().reset_index(drop=True)
    val_df = (
        df[(df["step"] > step_train_cutoff) & (df["step"] <= step_val_cutoff)]
        .copy()
        .reset_index(drop=True)
    )
    test_df = df[df["step"] > step_val_cutoff].copy().reset_index(drop=True)

    print(
        f"Chronological Split (Natural Prevalence):\n"
        f"  Train: steps 1..{step_train_cutoff} ({len(train_df):,} rows, {train_df[TARGET_COL].sum():,} frauds - {train_df[TARGET_COL].mean()*100:.4f}%)\n"
        f"  Val  : steps {step_train_cutoff+1}..{step_val_cutoff} ({len(val_df):,} rows, {val_df[TARGET_COL].sum():,} frauds - {val_df[TARGET_COL].mean()*100:.4f}%)\n"
        f"  Test : steps {step_val_cutoff+1}..743 ({len(test_df):,} rows, {test_df[TARGET_COL].sum():,} frauds - {test_df[TARGET_COL].mean()*100:.4f}%)"
    )

    return train_df, val_df, test_df
