"""
Precompute and attach 8 strictly past-only network graph features to processed parquets.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd

from ml.config import (
    PAYSIM_RAW_PATH,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    STEP_TRAIN_MAX,
    STEP_VAL_MAX,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
)
from ml.network import NETWORK_FEATURE_COLUMNS, compute_network_features


def main():
    print("=" * 70)
    print("  FRAUDSHIELD: PRECOMPUTING LEAK-FREE NETWORK GRAPH FEATURES")
    print("=" * 70)

    t0 = time.time()
    print(f"Loading full timeline from {PAYSIM_RAW_PATH}...")
    raw_df = pd.read_csv(
        PAYSIM_RAW_PATH,
        usecols=["step", "nameOrig", "nameDest"],
        dtype={"step": "int32", "nameOrig": "string", "nameDest": "string"},
    )
    print(f"Loaded {len(raw_df):,} transactions in {time.time()-t0:.2f}s.")

    # Chronological sort guarantee
    raw_df = raw_df.sort_values("step").reset_index(drop=True)

    print("Computing vectorised network graph features...")
    t1 = time.time()
    net_features = compute_network_features(raw_df)
    print(f"Graph computation finished in {time.time()-t1:.2f}s.")

    for col, arr in net_features.items():
        raw_df[col] = arr

    # Split masks
    train_mask = raw_df["step"] <= STEP_TRAIN_MAX
    val_mask = (raw_df["step"] > STEP_TRAIN_MAX) & (raw_df["step"] <= STEP_VAL_MAX)
    test_mask = raw_df["step"] > STEP_VAL_MAX

    print(f"Split distributions: Train={train_mask.sum():,}, Val={val_mask.sum():,}, Test={test_mask.sum():,}")

    # Update train.parquet
    print("Updating train.parquet...")
    train_df = pd.read_parquet(TRAIN_DATA_PATH)
    for col in NETWORK_FEATURE_COLUMNS:
        train_df[col] = raw_df.loc[train_mask, col].to_numpy()
    train_df.to_parquet(TRAIN_DATA_PATH, compression="snappy", index=False)

    # Update val.parquet
    print("Updating val.parquet...")
    val_df = pd.read_parquet(VAL_DATA_PATH)
    for col in NETWORK_FEATURE_COLUMNS:
        val_df[col] = raw_df.loc[val_mask, col].to_numpy()
    val_df.to_parquet(VAL_DATA_PATH, compression="snappy", index=False)

    # Update test.parquet
    print("Updating test.parquet...")
    test_df = pd.read_parquet(TEST_DATA_PATH)
    for col in NETWORK_FEATURE_COLUMNS:
        test_df[col] = raw_df.loc[test_mask, col].to_numpy()
    test_df.to_parquet(TEST_DATA_PATH, compression="snappy", index=False)

    # Update train_undersampled.parquet
    print("Updating train_undersampled.parquet...")
    train_us_df = pd.read_parquet(PROCESSED_DATA_DIR / "train_undersampled.parquet")
    # Match undersampled rows via (nameOrig, nameDest, step) or index
    # We re-generate train_undersampled cleanly from train_df to guarantee exact alignment
    fraud_train = train_df[train_df["isFraud"] == 1]
    legit_train = train_df[train_df["isFraud"] == 0]
    n_sample_legit = min(len(legit_train), len(fraud_train) * 10)
    sampled_legit = legit_train.sample(n=n_sample_legit, random_state=RANDOM_STATE)
    train_us_new = pd.concat([fraud_train, sampled_legit], ignore_index=True).sort_values("step").reset_index(drop=True)
    train_us_new.to_parquet(PROCESSED_DATA_DIR / "train_undersampled.parquet", compression="snappy", index=False)

    print(f"\nAll parquets updated with {len(NETWORK_FEATURE_COLUMNS)} network features!")
    print(f"Total time elapsed: {time.time()-t0:.2f}s.")


if __name__ == "__main__":
    main()

