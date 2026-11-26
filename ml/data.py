from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd

from ml.config import (
    ACTIVE_DATA_PATH,
    PAYSIM_RAW_PATH,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    STEP_TRAIN_MAX,
    STEP_VAL_MAX,
    SYNTHETIC_RAW_PATH,
    TARGET_COL,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
)
from ml.features import (
    FEATURE_COLUMNS,
    compute_receiver_features_vectorized,
    split_data_chronological,
)

PAYSIM_DTYPES = {
    "step": "int32",
    "type": "category",
    "amount": "float32",
    "nameOrig": "string",
    "oldbalanceOrg": "float32",
    "newbalanceOrig": "float32",
    "nameDest": "string",
    "oldbalanceDest": "float32",
    "newbalanceDest": "float32",
    "isFraud": "int8",
    "isFlaggedFraud": "int8",
}


def get_data_filepath(override_path: Path | str | None = None) -> Path:
    if override_path:
        p = Path(override_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Requested dataset path does not exist: {override_path}")

    if PAYSIM_RAW_PATH.exists():
        return PAYSIM_RAW_PATH
    if SYNTHETIC_RAW_PATH.exists():
        return SYNTHETIC_RAW_PATH
    raise FileNotFoundError(
        f"Neither {PAYSIM_RAW_PATH} nor {SYNTHETIC_RAW_PATH} was found. "
        "Run `python -m ml.synthetic` first or provide paysim.csv."
    )


def load_raw_data(data_path: Path | str | None = None, sample_size: int | None = None) -> pd.DataFrame:
    file_path = get_data_filepath(data_path)
    print(f"Loading dataset from: {file_path}")
    if sample_size is not None:
        df = pd.read_csv(file_path, dtype=PAYSIM_DTYPES, nrows=sample_size)
    else:
        df = pd.read_csv(file_path, dtype=PAYSIM_DTYPES)
    print(f"Loaded {len(df):,} transactions. Frauds: {df[TARGET_COL].sum():,} ({df[TARGET_COL].mean()*100:.4f}%).")
    return df


def check_data_quality(df: pd.DataFrame) -> dict:
    return {
        "total_records": len(df),
        "total_nulls": int(df.isnull().sum().sum()),
        "null_details": df.isnull().sum().to_dict(),
        "negative_amounts": int((df["amount"] < 0).sum()) if "amount" in df else 0,
        "fraud_count": int(df["isFraud"].sum()) if "isFraud" in df else 0,
        "fraud_by_type": df.groupby("type", observed=False)["isFraud"].sum().to_dict() if ("type" in df and "isFraud" in df) else {},
    }


def build_and_save_processed_data(
    data_path: Path | str | None = None,
    save_dir: Path = PROCESSED_DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    save_dir.mkdir(parents=True, exist_ok=True)
    df = load_raw_data(data_path)

    df = df.sort_values(by=["step"]).reset_index(drop=True)

    print("Computing vectorised receiver history features over full timeline...")
    receiver_feats = compute_receiver_features_vectorized(df)
    for col_name, arr in receiver_feats.items():
        df[col_name] = arr

    print("Computing pre-transaction features...")
    amt = df["amount"].to_numpy(dtype=np.float32)
    old_org = df["oldbalanceOrg"].to_numpy(dtype=np.float32)
    old_dest = df["oldbalanceDest"].to_numpy(dtype=np.float32)
    step_arr = df["step"].to_numpy(dtype=np.int32)
    hour = (step_arr % 24).astype(np.int8)

    df["hour_of_day"] = hour
    df["sin_hour"] = np.sin(2.0 * np.pi * hour / 24.0).astype(np.float32)
    df["cos_hour"] = np.cos(2.0 * np.pi * hour / 24.0).astype(np.float32)
    df["is_night"] = ((hour >= 0) & (hour <= 5)).astype(np.int8)
    df["log_amount"] = np.log1p(amt).astype(np.float32)

    df["orig_emptied"] = ((old_org > 0.0) & (amt >= old_org)).astype(np.int8)
    df["amount_to_balance_ratio"] = (amt / np.maximum(old_org, 1.0)).astype(np.float32)
    df["is_dest_merchant"] = df["nameDest"].str.startswith("M").astype(np.int8)
    df["dest_balance_zero"] = (old_dest == 0.0).astype(np.int8)

    for t in ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]:
        df[f"type_{t}"] = (df["type"] == t).astype(np.int8)

    save_columns = FEATURE_COLUMNS + [TARGET_COL, "step", "type", "nameOrig", "nameDest"]

    train_df, val_df, test_df = split_data_chronological(df)
    del df

    train_df = train_df[save_columns].copy()
    val_df = val_df[save_columns].copy()
    test_df = test_df[save_columns].copy()

    fraud_train = train_df[train_df[TARGET_COL] == 1]
    legit_train = train_df[train_df[TARGET_COL] == 0]
    n_sample_legit = min(len(legit_train), len(fraud_train) * 10)
    sampled_legit = legit_train.sample(n=n_sample_legit, random_state=RANDOM_STATE)
    train_undersampled = pd.concat([fraud_train, sampled_legit], ignore_index=True).sort_values("step").reset_index(drop=True)

    print(f"Saving parquets to {save_dir}...")
    train_df.to_parquet(save_dir / "train.parquet", compression="snappy", index=False)
    train_undersampled.to_parquet(save_dir / "train_undersampled.parquet", compression="snappy", index=False)
    val_df.to_parquet(save_dir / "val.parquet", compression="snappy", index=False)
    test_df.to_parquet(save_dir / "test.parquet", compression="snappy", index=False)

    print("All processed parquets successfully generated!")
    return train_df, val_df, test_df


if __name__ == "__main__":
    build_and_save_processed_data()
