from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd

from ml.config import (
    CHANNELS,
    PAYSIM_RAW_PATH,
    PROCESSED_DATA_DIR,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
)
from ml.data import load_raw_data, PAYSIM_DTYPES
from ml.features import engineer_features, split_data_chronological
from ml.predict import predict as paysim_predict

CHANNEL_INFO = CHANNELS["transfer_wallet"]


def load(data_path: Path | str | None = None, sample_size: int | None = None) -> pd.DataFrame:
    return load_raw_data(data_path=data_path or CHANNEL_INFO["dataset_path"], sample_size=sample_size)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    return engineer_features(df)


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return split_data_chronological(df)


def train(**kwargs: Any) -> dict[str, Any]:
    from ml.train import run_training
    return run_training(**kwargs)


def evaluate(**kwargs: Any) -> dict[str, Any]:
    from ml.evaluate import run_full_evaluation
    return run_full_evaluation(**kwargs)


def predict(
    payload: dict[str, Any],
    top_k: int = 3,
    history_store: Any | None = None,
    commit_to_store: bool = False,
) -> dict[str, Any]:
    res = paysim_predict(
        payload,
        top_k=top_k,
        history_store=history_store,
        commit_to_store=commit_to_store,
    )
    res["channel"] = "transfer_wallet"
    res["data_source"] = CHANNEL_INFO["data_source"]
    return res

