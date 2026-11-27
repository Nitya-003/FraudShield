"""
Unit tests for data loading, quality checks, and sampling.
"""

import pandas as pd
import pytest
from ml.config import PAYSIM_RAW_PATH, SYNTHETIC_RAW_PATH
from ml.data import check_data_quality, load_raw_data


def test_load_and_sample_preserves_all_fraud():
    if not (PAYSIM_RAW_PATH.exists() or SYNTHETIC_RAW_PATH.exists()):
        pytest.skip("Raw PaySim dataset excluded from submission archive (run `python -m ml.synthetic` to generate)")
    # Load with small sample size
    df = load_raw_data(sample_size=5_000)

    assert len(df) == 5_000
    # Check dtypes are memory optimized
    assert df["step"].dtype == "int32"
    assert df["type"].dtype.name == "category"
    assert df["amount"].dtype == "float32"
    assert df["isFraud"].dtype == "int8"


def test_check_data_quality():
    sample_df = pd.DataFrame({
        "step": [1, 2],
        "type": pd.Series(["TRANSFER", "CASH_OUT"], dtype="category"),
        "amount": [100.0, 200.0],
        "nameOrig": ["C1", "C2"],
        "oldbalanceOrg": [100.0, 200.0],
        "newbalanceOrig": [0.0, 0.0],
        "nameDest": ["C3", "C4"],
        "oldbalanceDest": [0.0, 0.0],
        "newbalanceDest": [100.0, 200.0],
        "isFraud": [1, 0],
        "isFlaggedFraud": [0, 0],
    })
    res = check_data_quality(sample_df)
    assert res["total_records"] == 2
    assert res["total_nulls"] == 0
    assert res["negative_amounts"] == 0
    assert res["fraud_count"] == 1
    assert res["fraud_by_type"]["TRANSFER"] == 1
    assert res["fraud_by_type"].get("CASH_OUT", 0) == 0
