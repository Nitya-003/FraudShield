"""
Tests for synthetic PaySim data generator.
"""

import pandas as pd
from ml.config import SYNTHETIC_RAW_PATH
from ml.synthetic import generate_synthetic_paysim


def test_synthetic_generator_output(tmp_path):
    out_file = tmp_path / "test_synth.csv"
    df = generate_synthetic_paysim(n_samples=5_000, fraud_rate=0.004, seed=42, output_path=out_file)

    assert out_file.exists()
    assert len(df) == 5_000
    assert df["isFraud"].sum() == 20
    assert df.isnull().sum().sum() == 0

    expected_cols = [
        "step", "type", "amount", "nameOrig", "oldbalanceOrg",
        "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest",
        "isFraud", "isFlaggedFraud"
    ]
    assert list(df.columns) == expected_cols

    fraud_types = set(df[df["isFraud"] == 1]["type"].unique())
    assert fraud_types.issubset({"TRANSFER", "CASH_OUT"})

