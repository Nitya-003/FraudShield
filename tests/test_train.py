"""
Tests for model artifacts, metadata integrity, and inference compatibility.
"""

import json
import joblib
import numpy as np
import pytest

from ml.config import METADATA_FILE, MODEL_FILE
from ml.train import CLEAN_MODEL_FEATURES


def test_champion_model_and_metadata_exist():
    assert MODEL_FILE.exists(), f"Champion model missing at {MODEL_FILE}"
    assert METADATA_FILE.exists(), f"Metadata json missing at {METADATA_FILE}"


def test_metadata_structure_and_keys():
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)

    required_keys = [
        "model_name",
        "champion_strategy",
        "features",
        "feature_count",
        "validation_metrics",
        "thresholds",
        "library_versions",
    ]
    for k in required_keys:
        assert k in meta, f"Missing key '{k}' in metadata.json"

    assert meta["feature_count"] == len(CLEAN_MODEL_FEATURES)
    assert meta["features"] == CLEAN_MODEL_FEATURES
    assert meta["validation_metrics"]["pr_auc"] > 0.85


def test_champion_model_prediction():
    artifact = joblib.load(MODEL_FILE)
    model = artifact["model"] if isinstance(artifact, dict) else artifact
    n_features = len(CLEAN_MODEL_FEATURES)

    # Synthetic batch of 5 records
    dummy_input = np.random.randn(5, n_features).astype(np.float32)
    probs = model.predict_proba(dummy_input)

    assert probs.shape == (5, 2)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_benchmark_baselines_present_and_sound():
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert "benchmarks" in meta
    bench = meta["benchmarks"]
    assert "Rule 1 (Amount > 50k)" in bench
    assert "Rule 2 (Channel & Empty)" in bench
    assert "Rule 3 (Channel & Empty & DestZero)" in bench
    assert "Decision Tree (Depth 3)" in bench
    assert "Tuned LightGBM (Champion)" in bench
    assert bench["Rule 2 (Channel & Empty)"]["fp"] < 200000
    assert bench["Tuned LightGBM (Champion)"]["fp"] < bench["Rule 3 (Channel & Empty & DestZero)"]["fp"]
    assert bench["Tuned LightGBM (Champion)"]["fp"] < 100


