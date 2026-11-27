"""
Tests for Phase 4 evaluation artifacts, figures, and scored sample.
"""

import json
from pathlib import Path
import pandas as pd
import pytest

from ml.config import FIGURES_DIR, METADATA_FILE, TEST_SAMPLE_PATH


def test_evaluation_figures_exist():
    expected_figures = [
        "06_test_precision_recall_curves.png",
        "07_champion_confusion_matrix.png",
        "08_model_comparison_metrics.png",
        "09_feature_importance.png",
    ]
    for fig_name in expected_figures:
        p = FIGURES_DIR / fig_name
        assert p.exists(), f"Missing figure: {fig_name}"
        assert p.stat().st_size > 1000, f"Figure {fig_name} is empty or corrupted"


def test_test_scored_sample_dataset():
    assert TEST_SAMPLE_PATH.exists(), f"Missing {TEST_SAMPLE_PATH}"
    df = pd.read_parquet(TEST_SAMPLE_PATH)

    assert len(df) >= 1000
    for col in ["risk_score", "band", "top_reasons", "amount", "type", "isFraud"]:
        assert col in df.columns, f"Missing column '{col}' in scored sample"

    assert set(df["band"].unique()).issubset({"Low", "Medium", "High"})
    assert df["risk_score"].min() >= 0.0
    assert df["risk_score"].max() <= 1.0


def test_metadata_contains_test_metrics():
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert "test_metrics" in meta
    assert meta["test_metrics"]["pr_auc"] > 0.90
    assert "risk_band_economics_inr" in meta
    assert "calibrated_thresholds" in meta

