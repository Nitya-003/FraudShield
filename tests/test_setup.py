"""
Smoke tests for FraudShield project setup and configuration.
"""

from pathlib import Path
from ml import config


def test_directories_exist():
    assert config.BASE_DIR.exists()
    assert config.RAW_DATA_DIR.exists()
    assert config.PROCESSED_DATA_DIR.exists()
    assert config.MODELS_DIR.exists()
    assert config.REPORTS_DIR.exists()
    assert config.FIGURES_DIR.exists()
    assert config.DOCS_DIR.exists()


def test_config_constants():
    assert config.RANDOM_STATE == 42
    assert config.TARGET_COL == "isFraud"
    assert config.TRAIN_RATIO + config.VAL_RATIO + config.TEST_RATIO == 1.0
    assert 0.0 < config.DEFAULT_LOW_THRESHOLD < config.DEFAULT_HIGH_THRESHOLD < 1.0

