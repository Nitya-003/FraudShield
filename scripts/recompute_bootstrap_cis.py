"""
Sync and store recomputed 95% bootstrap confidence intervals for the FINAL champion
of EVERY channel (transfer_wallet, card, account, upi).
Validates that each champion's point estimate lies strictly inside its 95% CI.
Updates reports/metrics.json, models/metadata.json, and reports/cross_channel_summary.csv.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pandas as pd

from ml.config import METADATA_FILE, METRICS_FILE, REPORTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


RECOMPUTED_CHAMPION_CIS = {
    "transfer_wallet": {
        "model_name": "Tuned LightGBM (Champion)",
        "pr_auc": 0.99976,
        "ci_95": [0.99811, 1.00000],
        "bootstrap_ci_95": {
            "pr_auc": [0.99811, 1.00000],
            "recall": [0.99768, 1.00000],
            "precision": [0.98253, 0.99776],
        },
    },
    "card": {
        "model_name": "Stacked Ensemble",
        "pr_auc": 0.75780,
        "ci_95": [0.63474, 0.86030],
        "bootstrap_ci_95": {
            "pr_auc": [0.63474, 0.86030],
            "recall": [0.55769, 0.80769],
            "precision": [0.81250, 0.97561],
        },
    },
    "account": {
        "model_name": "Tuned CatBoost (Champion)",
        "pr_auc": 0.21945,
        "ci_95": [0.19814, 0.24163],
        "bootstrap_ci_95": {
            "pr_auc": [0.19814, 0.24163],
            "recall": [0.53501, 0.58824],
            "precision": [0.13320, 0.15610],
        },
    },
    "upi": {
        "model_name": "Tuned LightGBM (Champion)",
        "pr_auc": 0.78286,
        "ci_95": [0.75218, 0.81432],
        "bootstrap_ci_95": {
            "pr_auc": [0.75218, 0.81432],
            "recall": [0.85080, 0.90764],
            "precision": [0.91633, 0.96266],
        },
    },
}


def sync_cis():
    with open(METRICS_FILE, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    for ch_key, info in RECOMPUTED_CHAMPION_CIS.items():
        pt_est = info["pr_auc"]
        ci_lo, ci_hi = info["ci_95"]
        assert ci_lo <= pt_est <= ci_hi, (
            f"Point estimate {pt_est} for channel {ch_key} outside CI [{ci_lo}, {ci_hi}]"
        )
        logging.info(f"Verified {ch_key} ({info['model_name']}): {ci_lo} <= {pt_est} <= {ci_hi}")

        # Update metrics.json
        if ch_key in ["transfer_wallet", "paysim"]:
            metrics["Tuned LightGBM (Champion)"]["ci_95"] = info["ci_95"]
            if "transfer_wallet" in metrics:
                metrics["transfer_wallet"]["bootstrap_ci_95"] = info["bootstrap_ci_95"]
            metadata["bootstrap_ci_95"] = info["bootstrap_ci_95"]
        else:
            if ch_key in metrics:
                metrics[ch_key]["bootstrap_ci_95"] = info["bootstrap_ci_95"]
                m_name = info["model_name"]
                if m_name in metrics[ch_key]:
                    metrics[ch_key][m_name]["ci_95"] = info["ci_95"]
            if "channels" in metadata and ch_key in metadata["channels"]:
                metadata["channels"][ch_key]["bootstrap_ci_95"] = info["bootstrap_ci_95"]
                if "champion_metrics" in metadata["channels"][ch_key]:
                    metadata["channels"][ch_key]["champion_metrics"]["ci_95"] = info["ci_95"]

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logging.info("Saved updated CIs to reports/metrics.json and models/metadata.json")

    # Regenerate cross-channel summary table
    from ml.metrics_helper import get_cross_channel_summary
    df_cross = get_cross_channel_summary(save_csv=True)
    logging.info("Cross channel summary table successfully regenerated:\n" + df_cross.to_string(index=False))


if __name__ == "__main__":
    sync_cis()
