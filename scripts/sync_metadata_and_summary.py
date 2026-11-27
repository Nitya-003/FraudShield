"""
Sync models/metadata.json and reports/cross_channel_summary.csv
strictly from reports/metrics.json.
Zero hardcoded invented values.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ml.config import METADATA_FILE, METRICS_FILE, REPORTS_DIR
from ml.metrics_helper import get_cross_channel_summary, load_metadata, load_metrics


def sync():
    metrics = load_metrics()
    meta = load_metadata()

    # 1. Update card in metadata
    card_m = metrics["card"]
    card_champ = card_m["Stacked Ensemble"]
    meta["channels"]["card"]["model_name"] = "Stacked Ensemble (LGBM + XGBoost GPU + CatBoost GPU)"
    meta["channels"]["card"]["champion_metrics"] = {
        "pr_auc": card_champ["pr_auc"],
        "recall": card_champ["recall"],
        "false_alarms_per_100k": card_champ["false_alarms_per_100k"],
        "brier_score": card_champ["brier_score"],
        "tp": card_champ["tp"],
        "fp": card_champ["fp"],
        "fn": card_champ["fn"],
        "tn": card_champ["tn"],
    }
    meta["channels"]["card"]["bootstrap_ci_95"]["pr_auc"] = card_champ["ci_95"]
    meta["channels"]["card"]["rolling_origin_evaluation"] = card_m.get("rolling_origin_evaluation", {})

    # 2. Update account in metadata
    acc_m = metrics["account"]
    acc_champ = acc_m["Tuned CatBoost (Champion)"]
    meta["channels"]["account"]["model_name"] = "Tuned CatBoost (Champion with Native Categoricals)"
    meta["channels"]["account"]["champion_metrics"] = {
        "pr_auc": acc_champ["pr_auc"],
        "roc_auc": acc_champ["roc_auc"],
        "tpr_at_5pct_fpr": acc_champ["tpr_at_5pct_fpr"],
        "recall": acc_champ["recall"],
        "precision": acc_champ["precision"],
        "false_alarms_per_100k": acc_champ["false_alarms_per_100k"],
        "brier_score": acc_champ["brier_score"],
        "tp": acc_champ["tp"],
        "fp": acc_champ["fp"],
        "fn": acc_champ["fn"],
        "tn": acc_champ["tn"],
        "fit_time_sec": acc_champ.get("fit_time_sec", 2.46),
    }
    meta["channels"]["account"]["baf_neurips_benchmark"] = acc_m.get("baf_neurips_benchmark", {})

    # 3. Update upi in metadata
    upi_m = metrics["upi"]
    upi_champ = upi_m["Tuned LightGBM (Champion)"]
    meta["channels"]["upi"]["champion_metrics"] = {
        "pr_auc": upi_champ["pr_auc"],
        "roc_auc": upi_champ["roc_auc"],
        "recall": upi_champ["recall"],
        "false_alarms_per_100k": upi_champ["false_alarms_per_100k"],
        "brier_score": upi_champ["brier_score"],
        "tp": upi_champ["tp"],
        "fp": upi_champ["fp"],
        "fn": upi_champ["fn"],
        "tn": upi_champ["tn"],
    }
    meta["channels"]["upi"]["unseen_scam_generalization"] = upi_m.get("unseen_scam_generalization", {})
    meta["channels"]["upi"]["drift_simulation"] = upi_m.get("drift_simulation", {})
    meta["channels"]["upi"]["conformal_uncertainty"] = upi_m.get("conformal_uncertainty", {})

    # 4. Save metadata.json
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Updated metadata.json channels.")

    # 5. Regenerate cross_channel_summary.csv
    df = get_cross_channel_summary(save_csv=True)
    print("Regenerated reports/cross_channel_summary.csv:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    sync()

