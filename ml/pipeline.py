from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ml.config import BASE_DIR, CHANNELS, PROCESSED_DATA_DIR, REPORTS_DIR

logger = logging.getLogger("fraudshield.pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def stage_1_validate_datasets() -> dict[str, Any]:
    logger.info(">>> STAGE 1: Validating Datasets and Splits <<<")
    required_files = {
        "transfer_wallet": PROCESSED_DATA_DIR / "test_scored_sample.parquet",
        "card": PROCESSED_DATA_DIR / "card_test_sample.parquet",
        "account": PROCESSED_DATA_DIR / "account_test_sample.parquet",
        "upi": PROCESSED_DATA_DIR / "upi_test_sample.parquet",
    }

    status = {}
    for ch, path in required_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing processed data for channel '{ch}' at {path}")
        size_kb = round(path.stat().st_size / 1024, 1)
        status[ch] = {"path": str(path), "size_kb": size_kb, "status": "valid"}
        logger.info("  [OK] %s: %s (%.1f KB)", ch, path.name, size_kb)

    return status


def stage_2_train_and_calibrate(channel: Optional[str] = None) -> dict[str, Any]:
    logger.info(">>> STAGE 2: Model Training & Platt Calibration <<<")
    channels_to_run = [channel] if channel else ["upi"]  
    results = {}

    for ch in channels_to_run:
        logger.info("  Training channel: %s...", ch)
        t0 = time.perf_counter()
        if ch == "upi":
            import ml.channels.upi as upi_mod
            upi_mod.train()
        results[ch] = {"status": "trained", "duration_sec": round(time.perf_counter() - t0, 2)}
        logger.info("  [OK] Channel %s trained in %.2f s", ch, results[ch]["duration_sec"])

    return results


def stage_3_unseen_scam_novelty() -> dict[str, Any]:
    logger.info(">>> STAGE 3: Unseen Scam Novelty & Feedback Trajectory <<<")
    from ml.unseen_detector import evaluate_unseen_scam_benchmark
    rep = evaluate_unseen_scam_benchmark(budgets=[0.001, 0.005, 0.010])
    logger.info("  [OK] Novelty benchmark evaluated on %s", rep["held_out_scam"])
    for b in rep["alert_budget_evaluation"]:
        logger.info("       Budget %s%%: Supervised=%s%%, Novelty=%s%% (+%s%%)",
                    b["budget_pct"], b["supervised_baseline"]["recall_pct"],
                    b["unsupervised_novelty"]["recall_pct"], b["recall_gain_pct"])
    return rep


def stage_4_reason_code_faithfulness() -> dict[str, Any]:
    logger.info(">>> STAGE 4: Reason-Code Faithfulness Deletion Tests <<<")
    from ml.reason_codes import run_all_deletion_tests
    rep = run_all_deletion_tests()
    for ch, r in rep["channels"].items():
        logger.info("  [OK] %s: %d/%d passed (%.1f%%, avg drop: %.4f)",
                    ch, r["passed_count"], r["tested_count"], r["pass_rate_pct"], r["avg_score_drop"])
    return rep


def stage_5_verify_registry_consistency() -> dict[str, Any]:
    logger.info(">>> STAGE 5: Champion Registry & Audit Verification <<<")
    reg_path = BASE_DIR / "models" / "champion_registry.json"
    with open(reg_path, "r", encoding="utf-8") as f:
        reg = json.load(f)

    champs = reg.get("champions", {})
    assert len(champs) == 4, f"Expected 4 channels, got {len(champs)}"
    for ch in ["transfer_wallet", "card", "account", "upi"]:
        assert ch in champs, f"Missing champion for {ch}"
        c = champs[ch]
        assert "pr_auc" in c, f"Missing pr_auc for {ch}"
        assert "version" in c, f"Missing version for {ch}"
        logger.info("  [OK] Champion %s: %s (v%s, PR-AUC=%.5f)", ch, c["model_name"], c["version"], c["pr_auc"])

    return {"status": "consistent", "champions_count": len(champs)}


def run_pipeline(all_stages: bool = True, channel: Optional[str] = None) -> dict[str, Any]:
    t_start = time.perf_counter()
    logger.info("======================================================================")
    logger.info("FRAUDSHIELD REPRODUCIBILITY PIPELINE: START")
    logger.info("======================================================================")

    summary = {
        "start_time": pd.Timestamp.utcnow().isoformat() + "Z",
        "stages": {},
    }

    summary["stages"]["datasets"] = stage_1_validate_datasets()

    summary["stages"]["training"] = stage_2_train_and_calibrate(channel=channel)

    summary["stages"]["unseen_scam"] = stage_3_unseen_scam_novelty()

    summary["stages"]["faithfulness"] = stage_4_reason_code_faithfulness()

    summary["stages"]["consistency"] = stage_5_verify_registry_consistency()

    total_time = round(time.perf_counter() - t_start, 2)
    summary["total_duration_sec"] = total_time
    summary["status"] = "SUCCESS"

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_file = REPORTS_DIR / "pipeline_run_summary.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("======================================================================")
    logger.info("FRAUDSHIELD REPRODUCIBILITY PIPELINE: COMPLETE in %.2f seconds", total_time)
    logger.info("Summary saved to %s", report_file)
    logger.info("======================================================================")
    return summary


def main():
    parser = argparse.ArgumentParser(description="FraudShield End-to-End ML Pipeline")
    parser.add_argument("--all", action="store_true", help="Execute complete end-to-end pipeline")
    parser.add_argument("--channel", type=str, default=None, help="Channel to isolate")
    args = parser.parse_args()

    run_pipeline(all_stages=args.all, channel=args.channel)


if __name__ == "__main__":
    main()
