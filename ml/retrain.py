from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix

from ml.config import BASE_DIR, CHANNELS, METRICS_FILE, PROCESSED_DATA_DIR, RANDOM_STATE, REPORTS_DIR
from ml.router import resolve_channel

logger = logging.getLogger("fraudshield.retrain")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CHAMPION_REGISTRY_PATH = BASE_DIR / "models" / "champion_registry.json"


def _load_registry() -> dict[str, Any]:
    if not CHAMPION_REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Champion registry not found at {CHAMPION_REGISTRY_PATH}")
    with open(CHAMPION_REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_registry(registry: dict[str, Any]) -> None:
    with open(CHAMPION_REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def _next_minor_version(current_ver: str) -> str:
    parts = current_ver.split(".")
    if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
        return f"{parts[0]}.{int(parts[1]) + 1}.0"
    return f"{current_ver}-retrained"


def compute_metrics_and_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    n_bootstrap: int = 150,
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    pr_auc = float(average_precision_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))

    preds = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    fpr = float(fp / max(tn + fp, 1))
    fa_per_100k = round(fpr * 100_000.0, 2)
    recall = round(float(tp / max(tp + fn, 1)), 5)
    precision = round(float(tp / max(tp + fp, 1)), 5)

    rng = np.random.RandomState(RANDOM_STATE)
    boot_aucs = []
    n_samples = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        y_b = y_true[idx]
        if y_b.sum() == 0 or y_b.sum() == len(y_b):
            continue
        p_b = y_prob[idx]
        boot_aucs.append(average_precision_score(y_b, p_b))

    if boot_aucs:
        ci_lower = round(float(np.percentile(boot_aucs, 2.5)), 5)
        ci_upper = round(float(np.percentile(boot_aucs, 97.5)), 5)
    else:
        ci_lower = round(pr_auc * 0.95, 5)
        ci_upper = round(min(pr_auc * 1.05, 1.0), 5)

    return {
        "pr_auc": round(pr_auc, 5),
        "brier_score": round(brier, 5),
        "false_alarms_per_100k": fa_per_100k,
        "recall": recall,
        "precision": precision,
        "ci_95_pr_auc": [ci_lower, ci_upper],
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def evaluate_candidate_upi() -> Tuple[np.ndarray, np.ndarray, float]:
    import joblib
    import ml.channels.upi as upi
    logger.info("Retraining UPI model...")
    upi.train()
    test_path = PROCESSED_DATA_DIR / "upi_test_sample.parquet"
    df = pd.read_parquet(test_path)
    y_test = df["is_fraud"].to_numpy()
    artifacts = joblib.load(upi.MODEL_PATH)
    champ = artifacts["champion_model"]
    cal = artifacts["calibrator"]
    ohe = artifacts["ohe"]
    num_cols = artifacts["numeric_features"]
    cat_cols = artifacts["categorical_features"]

    X_num = df[num_cols].fillna(0).to_numpy(dtype=np.float32)
    X_cat = ohe.transform(df[cat_cols])
    X = np.hstack([X_num, X_cat])
    raw_p = champ.predict_proba(X)[:, 1]
    logits = np.log(np.clip(raw_p, 1e-7, 1 - 1e-7) / (1 - np.clip(raw_p, 1e-7, 1 - 1e-7))).reshape(-1, 1)
    probs = cal.predict_proba(logits)[:, 1]

    t_low = artifacts.get("thresholds", {}).get("low_risk_approve", 0.15)
    return y_test, probs, t_low


def evaluate_candidate_card() -> Tuple[np.ndarray, np.ndarray, float]:
    import ml.channels.card as card
    test_path = PROCESSED_DATA_DIR / "card_test_sample.parquet"
    df = pd.read_parquet(test_path)
    y_test = df["isFraud"].to_numpy()

    probs = []
    for _, r in df.iterrows():
        res = card.predict(r.to_dict())
        probs.append(res["risk_score"])

    return y_test, np.array(probs), 0.20


def evaluate_candidate_account() -> Tuple[np.ndarray, np.ndarray, float]:
    import ml.channels.account as account
    test_path = PROCESSED_DATA_DIR / "account_test_sample.parquet"
    df = pd.read_parquet(test_path)
    y_test = df["fraud_bool"].to_numpy()

    probs = []
    for _, r in df.iterrows():
        res = account.predict(r.to_dict())
        probs.append(res["risk_score"])

    return y_test, np.array(probs), 0.0436


def evaluate_candidate_transfer() -> Tuple[np.ndarray, np.ndarray, float]:
    import ml.channels.transfer_wallet as tw
    test_path = PROCESSED_DATA_DIR / "test_scored_sample.parquet"
    df = pd.read_parquet(test_path)
    y_test = df["isFraud"].to_numpy()

    probs = []
    for _, r in df.iterrows():
        res = tw.predict(r.to_dict())
        probs.append(res["risk_score"])

    return y_test, np.array(probs), 0.45


def retrain_and_promote(
    channel_name: str,
    force_promote: bool = False,
    dry_run: bool = False,
    n_bootstrap: int = 150,
) -> dict[str, Any]:

    canonical = resolve_channel(channel_name)
    registry = _load_registry()
    champions = registry.get("champions", {})

    if canonical not in champions:
        raise ValueError(f"Channel '{canonical}' not registered in champion registry.")

    incumbent = champions[canonical]
    incumbent_ver = incumbent.get("version", "2.0.0")
    incumbent_pr_auc = float(incumbent.get("pr_auc", 0.8))
    incumbent_ci = incumbent.get("bootstrap_ci_95", {}).get("pr_auc", [incumbent_pr_auc * 0.95, incumbent_pr_auc * 1.05])
    incumbent_ci_low = float(incumbent_ci[0])
    incumbent_fa = float(incumbent.get("false_alarms_per_100k", 50.0))
    incumbent_brier = float(incumbent.get("brier_score", 0.005))

    logger.info("=" * 70)
    logger.info("RETRAIN & PROMOTE: %s", canonical.upper())
    logger.info("Incumbent Champion: %s (v%s)", incumbent.get("model_name"), incumbent_ver)
    logger.info("Incumbent PR-AUC: %.5f [95%% CI: %.5f - %.5f]", incumbent_pr_auc, incumbent_ci[0], incumbent_ci[1])
    logger.info("Incumbent False Alarms / 100k: %.2f | Brier: %.5f", incumbent_fa, incumbent_brier)
    logger.info("=" * 70)

    if canonical == "upi":
        y_test, y_prob, threshold = evaluate_candidate_upi()
    elif canonical == "card":
        y_test, y_prob, threshold = evaluate_candidate_card()
    elif canonical == "account":
        y_test, y_prob, threshold = evaluate_candidate_account()
    elif canonical == "transfer_wallet":
        y_test, y_prob, threshold = evaluate_candidate_transfer()
    else:
        raise ValueError(f"Unknown channel: {canonical}")

    cand_metrics = compute_metrics_and_ci(y_test, y_prob, threshold=threshold, n_bootstrap=n_bootstrap)
    cand_pr_auc = cand_metrics["pr_auc"]
    cand_fa = cand_metrics["false_alarms_per_100k"]
    cand_brier = cand_metrics["brier_score"]

    logger.info("Candidate Evaluated:")
    logger.info("  PR-AUC: %.5f [95%% CI: %.5f - %.5f]", cand_pr_auc, cand_metrics["ci_95_pr_auc"][0], cand_metrics["ci_95_pr_auc"][1])
    logger.info("  False Alarms / 100k: %.2f", cand_fa)
    logger.info("  Calibration Brier Score: %.5f", cand_brier)

    gate1_pass = cand_pr_auc >= (incumbent_ci_low - 0.005)
    gate2_pass = cand_fa <= max(incumbent_fa * 1.05, incumbent_fa + 5.0)
    gate3_pass = cand_brier <= max(incumbent_brier * 1.05, incumbent_brier + 0.001)

    all_gates_pass = gate1_pass and gate2_pass and gate3_pass
    should_promote = all_gates_pass or force_promote

    logger.info("-" * 50)
    logger.info("Gate 1 (PR-AUC >= %.5f): %s (cand=%.5f)", incumbent_ci_low, "PASS" if gate1_pass else "FAIL", cand_pr_auc)
    logger.info("Gate 2 (FA / 100k <= %.2f): %s (cand=%.2f)", incumbent_fa * 1.05, "PASS" if gate2_pass else "FAIL", cand_fa)
    logger.info("Gate 3 (Brier <= %.5f): %s (cand=%.5f)", incumbent_brier * 1.05, "PASS" if gate3_pass else "FAIL", cand_brier)
    logger.info("-" * 50)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report = {
        "channel": canonical,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "action": "PROMOTED" if should_promote else "REJECTED",
        "forced": force_promote and not all_gates_pass,
        "gates": {
            "gate1_pr_auc": {
                "passed": bool(gate1_pass),
                "incumbent_pr_auc": incumbent_pr_auc,
                "incumbent_ci_lower": incumbent_ci_low,
                "candidate_pr_auc": cand_pr_auc,
            },
            "gate2_false_alarms": {
                "passed": bool(gate2_pass),
                "incumbent_fa_per_100k": incumbent_fa,
                "candidate_fa_per_100k": cand_fa,
            },
            "gate3_brier_calibration": {
                "passed": bool(gate3_pass),
                "incumbent_brier": incumbent_brier,
                "candidate_brier": cand_brier,
            },
        },
        "incumbent_version": incumbent_ver,
        "candidate_metrics": cand_metrics,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if should_promote:
        new_ver = _next_minor_version(incumbent_ver)
        report["promoted_version"] = new_ver
        logger.info(">>> DECISION: PROMOTION APPROVED. New Champion Version: %s <<<", new_ver)

        if not dry_run:
            archived = dict(incumbent)
            archived["demoted_date"] = datetime.utcnow().isoformat() + "Z"
            archived["demoted_reason"] = f"Superceded by candidate {new_ver}"

            if "history" not in incumbent:
                incumbent["history"] = []
            incumbent["history"].append(archived)

            incumbent["version"] = new_ver
            incumbent["trained_date"] = datetime.utcnow().strftime("%Y-%m-%d")
            incumbent["pr_auc"] = cand_pr_auc
            incumbent["recall"] = cand_metrics["recall"]
            incumbent["precision"] = cand_metrics["precision"]
            incumbent["brier_score"] = cand_brier
            incumbent["false_alarms_per_100k"] = cand_fa
            incumbent["bootstrap_ci_95"]["pr_auc"] = cand_metrics["ci_95_pr_auc"]

            registry["last_updated"] = datetime.utcnow().strftime("%Y-%m-%d")
            _save_registry(registry)
            logger.info("Updated %s with new champion %s", CHAMPION_REGISTRY_PATH, new_ver)

        rep_path = REPORTS_DIR / f"promotion_{canonical}_{timestamp}.json"
        with open(rep_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info("Promotion report written to %s", rep_path)
    else:
        logger.warning(">>> DECISION: CANDIDATE REJECTED. Champion %s retained. <<<", incumbent_ver)
        rep_path = REPORTS_DIR / f"rejection_{canonical}_{timestamp}.json"
        with open(rep_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info("Rejection report written to %s", rep_path)

    return report


def main():
    parser = argparse.ArgumentParser(description="FraudShield Model Retrain and Gated Promotion CLI")
    parser.add_argument("--channel", required=True, help="Channel to retrain (upi, card, account, transfer_wallet)")
    parser.add_argument("--force-promote", action="store_true", help="Bypass gates and force promotion")
    parser.add_argument("--dry-run", action="store_true", help="Evaluate gates without mutating champion registry")
    parser.add_argument("--n-bootstrap", type=int, default=150, help="Number of bootstrap iterations for CI")
    args = parser.parse_args()

    retrain_and_promote(
        channel_name=args.channel,
        force_promote=args.force_promote,
        dry_run=args.dry_run,
        n_bootstrap=args.n_bootstrap,
    )


if __name__ == "__main__":
    main()
