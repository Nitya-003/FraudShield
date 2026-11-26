from __future__ import annotations

import copy
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd

from ml.history_store import AccountHistoryStore
from ml.metrics_helper import get_calibrated_thresholds
from ml.router import predict, resolve_channel

_ATTACK_EXPERIMENT_LOGS: list[dict[str, Any]] = []


def get_attack_presets() -> list[dict[str, Any]]:
    return [
        {
            "id": "paysim_full_drain",
            "name": "PaySim: Full Account Drain (₹10,000,000)",
            "channel": "transfer_wallet",
            "description": "Victim account drained completely in a single high-value transfer.",
            "payload": {
                "amount": 10000000.0,
                "oldbalanceOrg": 10000000.0,
                "oldbalanceDest": 0.0,
                "type": "TRANSFER",
                "nameOrig": "C992817263",
                "nameDest": "C182736451",
                "step": 380,
            },
        },
        {
            "id": "upi_extortion_scam",
            "name": "UPI: Digital Arrest Extortion Scam (₹85,000)",
            "channel": "upi",
            "description": "Coerced high-value collect-request transfer to an unknown payee.",
            "payload": {
                "amount": 85000.0,
                "payer_vpa": "user_elderly@oksbi",
                "payee_vpa": "mule_cyber_collect@paytm",
                "type": "COLLECT",
                "payment_mode": "COLLECT",
                "is_payer_off_hours": 1,
                "is_collect_request": True,
                "device_takeover_risk": 1,
                "payer_amount_ratio": 4.5,
                "amount_to_median_ratio": 8.5,
                "is_extreme_amount": 1,
                "first_time_high_value": 1,
            },
        },
        {
            "id": "card_stolen_burst",
            "name": "Card: High-Velocity Stolen Card (₹250,000)",
            "channel": "card",
            "description": "Unauthorized luxury purchase on a compromised credit card.",
            "payload": {
                "amount": 250000.0,
                "card_id": "card_stolen_8819",
                "merchant_category": "electronics",
                "is_foreign_country": 1,
                "foreign_transaction": 1,
                "hour_of_day": 3,
                "is_night": 1,
            },
        },
        {
            "id": "paysim_medium_drain",
            "name": "PaySim: Moderate Drain (₹1,500,000)",
            "channel": "transfer_wallet",
            "description": "Moderate balance extraction on an active business account.",
            "payload": {
                "amount": 1500000.0,
                "oldbalanceOrg": 1500000.0,
                "oldbalanceDest": 25000.0,
                "type": "TRANSFER",
                "nameOrig": "C448192031",
                "nameDest": "C667346055",
                "step": 400,
            },
        },
    ]


def simulate_attack_scenario(
    channel: str,
    base_payload: dict[str, Any],
    attack_type: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    canonical_ch = resolve_channel(channel)
    low_th, high_th = get_calibrated_thresholds(canonical_ch)

    store_base = AccountHistoryStore(window_hours=24)
    base_res = predict(canonical_ch, base_payload, top_k=3, history_store=store_base, commit_to_store=False)
    base_score = float(base_res.get("risk_score", 0.0))
    base_band = str(base_res.get("band", "Low"))
    base_action = str(base_res.get("action", "Approve"))

    k_splits = int(params.get("k_splits", 3 if attack_type in ["split_transfers", "change_timing", "combined"] else 1))
    split_mode = str(params.get("split_mode", "equal"))
    time_gap_hours = int(params.get("time_gap_hours", 1 if attack_type == "change_timing" else 0))
    amt_fraction = float(params.get("amount_fraction", 0.5 if attack_type == "reduce_amount" else 1.0))
    use_mule_hop = bool(params.get("use_mule_hop", attack_type == "add_hop"))

    orig_amt = float(base_payload.get("amount", base_payload.get("Amount", 100000.0)))
    total_target_amt = orig_amt * amt_fraction
    orig_bal = float(base_payload.get("oldbalanceOrg", total_target_amt))

    if k_splits <= 1:
        split_amounts = [total_target_amt]
    else:
        if split_mode == "front_loaded":
            weights = [0.5] + [(0.5 / (k_splits - 1))] * (k_splits - 1)
        elif split_mode == "back_loaded":
            weights = [(0.4 / (k_splits - 1))] * (k_splits - 1) + [0.6]
        else:  
            weights = [1.0 / k_splits] * k_splits
        split_amounts = [round(total_target_amt * w, 2) for w in weights]

    isolated_store = AccountHistoryStore(window_hours=24)
    simulated_txs = []
    base_step = int(base_payload.get("step", 380))
    running_balance = orig_bal

    for i, part_amt in enumerate(split_amounts):
        step_val = base_step + (i * time_gap_hours)
        tx_payload = copy.deepcopy(base_payload)
        tx_payload["amount"] = part_amt
        tx_payload["step"] = step_val
        tx_payload["oldbalanceOrg"] = running_balance

        if use_mule_hop and i == 0:
            tx_payload["nameDest"] = "C_MULE_INTERMEDIARY_HOP1"
            if canonical_ch == "upi":
                tx_payload["payee_vpa"] = "mule_intermediary@okaxis"

        if canonical_ch == "upi":
            payer_avg = max(orig_amt / 4.0, 5000.0)
            tx_payload["payer_amount_ratio"] = round(part_amt / payer_avg, 2)
            tx_payload["amount_to_median_ratio"] = round(part_amt / 5000.0, 2)
            tx_payload["is_extreme_amount"] = 1 if part_amt > 50000.0 else 0

        t_res = predict(canonical_ch, tx_payload, top_k=2, history_store=isolated_store, commit_to_store=True)

        sc = float(t_res.get("risk_score", 0.0))
        bd = str(t_res.get("band", "Low"))
        act = str(t_res.get("action", "Approve"))
        top_factor = t_res.get("top_reasons", ["Normal authorization"])[0] if t_res.get("top_reasons") else "Normal pattern"

        simulated_txs.append({
            "transfer_index": i + 1,
            "step": step_val,
            "amount": part_amt,
            "amount_fraction": round((part_amt / orig_amt) * 100, 1),
            "remaining_balance": max(running_balance - part_amt, 0.0),
            "risk_score": sc,
            "band": bd,
            "action": act,
            "primary_signal": top_factor,
        })

        running_balance = max(running_balance - part_amt, 0.0)

    scores = [t["risk_score"] for t in simulated_txs]
    max_score = max(scores)
    mean_score = sum(scores) / len(scores)

    any_high = any(t["band"] == "High" for t in simulated_txs)
    any_med = any(t["band"] == "Medium" for t in simulated_txs)

    if any_high:
        verdict = "BLOCKED"
        verdict_color = "#dc2626"
        verdict_summary = "Fraud Intercepted: Model caught the evasion pattern and triggered Decline & Alert."
    elif any_med:
        verdict = "STEP_UP_CHALLENGED"
        verdict_color = "#d97706"
        verdict_summary = "Partial Defense: Model detected anomalous behavior and required Step-Up OTP challenge."
    else:
        verdict = "EVADED"
        verdict_color = "#16a34a"
        verdict_summary = "Evasion Successful: All transformed transfers scored below threshold and were Auto-Approved."

    explanation_parts = []
    if k_splits > 1:
        if time_gap_hours >= 4:
            explanation_parts.append(
                f"Spacing transfers {time_gap_hours} hours apart exploited the sliding 1-hour window expiration. "
                "The model's short-term velocity features decayed between transactions."
            )
        elif any_high:
            explanation_parts.append(
                f"Splitting into {k_splits} transfers reduced individual transaction size, but the sequence-aware "
                "cumulative outflow tracker flagged the multi-transfer balance drain."
            )
        else:
            explanation_parts.append(
                f"Splitting into {k_splits} transfers successfully distributed risk below the high-risk threshold."
            )

    if amt_fraction <= 0.20:
        explanation_parts.append(
            f"Reducing extraction to {int(amt_fraction*100)}% kept the amount-to-balance ratio within normal variance."
        )

    if use_mule_hop:
        explanation_parts.append(
            "Routing through an intermediate mule account distributed counterparty concentration."
        )

    if not explanation_parts:
        explanation_parts.append(
            f"The transformed transaction registered a peak risk score of {max_score:.4f} compared to {base_score:.4f} baseline."
        )

    mechanics_explanation = " ".join(explanation_parts)

    result_docket = {
        "channel": canonical_ch,
        "attack_type": attack_type,
        "params": params,
        "base_score": base_score,
        "base_band": base_band,
        "base_action": base_action,
        "peak_attack_score": max_score,
        "mean_attack_score": round(mean_score, 4),
        "verdict": verdict,
        "verdict_color": verdict_color,
        "verdict_summary": verdict_summary,
        "mechanics_explanation": mechanics_explanation,
        "transfers": simulated_txs,
        "thresholds": {"low": low_th, "high": high_th},
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
    }

    log_attack_experiment({
        "timestamp": result_docket["timestamp"],
        "channel": canonical_ch,
        "attack_type": attack_type,
        "k_splits": k_splits,
        "time_gap_h": time_gap_hours,
        "amt_pct": int(amt_fraction * 100),
        "base_score": round(base_score, 4),
        "peak_score": round(max_score, 4),
        "verdict": verdict,
    })

    return result_docket


def log_attack_experiment(entry: dict[str, Any]) -> None:
    _ATTACK_EXPERIMENT_LOGS.insert(0, entry)
    if len(_ATTACK_EXPERIMENT_LOGS) > 100:
        _ATTACK_EXPERIMENT_LOGS.pop()


def get_attack_history() -> pd.DataFrame:
    if not _ATTACK_EXPERIMENT_LOGS:
        return pd.DataFrame(columns=[
            "timestamp", "channel", "attack_type", "k_splits", "time_gap_h",
            "amt_pct", "base_score", "peak_score", "verdict"
        ])
    return pd.DataFrame(_ATTACK_EXPERIMENT_LOGS)

