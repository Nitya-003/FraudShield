from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd

from ml.config import (
    DEFAULT_HIGH_THRESHOLD,
    DEFAULT_LOW_THRESHOLD,
    MAX_DAILY_REVIEW_CAPACITY,
)


def route_transaction_action(score: float, amount: float, low_th: float = 0.30, high_th: float = 0.75) -> tuple[str, str, str]:
    if score < low_th:
        return "Low", "Auto-Approve", "None (Frictionless clearing)"
    elif score <= high_th:
        if amount > 100_000:
            return "Medium", "Step-Up (OTP)", "SMS OTP + Biometric Re-auth"
        return "Medium", "Step-Up (OTP)", "SMS OTP Challenge"
    else:
        if score >= 0.95 or amount >= 500_000:
            return "High", "Decline & Alert", "Immediate Block & Law Enforcement Freeze"
        return "High", "Manual Review Queue", "Analyst Outbound Verification Call"


def build_analyst_review_queue(
    df: pd.DataFrame,
    scores: np.ndarray,
    low_th: float = DEFAULT_LOW_THRESHOLD,
    high_th: float = DEFAULT_HIGH_THRESHOLD,
    daily_capacity: int = MAX_DAILY_REVIEW_CAPACITY,
    duration_days: float = 15.17,
) -> pd.DataFrame:
    total_capacity = int(daily_capacity * duration_days)

    amounts = df["amount"].to_numpy(dtype=float)
    bands = []
    policies = []
    actions = []

    for s, a in zip(scores, amounts):
        band, pol, act = route_transaction_action(s, a, low_th, high_th)
        bands.append(band)
        policies.append(pol)
        actions.append(act)

    queue_df = pd.DataFrame({
        "step": df["step"].to_numpy() if "step" in df else np.zeros(len(df), dtype=int),
        "nameOrig": df["nameOrig"].to_numpy() if "nameOrig" in df else [f"C{i}" for i in range(len(df))],
        "nameDest": df["nameDest"].to_numpy() if "nameDest" in df else [f"M{i}" for i in range(len(df))],
        "type": df["type"].to_numpy() if "type" in df else ["TRANSFER"] * len(df),
        "amount": amounts,
        "risk_score": np.round(scores, 4),
        "risk_band": bands,
        "policy": policies,
        "action": actions,
        "expected_loss_inr": np.round(scores * amounts, 2),
        "is_fraud_ground_truth": df["isFraud"].to_numpy() if "isFraud" in df else np.zeros(len(df), dtype=int),
    })

    reviewable = queue_df[queue_df["risk_band"].isin(["Medium", "High"])].copy()

    reviewable = reviewable.sort_values("expected_loss_inr", ascending=False).reset_index(drop=True)

    reviewable["within_capacity"] = False
    reviewable.loc[:total_capacity - 1, "within_capacity"] = True
    reviewable["queue_rank"] = np.arange(1, len(reviewable) + 1)

    return reviewable

