from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from ml.config import METRICS_FILE, PROCESSED_DATA_DIR, REPORTS_DIR


def calculate_psi(
    expected: np.ndarray | pd.Series,
    actual: np.ndarray | pd.Series,
    num_bins: int = 10,
    eps: float = 1e-4,
) -> float:
    exp_clean = np.asarray(expected, dtype=np.float64)
    act_clean = np.asarray(actual, dtype=np.float64)

    exp_clean = exp_clean[np.isfinite(exp_clean)]
    act_clean = act_clean[np.isfinite(act_clean)]

    if len(exp_clean) == 0 or len(act_clean) == 0:
        return 0.0

    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(exp_clean, quantiles)
    bin_edges[0] -= 1e-5
    bin_edges[-1] += 1e-5

    bin_edges = np.unique(bin_edges)
    if len(bin_edges) <= 1:
        return 0.0

    exp_counts, _ = np.histogram(exp_clean, bins=bin_edges)
    act_counts, _ = np.histogram(act_clean, bins=bin_edges)

    exp_pct = exp_counts / max(exp_counts.sum(), 1)
    act_pct = act_counts / max(act_counts.sum(), 1)

    exp_pct = np.clip(exp_pct, eps, 1.0)
    act_pct = np.clip(act_pct, eps, 1.0)

    exp_pct /= exp_pct.sum()
    act_pct /= act_pct.sum()

    psi_val = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(round(max(0.0, psi_val), 5))


def classify_psi_drift(psi_val: float) -> str:
    if psi_val < 0.10:
        return "Stable (No Drift)"
    elif psi_val < 0.25:
        return "Moderate Drift (Warning)"
    else:
        return "Significant Drift (Retrain)"


def compute_feature_drift_summary(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: list[str],
) -> list[dict[str, Any]]:
    results = []
    for col in feature_cols:
        if col in df_train.columns and col in df_test.columns:
            psi = calculate_psi(df_train[col], df_test[col])
            status_str = classify_psi_drift(psi)
            results.append({
                "feature": col,
                "psi": psi,
                "status": status_str,
                "action": "Maintain" if psi < 0.10 else ("Monitor" if psi < 0.25 else "Retrain"),
            })
    return sorted(results, key=lambda x: x["psi"], reverse=True)


def compute_score_distribution_over_time(
    df: pd.DataFrame,
    time_col: str = "step",
    score_col: str = "risk_score",
    time_bin_hours: int = 24,
) -> pd.DataFrame:

    data = df.copy()
    if time_col not in data.columns or score_col not in data.columns:
        return pd.DataFrame()

    data["time_window"] = (data[time_col] // time_bin_hours).astype(int)
    grouped = data.groupby("time_window")

    traj = grouped[score_col].agg(
        mean_score="mean",
        median_score="median",
        p95_score=lambda x: float(np.percentile(x, 95)),
        total_tx="count",
    ).reset_index()

    traj["mean_score"] = traj["mean_score"].round(4)
    traj["median_score"] = traj["median_score"].round(4)
    traj["p95_score"] = traj["p95_score"].round(4)
    return traj


def compute_alert_rate_over_time(
    df: pd.DataFrame,
    time_col: str = "step",
    band_col: str = "band",
    time_bin_hours: int = 24,
) -> pd.DataFrame:
    
    data = df.copy()
    if time_col not in data.columns or band_col not in data.columns:
        return pd.DataFrame()

    data["time_window"] = (data[time_col] // time_bin_hours).astype(int)
    grouped = data.groupby("time_window")

    res = []
    for tw, grp in grouped:
        total = len(grp)
        high_cnt = int((grp[band_col] == "High").sum())
        med_cnt = int((grp[band_col] == "Medium").sum())
        low_cnt = int((grp[band_col] == "Low").sum())
        alert_rate = (high_cnt + med_cnt) / max(total, 1) * 100.0
        decline_rate = high_cnt / max(total, 1) * 100.0

        res.append({
            "day": int(tw),
            "total_transactions": total,
            "auto_approved": low_cnt,
            "step_up_otp": med_cnt,
            "declined": high_cnt,
            "alert_rate_pct": round(alert_rate, 3),
            "decline_rate_pct": round(decline_rate, 3),
        })

    return pd.DataFrame(res)

