from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ml.config import (
    BASE_DIR,
    CHANNELS,
    METADATA_FILE,
    METRICS_FILE,
    REPORTS_DIR,
)

CROSS_CHANNEL_CSV = REPORTS_DIR / "cross_channel_summary.csv"

CHANNEL_ALIASES = {
    "card": "card",
    "credit_card": "card",
    "creditcard": "card",
    "debit_card": "card",
    "cards": "card",
    "card_payments": "card",
    "transfer_wallet": "transfer_wallet",
    "transfer": "transfer_wallet",
    "wallet": "transfer_wallet",
    "paysim": "transfer_wallet",
    "bank_transfer": "transfer_wallet",
    "account": "account",
    "bank_account": "account",
    "baf": "account",
    "account_opening": "account",
    "upi": "upi",
    "bhim": "upi",
    "upi_transactions": "upi",
}


def _resolve_key(channel: str | None) -> str:
    if not channel:
        return "transfer_wallet"
    key = str(channel).strip().lower().replace("-", "_").replace(" ", "_")
    return CHANNEL_ALIASES.get(key, key)


_METRICS_CACHE: dict[str, Any] | None = None
_METRICS_MTIME: float = 0.0

_METADATA_CACHE: dict[str, Any] | None = None
_METADATA_MTIME: float = 0.0


def load_metrics(force_reload: bool = False) -> dict[str, Any]:
    global _METRICS_CACHE, _METRICS_MTIME
    if not METRICS_FILE.exists():
        raise FileNotFoundError(f"Metrics file missing at {METRICS_FILE}")
    mtime = METRICS_FILE.stat().st_mtime
    if not force_reload and _METRICS_CACHE is not None and mtime == _METRICS_MTIME:
        return _METRICS_CACHE
    with open(METRICS_FILE, "r", encoding="utf-8") as f:
        _METRICS_CACHE = json.load(f)
    _METRICS_MTIME = mtime
    return _METRICS_CACHE


def load_metadata(force_reload: bool = False) -> dict[str, Any]:
    global _METADATA_CACHE, _METADATA_MTIME
    if not METADATA_FILE.exists():
        raise FileNotFoundError(f"Metadata file missing at {METADATA_FILE}")
    mtime = METADATA_FILE.stat().st_mtime
    if not force_reload and _METADATA_CACHE is not None and mtime == _METADATA_MTIME:
        return _METADATA_CACHE
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        _METADATA_CACHE = json.load(f)
    _METADATA_MTIME = mtime
    return _METADATA_CACHE


def get_channel_data_source(channel: str = "transfer_wallet") -> str:
    canonical = _resolve_key(channel)
    return CHANNELS.get(canonical, {}).get("data_source", "Simulated (agent-based)")


def get_api_latency() -> dict[str, Any]:
    metrics = load_metrics()
    return metrics.get("api_latency", {})


def get_channel_metrics(channel: str = "transfer_wallet") -> dict[str, Any]:
    metrics = load_metrics()
    canonical = _resolve_key(channel)

    if canonical in ["transfer_wallet", "paysim"]:
        if "transfer_wallet" in metrics:
            return metrics["transfer_wallet"]
        return metrics

    if canonical in metrics:
        return metrics[canonical]

    raise KeyError(f"Channel '{channel}' (resolved: '{canonical}') not found in metrics.json. Available: {list(metrics.keys())}")


def get_channel_metadata(channel: str = "transfer_wallet") -> dict[str, Any]:
    meta = load_metadata()
    canonical = _resolve_key(channel)
    channels_meta = meta.get("channels", {})

    if canonical in channels_meta:
        return channels_meta[canonical]

    if canonical in ["transfer_wallet", "paysim"]:
        return meta

    raise KeyError(f"Channel '{channel}' not found in metadata.json channels: {list(channels_meta.keys())}")


def get_calibrated_thresholds(channel: str = "transfer_wallet") -> tuple[float, float]:
    canonical = _resolve_key(channel)
    metrics = load_metrics()

    if canonical in ["transfer_wallet", "paysim"]:
        th = metrics.get("calibrated_thresholds", metrics.get("transfer_wallet", {}).get("thresholds", {}))
    else:
        ch_m = metrics.get(canonical, {})
        th = ch_m.get("thresholds", {})

    low = float(th.get("low_risk_approve", 0.40))
    high = float(th.get("high_risk_decline", 0.80))
    return low, high


def get_champion_metrics(channel: str = "transfer_wallet") -> dict[str, Any]:
    canonical = _resolve_key(channel)
    metrics = load_metrics()

    if canonical in ["transfer_wallet", "paysim"]:
        res = dict(metrics.get("Tuned LightGBM (Champion)", {}))
    elif canonical == "card":
        res = dict(metrics.get(canonical, {}).get("Stacked Ensemble", metrics.get(canonical, {}).get("Tuned LightGBM (Champion)", {})))
    elif canonical == "account":
        res = dict(metrics.get(canonical, {}).get("Tuned CatBoost (Champion)", metrics.get(canonical, {}).get("Tuned LightGBM (Champion)", {})))
    else:
        res = dict(metrics.get(canonical, {}).get("Tuned LightGBM (Champion)", {}))

    if "precision" not in res and "tp" in res and "fp" in res:
        res["precision"] = res["tp"] / max(1, res["tp"] + res["fp"])
    return res


def get_economics_summary(channel: str = "transfer_wallet") -> dict[str, Any]:
    canonical = _resolve_key(channel)
    metrics = load_metrics()

    if canonical in ["transfer_wallet", "paysim"]:
        return metrics.get("economics_natural_test", metrics.get("transfer_wallet", {}).get("economics", {}))

    ch_m = metrics.get(canonical, {})
    return ch_m.get("economics", {})


def get_cross_channel_summary(save_csv: bool = True) -> pd.DataFrame:
    metrics = load_metrics()
    metadata = load_metadata()
    channels_meta = metadata.get("channels", {})

    rows = []

    channel_defs = [
        ("card", "Card Payments", "Real, anonymised", "ULB Credit Card Fraud"),
        ("transfer_wallet", "Online Transfers & Wallets", "Simulated (agent-based)", "PaySim Mobile Money"),
        ("account", "Bank Accounts", "Privacy-preserving synthetic", "Bank Account Fraud (BAF NeurIPS 2022)"),
        ("upi", "UPI Transactions", "Simulated by us", "Synthetic Indian UPI Simulator"),
    ]

    for key, display_name, data_source, benchmark_name in channel_defs:
        if key in ["transfer_wallet", "paysim"]:
            champ = metrics.get("Tuned LightGBM (Champion)", {})
            econ = metrics.get("economics_natural_test", {})
            ci_data = metrics.get("transfer_wallet", {}).get("bootstrap_ci_95", metadata.get("bootstrap_ci_95", {}))
            ci_pr = ci_data.get("pr_auc", [0.99811, 1.0])
            test_n = int(econ.get("total_test_transactions", 918617))
            test_frauds = int(econ.get("total_test_fraud_count", 4006))
            prev = (test_frauds / max(test_n, 1)) * 100.0
            prot_pct = float(econ.get("fraud_value_protection_pct", 99.99))
        else:
            ch_data = metrics.get(key, {})
            champ = get_champion_metrics(key)
            econ = ch_data.get("economics", {})
            ci_data = ch_data.get("bootstrap_ci_95", {})
            if "ci_95" in champ:
                ci_pr = champ["ci_95"]
            else:
                ci_pr = ci_data.get("pr_auc", [0.0, 0.0])

            test_n = int(econ.get("total_test_transactions", econ.get("total_test_applications", 0)))
            test_frauds = int(econ.get("total_test_fraud_count", 0))
            prev = (test_frauds / max(test_n, 1)) * 100.0
            prot_pct = float(econ.get("fraud_value_protection_pct", 0.0))

        ci_str = f"[{ci_pr[0]:.5f}, {ci_pr[1]:.5f}]"
        rec = float(champ.get("recall", 0.0)) * 100.0
        far = float(champ.get("false_alarms_per_100k", 0.0))
        pr_auc = float(champ.get("pr_auc", 0.0))

        rows.append({
            "Channel": display_name,
            "Data source": data_source,
            "Benchmark name": benchmark_name,
            "Test rows": f"{test_n:,}",
            "Test prevalence": f"{prev:.3f}%",
            "Champion PR-AUC": f"{pr_auc:.5f}",
            "95% CI": ci_str,
            "Recall": f"{rec:.2f}%",
            "False alarms / 100k": f"{far:.1f}",
            "INR value protected %": f"{prot_pct:.2f}%",
        })

    df = pd.DataFrame(rows)

    if save_csv:
        CROSS_CHANNEL_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(CROSS_CHANNEL_CSV, index=False)

    return df


def compute_false_alarms_per_100k(fp: int, total_transactions: int) -> float:
    if total_transactions <= 0:
        return 0.0
    return round(float(fp / total_transactions * 100000.0), 1)


def format_inr(amount: float) -> str:
    return f"₹{amount:,.2f}"


def format_pct(value: float, decimals: int = 2) -> str:
    return f"{value * 100:.{decimals}f}%"
