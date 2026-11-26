from __future__ import annotations

import io
import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from ml.metrics_helper import get_calibrated_thresholds
from ml.router import predict, resolve_channel

logger = logging.getLogger(__name__)


CHANNEL_REQUIRED_COLUMNS = {
    "transfer_wallet": ["amount", "oldbalanceOrg", "type"],
    "card": ["Amount"],
    "account": ["income", "proposed_credit_limit"],
    "upi": ["amount", "txn_type"],
}

CHANNEL_TEMPLATES = {
    "transfer_wallet": pd.DataFrame([
        {"step": 743, "type": "TRANSFER", "amount": 181.0, "nameOrig": "C824064320", "oldbalanceOrg": 181.0, "newbalanceOrig": 0.0, "nameDest": "C215812169", "oldbalanceDest": 0.0, "newbalanceDest": 0.0},
        {"step": 743, "type": "CASH_OUT", "amount": 181.0, "nameOrig": "C840083671", "oldbalanceOrg": 181.0, "newbalanceOrig": 0.0, "nameDest": "C38997010", "oldbalanceDest": 21182.0, "newbalanceDest": 21363.0},
        {"step": 743, "type": "PAYMENT", "amount": 5420.5, "nameOrig": "C102384729", "oldbalanceOrg": 12000.0, "newbalanceOrig": 6579.5, "nameDest": "M48392019", "oldbalanceDest": 0.0, "newbalanceDest": 0.0},
        {"step": 743, "type": "TRANSFER", "amount": 4500000.0, "nameOrig": "C992837192", "oldbalanceOrg": 4500000.0, "newbalanceOrig": 0.0, "nameDest": "C110293847", "oldbalanceDest": 0.0, "newbalanceDest": 0.0},
        {"step": 743, "type": "CASH_IN", "amount": 25000.0, "nameOrig": "C392819201", "oldbalanceOrg": 5000.0, "newbalanceOrig": 30000.0, "nameDest": "C482910293", "oldbalanceDest": 50000.0, "newbalanceDest": 25000.0},
    ]),
    "card": pd.DataFrame([
        {"Time": 45000.0, "Amount": 149.50, "V1": -1.2, "V2": 0.5, "V3": -0.8, "V4": 1.1, "V14": -5.2, "V10": -2.1, "V12": -1.8, "V17": -3.5},
        {"Time": 45010.0, "Amount": 12.00, "V1": 0.2, "V2": 0.1, "V3": 0.4, "V4": -0.2, "V14": 0.1, "V10": 0.0, "V12": 0.1, "V17": 0.0},
        {"Time": 45025.0, "Amount": 850.00, "V1": -2.5, "V2": 1.8, "V3": -3.1, "V4": 3.4, "V14": -8.5, "V10": -4.2, "V12": -5.1, "V17": -6.2},
    ]),
    "account": pd.DataFrame([
        {"income": 0.4, "name_email_similarity": 0.15, "prev_address_months_count": -1, "current_address_months_count": 12, "days_since_request": 0.01, "proposed_credit_limit": 1500.0, "payment_type": "AB", "employment_status": "CA", "housing_status": "BC", "source": "INTERNET", "device_os": "windows"},
        {"income": 0.8, "name_email_similarity": 0.95, "prev_address_months_count": 24, "current_address_months_count": 48, "days_since_request": 0.01, "proposed_credit_limit": 5000.0, "payment_type": "AA", "employment_status": "CB", "housing_status": "BE", "source": "INTERNET", "device_os": "macintosh"},
    ]),
    "upi": pd.DataFrame([
        {"amount": 25000.0, "hour": 23.0, "txn_type": "COLLECT_REQUEST", "payer_vpa": "user1@okhdfcbank", "payee_vpa": "fraud1@paytm", "is_first_time_payee": 1, "is_new_device": 1, "remote_access_app_flag": 1, "velocity_1h": 4},
        {"amount": 500.0, "hour": 14.0, "txn_type": "PAY", "payer_vpa": "user2@oksbi", "payee_vpa": "store@paytm", "is_first_time_payee": 0, "is_new_device": 0, "remote_access_app_flag": 0, "velocity_1h": 0},
    ]),
}


def get_sample_template_csv(channel: str = "transfer_wallet") -> str:
    canonical = resolve_channel(channel)
    df = CHANNEL_TEMPLATES.get(canonical, CHANNEL_TEMPLATES["transfer_wallet"])
    return df.to_csv(index=False)


def score_batch(
    df: pd.DataFrame,
    channel: str = "transfer_wallet",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    canonical = resolve_channel(channel)
    req_cols = CHANNEL_REQUIRED_COLUMNS.get(canonical, [])
    missing = [c for c in req_cols if c not in df.columns]

    if missing:
        raise ValueError(
            f"Uploaded CSV is missing required columns for '{canonical}': {missing}.\n"
            f"Expected at least: {req_cols}"
        )

    t0 = time.perf_counter()
    scored_records = []
    low_th, high_th = get_calibrated_thresholds(canonical)

    for _, row in df.iterrows():
        payload = row.to_dict()
        try:
            res = predict(channel=canonical, payload=payload, top_k=2, commit_to_store=False)
            scored_records.append({
                "risk_score": res["risk_score"],
                "risk_band": res["band"],
                "action": res["action"],
                "primary_reason": res["top_reasons"][0] if res.get("top_reasons") else "Normal parameters",
                "counterfactual": res.get("counterfactual", ""),
            })
        except (ValueError, KeyError, TypeError, RuntimeError) as e:
            logger.warning("Error scoring batch row %s: %s", payload.get("ID", payload.get("txn_id", "unknown")), e)
            scored_records.append({
                "risk_score": 0.50,
                "risk_band": "Review",
                "action": "Manual Review",
                "primary_reason": f"Execution warning: {str(e)[:50]}",
                "counterfactual": "",
            })

    total_time_ms = (time.perf_counter() - t0) * 1000.0
    scored_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(scored_records)], axis=1)

    total_tx = len(scored_df)
    high_count = int((scored_df["risk_band"] == "High").sum())
    med_count = int((scored_df["risk_band"] == "Medium").sum())
    low_count = int((scored_df["risk_band"] == "Low").sum())

    amt_col = "amount" if "amount" in scored_df.columns else ("Amount" if "Amount" in scored_df.columns else None)
    if amt_col:
        total_inr = float(scored_df[amt_col].sum())
        flagged_inr = float(scored_df[scored_df["risk_band"] == "High"][amt_col].sum())
    else:
        total_inr = 0.0
        flagged_inr = 0.0

    summary = {
        "channel": canonical,
        "total_transactions": total_tx,
        "high_risk_declines": high_count,
        "high_risk_pct": round((high_count / max(total_tx, 1)) * 100.0, 2),
        "medium_risk_challenges": med_count,
        "medium_risk_pct": round((med_count / max(total_tx, 1)) * 100.0, 2),
        "low_risk_approvals": low_count,
        "low_risk_pct": round((low_count / max(total_tx, 1)) * 100.0, 2),
        "total_volume_inr": total_inr,
        "flagged_volume_inr": flagged_inr,
        "total_time_ms": round(total_time_ms, 2),
        "avg_latency_ms": round(total_time_ms / max(total_tx, 1), 2),
        "thresholds_applied": {"low": low_th, "high": high_th},
    }

    return scored_df, summary

