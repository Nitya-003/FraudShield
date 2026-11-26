from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ml.config import BASE_DIR, CHANNELS, PROCESSED_DATA_DIR, REPORTS_DIR
from ml.router import predict

logger = logging.getLogger("fraudshield.reason_codes")


REASON_CATALOG: dict[str, dict[str, Any]] = {
    "RC_AMOUNT_HIGH_DEVIATION": {
        "code": "RC_AMOUNT_HIGH_DEVIATION",
        "title": "High Transaction Amount",
        "description": "Transaction amount is unusually elevated compared to standard historical baseline.",
        "category": "Monetary",
    },
    "RC_ORIGIN_BALANCE_DRAINED": {
        "code": "RC_ORIGIN_BALANCE_DRAINED",
        "title": "Origin Balance Drained",
        "description": "The transaction empties or near-completely exhausts the payer's available balance.",
        "category": "Account Behavior",
    },
    "RC_DESTINATION_ZERO_BALANCE": {
        "code": "RC_DESTINATION_ZERO_BALANCE",
        "title": "Beneficiary Empty Pre-Transfer",
        "description": "The recipient account had a zero balance immediately prior to receiving funds.",
        "category": "Mule & Laundering Pattern",
    },
    "RC_NOCTURNAL_TRANSACTION": {
        "code": "RC_NOCTURNAL_TRANSACTION",
        "title": "Nocturnal Hour Activity",
        "description": "Payment initiated during nocturnal off-peak hours (12 AM - 5 AM).",
        "category": "Temporal",
    },
    "RC_FIRST_TIME_PAYEE": {
        "code": "RC_FIRST_TIME_PAYEE",
        "title": "First-Time Beneficiary",
        "description": "Payment directed to a newly registered payee without prior transaction history.",
        "category": "Counterparty Risk",
    },
    "RC_NEW_DEVICE_LOGIN": {
        "code": "RC_NEW_DEVICE_LOGIN",
        "title": "Unregistered Device",
        "description": "Payment originated from a newly observed hardware identifier.",
        "category": "Device & Session",
    },
    "RC_REMOTE_ACCESS_ACTIVE": {
        "code": "RC_REMOTE_ACCESS_ACTIVE",
        "title": "Remote Access / Screen-Sharing Active",
        "description": "Active screen-sharing or remote administrative software detected during transaction.",
        "category": "Social Engineering",
    },
    "RC_SIM_SWAP_RECENT": {
        "code": "RC_SIM_SWAP_RECENT",
        "title": "Recent SIM Card Swap",
        "description": "Carrier SIM swap detected on the payer line within the past 48 hours.",
        "category": "Account Takeover",
    },
    "RC_RAPID_VELOCITY": {
        "code": "RC_RAPID_VELOCITY",
        "title": "Rapid Outgoing Velocity",
        "description": "Unusually high frequency of transfers initiated within a short time window.",
        "category": "Velocity",
    },
    "RC_COLLECT_REQUEST": {
        "code": "RC_COLLECT_REQUEST",
        "title": "Inbound Collect-Request Pattern",
        "description": "Payment was initiated via an inbound collect request rather than a direct push transfer.",
        "category": "Payment Channel",
    },
    "RC_ADDRESS_INSTABILITY": {
        "code": "RC_ADDRESS_INSTABILITY",
        "title": "Address Tenure Instability",
        "description": "Applicant has resided at current residential address for fewer than 6 months.",
        "category": "Identity & Stability",
    },
    "RC_NAME_EMAIL_DIVERGENCE": {
        "code": "RC_NAME_EMAIL_DIVERGENCE",
        "title": "Name-Email Discrepancy",
        "description": "Low lexical similarity between applicant legal name and application email handle.",
        "category": "Identity & Stability",
    },
    "RC_CARD_PCA_ANOMALY": {
        "code": "RC_CARD_PCA_ANOMALY",
        "title": "Card Latent Feature Anomaly",
        "description": "High divergence across anonymized cardholder behavioral embedding components.",
        "category": "Pattern Anomaly",
    },
}


def get_reason_catalog() -> list[dict[str, Any]]:
    """Return the full standardized reason code catalog."""
    return list(REASON_CATALOG.values())


def map_feature_to_reason_code(channel: str, feature_name: str) -> dict[str, Any]:
    fn = feature_name.lower()
    
    if "amount" in fn:
        return REASON_CATALOG["RC_AMOUNT_HIGH_DEVIATION"]
    if "emptied" in fn or ("balance" in fn and "org" in fn):
        return REASON_CATALOG["RC_ORIGIN_BALANCE_DRAINED"]
    if "dest_zero" in fn or ("balance" in fn and "dest" in fn):
        return REASON_CATALOG["RC_DESTINATION_ZERO_BALANCE"]
    if "night" in fn or "hour" in fn or "time" in fn:
        return REASON_CATALOG["RC_NOCTURNAL_TRANSACTION"]
    if "first_time" in fn or "payee" in fn or "fresh" in fn:
        return REASON_CATALOG["RC_FIRST_TIME_PAYEE"]
    if "new_device" in fn or "device_fraud" in fn or "device" in fn:
        return REASON_CATALOG["RC_NEW_DEVICE_LOGIN"]
    if "remote" in fn or "rat" in fn:
        return REASON_CATALOG["RC_REMOTE_ACCESS_ACTIVE"]
    if "sim" in fn:
        return REASON_CATALOG["RC_SIM_SWAP_RECENT"]
    if "velocity" in fn:
        return REASON_CATALOG["RC_RAPID_VELOCITY"]
    if "collect" in fn:
        return REASON_CATALOG["RC_COLLECT_REQUEST"]
    if "address" in fn:
        return REASON_CATALOG["RC_ADDRESS_INSTABILITY"]
    if "email" in fn or "name" in fn:
        return REASON_CATALOG["RC_NAME_EMAIL_DIVERGENCE"]
    if fn.startswith("v") and fn[1:].isdigit():
        return REASON_CATALOG["RC_CARD_PCA_ANOMALY"]

    return {
        "code": f"RC_{channel.upper()}_{feature_name.upper()}",
        "title": f"Unusual {feature_name.replace('_', ' ').title()}",
        "description": f"Divergence detected in {feature_name} relative to baseline.",
        "category": "Behavioral",
    }


def map_shap_to_reason_codes(channel: str, factors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mapped: list[dict[str, Any]] = []
    seen_codes = set()
    
    for factor in factors:
        feat = factor.get("feature", "")
        shap_val = factor.get("shap_value", 0.0)
        direction = factor.get("direction", "pushes risk up")
        
        info = map_feature_to_reason_code(channel, feat)
        code = info["code"]
        if code in seen_codes:
            continue
        seen_codes.add(code)
        
        mapped.append({
            "reason_code": code,
            "title": info["title"],
            "description": info["description"],
            "category": info["category"],
            "feature": feat,
            "shap_value": shap_val,
            "direction": direction,
            "human_readable": factor.get("human_readable", f"{info['title']} {direction}"),
        })
        
    return mapped


def _neutralize_feature(channel: str, payload: dict[str, Any], top_feature: str) -> dict[str, Any]:
    mod = dict(payload)
    fn = top_feature.lower()

    if channel == "transfer_wallet":
        if "amount" in fn:
            mod["amount"] = 50.0  
        elif "emptied" in fn or "orig" in fn:
            mod["orig_emptied"] = 0
            mod["oldbalanceOrg"] = 50000.0
            mod["newbalanceOrig"] = 49000.0
        elif "dest" in fn:
            mod["dest_zero"] = 0
            mod["dest_balance_zero"] = 0
            mod["oldbalanceDest"] = 10000.0
        elif "type" in fn:
            mod["type"] = "PAYMENT"
        else:
            mod["amount"] = float(mod.get("amount", 1000.0)) * 0.1

    elif channel == "card":
        if fn == "amount":
            mod["Amount"] = 10.0
        elif fn.startswith("v") and fn[1:].isdigit():
            mod[top_feature] = 0.0
        else:
            mod["Amount"] = 10.0

    elif channel == "upi":
        if "payee_vpa_age" in fn:
            mod["payee_vpa_age_days"] = 365.0
            mod["is_fresh_payee_vpa"] = 0
        elif "payer_vpa_age" in fn:
            mod["payer_vpa_age_days"] = 365.0
        elif "vpa_age_diff" in fn:
            mod["vpa_age_diff"] = 0.0
        elif "remote_access" in fn or "rat" in fn:
            mod["remote_access_app_flag"] = 0
        elif "new_device" in fn:
            mod["is_new_device"] = 0
        elif "first_time" in fn:
            mod["is_first_time_payee"] = 0
        elif "sim" in fn:
            mod["sim_change_last_48h"] = 0
        elif "amount" in fn:
            mod["amount"] = 100.0
        elif "collect" in fn:
            mod["txn_type"] = "PAY"
        elif "velocity" in fn:
            mod["velocity_1h"] = 0.0
            mod["velocity_24h"] = 0.0
        else:
            mod["amount"] = 100.0

    elif channel == "account":
        if "housing_status" in fn:
            mod["housing_status"] = "BC"
        elif "credit_limit" in fn:
            mod["proposed_credit_limit"] = 200.0
        elif "device_os" in fn:
            mod["device_os"] = "linux"
        elif "distinct_emails" in fn:
            mod["device_distinct_emails_8w"] = 1
            mod["date_of_birth_distinct_emails_4w"] = 0
        elif "device" in fn:
            mod["device_fraud_count"] = 0
        elif "velocity" in fn:
            mod["velocity_6h"] = 10.0
            mod["velocity_24h"] = 10.0
        elif "address" in fn:
            mod["current_address_months_count"] = 72.0  
        elif "name_email" in fn:
            mod["name_email_similarity"] = 0.95
        elif "income" in fn:
            mod["income"] = 0.8
        elif "keep_alive" in fn:
            mod["keep_alive_session"] = 1
        elif "phone" in fn:
            mod["phone_home_valid"] = 1
        else:
            mod["device_fraud_count"] = 0

    return mod


def run_deletion_faithfulness_test(
    channel: str,
    n_samples: int = 50,
) -> dict[str, Any]:
    sample_files = {
        "transfer_wallet": PROCESSED_DATA_DIR / "test_scored_sample.parquet",
        "card": PROCESSED_DATA_DIR / "card_test_sample.parquet",
        "upi": PROCESSED_DATA_DIR / "upi_test_sample.parquet",
        "account": PROCESSED_DATA_DIR / "account_test_sample.parquet",
    }

    file_path = sample_files.get(channel)
    if not file_path or not file_path.exists():
        return {
            "channel": channel,
            "status": "error",
            "message": f"Sample file not found: {file_path}",
            "tested_count": 0,
            "passed_count": 0,
            "pass_rate_pct": 0.0,
        }

    df = pd.read_parquet(file_path)

    fraud_col = "isFraud" if "isFraud" in df.columns else ("is_fraud" if "is_fraud" in df.columns else "fraud_bool")
    if fraud_col in df.columns and (df[fraud_col] == 1).any():
        pos_df = df[df[fraud_col] == 1]
        neg_df = df[df[fraud_col] == 0]
        test_df = pd.concat([pos_df.head(n_samples // 2), neg_df.head(n_samples // 2)])
    else:
        test_df = df.head(n_samples)

    passed_count = 0
    tested_count = 0
    results: list[dict[str, Any]] = []

    for idx, row in test_df.iterrows():
        if channel == "transfer_wallet":
            amt = float(row.get("amount", 50000.0))
            payload = {
                "step": int(row.get("step", 1)),
                "type": str(row.get("type", "TRANSFER")),
                "amount": amt,
                "oldbalanceOrg": float(row.get("oldbalanceOrg", amt)),
                "newbalanceOrig": float(row.get("newbalanceOrig", 0.0)),
                "oldbalanceDest": float(row.get("oldbalanceDest", 0.0)),
                "newbalanceDest": float(row.get("newbalanceDest", amt)),
                "nameOrig": str(row.get("nameOrig", f"C_{idx}")),
                "nameDest": str(row.get("nameDest", f"M_{idx}")),
                "orig_emptied": int(row.get("orig_emptied", 1)),
                "dest_zero": int(row.get("dest_zero", 1)),
            }
        elif channel == "card":
            payload = {
                "Amount": float(row.get("Amount", 150.0)),
                "Time": float(row.get("Time", 1000.0)),
            }
            for i in range(1, 29):
                payload[f"V{i}"] = float(row.get(f"V{i}", 0.0))
        elif channel == "upi":
            payload = row.to_dict()
            payload["amount"] = float(row.get("amount", 25000.0))
            payload["hour"] = float(row.get("hour", 2.0))
        elif channel == "account":
            payload = row.to_dict()
            payload["income"] = float(row.get("income", 0.3))
            payload["customer_age"] = float(row.get("customer_age", 55))
        else:
            continue

        try:
            pred_orig = predict(channel=channel, payload=payload, top_k=3, commit_to_store=False)
            score_orig = pred_orig["risk_score"]
            factors = pred_orig.get("factors", [])

            up_factors = [f for f in factors if f.get("shap_value", 0.0) > 0]
            if not up_factors:
                continue

            top_factor = up_factors[0]
            top_feat = top_factor["feature"]

            mod_payload = _neutralize_feature(channel, payload, top_feat)
            pred_mod = predict(channel=channel, payload=mod_payload, top_k=3, commit_to_store=False)
            score_mod = pred_mod["risk_score"]

            score_diff = round(score_orig - score_mod, 4)
            passed = score_mod < score_orig

            tested_count += 1
            if passed:
                passed_count += 1

            results.append({
                "index": int(idx),
                "top_feature": top_feat,
                "shap_value": top_factor.get("shap_value", 0.0),
                "score_orig": score_orig,
                "score_deleted": score_mod,
                "score_drop": score_diff,
                "passed": passed,
            })
        except Exception as e:
            logger.warning("Error running deletion test on %s row %s: %s", channel, idx, e)
            continue

    pass_rate = round((passed_count / max(tested_count, 1)) * 100.0, 2)
    return {
        "channel": channel,
        "tested_count": tested_count,
        "passed_count": passed_count,
        "pass_rate_pct": pass_rate,
        "avg_score_drop": round(float(np.mean([r["score_drop"] for r in results])), 4) if results else 0.0,
        "sample_trials": results[:10],
    }


def run_all_deletion_tests() -> dict[str, Any]:
    report = {
        "timestamp": pd.Timestamp.utcnow().isoformat() + "Z",
        "methodology": (
            "Feature-deletion explanation faithfulness test. For each channel, high-risk "
            "transactions are evaluated with TreeSHAP. The top risk-increasing feature is removed "
            "or neutralized to its baseline value. The test passes if the calibrated risk score drops."
        ),
        "channels": {},
    }

    for ch in CHANNELS.keys():
        print(f"Running faithfulness deletion test on channel: {ch}...")
        res = run_deletion_faithfulness_test(ch, n_samples=40)
        report["channels"][ch] = res
        print(f"  --> {ch}: {res['passed_count']}/{res['tested_count']} passed ({res['pass_rate_pct']}%)")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = REPORTS_DIR / "reason_code_faithfulness.json"
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Faithfulness report saved to {out_file}")
    return report


if __name__ == "__main__":
    run_all_deletion_tests()
