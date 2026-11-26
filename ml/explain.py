from __future__ import annotations

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from ml.config import (
    FIGURES_DIR,
    METADATA_FILE,
    MODEL_FILE,
    RANDOM_STATE,
    TARGET_COL,
    TEST_DATA_PATH,
)
from ml.features import FEATURE_COLUMNS, engineer_features

_CACHED_MODEL = None
_CACHED_CALIBRATOR = None
_CACHED_EXPLAINER = None


def get_model_and_explainer():
    global _CACHED_MODEL, _CACHED_CALIBRATOR, _CACHED_EXPLAINER
    if _CACHED_MODEL is None:
        payload = joblib.load(MODEL_FILE)
        if isinstance(payload, dict):
            _CACHED_MODEL = payload.get("model", payload)
            _CACHED_CALIBRATOR = payload.get("calibrator", None)
            if _CACHED_CALIBRATOR is not None and not hasattr(_CACHED_CALIBRATOR, "multi_class"):
                _CACHED_CALIBRATOR.multi_class = "auto"
        else:
            _CACHED_MODEL = payload
            _CACHED_CALIBRATOR = None
        _CACHED_EXPLAINER = shap.TreeExplainer(_CACHED_MODEL)
    return _CACHED_MODEL, _CACHED_CALIBRATOR, _CACHED_EXPLAINER


def humanize_shap_factor(feature_name: str, shap_val: float, raw_val: float) -> str:
    sign = "+" if shap_val > 0 else "-"
    delta = f"{sign}{abs(shap_val):.2f} risk"

    if feature_name == "orig_emptied":
        if raw_val == 1:
            return f"Transaction liquidates entire sender account balance ({delta})"
        return f"Sender retains remaining balance in account ({delta})"

    if feature_name == "amount" or feature_name == "log_amount":
        if raw_val > 100_000:
            return f"High transfer amount of ₹{raw_val:,.2f} ({delta})"
        return f"Routine transfer amount of ₹{raw_val:,.2f} ({delta})"

    if feature_name == "amount_to_balance_ratio":
        if raw_val >= 1.0:
            return f"Requested amount exceeds available sender balance ({delta})"
        return f"Requested amount is well within sender balance ({delta})"

    if feature_name == "type_TRANSFER":
        if raw_val == 1:
            return f"Outbound fund transfer payment channel ({delta})"
        return f"Non-transfer transaction channel ({delta})"

    if feature_name == "type_CASH_OUT":
        if raw_val == 1:
            return f"Cash-out withdrawal liquidation channel ({delta})"
        return f"Non-cashout transaction channel ({delta})"

    if feature_name in ("type_PAYMENT", "type_CASH_IN", "type_DEBIT"):
        if raw_val == 1:
            return f"Low-risk routine transaction channel ({delta})"
        return f"Standard channel type ({delta})"

    if feature_name == "oldbalanceOrg":
        return f"Sender starting balance of ₹{raw_val:,.2f} ({delta})"

    if feature_name == "oldbalanceDest":
        if raw_val == 0.0:
            return f"Recipient account starts with ₹0.00 balance ({delta})"
        return f"Recipient account has existing balance of ₹{raw_val:,.2f} ({delta})"

    if feature_name == "dest_balance_zero":
        if raw_val == 1:
            return f"Destination starts with zero balance ({delta})"
        return f"Destination maintains positive balance ({delta})"

    if feature_name == "is_dest_merchant":
        if raw_val == 1:
            return f"Recipient is a verified commercial merchant ({delta})"
        return f"Recipient is an unverified individual account ({delta})"

    if feature_name == "dest_past_tx_count":
        return f"Recipient has received {int(raw_val):,} previous transactions ({delta})"

    if feature_name == "dest_total_past_amount":
        return f"Recipient cumulative past volume of ₹{raw_val:,.2f} ({delta})"

    if feature_name == "dest_time_since_last_tx":
        if raw_val >= 999.0:
            return f"First-time recipient with no prior transaction history ({delta})"
        return f"{raw_val:.1f} hours elapsed since recipient's prior transaction ({delta})"

    if feature_name in ("dest_tx_last_1h", "dest_tx_last_6h", "dest_tx_last_24h"):
        window = feature_name.split("_")[-1]
        return f"Recipient received {int(raw_val)} transactions in past {window} ({delta})"

    if feature_name == "is_night":
        if raw_val == 1:
            return f"Nocturnal attack window (00:00–05:00) ({delta})"
        return f"Normal daytime operating hours ({delta})"

    if feature_name == "hour_of_day":
        return f"Transaction initiated at {int(raw_val):02d}:00 hours ({delta})"

    return f"Feature '{feature_name}' value {raw_val:.2f} ({delta})"


def explain_transaction(
    row_or_dict: pd.Series | dict | pd.DataFrame | np.ndarray,
    top_k: int = 3,
    precomputed_X: np.ndarray | None = None,
) -> tuple[list[str], list[dict]]:
    model, _, explainer = get_model_and_explainer()

    if precomputed_X is not None:
        X_arr = np.asarray(precomputed_X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
    elif isinstance(row_or_dict, np.ndarray):
        X_arr = row_or_dict if row_or_dict.ndim == 2 else row_or_dict.reshape(1, -1)
    elif isinstance(row_or_dict, pd.DataFrame) and all(c in row_or_dict.columns for c in FEATURE_COLUMNS):
        X_arr = row_or_dict[FEATURE_COLUMNS].to_numpy()
    else:
        if isinstance(row_or_dict, pd.Series):
            df_row = pd.DataFrame([row_or_dict])
        elif isinstance(row_or_dict, dict):
            df_row = pd.DataFrame([row_or_dict])
        else:
            df_row = row_or_dict.copy()

        X_row, _ = engineer_features(df_row, is_training=False)
        X_arr = X_row[FEATURE_COLUMNS].to_numpy()

    shap_vals = explainer.shap_values(X_arr)
    if isinstance(shap_vals, list):
        vals = shap_vals[1][0]
    elif shap_vals.ndim == 2:
        vals = shap_vals[0]
    else:
        vals = shap_vals[0]

    abs_indices = np.argsort(np.abs(vals))[::-1]

    top_reasons = []
    factor_details = []

    for idx in abs_indices:
        col = FEATURE_COLUMNS[idx]
        sh_val = float(vals[idx])
        raw_val = float(X_arr[0, idx])
        direction = "increases risk" if sh_val > 0 else "reduces risk"

        text = humanize_shap_factor(col, sh_val, raw_val)
        factor_details.append({
            "feature": col,
            "shap_value": round(sh_val, 4),
            "raw_value": round(raw_val, 4),
            "direction": direction,
            "description": text,
            "human_readable": text,
        })

        if len(top_reasons) < top_k:
            top_reasons.append(text)

    return top_reasons, factor_details


def generate_global_shap_figures(sample_size: int = 2000):
    print("Generating global TreeSHAP figures on test sample...")
    model, _, explainer = get_model_and_explainer()
    test_df = pd.read_parquet(TEST_DATA_PATH)


    np.random.seed(RANDOM_STATE)
    pos_idx = np.where(test_df[TARGET_COL] == 1)[0]
    neg_idx = np.where(test_df[TARGET_COL] == 0)[0]
    sample_pos = np.random.choice(pos_idx, size=min(len(pos_idx), 500), replace=False)
    sample_neg = np.random.choice(neg_idx, size=min(len(neg_idx), sample_size - len(sample_pos)), replace=False)
    sub_indices = np.concatenate([sample_pos, sample_neg])
    np.random.shuffle(sub_indices)

    X_sub = test_df.iloc[sub_indices][FEATURE_COLUMNS]
    shap_values = explainer.shap_values(X_sub)
    if isinstance(shap_values, list):
        vals = shap_values[1]
    else:
        vals = shap_values

    plt.figure(figsize=(10, 7))
    shap.summary_plot(vals, X_sub, max_display=15, show=False)
    plt.title("Global TreeSHAP Summary (Test Set Natural Distribution)", fontsize=13, weight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "10_shap_summary_beeswarm.png", dpi=200)
    plt.close()

    plt.figure(figsize=(9, 6))
    shap.summary_plot(vals, X_sub, plot_type="bar", max_display=15, show=False)
    plt.title("Global Mean |SHAP| Feature Attributions", fontsize=13, weight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "11_shap_feature_importance_bar.png", dpi=200)
    plt.close()

    print("Global SHAP figures saved successfully.")


if __name__ == "__main__":
    generate_global_shap_figures()
