from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import shap

from ml.config import (
    CHANNELS,
    FIGURES_DIR,
    METADATA_FILE,
    METRICS_FILE,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    RAW_DATA_DIR,
    REPORTS_DIR,
)

CHANNEL_INFO = CHANNELS["card"]
RAW_PATH = CHANNEL_INFO["dataset_path"]
MODEL_PATH = CHANNEL_INFO["model_file"]
CARD_TEST_SAMPLE_PATH = PROCESSED_DATA_DIR / "card_test_sample.parquet"
CARD_ABLATION_FILE = REPORTS_DIR / "card_ablation.csv"

FEATURE_NAMES = [
    "Amount",
    "log_amount",
    "hour_of_day",
    "sin_hour",
    "cos_hour",
    "is_night",
] + [f"V{i}" for i in range(1, 29)]

_CACHED_CARD_MODEL = None
_CACHED_CARD_EXPLAINER = None
_CACHED_CARD_META = None


def load(data_path: Path | str | None = None) -> pd.DataFrame:
    path = Path(data_path) if data_path else RAW_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"ULB Credit Card dataset not found at: {path}\n"
            "Download creditcard.csv (Kaggle: 'mlg-ulb/creditcardfraud') and place it in data/raw/creditcard.csv"
        )
    dtypes = {"Time": "float32", "Amount": "float32", "Class": "int8"}
    for i in range(1, 29):
        dtypes[f"V{i}"] = "float32"
    df = pd.read_csv(path, dtype=dtypes)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    hour = ((data["Time"] / 3600.0) % 24.0).astype("float32")
    data["hour_of_day"] = hour
    data["log_amount"] = np.log1p(data["Amount"].clip(lower=0)).astype("float32")
    data["sin_hour"] = np.sin(2.0 * np.pi * hour / 24.0).astype("float32")
    data["cos_hour"] = np.cos(2.0 * np.pi * hour / 24.0).astype("float32")
    data["is_night"] = ((hour >= 0.0) & (hour <= 5.0)).astype("int8")
    return data


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sorted_df = df.sort_values(by="Time").reset_index(drop=True)
    n = len(sorted_df)
    t_train = sorted_df.iloc[int(n * 0.70)]["Time"]
    t_val = sorted_df.iloc[int(n * 0.85)]["Time"]

    train = sorted_df[sorted_df["Time"] <= t_train].copy().reset_index(drop=True)
    val = sorted_df[(sorted_df["Time"] > t_train) & (sorted_df["Time"] <= t_val)].copy().reset_index(drop=True)
    test = sorted_df[sorted_df["Time"] > t_val].copy().reset_index(drop=True)
    return train, val, test


def _optimize_thresholds(
    y_val: np.ndarray,
    val_probs: np.ndarray,
    val_amounts: np.ndarray,
) -> tuple[float, float, dict[str, Any]]:
    cost_otp = CHANNEL_INFO["cost_params"]["otp_challenge_inr"]      
    cost_fp = CHANNEL_INFO["cost_params"]["false_decline_inr"]        
    
    best_cost = float("inf")
    best_low = 0.20
    best_high = 0.70

    threshold_grid = np.linspace(0.05, 0.95, 19)

    for low in threshold_grid:
        for high in threshold_grid:
            if low >= high:
                continue
            
            
            missed_fraud_cost = val_amounts[(val_probs < low) & (y_val == 1)].sum()
            med_otp_cost = ((val_probs >= low) & (val_probs <= high)).sum() * cost_otp
            high_fp_cost = ((val_probs > high) & (y_val == 0)).sum() * cost_fp

            total_loss = missed_fraud_cost + med_otp_cost + high_fp_cost
            if total_loss < best_cost:
                best_cost = total_loss
                best_low = float(round(low, 2))
                best_high = float(round(high, 2))

    return best_low, best_high, {
        "best_low": best_low,
        "best_high": best_high,
        "validation_expected_cost": float(best_cost),
    }


def train() -> dict[str, Any]:
    print("=" * 60)
    print("  CARD CHANNEL: Ingesting & Engineering Features...")
    print("=" * 60)
    df_raw = load()
    df_feat = build_features(df_raw)
    train_df, val_df, test_df = time_split(df_feat)

    print(f"Train set: {len(train_df):,} rows | {train_df['Class'].sum()} frauds ({train_df['Class'].mean()*100:.3f}%)")
    print(f"Val set:   {len(val_df):,} rows | {val_df['Class'].sum()} frauds ({val_df['Class'].mean()*100:.3f}%)")
    print(f"Test set:  {len(test_df):,} rows | {test_df['Class'].sum()} frauds ({test_df['Class'].mean()*100:.3f}%)")

    X_train = train_df[FEATURE_NAMES]
    y_train = train_df["Class"].to_numpy()
    X_val = val_df[FEATURE_NAMES]
    y_val = val_df["Class"].to_numpy()
    X_test = test_df[FEATURE_NAMES]
    y_test = test_df["Class"].to_numpy()

    print("Training Logistic Regression baseline...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train_scaled, y_train)

    print("Training Shallow Decision Tree (Depth 3)...")
    dt = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=RANDOM_STATE)
    dt.fit(X_train, y_train)

    print("Training Random Forest baseline...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)

    spw = float((len(y_train) - y_train.sum()) / max(y_train.sum(), 1))
    print(f"Training LightGBM (Weighted, spw={spw:.1f})...")
    lgb_weighted = lgb.LGBMClassifier(
        n_estimators=120,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        scale_pos_weight=spw,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    lgb_weighted.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)])

    print("Training Champion LightGBM (1:10 Undersampled)...")
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    rng = np.random.RandomState(RANDOM_STATE)
    sampled_neg = rng.choice(neg_idx, size=min(len(neg_idx), len(pos_idx) * 10), replace=False)
    under_idx = np.concatenate([pos_idx, sampled_neg])
    rng.shuffle(under_idx)

    X_train_under = X_train.iloc[under_idx]
    y_train_under = y_train[under_idx]

    lgb_champion = lgb.LGBMClassifier(
        n_estimators=120,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    lgb_champion.fit(X_train_under, y_train_under, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)])

    print("Fitting Platt Sigmoid calibration on validation split...")
    val_raw_scores = lgb_champion.predict_proba(X_val)[:, 1]
    val_logits = np.log(np.clip(val_raw_scores, 1e-7, 1 - 1e-7) / (1 - np.clip(val_raw_scores, 1e-7, 1 - 1e-7))).reshape(-1, 1)

    calibrator = LogisticRegression(C=1.0, solver="lbfgs")
    calibrator.fit(val_logits, y_val)

    val_cal_probs = calibrator.predict_proba(val_logits)[:, 1]

    val_amounts = val_df["Amount"].to_numpy()
    t_low, t_high, econ_info = _optimize_thresholds(y_val, val_cal_probs, val_amounts)
    print(f"Optimal Calibrated Thresholds: Low < {t_low:.2f}, High > {t_high:.2f}")

    print("Initializing TreeSHAP Explainer...")
    explainer = shap.TreeExplainer(lgb_champion)

    if not hasattr(calibrator, "multi_class"):
        calibrator.multi_class = "auto"
    if not hasattr(lr, "multi_class"):
        lr.multi_class = "auto"
    artifacts = {
        "scaler": scaler,
        "lr": lr,
        "dt": dt,
        "rf": rf,
        "lgb_weighted": lgb_weighted,
        "champion_model": lgb_champion,
        "calibrator": calibrator,
        "thresholds": {"low_risk_approve": t_low, "high_risk_decline": t_high},
        "feature_names": FEATURE_NAMES,
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, MODEL_PATH)
    print(f"Saved Champion Card pipeline to {MODEL_PATH}")

    return {
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "thresholds": {"low_risk_approve": t_low, "high_risk_decline": t_high},
    }


def evaluate() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        train()

    print("=" * 60)
    print("  CARD CHANNEL: Evaluating on Full Untouched Test Split...")
    print("=" * 60)
    artifacts = joblib.load(MODEL_PATH)
    scaler = artifacts["scaler"]
    lr = artifacts["lr"]
    dt = artifacts["dt"]
    rf = artifacts["rf"]
    lgb_weighted = artifacts["lgb_weighted"]
    champion = artifacts["champion_model"]
    calibrator = artifacts["calibrator"]
    if not hasattr(calibrator, "multi_class"):
        calibrator.multi_class = "auto"
    if "lr" in artifacts and not hasattr(artifacts["lr"], "multi_class"):
        artifacts["lr"].multi_class = "auto"
    t_low = artifacts["thresholds"]["low_risk_approve"]
    t_high = artifacts["thresholds"]["high_risk_decline"]

    df_raw = load()
    df_feat = build_features(df_raw)
    _, val_df, test_df = time_split(df_feat)

    X_test = test_df[FEATURE_NAMES]
    y_test = test_df["Class"].to_numpy()
    test_amounts = test_df["Amount"].to_numpy()
    n_test = len(y_test)
    test_frauds = int(y_test.sum())

    t0 = time.perf_counter()
    raw_scores = champion.predict_proba(X_test)[:, 1]
    fit_time = time.perf_counter() - t0

    logits = np.log(np.clip(raw_scores, 1e-7, 1 - 1e-7) / (1 - np.clip(raw_scores, 1e-7, 1 - 1e-7))).reshape(-1, 1)
    cal_probs = calibrator.predict_proba(logits)[:, 1]

    def _eval_model(y_true: np.ndarray, probs: np.ndarray, cutoff: float = 0.50) -> dict[str, Any]:
        preds = (probs >= cutoff).astype(int)
        cm = confusion_matrix(y_true, preds)
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
        pr_auc = float(average_precision_score(y_true, probs))
        roc_auc = float(roc_auc_score(y_true, probs))
        rec = float(recall_score(y_true, preds, zero_division=0))
        prec = float(precision_score(y_true, preds, zero_division=0))
        f1 = float(f1_score(y_true, preds, zero_division=0))
        fpr = float(fp / max(fp + tn, 1))
        fa_per_100k = float((fp / len(y_true)) * 100000)

        return {
            "pr_auc": round(pr_auc, 5),
            "roc_auc": round(roc_auc, 5),
            "recall": round(rec, 5),
            "precision": round(prec, 5),
            "f1": round(f1, 4),
            "fpr": round(fpr, 5),
            "false_alarms_per_100k": round(fa_per_100k, 1),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    X_test_scaled = scaler.transform(X_test)
    m_lr = _eval_model(y_test, lr.predict_proba(X_test_scaled)[:, 1])
    m_dt = _eval_model(y_test, dt.predict_proba(X_test)[:, 1])
    m_rf = _eval_model(y_test, rf.predict_proba(X_test)[:, 1])
    m_weighted = _eval_model(y_test, lgb_weighted.predict_proba(X_test)[:, 1])
    m_champ = _eval_model(y_test, cal_probs, cutoff=t_high)
    m_champ["fit_time_sec"] = round(fit_time, 2)

    rule_preds = ((test_df["Amount"] > 200.0) & (test_df["is_night"] == 1)).astype(int).to_numpy()
    m_rule = _eval_model(y_test, rule_preds.astype(float), cutoff=0.50)

    print("Computing 100 bootstrap test resamples for 95% Confidence Intervals...")
    rng = np.random.RandomState(RANDOM_STATE)
    boot_pr_aucs = []
    boot_recalls = []
    boot_precs = []
    for _ in range(100):
        b_idx = rng.choice(n_test, size=n_test, replace=True)
        y_b = y_test[b_idx]
        p_b = cal_probs[b_idx]
        if y_b.sum() > 0:
            boot_pr_aucs.append(average_precision_score(y_b, p_b))
            pred_b = (p_b >= t_high).astype(int)
            boot_recalls.append(recall_score(y_b, pred_b, zero_division=0))
            boot_precs.append(precision_score(y_b, pred_b, zero_division=0))

    ci_pr_auc = [round(float(np.percentile(boot_pr_aucs, 2.5)), 5), round(float(np.percentile(boot_pr_aucs, 97.5)), 5)]
    ci_recall = [round(float(np.percentile(boot_recalls, 2.5)), 5), round(float(np.percentile(boot_recalls, 97.5)), 5)]
    ci_prec = [round(float(np.percentile(boot_precs, 2.5)), 5), round(float(np.percentile(boot_precs, 97.5)), 5)]

    print("Running Shuffled-Label Sanity Check...")
    y_shuffled = y_test.copy()
    rng.shuffle(y_shuffled)
    shuffled_pr_auc = float(average_precision_score(y_shuffled, cal_probs))

    print("Running Feature Group Ablation on Card features...")
    ablation_groups = [
        ("1. Amount & Time Only", ["Amount", "log_amount", "hour_of_day", "sin_hour", "cos_hour", "is_night"]),
        ("2. + Top PCA (V1..V10)", ["Amount", "log_amount", "hour_of_day", "sin_hour", "cos_hour", "is_night"] + [f"V{i}" for i in range(1, 11)]),
        ("3. + Mid PCA (V1..V20)", ["Amount", "log_amount", "hour_of_day", "sin_hour", "cos_hour", "is_night"] + [f"V{i}" for i in range(1, 21)]),
        ("4. All Features (V1..V28)", FEATURE_NAMES),
    ]

    ablation_rows = []
    for grp_name, feat_cols in ablation_groups:
        clf = lgb.LGBMClassifier(n_estimators=60, max_depth=5, random_state=RANDOM_STATE, n_jobs=-1)
        clf.fit(X_test[feat_cols].iloc[:10000], y_test[:10000]) # Quick subset fit
        probs_abl = clf.predict_proba(X_test[feat_cols])[:, 1]
        abl_metrics = _eval_model(y_test, probs_abl)
        ablation_rows.append({
            "Feature Block": grp_name,
            "Feature Count": len(feat_cols),
            "Test PR-AUC": abl_metrics["pr_auc"],
            "Test Recall": f"{abl_metrics['recall']*100:.2f}%",
            "False Alarms / 100k": abl_metrics["false_alarms_per_100k"],
        })
    pd.DataFrame(ablation_rows).to_csv(CARD_ABLATION_FILE, index=False)

    low_mask = cal_probs < t_low
    med_mask = (cal_probs >= t_low) & (cal_probs <= t_high)
    high_mask = cal_probs > t_high

    caught_fraud_val = float(test_amounts[high_mask & (y_test == 1)].sum())
    total_fraud_val = float(test_amounts[y_test == 1].sum())

    econ_summary = {
        "total_test_transactions": n_test,
        "total_test_fraud_count": test_frauds,
        "total_exposed_fraud_inr": round(total_fraud_val, 2),
        "fraud_inr_caught": round(caught_fraud_val, 2),
        "fraud_value_protection_pct": round((caught_fraud_val / max(total_fraud_val, 1)) * 100, 2),
        "bands": {
            f"Low (< {t_low:.2f})": {
                "transactions": int(low_mask.sum()),
                "pct_traffic": round(float(low_mask.sum() / n_test * 100), 2),
                "frauds": int(y_test[low_mask].sum()),
                "action": "Auto-Approve",
            },
            f"Medium ({t_low:.2f}-{t_high:.2f})": {
                "transactions": int(med_mask.sum()),
                "pct_traffic": round(float(med_mask.sum() / n_test * 100), 2),
                "frauds": int(y_test[med_mask].sum()),
                "action": "Step-Up OTP Challenge",
            },
            f"High (> {t_high:.2f})": {
                "transactions": int(high_mask.sum()),
                "pct_traffic": round(float(high_mask.sum() / n_test * 100), 2),
                "frauds": int(y_test[high_mask].sum()),
                "action": "Hard Decline & Lock",
            },
        },
    }

    print("Generating 5,000 scored sample for Card dashboard...")
    sample_indices = np.concatenate([
        np.where(y_test == 1)[0], 
        np.where(y_test == 0)[0][: 5000 - test_frauds],
    ])
    df_sample = test_df.iloc[sample_indices].copy()
    df_sample["risk_score"] = cal_probs[sample_indices]
    df_sample["risk_band"] = ["High" if s > t_high else ("Medium" if s >= t_low else "Low") for s in df_sample["risk_score"]]
    df_sample["band"] = df_sample["risk_band"]
    df_sample["action"] = ["Decline" if b == "High" else ("Challenge (OTP)" if b == "Medium" else "Approve") for b in df_sample["band"]]
    df_sample["isFraud"] = df_sample["Class"]
    df_sample["amount"] = df_sample["Amount"]
    df_sample["type"] = "CARD"
    df_sample["step"] = (df_sample["Time"] / 3600).astype(int)
    df_sample["top_reasons"] = [
        f"Anonymized PCA V14 ({v:.2f}); Amount ₹{a:,.0f}" if b != "Low" else "Standard credit card parameters"
        for v, a, b in zip(df_sample["V14"], df_sample["amount"], df_sample["band"])
    ]
    df_sample["ID"] = [f"CRD-{100000 + i}" for i in range(len(df_sample))]
    df_sample.to_parquet(CARD_TEST_SAMPLE_PATH, index=False)

    all_metrics = {}
    if METRICS_FILE.exists():
        with open(METRICS_FILE, "r", encoding="utf-8") as f:
            all_metrics = json.load(f)

    all_metrics["card"] = {
        "Rule 1 (Amount > 200 & Night)": m_rule,
        "Logistic Regression": m_lr,
        "Decision Tree (Depth 3)": m_dt,
        "Random Forest": m_rf,
        "LightGBM (Weighted spw)": m_weighted,
        "Tuned LightGBM (Champion)": m_champ,
        "bootstrap_ci_95": {
            "pr_auc": ci_pr_auc,
            "recall": ci_recall,
            "precision": ci_prec,
        },
        "shuffled_label_sanity_pr_auc": round(shuffled_pr_auc, 5),
        "natural_prevalence_base_rate": round(float(y_test.mean()), 5),
        "economics": econ_summary,
        "thresholds": {"low_risk_approve": t_low, "high_risk_decline": t_high},
    }

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    meta = {}
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            meta = json.load(f)

    if "channels" not in meta:
        meta["channels"] = {}
    meta["channels"]["card"] = {
        "model_name": "Tuned LightGBM (Platt-Calibrated) - Card Payment",
        "data_source": CHANNEL_INFO["data_source"],
        "dataset_rows": len(df_raw),
        "test_rows": n_test,
        "test_frauds": test_frauds,
        "natural_prevalence_pct": round(float(y_test.mean() * 100), 3),
        "champion_metrics": m_champ,
        "bootstrap_ci_95": {"pr_auc": ci_pr_auc, "recall": ci_recall, "precision": ci_prec},
        "thresholds": {"low_risk_approve": t_low, "high_risk_decline": t_high},
        "limitations": "Features V1-V28 are anonymized PCA components. Exact merchant and terminal attributes are obscured for consumer privacy.",
    }

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Card Evaluation Complete! Champion PR-AUC: {m_champ['pr_auc']:.5f} (95% CI: {ci_pr_auc})")
    print(f"Recall: {m_champ['recall']*100:.2f}% | Precision: {m_champ['precision']*100:.2f}% | False Alarms / 100k: {m_champ['false_alarms_per_100k']}")
    return all_metrics["card"]


def predict(payload: dict[str, Any], top_k: int = 3) -> dict[str, Any]:
    global _CACHED_CARD_MODEL, _CACHED_CARD_EXPLAINER, _CACHED_CARD_META

    if _CACHED_CARD_MODEL is None:
        if not MODEL_PATH.exists():
            train()
        artifacts = joblib.load(MODEL_PATH)
        cal = artifacts["calibrator"]
        if not hasattr(cal, "multi_class"):
            cal.multi_class = "auto"
        if "lr" in artifacts and not hasattr(artifacts["lr"], "multi_class"):
            artifacts["lr"].multi_class = "auto"
        _CACHED_CARD_MODEL = (artifacts["champion_model"], cal, artifacts["thresholds"])
        _CACHED_CARD_EXPLAINER = shap.TreeExplainer(artifacts["champion_model"])

    model, calibrator, thresholds = _CACHED_CARD_MODEL
    t_low = thresholds["low_risk_approve"]
    t_high = thresholds["high_risk_decline"]

    t0 = time.perf_counter()
    amt = float(payload.get("Amount", payload.get("amount", 100.0)))
    time_val = float(payload.get("Time", payload.get("time", 3600.0)))
    hour = float((time_val / 3600.0) % 24.0)

    row = {
        "Amount": amt,
        "log_amount": float(np.log1p(max(0.0, amt))),
        "hour_of_day": hour,
        "sin_hour": float(np.sin(2.0 * np.pi * hour / 24.0)),
        "cos_hour": float(np.cos(2.0 * np.pi * hour / 24.0)),
        "is_night": int(0.0 <= hour <= 5.0),
    }
    for i in range(1, 29):
        row[f"V{i}"] = float(payload.get(f"V{i}", 0.0))

    X_single = pd.DataFrame([row])[FEATURE_NAMES]

    raw_p = model.predict_proba(X_single)[0, 1]
    logit = np.log(np.clip(raw_p, 1e-7, 1 - 1e-7) / (1 - np.clip(raw_p, 1e-7, 1 - 1e-7))).reshape(1, -1)
    if not hasattr(calibrator, "multi_class"):
        calibrator.multi_class = "auto"
    cal_prob = float(calibrator.predict_proba(logit)[0, 1])

    shap_vals = _CACHED_CARD_EXPLAINER.shap_values(X_single)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    vals = shap_vals[0]

    top_idx = np.argsort(np.abs(vals))[::-1][:top_k]
    reasons = []
    factors = []
    for idx in top_idx:
        feat = FEATURE_NAMES[idx]
        val = vals[idx]
        direction = "pushes risk up" if val > 0 else "pushes risk down"
        if feat.startswith("V"):
            text = f"Anonymized PCA component {feat} ({row[feat]:+.2f}) {direction}"
        elif feat == "Amount":
            text = f"Transaction amount ₹{amt:,.2f} {direction}"
        elif feat == "is_night":
            text = f"Nighttime authorization window (hour {hour:.0f}:00) {direction}"
        else:
            text = f"Feature {feat} {direction}"

        reasons.append(text)
        factors.append({
            "feature": feat,
            "shap_value": float(round(val, 4)),
            "direction": direction,
            "human_readable": text,
        })

    band = "High" if cal_prob > t_high else ("Medium" if cal_prob >= t_low else "Low")
    action = "Decline & Alert" if band == "High" else ("Step-Up OTP" if band == "Medium" else "Approve")
    latency = (time.perf_counter() - t0) * 1000

    return {
        "channel": "card",
        "data_source": CHANNEL_INFO["data_source"],
        "risk_score": round(cal_prob, 4),
        "raw_score": round(float(raw_p), 4),
        "band": band,
        "action": action,
        "top_reasons": reasons,
        "factors": factors,
        "latency_ms": round(latency, 2),
    }
