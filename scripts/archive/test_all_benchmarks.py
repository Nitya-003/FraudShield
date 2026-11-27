import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
)

from ml.features import FEATURE_COLUMNS

print("Loading parquets...")
train_df = pd.read_parquet("data/processed/train.parquet")
train_us = pd.read_parquet("data/processed/train_undersampled.parquet")
val_df = pd.read_parquet("data/processed/val.parquet")
test_df = pd.read_parquet("data/processed/test.parquet")

X_train_full = train_df[FEATURE_COLUMNS].to_numpy()
y_train_full = train_df["isFraud"].to_numpy()

X_train_us = train_us[FEATURE_COLUMNS].to_numpy()
y_train_us = train_us["isFraud"].to_numpy()

X_val = val_df[FEATURE_COLUMNS].to_numpy()
y_val = val_df["isFraud"].to_numpy()

X_test = test_df[FEATURE_COLUMNS].to_numpy()
y_test = test_df["isFraud"].to_numpy()

print(f"Full Train: {len(y_train_full):,} rows ({y_train_full.sum():,} fraud)")
print(f"Undersampled Train: {len(y_train_us):,} rows ({y_train_us.sum():,} fraud)")
print(f"Val (Natural): {len(y_val):,} rows ({y_val.sum():,} fraud - {y_val.mean()*100:.4f}%)")
print(f"Test (Natural): {len(y_test):,} rows ({y_test.sum():,} fraud - {y_test.mean()*100:.4f}%)")

results = []

def eval_preds(name, y_true, probs, preds, fit_time=0.0):
    pr_auc = float(average_precision_score(y_true, probs)) if probs is not None else float("nan")
    roc_auc = float(roc_auc_score(y_true, probs)) if probs is not None else float("nan")
    rec = float(recall_score(y_true, preds))
    prec = float(precision_score(y_true, preds, zero_division=0))
    f1 = float(f1_score(y_true, preds, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    fpr = float(fp / (fp + tn))
    fa_per_100k = float(fpr * 100000)
    print(f"{name:32s} | PR-AUC: {pr_auc:.5f} | Rec: {rec*100:.2f}% | Prec: {prec*100:.2f}% | F1: {f1:.4f} | FPR: {fpr*100:.4f}% | FP: {fp:,}")
    return {
        "model": name,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "recall": rec,
        "precision": prec,
        "f1": f1,
        "fpr": fpr,
        "false_alarms_per_100k": fa_per_100k,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "fit_time_sec": round(fit_time, 2),
    }

print("\n--- TEST SET BENCHMARK (NATURAL PREVALENCE 918,617 ROWS) ---")

# 1. Rule 1: amount > 50,000
pred_r1 = (test_df["amount"] > 50000).astype(int).to_numpy()
eval_preds("Rule 1 (Amount > 50k)", y_test, pred_r1.astype(float), pred_r1)

# 2. Rule 2: type in (TRANSFER, CASH_OUT) and orig_emptied
pred_r2 = ((test_df["type_TRANSFER"] == 1) | (test_df["type_CASH_OUT"] == 1)) & (test_df["orig_emptied"] == 1)
pred_r2 = pred_r2.astype(int).to_numpy()
eval_preds("Rule 2 (Channel & Empty)", y_test, pred_r2.astype(float), pred_r2)

# 3. Logistic Regression (Trained on undersampled train with RobustScaler)
scaler = RobustScaler()
X_tr_s = scaler.fit_transform(X_train_us)
X_te_s = scaler.transform(X_test)
t0 = time.time()
lr = LogisticRegression(class_weight="balanced", max_iter=200, random_state=42)
lr.fit(X_tr_s, y_train_us)
t_lr = time.time() - t0
lr_probs = lr.predict_proba(X_te_s)[:, 1]
eval_preds("Logistic Regression", y_test, lr_probs, (lr_probs >= 0.5).astype(int), t_lr)

# 4. Random Forest (100 trees, max_depth=12)
t0 = time.time()
rf = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1)
rf.fit(X_train_us, y_train_us)
t_rf = time.time() - t0
rf_probs = rf.predict_proba(X_test)[:, 1]
eval_preds("Random Forest (100 trees)", y_test, rf_probs, (rf_probs >= 0.5).astype(int), t_rf)

# 5. LightGBM (Full Train, Cost-Weighted)
t0 = time.time()
lgb_cw = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.08, num_leaves=31, scale_pos_weight=50, random_state=42, n_jobs=-1, verbose=-1)
lgb_cw.fit(X_train_full, y_train_full)
t_cw = time.time() - t0
lgb_cw_probs = lgb_cw.predict_proba(X_test)[:, 1]
eval_preds("LightGBM (Full Train, spw=50)", y_test, lgb_cw_probs, (lgb_cw_probs >= 0.5).astype(int), t_cw)

# 6. LightGBM (Undersampled 1:10, Champion Candidate)
t0 = time.time()
lgb_us = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.08, num_leaves=31, random_state=42, n_jobs=-1, verbose=-1)
lgb_us.fit(X_train_us, y_train_us)
t_us = time.time() - t0
lgb_us_probs = lgb_us.predict_proba(X_test)[:, 1]
eval_preds("LightGBM (Undersampled 1:10)", y_test, lgb_us_probs, (lgb_us_probs >= 0.5).astype(int), t_us)

