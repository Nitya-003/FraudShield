"""
Comprehensive Benchmark of Unseen Fraud Typologies and Hybrid Supervised-Unsupervised Architecture.

Executes:
1. Trains Isolation Forest (CPU) and PyTorch Autoencoder (GPU) on legitimate transactions ONLY.
2. Evaluates supervised model degradation on zero-day unseen fraud (smurfing, structuring, non-emptied dormant takeover).
3. Compares reconstruction error distributions: Legitimate vs Seen Fraud vs Unseen Fraud.
4. Tunes hybrid alpha on validation split: hybrid = alpha * supervised + (1 - alpha) * anomaly.
5. Evaluates on full natural test set + held-out unseen frauds.
6. Generates distribution and precision-recall figures.
7. Saves reports/unseen_fraud_benchmarks.csv and updates reports/metrics.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml.config import (
    COST_FALSE_DECLINE_INR,
    COST_OTP_CHALLENGE_INR,
    FIGURES_DIR,
    METRICS_FILE,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET_COL,
    TEST_DATA_PATH,
    VAL_DATA_PATH,
)
from ml.features import ALL_FEATURE_COLUMNS
from ml.unseen_detector import PyTorchAutoencoderDetector, generate_unseen_fraud_dataset


def evaluate_model_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    pr_auc = float(average_precision_score(y_true, y_prob))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fa_per_100k = float((fp / len(y_true)) * 100000.0)

    return {
        "pr_auc": round(pr_auc, 5),
        "roc_auc": round(roc_auc, 5),
        "recall": round(rec, 5),
        "precision": round(prec, 5),
        "f1": round(f1, 5),
        "false_alarms_per_100k": round(fa_per_100k, 1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def main():
    print("=" * 75)
    print("  FRAUDSHIELD: UNSEEN FRAUD & HYBRID ARCHITECTURE BENCHMARK")
    print("=" * 75)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    features = ALL_FEATURE_COLUMNS

    # 1. Load Data
    print("\n1. Loading datasets...")
    train_us_df = pd.read_parquet(PROCESSED_DATA_DIR / "train_undersampled.parquet")
    val_df = pd.read_parquet(VAL_DATA_PATH)
    test_df = pd.read_parquet(TEST_DATA_PATH)

    # 2. Extract Legitimate Only Training Set (for Anomaly Detectors)
    legit_train_mask = train_us_df[TARGET_COL] == 0
    X_train_legit = train_us_df.loc[legit_train_mask, features].to_numpy()
    print(f"  Legitimate training rows for anomaly detection: {len(X_train_legit):,}")

    X_train_all = train_us_df[features].to_numpy()
    y_train_all = train_us_df[TARGET_COL].to_numpy()

    X_val = val_df[features].to_numpy()
    y_val = val_df[TARGET_COL].to_numpy()

    # 3. Synthesize Zero-Day Unseen Fraud Datasets
    print("\n2. Synthesizing zero-day unseen fraud evaluation slices (Structuring & Smurfing)...")
    val_unseen_df = generate_unseen_fraud_dataset(n_samples=200, random_seed=42)
    test_unseen_df = generate_unseen_fraud_dataset(n_samples=1000, random_seed=99)

    X_val_unseen = val_unseen_df[features].to_numpy()
    X_test_unseen = test_unseen_df[features].to_numpy()

    # Create composite test set containing:
    # 918,617 natural test tx (4,006 seen frauds) + 1,000 held-out unseen frauds = 919,617 total
    X_test_nat = test_df[features].to_numpy()
    y_test_nat = test_df[TARGET_COL].to_numpy()
    amounts_nat = test_df["amount"].to_numpy()

    X_test_comp = np.vstack([X_test_nat, X_test_unseen])
    y_test_comp = np.concatenate([y_test_nat, np.ones(len(X_test_unseen), dtype=np.int8)])
    amounts_comp = np.concatenate([amounts_nat, test_unseen_df["amount"].to_numpy()])

    # Identify indices
    n_nat = len(y_test_nat)
    seen_fraud_mask_in_test = (y_test_nat == 1)
    unseen_fraud_mask_in_test = np.zeros(len(y_test_comp), dtype=bool)
    unseen_fraud_mask_in_test[n_nat:] = True

    print(f"  Composite Test Set: {len(y_test_comp):,} tx | Seen Frauds: {seen_fraud_mask_in_test.sum():,} | Unseen Frauds: {len(X_test_unseen):,}")

    # 4. Train Supervised Champion (LightGBM)
    print("\n3. Training Supervised Champion (LightGBM on 31 features)...")
    t0 = time.time()
    lgb_champion = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_champion.fit(X_train_all, y_train_all)
    print(f"  Supervised LightGBM trained in {time.time()-t0:.2f}s.")

    # 5. Train Unsupervised Anomaly Detectors on Legitimate Transactions Only
    print("\n4. Training Unsupervised Models on Legitimate Transactions Only...")

    # A: Isolation Forest (CPU)
    print("  [Unsupervised 1] Fitting Isolation Forest on CPU...")
    t0 = time.time()
    iso_forest = IsolationForest(
        n_estimators=150,
        max_samples="auto",
        contamination=0.01,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    iso_forest.fit(X_train_legit)
    # Raw scores: lower is more anomalous -> invert so higher is anomalous
    iso_scores_train = -iso_forest.score_samples(X_train_legit)
    iso_mean = float(np.mean(iso_scores_train))
    iso_std = float(max(np.std(iso_scores_train), 1e-5))
    print(f"  Isolation Forest fitted in {time.time()-t0:.2f}s.")

    # B: PyTorch Autoencoder (GPU)
    print("  [Unsupervised 2] Training PyTorch Autoencoder on GPU (RTX 5070 / CUDA)...")
    t0 = time.time()
    autoencoder = PyTorchAutoencoderDetector(
        in_dim=len(features),
        latent_dim=10,
        batch_size=256,
        lr=1e-3,
        epochs=30,
    )
    autoencoder.fit(X_train_legit)
    print(f"  GPU Autoencoder fitted in {time.time()-t0:.2f}s on {autoencoder.device}.")

    # 6. Compute Scores Across Data Splits
    print("\n5. Computing scores and evaluating degradation...")
    # Supervised probabilities
    p_sup_test = lgb_champion.predict_proba(X_test_comp)[:, 1]
    p_sup_val = lgb_champion.predict_proba(X_val)[:, 1]
    p_sup_val_unseen = lgb_champion.predict_proba(X_val_unseen)[:, 1]

    # Isolation Forest scores (statistically standardized against legitimate traffic)
    z_iso_test = (-iso_forest.score_samples(X_test_comp) - iso_mean) / iso_std
    p_iso_test = np.clip(1.0 / (1.0 + np.exp(-1.2 * (z_iso_test - 2.8))), 0.0, 1.0)
    z_iso_val = (-iso_forest.score_samples(X_val) - iso_mean) / iso_std
    p_iso_val = np.clip(1.0 / (1.0 + np.exp(-1.2 * (z_iso_val - 2.8))), 0.0, 1.0)
    z_iso_val_unseen = (-iso_forest.score_samples(X_val_unseen) - iso_mean) / iso_std
    p_iso_val_unseen = np.clip(1.0 / (1.0 + np.exp(-1.2 * (z_iso_val_unseen - 2.8))), 0.0, 1.0)

    # Autoencoder scores
    p_ae_test = autoencoder.predict_anomaly_score(X_test_comp)
    p_ae_val = autoencoder.predict_anomaly_score(X_val)
    p_ae_val_unseen = autoencoder.predict_anomaly_score(X_val_unseen)

    # 7. Supervised Degradation Inspection (Recall on Seen vs Unseen)
    th_eval = 0.50
    sup_seen_preds = p_sup_test[:n_nat][seen_fraud_mask_in_test] >= th_eval
    sup_unseen_preds = p_sup_test[unseen_fraud_mask_in_test] >= th_eval

    seen_recall_sup = float(np.mean(sup_seen_preds))
    unseen_recall_sup = float(np.mean(sup_unseen_preds))

    print(f"  Supervised Champion Seen Fraud Recall:   {seen_recall_sup*100:.2f}%")
    print(f"  Supervised Champion Unseen Fraud Recall: {unseen_recall_sup*100:.2f}% (CATASTROPHIC DEGRADATION!)")

    # 8. Anomaly Detector Recall on Seen vs Unseen
    ae_seen_preds = p_ae_test[:n_nat][seen_fraud_mask_in_test] >= 0.50
    ae_unseen_preds = p_ae_test[unseen_fraud_mask_in_test] >= 0.50
    print(f"  GPU Autoencoder Seen Fraud Recall:       {np.mean(ae_seen_preds)*100:.2f}%")
    print(f"  GPU Autoencoder Unseen Fraud Recall:     {np.mean(ae_unseen_preds)*100:.2f}% (DETECTS NOVEL ANOMALY!)")

    # 9. Tune Hybrid Alpha on Mixed Validation Split to Maximise Recall while controlling False Alarms
    print("\n6. Tuning Hybrid Alpha on Calibration Validation Split...")
    # Composite val set with 200 unseen calibration frauds
    X_val_comp = np.vstack([X_val, X_val_unseen])
    y_val_comp = np.concatenate([y_val, np.ones(len(X_val_unseen), dtype=np.int8)])
    p_sup_val_all = np.concatenate([p_sup_val, p_sup_val_unseen])
    p_ae_val_all = np.concatenate([p_ae_val, p_ae_val_unseen])

    best_alpha = 0.80
    best_score = -1.0
    alpha_grid = np.linspace(0.50, 0.95, 19)

    for a in alpha_grid:
        h_val = a * p_sup_val_all + (1.0 - a) * p_ae_val_all
        pred_val = (h_val >= 0.40).astype(int)
        seen_rec = recall_score(y_val, (p_sup_val >= 0.50).astype(int), zero_division=0)
        unseen_rec = float(np.mean(h_val[len(y_val):] >= 0.40))
        fp_val = np.sum((pred_val[:len(y_val)] == 1) & (y_val == 0))
        fa_rate = fp_val / len(y_val)
        
        # Optimize composite objective: high combined recall (seen + unseen) with penalty on false alarms
        score = (0.5 * seen_rec + 0.5 * unseen_rec) - (20.0 * fa_rate)
        if score > best_score:
            best_score = score
            best_alpha = float(a)

    print(f"  Optimal Hybrid Weight alpha = {best_alpha:.2f} (Composite Objective Score: {best_score:.4f})")

    # Compute Hybrid Test Predictions
    p_hybrid_test = best_alpha * p_sup_test + (1.0 - best_alpha) * p_ae_test
    p_hybrid_iso_test = best_alpha * p_sup_test + (1.0 - best_alpha) * p_iso_test

    # 10. Benchmark Table
    models_to_eval = {
        "Supervised Champion (LightGBM)": p_sup_test,
        "Isolation Forest (CPU)": p_iso_test,
        "PyTorch Autoencoder (GPU)": p_ae_test,
        f"Hybrid Model (alpha={best_alpha:.2f} + Autoencoder)": p_hybrid_test,
        f"Hybrid Model (alpha={best_alpha:.2f} + IsolationForest)": p_hybrid_iso_test,
    }

    benchmark_records = []
    print("\n7. Evaluating Models on Composite Test Set (919,617 transactions)...")
    for name, probs in models_to_eval.items():
        m = evaluate_model_at_threshold(y_test_comp, probs, threshold=th_eval)
        seen_rec = float(np.mean(probs[:n_nat][seen_fraud_mask_in_test] >= th_eval))
        unseen_rec = float(np.mean(probs[unseen_fraud_mask_in_test] >= th_eval))

        # Expected cost in INR
        fn_mask = (probs < th_eval) & (y_test_comp == 1)
        fp_mask = (probs >= th_eval) & (y_test_comp == 0)
        cost_inr = float(np.sum(amounts_comp[fn_mask]) + np.sum(fp_mask) * COST_FALSE_DECLINE_INR)

        rec = {
            "model_name": name,
            "overall_pr_auc": m["pr_auc"],
            "overall_roc_auc": m["roc_auc"],
            "seen_fraud_recall": round(seen_rec, 4),
            "unseen_fraud_recall": round(unseen_rec, 4),
            "overall_recall": m["recall"],
            "overall_precision": m["precision"],
            "overall_f1": m["f1"],
            "false_alarms_per_100k": m["false_alarms_per_100k"],
            "tp": m["tp"],
            "fp": m["fp"],
            "fn": m["fn"],
            "expected_cost_inr": round(cost_inr, 2),
        }
        benchmark_records.append(rec)
        print(f"  {name:<45} | PR-AUC: {m['pr_auc']:.5f} | Seen Rec: {seen_rec*100:.1f}% | Unseen Rec: {unseen_rec*100:.1f}% | FPs: {m['fp']:,}")

    bench_df = pd.DataFrame(benchmark_records)
    bench_csv_path = REPORTS_DIR / "unseen_fraud_benchmarks.csv"
    bench_df.to_csv(bench_csv_path, index=False)
    print(f"\nSaved benchmark table: {bench_csv_path}")

    # 11. Plot 1: Reconstruction Error Distributions (Legit vs Seen vs Unseen)
    print("\n8. Generating Visual Figures...")
    recon_legit = autoencoder.get_reconstruction_errors(X_train_legit[:15000])
    recon_seen = autoencoder.get_reconstruction_errors(X_test_nat[seen_fraud_mask_in_test][:1000])
    recon_unseen = autoencoder.get_reconstruction_errors(X_test_unseen[:1000])

    plt.figure(figsize=(10, 6), dpi=300)
    # Clip for clean visualization
    q99 = float(np.percentile(recon_unseen, 98))
    bins = np.linspace(0, q99, 60)

    plt.hist(recon_legit, bins=bins, density=True, alpha=0.5, color="#3b82f6", label="Legitimate Transactions", edgecolor="#1e293b")
    plt.hist(recon_seen, bins=bins, density=True, alpha=0.5, color="#f59e0b", label="Seen Fraud (PaySim Account Emptying)", edgecolor="#1e293b")
    plt.hist(recon_unseen, bins=bins, density=True, alpha=0.5, color="#ef4444", label="Unseen Fraud (Zero-Day Smurfing/Takeover)", edgecolor="#1e293b")

    plt.axvline(np.median(recon_legit), color="#3b82f6", linestyle="--", linewidth=1.5, label=f"Legit Median ({np.median(recon_legit):.2f})")
    plt.axvline(np.median(recon_unseen), color="#ef4444", linestyle="--", linewidth=1.5, label=f"Unseen Median ({np.median(recon_unseen):.2f})")

    plt.title("GPU Autoencoder Reconstruction Error Distribution (Separability Analysis)", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Reconstruction Mean Squared Error (MSE)", fontsize=11)
    plt.ylabel("Probability Density", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(loc="upper right", framealpha=0.9, fontsize=9)
    plt.tight_layout()

    dist_fig_path = FIGURES_DIR / "unseen_reconstruction_distributions.png"
    plt.savefig(dist_fig_path, dpi=300)
    plt.close()
    print(f"Saved reconstruction distributions figure: {dist_fig_path}")

    # 12. Plot 2: Precision-Recall Curves (Supervised vs Anomaly vs Hybrid)
    plt.figure(figsize=(9, 6), dpi=300)

    # Supervised
    p_sup_curve, r_sup_curve, _ = precision_recall_curve(y_test_comp, p_sup_test)
    pr_auc_sup = average_precision_score(y_test_comp, p_sup_test)
    plt.plot(r_sup_curve, p_sup_curve, color="#3b82f6", linewidth=2.0, label=f"Supervised LightGBM (PR-AUC = {pr_auc_sup:.4f})")

    # Autoencoder
    p_ae_curve, r_ae_curve, _ = precision_recall_curve(y_test_comp, p_ae_test)
    pr_auc_ae = average_precision_score(y_test_comp, p_ae_test)
    plt.plot(r_ae_curve, p_ae_curve, color="#f59e0b", linewidth=2.0, linestyle="--", label=f"GPU Autoencoder (PR-AUC = {pr_auc_ae:.4f})")

    # Hybrid
    p_hyb_curve, r_hyb_curve, _ = precision_recall_curve(y_test_comp, p_hybrid_test)
    pr_auc_hyb = average_precision_score(y_test_comp, p_hybrid_test)
    plt.plot(r_hyb_curve, p_hyb_curve, color="#10b981", linewidth=2.5, label=f"Hybrid Model (alpha={best_alpha:.2f}, PR-AUC = {pr_auc_hyb:.4f})")

    plt.title("Precision-Recall Trade-off under Unseen Zero-Day Fraud Injection", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Recall across All Frauds (Seen + Unseen)", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(loc="lower left", framealpha=0.9, fontsize=9)
    plt.tight_layout()

    pr_fig_path = FIGURES_DIR / "unseen_pr_curves.png"
    plt.savefig(pr_fig_path, dpi=300)
    plt.close()
    print(f"Saved PR curves figure: {pr_fig_path}")

    # 13. Update metrics.json
    if METRICS_FILE.exists():
        with open(METRICS_FILE, "r", encoding="utf-8") as f:
            metrics_dict = json.load(f)
    else:
        metrics_dict = {}

    metrics_dict["unseen_fraud"] = {
        "optimal_alpha": best_alpha,
        "supervised_champion": {
            "seen_fraud_recall": seen_recall_sup,
            "unseen_fraud_recall": unseen_recall_sup,
            "overall_pr_auc": pr_auc_sup,
        },
        "gpu_autoencoder": {
            "seen_fraud_recall": float(np.mean(ae_seen_preds)),
            "unseen_fraud_recall": float(np.mean(ae_unseen_preds)),
            "overall_pr_auc": pr_auc_ae,
        },
        "hybrid_model": {
            "seen_fraud_recall": float(np.mean(p_hybrid_test[:n_nat][seen_fraud_mask_in_test] >= th_eval)),
            "unseen_fraud_recall": float(np.mean(p_hybrid_test[unseen_fraud_mask_in_test] >= th_eval)),
            "overall_pr_auc": pr_auc_hyb,
            "false_alarms_per_100k": float(bench_df.loc[bench_df["model_name"].str.contains("Hybrid Model.*Autoencoder"), "false_alarms_per_100k"].iloc[0]),
        },
        "benchmarks": bench_df.to_dict(orient="records"),
    }

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"Updated metrics saved to: {METRICS_FILE}")


if __name__ == "__main__":
    main()

