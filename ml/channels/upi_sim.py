from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd

from ml.config import RANDOM_STATE, RAW_DATA_DIR

SYNTHETIC_UPI_PATH = RAW_DATA_DIR / "upi_synthetic.csv"

POPULAR_HANDLES = [
    "okhdfcbank",  
    "oksbi",     
    "okicici",     
    "okaxis",      
    "paytm",       
    "ybl",         
    "ibl",        
    "axl",         
    "barodampay",  
    "pnb",         
    "aubank",      
]

TXN_TYPES = ["PAY", "COLLECT_REQUEST", "QR_CODE"]


def generate_upi_dataset(
    num_records: int = 1_000_000,
    target_fraud_prevalence: float = 0.0040,
    label_noise_pct: float = 0.05,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:

    rng = np.random.RandomState(random_state)
    start_date = datetime(2026, 9, 1, 0, 0, 0)

    print(f"Generating {num_records:,} synthetic UPI transactions across 60 days...")
    t0 = time.time()

    day_weights = np.ones(60, dtype=np.float64)
    day_weights[24:27] = 1.45  
    day_weights[0] = 1.25      
    day_weights[30] = 1.25     
    day_probs = day_weights / day_weights.sum()

    days = rng.choice(np.arange(1, 61), size=num_records, p=day_probs)

    hour_weights = np.array([
        0.015, 0.008, 0.005, 0.004, 0.005, 0.010,  
        0.025, 0.045, 0.065, 0.075, 0.080, 0.085,  
        0.080, 0.075, 0.070, 0.075, 0.080, 0.085,  
        0.090, 0.085, 0.060, 0.045, 0.035, 0.022,  
    ])
    hour_probs = hour_weights / hour_weights.sum()
    hours = rng.choice(np.arange(24), size=num_records, p=hour_probs)

    minutes = rng.randint(0, 60, size=num_records)
    seconds = rng.randint(0, 60, size=num_records)

    num_payers = 50_000
    num_payees = 80_000

    payer_ids = rng.randint(1, num_payers + 1, size=num_records)
    payee_ids = rng.randint(1, num_payees + 1, size=num_records)

    payer_handle_idx = rng.randint(0, len(POPULAR_HANDLES), size=num_records)
    payee_handle_idx = rng.randint(0, len(POPULAR_HANDLES), size=num_records)

    payer_handles = [POPULAR_HANDLES[i] for i in payer_handle_idx]
    payee_handles = [POPULAR_HANDLES[i] for i in payee_handle_idx]

    txn_type_probs = [0.85, 0.08, 0.07]
    txn_type_idx = rng.choice([0, 1, 2], size=num_records, p=txn_type_probs)
    txn_types = np.array(TXN_TYPES)[txn_type_idx]

    raw_amounts = rng.lognormal(mean=5.5, sigma=1.3, size=num_records)
    raw_amounts = np.clip(raw_amounts, 5.0, 75_000.0)

    round_mask = rng.rand(num_records) < 0.42
    round_choices = np.array([10, 20, 50, 100, 200, 500, 1000, 1500, 2000, 2500, 5000, 10000, 20000, 50000])
    raw_amounts[round_mask] = rng.choice(round_choices, size=round_mask.sum())

    rent_mask = ((days == 1) | (days == 31)) & (rng.rand(num_records) < 0.04)
    raw_amounts[rent_mask] = rng.choice([15000, 18000, 22000, 25000, 30000, 45000, 60000], size=rent_mask.sum())

    diwali_mask = ((days >= 25) & (days <= 27)) & (rng.rand(num_records) < 0.06)
    raw_amounts[diwali_mask] = rng.choice([2500, 5000, 10000, 15000, 21000, 50000], size=diwali_mask.sum())

    payer_vpa_age = rng.randint(15, 1200, size=num_records).astype(np.int32)
    payee_vpa_age = rng.randint(10, 1200, size=num_records).astype(np.int32)

    is_first_time = (rng.rand(num_records) < 0.35).astype(np.int8)
    is_new_device = (rng.rand(num_records) < 0.035).astype(np.int8)
    sim_change_last_48h = (rng.rand(num_records) < 0.008).astype(np.int8)
    remote_access_flag = (rng.rand(num_records) < 0.0015).astype(np.int8)

    velocity_1h = rng.poisson(lam=0.45, size=num_records).astype(np.int16)
    velocity_24h = velocity_1h + rng.poisson(lam=2.1, size=num_records).astype(np.int16)

    total_frauds = int(num_records * target_fraud_prevalence)
    scam_types = np.array(["LEGITIMATE"] * num_records, dtype=object)
    true_fraud = np.zeros(num_records, dtype=np.int8)

    train_mask = days <= 42
    val_mask = (days >= 43) & (days <= 51)
    test_mask = days >= 52

    n_train_frauds = int(total_frauds * 0.70)
    n_val_frauds = int(total_frauds * 0.15)
    n_test_frauds = total_frauds - n_train_frauds - n_val_frauds

    train_indices = np.where(train_mask)[0]
    val_indices = np.where(val_mask)[0]
    test_indices = np.where(test_mask)[0]

    fraud_train_idx = rng.choice(train_indices, size=n_train_frauds, replace=False)
    fraud_val_idx = rng.choice(val_indices, size=n_val_frauds, replace=False)
    fraud_test_idx = rng.choice(test_indices, size=n_test_frauds, replace=False)

    all_fraud_idx = np.concatenate([fraud_train_idx, fraud_val_idx, fraud_test_idx])
    true_fraud[all_fraud_idx] = 1

    def apply_standard_scams(indices: np.ndarray) -> None:
        p_scams = rng.rand(len(indices))
        for i, idx in enumerate(indices):
            p = p_scams[i]
            if p < 0.35:
                scam_types[idx] = "COLLECT_REQUEST_SCAM"
                txn_types[idx] = "COLLECT_REQUEST"
                raw_amounts[idx] = rng.choice([2999.0, 4999.0, 9999.0, 14999.0, 24999.0])
                payee_vpa_age[idx] = rng.randint(1, 7)  
                is_first_time[idx] = 1
                velocity_1h[idx] = rng.randint(1, 3)
            elif p < 0.60:
                scam_types[idx] = "QR_PHISHING"
                txn_types[idx] = "QR_CODE"
                raw_amounts[idx] = rng.choice([1200.0, 3500.0, 5000.0, 8500.0, 15000.0])
                payee_vpa_age[idx] = rng.randint(1, 5)
                is_first_time[idx] = 1
            elif p < 0.85:
                scam_types[idx] = "REMOTE_ACCESS_TAKEOVER"
                txn_types[idx] = "PAY"
                raw_amounts[idx] = rng.choice([9500.0, 19500.0, 25000.0, 49000.0])
                remote_access_flag[idx] = 1
                is_first_time[idx] = 1
                payee_vpa_age[idx] = rng.randint(1, 10)
                velocity_1h[idx] = rng.randint(3, 7)  
                velocity_24h[idx] = velocity_1h[idx] + rng.randint(2, 5)
            else:
                scam_types[idx] = "SIM_SWAP_BURST"
                txn_types[idx] = "PAY"
                raw_amounts[idx] = rng.choice([25000.0, 35000.0, 50000.0, 75000.0])
                sim_change_last_48h[idx] = 1
                is_new_device[idx] = 1
                is_first_time[idx] = 1
                payee_vpa_age[idx] = rng.randint(1, 14)
                velocity_1h[idx] = rng.randint(2, 6)
                velocity_24h[idx] = velocity_1h[idx] + rng.randint(3, 8)

    apply_standard_scams(fraud_train_idx)
    apply_standard_scams(fraud_val_idx)

    p_test_scams = rng.rand(len(fraud_test_idx))
    standard_test_idx = fraud_test_idx[p_test_scams >= 0.20]
    unseen_test_idx = fraud_test_idx[p_test_scams < 0.20]

    apply_standard_scams(standard_test_idx)

    for idx in unseen_test_idx:
        scam_types[idx] = "UNSEEN_DIGITAL_ARREST_EXTORTION"
        txn_types[idx] = "PAY"
        raw_amounts[idx] = rng.choice([25000.0, 35000.0, 50000.0, 75000.0, 95000.0]) 
        is_first_time[idx] = 1
        payee_vpa_age[idx] = rng.randint(10, 90)  
        is_new_device[idx] = 0                    
        sim_change_last_48h[idx] = 0              
        remote_access_flag[idx] = 0               
        velocity_1h[idx] = rng.randint(0, 2)     

    reported_fraud = true_fraud.copy()
    noise_mask = (true_fraud == 1) & (rng.rand(num_records) < label_noise_pct)
    reported_fraud[noise_mask] = 0  

    print("Formatting timestamps and VPA identifiers...")
    timestamps = [
        (start_date + timedelta(days=int(d) - 1, hours=int(h), minutes=int(m), seconds=int(s))).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        for d, h, m, s in zip(days, hours, minutes, seconds)
    ]

    payer_vpas = [f"user_{uid:05d}@{hndl}" for uid, hndl in zip(payer_ids, payer_handles)]
    payee_vpas = [f"payee_{mid:05d}@{hndl}" for mid, hndl in zip(payee_ids, payee_handles)]
    txn_ids = [f"UPI{i+1:07d}" for i in range(num_records)]

    amounts = np.round(raw_amounts, 2)
    is_round = ((amounts % 100.0 == 0.0) | (amounts % 500.0 == 0.0)).astype(np.int8)

    df = pd.DataFrame({
        "txn_id": txn_ids,
        "timestamp": timestamps,
        "day": days.astype(np.int16),
        "hour": hours.astype(np.int16),
        "payer_vpa": payer_vpas,
        "payer_handle": payer_handles,
        "payee_vpa": payee_vpas,
        "payee_handle": payee_handles,
        "txn_type": txn_types,
        "amount": amounts.astype(np.float32),
        "is_round_amount": is_round,
        "payer_vpa_age_days": payer_vpa_age,
        "payee_vpa_age_days": payee_vpa_age,
        "is_first_time_payee": is_first_time,
        "is_new_device": is_new_device,
        "sim_change_last_48h": sim_change_last_48h,
        "remote_access_app_flag": remote_access_flag,
        "velocity_1h": velocity_1h,
        "velocity_24h": velocity_24h,
        "scam_type": scam_types,
        "is_fraud": reported_fraud,
        "true_fraud_label": true_fraud,
    })

    df = df.sort_values(by=["day", "timestamp"]).reset_index(drop=True)
    df["txn_id"] = [f"UPI{i+1:07d}" for i in range(num_records)]

    elapsed = time.time() - t0
    n_reported = df["is_fraud"].sum()
    n_true = df["true_fraud_label"].sum()
    print(f"Generated {len(df):,} UPI transactions in {elapsed:.2f}s.")
    print(f"Total True Frauds: {n_true:,} ({n_true/num_records*100:.3f}%)")
    print(f"Reported Frauds (after {label_noise_pct*100:.0f}% label noise): {n_reported:,} ({n_reported/num_records*100:.3f}%)")
    print("Scam type distribution:\n", df["scam_type"].value_counts())

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic Indian UPI transactions stream.")
    parser.add_argument("--records", type=int, default=1_000_000, help="Number of records to generate (default: 1M)")
    parser.add_argument("--output", type=str, default=str(SYNTHETIC_UPI_PATH), help="Output CSV path")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_upi_dataset(num_records=args.records)

    print(f"Saving dataset to {out_path}...")
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write("# DATA SOURCE: SIMULATED BY FRAUDSHIELD (Synthetic Indian UPI Transaction Stream)\n")
        f.write("# TIMEFRAME: 60 Days | NATURAL FRAUD PREVALENCE: ~0.40% | LABEL NOISE: 5%\n")
        f.write("# NOTICE: For academic evaluation only. Contains concept drift in Test days 52..60.\n")
        df.to_csv(f, index=False)

    print(f"Successfully saved {len(df):,} records to {out_path} ({out_path.stat().st_size / (1024*1024):.1f} MB).")


if __name__ == "__main__":
    main()
