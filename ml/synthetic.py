from __future__ import annotations

import argparse
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd

from ml.config import RANDOM_STATE, SYNTHETIC_RAW_PATH, TRANSACTION_TYPES


def generate_synthetic_paysim(
    n_samples: int = 250_000,
    fraud_rate: float = 0.002,
    seed: int = RANDOM_STATE,
    output_path: Path | str | None = None,
) -> pd.DataFrame:

    rng = np.random.default_rng(seed)

    print(f"Generating {n_samples:,} synthetic transactions with ~{fraud_rate*100:.2f}% fraud...")

    hours = np.tile(np.arange(24), n_samples // 24 + 1)[:n_samples]
    day_weights = np.array([
        0.015, 0.010, 0.008, 0.007, 0.008, 0.012, 
        0.025, 0.040, 0.055, 0.070, 0.075, 0.070, 
        0.070, 0.065, 0.060, 0.060, 0.065, 0.070, 
        0.070, 0.060, 0.050, 0.040, 0.025, 0.020  
    ])
    day_weights /= day_weights.sum()

    days = rng.integers(0, 30, size=n_samples)
    sampled_hours = rng.choice(np.arange(24), size=n_samples, p=day_weights)
    steps = days * 24 + sampled_hours + 1
    steps = np.sort(steps)

    type_probs = [0.22, 0.35, 0.01, 0.34, 0.08] 
    tx_types = rng.choice(TRANSACTION_TYPES, size=n_samples, p=type_probs)

    n_fraud = int(n_samples * fraud_rate)
    fraud_eligible_indices = np.where(np.isin(tx_types, ["TRANSFER", "CASH_OUT"]))[0]

    fraud_indices = rng.choice(fraud_eligible_indices, size=n_fraud, replace=False)
    is_fraud = np.zeros(n_samples, dtype=np.int8)
    is_fraud[fraud_indices] = 1

    amounts = np.zeros(n_samples, dtype=np.float64)
    legit_mask = is_fraud == 0
    fraud_mask = is_fraud == 1

    amounts[legit_mask] = rng.lognormal(mean=9.5, sigma=1.2, size=legit_mask.sum())
    amounts[legit_mask] = np.clip(amounts[legit_mask], 5.0, 1_500_000.0)

    amounts[fraud_mask] = rng.lognormal(mean=12.2, sigma=1.0, size=fraud_mask.sum())
    amounts[fraud_mask] = np.clip(amounts[fraud_mask], 10_000.0, 10_000_000.0)

    amounts = np.round(amounts, 2)

    n_unique_senders = int(n_samples * 0.85)
    sender_pool = [f"C{100000000 + i}" for i in range(n_unique_senders)]
    name_orig = rng.choice(sender_pool, size=n_samples)

    n_unique_dest = int(n_samples * 0.70)
    dest_pool_cust = [f"C{500000000 + i}" for i in range(n_unique_dest // 2)]
    dest_pool_merch = [f"M{800000000 + i}" for i in range(n_unique_dest // 2)]

    name_dest = np.empty(n_samples, dtype=object)
    payment_mask = tx_types == "PAYMENT"
    name_dest[payment_mask] = rng.choice(dest_pool_merch, size=payment_mask.sum())
    name_dest[~payment_mask] = rng.choice(dest_pool_cust, size=(~payment_mask).sum())

    oldbalance_orig = np.zeros(n_samples, dtype=np.float64)
    newbalance_orig = np.zeros(n_samples, dtype=np.float64)
    oldbalance_dest = np.zeros(n_samples, dtype=np.float64)
    newbalance_dest = np.zeros(n_samples, dtype=np.float64)

    base_sender_balance = rng.lognormal(mean=10.0, sigma=1.5, size=n_samples)
    base_sender_balance = np.clip(base_sender_balance, 0.0, 15_000_000.0)
    oldbalance_orig = np.round(base_sender_balance, 2)

    for i in range(n_samples):
        t = tx_types[i]
        amt = amounts[i]
        fraud = is_fraud[i]

        if fraud == 1:
            oldbalance_orig[i] = amt
            newbalance_orig[i] = 0.0
            oldbalance_dest[i] = np.round(rng.uniform(0, 100_000), 2)
            if rng.random() < 0.6:
                newbalance_dest[i] = oldbalance_dest[i]  
            else:
                newbalance_dest[i] = round(oldbalance_dest[i] + amt, 2)
        else:
            if t in ["CASH_OUT", "TRANSFER"]:
                if oldbalance_orig[i] < amt:
                    oldbalance_orig[i] = round(amt + rng.uniform(10, 5000), 2)
                newbalance_orig[i] = max(0.0, round(oldbalance_orig[i] - amt, 2))
                oldbalance_dest[i] = round(rng.lognormal(mean=9.0, sigma=1.5), 2)
                newbalance_dest[i] = round(oldbalance_dest[i] + amt, 2)
            elif t == "PAYMENT":
                oldbalance_dest[i] = 0.0
                newbalance_dest[i] = 0.0
                if oldbalance_orig[i] < amt:
                    oldbalance_orig[i] = round(amt + rng.uniform(5, 500), 2)
                newbalance_orig[i] = max(0.0, round(oldbalance_orig[i] - amt, 2))
            elif t == "CASH_IN":
                newbalance_orig[i] = round(oldbalance_orig[i] + amt, 2)
                oldbalance_dest[i] = round(rng.lognormal(mean=8.5, sigma=1.5), 2)
                newbalance_dest[i] = max(0.0, round(oldbalance_dest[i] - amt, 2))
            else:  
                newbalance_orig[i] = max(0.0, round(oldbalance_orig[i] - amt, 2))
                oldbalance_dest[i] = round(rng.lognormal(mean=8.0, sigma=1.2), 2)
                newbalance_dest[i] = round(oldbalance_dest[i] + amt, 2)

    is_flagged = np.zeros(n_samples, dtype=np.int8)
    flag_cond = (tx_types == "TRANSFER") & (amounts > 200_000.0)
    is_flagged[flag_cond & (rng.random(n_samples) < 0.05)] = 1

    df = pd.DataFrame({
        "step": steps,
        "type": tx_types,
        "amount": amounts,
        "nameOrig": name_orig,
        "oldbalanceOrg": oldbalance_orig,
        "newbalanceOrig": newbalance_orig,
        "nameDest": name_dest,
        "oldbalanceDest": oldbalance_dest,
        "newbalanceDest": newbalance_dest,
        "isFraud": is_fraud,
        "isFlaggedFraud": is_flagged,
    })

    if output_path is None:
        output_path = SYNTHETIC_RAW_PATH
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Saved {len(df):,} transactions to {output_path}")
    print(f"Class distribution: {df['isFraud'].value_counts().to_dict()} "
          f"({df['isFraud'].mean()*100:.3f}% fraud)")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic PaySim dataset")
    parser.add_argument("--samples", type=int, default=250_000, help="Number of samples")
    parser.add_argument("--fraud_rate", type=float, default=0.002, help="Fraud rate (default: 0.002)")
    parser.add_argument("--output", type=str, default=str(SYNTHETIC_RAW_PATH), help="Output CSV path")
    args = parser.parse_args()

    generate_synthetic_paysim(
        n_samples=args.samples,
        fraud_rate=args.fraud_rate,
        output_path=args.output,
    )

