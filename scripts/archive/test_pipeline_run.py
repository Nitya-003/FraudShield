import time, os, psutil, sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from collections import defaultdict
from ml.config import PAYSIM_RAW_PATH, PROCESSED_DATA_DIR, STEP_TRAIN_MAX, STEP_VAL_MAX

def check_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

print(f"Initial Memory: {check_memory():.1f} MB")
t0 = time.time()

# 1. Load full raw dataset with minimal memory
dtypes = {
    "step": "int32",
    "type": "category",
    "amount": "float32",
    "nameOrig": "string",
    "oldbalanceOrg": "float32",
    "newbalanceOrig": "float32",
    "nameDest": "string",
    "oldbalanceDest": "float32",
    "newbalanceDest": "float32",
    "isFraud": "int8",
    "isFlaggedFraud": "int8",
}

print("Loading PaySim CSV...")
df = pd.read_csv(PAYSIM_RAW_PATH, dtype=dtypes)
t1 = time.time()
print(f"Loaded {len(df):,} rows in {t1 - t0:.2f}s. Memory: {check_memory():.1f} MB")

# Sort chronologically by step
df = df.sort_values(by=["step"]).reset_index(drop=True)

# 2. Extract receiver features over full history
print("Computing receiver past-only features across 6.36M transactions...")
steps = df["step"].to_numpy(dtype=np.int32)
amounts = df["amount"].to_numpy(dtype=np.float32)
dest_codes, _ = pd.factorize(df["nameDest"])

n = len(df)
past_count = np.zeros(n, dtype=np.int32)
past_amount = np.zeros(n, dtype=np.float32)
time_since = np.full(n, 999.0, dtype=np.float32)
last_1h = np.zeros(n, dtype=np.int16)
last_6h = np.zeros(n, dtype=np.int16)
last_24h = np.zeros(n, dtype=np.int16)

cum_counts = defaultdict(int)
cum_amounts = defaultdict(float)
last_steps = {}
recent_steps = defaultdict(list)

for i in range(n):
    d = dest_codes[i]
    st = steps[i]
    amt = amounts[i]
    
    if d in last_steps:
        past_count[i] = cum_counts[d]
        past_amount[i] = cum_amounts[d]
        time_since[i] = st - last_steps[d]
        
        h = recent_steps[d]
        cutoff = st - 24
        idx = 0
        while idx < len(h) and h[idx] < cutoff:
            idx += 1
        if idx > 0:
            del h[:idx]
        c1, c6, c24 = 0, 0, 0
        for past_st in h:
            diff = st - past_st
            if diff <= 1:
                c1 += 1
            if diff <= 6:
                c6 += 1
            if diff <= 24:
                c24 += 1
        last_1h[i] = c1
        last_6h[i] = c6
        last_24h[i] = c24
        
        cum_counts[d] += 1
        cum_amounts[d] += amt
        last_steps[d] = st
        h.append(st)
    else:
        last_steps[d] = st
        cum_counts[d] = 1
        cum_amounts[d] = amt
        recent_steps[d] = [st]

t2 = time.time()
print(f"Receiver features computed in {t2 - t1:.2f}s. Memory: {check_memory():.1f} MB")

# Assign receiver features
df["dest_past_tx_count"] = past_count
df["dest_total_past_amount"] = past_amount
df["dest_time_since_last_tx"] = time_since
df["dest_tx_last_1h"] = last_1h
df["dest_tx_last_6h"] = last_6h
df["dest_tx_last_24h"] = last_24h

del past_count, past_amount, time_since, last_1h, last_6h, last_24h, steps, amounts, dest_codes, cum_counts, cum_amounts, last_steps, recent_steps

# 3. Pre-transaction features
print("Engineering pre-transaction features...")
amt = df["amount"].to_numpy(dtype=np.float32)
old_org = df["oldbalanceOrg"].to_numpy(dtype=np.float32)
old_dest = df["oldbalanceDest"].to_numpy(dtype=np.float32)
step_arr = df["step"].to_numpy(dtype=np.int32)
hour = (step_arr % 24).astype(np.int8)

df["hour_of_day"] = hour
df["sin_hour"] = np.sin(2.0 * np.pi * hour / 24.0).astype(np.float32)
df["cos_hour"] = np.cos(2.0 * np.pi * hour / 24.0).astype(np.float32)
df["is_night"] = ((hour >= 0) & (hour <= 5)).astype(np.int8)
df["log_amount"] = np.log1p(amt).astype(np.float32)

# Pre-transaction account & status
df["orig_emptied"] = (amt >= old_org).astype(np.int8)
df["amount_to_balance_ratio"] = (amt / np.maximum(old_org, 1.0)).astype(np.float32)
df["is_dest_merchant"] = df["nameDest"].str.startswith("M").astype(np.int8)
df["dest_balance_zero"] = (old_dest == 0.0).astype(np.int8)

# One-hot transaction types
for t in ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]:
    df[f"type_{t}"] = (df["type"] == t).astype(np.int8)

t3 = time.time()
print(f"Pre-transaction features engineered in {t3 - t2:.2f}s. Memory: {check_memory():.1f} MB")

# Split by chronological steps
print(f"Splitting at Step {STEP_TRAIN_MAX} and Step {STEP_VAL_MAX}...")
train_df = df[df["step"] <= STEP_TRAIN_MAX].copy()
val_df = df[(df["step"] > STEP_TRAIN_MAX) & (df["step"] <= STEP_VAL_MAX)].copy()
test_df = df[df["step"] > STEP_VAL_MAX].copy()
del df

print(f"Train: {len(train_df):,} rows, {train_df['isFraud'].sum():,} frauds ({train_df['isFraud'].mean()*100:.4f}%)")
print(f"Val:   {len(val_df):,} rows, {val_df['isFraud'].sum():,} frauds ({val_df['isFraud'].mean()*100:.4f}%)")
print(f"Test:  {len(test_df):,} rows, {test_df['isFraud'].sum():,} frauds ({test_df['isFraud'].mean()*100:.4f}%)")

print("Pipeline test successful!")
