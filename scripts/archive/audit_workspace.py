import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

def run_audit():
    print("=== PROBLEM A: Sampled vs Full Fraud Rate ===")
    train_df = pd.read_parquet("data/processed/train.parquet")
    val_df = pd.read_parquet("data/processed/val.parquet")
    test_df = pd.read_parquet("data/processed/test.parquet")
    
    print(f"Train split: {len(train_df):,} rows, {train_df['isFraud'].sum():,} frauds ({train_df['isFraud'].mean()*100:.3f}%)")
    print(f"Val split:   {len(val_df):,} rows, {val_df['isFraud'].sum():,} frauds ({val_df['isFraud'].mean()*100:.3f}%)")
    print(f"Test split:  {len(test_df):,} rows, {test_df['isFraud'].sum():,} frauds ({test_df['isFraud'].mean()*100:.3f}%)")

    print("\n=== PROBLEM B: Feature Importances of Sender Velocity ===")
    model_data = joblib.load("models/fraudshield_champion.joblib")
    meta = json.loads(Path("models/metadata.json").read_text(encoding="utf-8"))
    feature_names = meta.get("feature_names", [])
    
    # check model type
    model = model_data if hasattr(model_data, "feature_importances_") else model_data.get("model", model_data)
    importances = model.feature_importances_
    fi_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)
    print(fi_df.to_string(index=False))

    print("\n=== PROBLEM C: Inconsistencies & Features Count ===")
    print(f"Features in metadata.json: {len(feature_names)}")
    from ml.features import engineer_features
    # dummy row to check engineered columns
    dummy = pd.DataFrame([{
        "step": 10,
        "type": "PAYMENT",
        "amount": 100.0,
        "nameOrig": "C123",
        "oldbalanceOrg": 500.0,
        "newbalanceOrig": 400.0,
        "nameDest": "M456",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
        "isFraud": 0,
        "isFlaggedFraud": 0
    }])
    feats, _ = engineer_features(dummy, is_training=False)
    print(f"Features produced by ml/features.py: {len(feats.columns)}")
    print(f"Columns: {list(feats.columns)}")

if __name__ == "__main__":
    run_audit()

