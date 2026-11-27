import pandas as pd

def eval_rule(name, pred, y):
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int(((~pred) & (y == 1)).sum())
    tn = int(((~pred) & (y == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    print(f"{name}:")
    print(f"  TP={tp:,}, FP={fp:,}, FN={fn:,}, TN={tn:,}")
    print(f"  Recall={rec*100:.2f}%, Precision={prec*100:.4f}%, F1={f1:.4f}, FPR={fpr*100:.2f}%")

df = pd.read_csv("data/raw/paysim.csv", usecols=["type", "amount", "oldbalanceOrg", "isFraud"])
df["orig_emptied_pre"] = df["amount"] >= df["oldbalanceOrg"]
rule1 = df["amount"] > 50000
rule2 = df["type"].isin(["TRANSFER", "CASH_OUT"]) & df["orig_emptied_pre"]

print(f"Total rows: {len(df):,}, Total fraud: {df['isFraud'].sum():,}")
eval_rule("Rule 1 (Amount > 50k)", rule1, df["isFraud"])
eval_rule("Rule 2 (TRANSFER/CASH_OUT & Empty)", rule2, df["isFraud"])

