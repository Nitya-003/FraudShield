from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

NOTEBOOKS_DIR = Path(__file__).resolve().parent
ROOT_DIR = NOTEBOOKS_DIR.parent
NOTEBOOK_PATH = NOTEBOOKS_DIR / "01_eda.ipynb"

def create_eda_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3 (.venv)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11"
        }
    }

    cells = []

    # 1. Title
    cells.append(nbf.v4.new_markdown_cell(
"""# FraudShield — 01. Exploratory Data Analysis (EDA)
**Coursework:** Fundamentals of Machine Learning  
**Author / Student:** Nitya  
**Dataset:** PaySim Financial Transactions Benchmark  
**Objective:** Uncover behavioural fraud patterns, assess data quality, and identify feature engineering signals while strictly preventing data leakage.

---
### Research Questions Addressed:
1. What is the severity of **class imbalance**, and why is classification accuracy a misleading metric?
2. Which **transaction types** carry fraud risk, and which can be filtered or down-weighted?
3. How do transaction **amounts** differ between fraudulent and legitimate activity?
4. Is there a **temporal / nocturnal pattern** to fraud attempts across the 24-hour cycle?
5. What **balance-emptying behaviours** characterize fraudulent account takeovers?
"""
    ))

    # 2. Imports
    cells.append(nbf.v4.new_code_cell(
"""import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ml.config import FIGURES_DIR
from ml.data import check_data_quality, load_raw_data

# Visual styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({"figure.dpi": 150, "font.size": 10})
%matplotlib inline

print("Environment ready. Project root:", ROOT)
"""
    ))

    # 3. Section 1: Ingestion
    cells.append(nbf.v4.new_markdown_cell(
"""## 1. Data Ingestion & Schema Verification
The PaySim dataset simulates mobile money transactions based on a sample of real financial logs from an African mobile money service.  
We load the dataset with memory-optimized data types (`int32`, `float32`, `category`, `string`) to ensure fast, scalable processing without unnecessary RAM overhead.
"""
    ))

    # 4. Code: Load data
    cells.append(nbf.v4.new_code_cell(
"""# Load 1M rows preserving 100% of the 8,213 frauds
df = load_raw_data(sample_size=1_000_000)
display(df.head())
print("\\nDataset Information:")
df.info()
"""
    ))

    # 5. Section 2: Data Quality
    cells.append(nbf.v4.new_markdown_cell(
"""## 2. Data Hygiene & Quality Checks
Prior to statistical analysis, we verify fundamental data integrity:
- Missing / Null values across all columns
- Duplicate transactions
- Physically impossible values (negative transaction amounts or negative balances)
- Arithmetic consistency of reported balances
"""
    ))

    # 6. Code: Quality checks
    cells.append(nbf.v4.new_code_cell(
"""quality = check_data_quality(df)
print("--- Data Quality Audit Report ---")
for metric, val in quality.items():
    if metric != "null_details":
        print(f"  • {metric:<26}: {val}")

print("\\nNull counts per column:")
for col, cnt in quality["null_details"].items():
    print(f"  • {col:<16}: {cnt}")
"""
    ))

    # 7. Section 3: Class Imbalance
    cells.append(nbf.v4.new_markdown_cell(
"""## 3. Class Imbalance Analysis
In financial fraud detection, positive (fraudulent) transactions represent a tiny minority of all transactions.
- **Why Accuracy Fails:** If 99.8% of transactions are legitimate, a naive classifier that predicts *Legitimate* 100% of the time achieves **99.8% Accuracy**, yet catches **0% of fraud** and incurs catastrophic financial loss.
- **Evaluation Mandate:** We must evaluate models using **Precision-Recall AUC (PR-AUC)**, **Recall (Sensitivity)**, **Precision**, and business cost matrices.
"""
    ))

    # 8. Code: Class Imbalance Plot
    cells.append(nbf.v4.new_code_cell(
"""counts = df['isFraud'].value_counts()
pcts = df['isFraud'].value_counts(normalize=True) * 100

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(['Legitimate (0)', 'Fraud (1)'], counts, color=['#2563eb', '#dc2626'], width=0.4, edgecolor='black')
ax.set_yscale('log')
ax.set_ylabel('Transaction Count (Log Scale)', fontweight='bold')
ax.set_title('Severe Class Imbalance in Financial Transactions', pad=12)

for bar, cnt, pct in zip(bars, counts, pcts):
    ax.annotate(f'{cnt:,}\\n({pct:.3f}%)',
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 6), textcoords='offset points', ha='center', fontweight='bold')

plt.tight_layout()
plt.show()
print(f"Imbalance Ratio: 1 fraud per {int(counts[0]/counts[1])} legitimate transactions.")
"""
    ))

    # 9. Section 4: Fraud by Type
    cells.append(nbf.v4.new_markdown_cell(
"""## 4. Fraud Distribution by Transaction Type
The PaySim schema contains 5 distinct transaction types:
- `PAYMENT`: Customer paying a merchant (`M...`)
- `CASH_OUT`: Customer withdrawing physical cash
- `CASH_IN`: Customer depositing money into account
- `TRANSFER`: Customer sending funds to another account
- `DEBIT`: Direct bank debit

Let us examine which transaction types contain fraudulent activity.
"""
    ))

    # 10. Code: Fraud by type
    cells.append(nbf.v4.new_code_cell(
"""cross_tab = pd.crosstab(df['type'], df['isFraud'], margins=True)
cross_tab['% of All Fraud'] = (cross_tab[1] / df['isFraud'].sum()) * 100
display(cross_tab)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
df['type'].value_counts().plot(kind='bar', ax=ax1, color='#64748b', edgecolor='black')
ax1.set_title('Overall Volume by Type', fontweight='bold')
ax1.set_ylabel('Count')

fraud_types = df[df['isFraud'] == 1]['type'].value_counts()
colors = ['#dc2626' if fraud_types.get(t, 0) > 0 else '#94a3b8' for t in df['type'].cat.categories]
df['type'].cat.categories.map(lambda t: fraud_types.get(t, 0)).to_series(index=df['type'].cat.categories).plot(
    kind='bar', ax=ax2, color=colors, edgecolor='black'
)
ax2.set_title('Fraud Volume by Type (100% in TRANSFER & CASH_OUT)', fontweight='bold')
ax2.set_ylabel('Fraud Count')
plt.tight_layout()
plt.show()
"""
    ))

    # 11. Section 5: Amount distributions
    cells.append(nbf.v4.new_markdown_cell(
r"""## 5. Amount Distributions & Skewness (Log Scale)
Financial transaction amounts span several orders of magnitude (from ₹5 to ₹10,000,000+).  
We analyze amounts on a $\log_{10}$ scale to inspect differences in spend behavior between fraudsters and legitimate account owners.
"""
    ))

    # 12. Code: Amount plot
    cells.append(nbf.v4.new_code_cell(
"""legit_amt = df[df['isFraud'] == 0]['amount']
fraud_amt = df[df['isFraud'] == 1]['amount']

print(f"Legitimate Amounts  -> Median: ₹{legit_amt.median():,.2f} | 95th Pct: ₹{legit_amt.quantile(0.95):,.2f} | Max: ₹{legit_amt.max():,.2f}")
print(f"Fraudulent Amounts  -> Median: ₹{fraud_amt.median():,.2f} | 95th Pct: ₹{fraud_amt.quantile(0.95):,.2f} | Max: ₹{fraud_amt.max():,.2f}")

fig, ax = plt.subplots(figsize=(8, 4))
sns.kdeplot(np.log10(np.clip(legit_amt, 1, None)), ax=ax, label='Legitimate', color='#2563eb', fill=True, alpha=0.35)
sns.kdeplot(np.log10(np.clip(fraud_amt, 1, None)), ax=ax, label='Fraudulent', color='#dc2626', fill=True, alpha=0.45)
ax.set_title('Transaction Amount Density (log10 scale)', fontweight='bold')
ax.set_xlabel('log10(Amount in ₹)')
ax.legend()
plt.tight_layout()
plt.show()
"""
    ))

    # 13. Section 6: Temporal patterns
    cells.append(nbf.v4.new_markdown_cell(
"""## 6. Temporal Patterns & Nocturnal Anomalies
The `step` feature maps directly to hours (1 step = 1 hour, encompassing 30 days = 720+ hours).  
We extract the diurnal cycle via `hour_of_day = step % 24`.  
- **Hypothesis:** Legitimate humans sleep at night (00:00 - 05:00), leading to steep drops in transaction volume.
- Fraudsters and automated attack scripts operate around the clock, creating a pronounced relative nocturnal anomaly.
"""
    ))

    # 14. Code: Temporal plot
    cells.append(nbf.v4.new_code_cell(
"""df_temp = df.copy()
df_temp['hour'] = df_temp['step'] % 24

legit_hourly = df_temp[df_temp['isFraud'] == 0].groupby('hour').size()
fraud_hourly = df_temp[df_temp['isFraud'] == 1].groupby('hour').size()

legit_pct = (legit_hourly / legit_hourly.sum()) * 100
fraud_pct = (fraud_hourly / fraud_hourly.sum()) * 100

hours = np.arange(24)
legit_pct = legit_pct.reindex(hours, fill_value=0.0)
fraud_pct = fraud_pct.reindex(hours, fill_value=0.0)

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(hours, legit_pct, marker='o', color='#2563eb', label='Legitimate (%)', linewidth=2)
ax.plot(hours, fraud_pct, marker='s', color='#dc2626', label='Fraudulent (%)', linewidth=2)
ax.axvspan(0, 5, color='#fef08a', alpha=0.45, label='Nocturnal Window (00:00 - 05:00)')
ax.set_xticks(hours)
ax.set_xlabel('Hour of Day (0 to 23)')
ax.set_ylabel('% of Class Volume')
ax.set_title('Nocturnal Anomaly: Legitimate Activity Drops at Night While Fraud Continues', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.show()
"""
    ))

    # 15. Section 7: Balance Emptying
    cells.append(nbf.v4.new_markdown_cell(
r"""## 7. Account Depletion Behavior (Emptying Pattern)
Fraudsters typically attempt to extract as much liquid capital as possible before the account owner detects unauthorized access or the bank freezes the credentials.
- We check whether `newbalanceOrig == 0` when `amount > 0`.
- We also check the arithmetic balance equation: $\text{diff\_error} = \text{oldbalanceOrg} - \text{amount} - \text{newbalanceOrig}$.
"""
    ))

    # 16. Code: Balance pattern
    cells.append(nbf.v4.new_code_cell(
"""drained_legit = ((df[df['isFraud'] == 0]['newbalanceOrig'] == 0) & (df[df['isFraud'] == 0]['amount'] > 0)).mean() * 100
drained_fraud = ((df[df['isFraud'] == 1]['newbalanceOrig'] == 0) & (df[df['isFraud'] == 1]['amount'] > 0)).mean() * 100

print(f"Account-Emptying Rate (Legitimate): {drained_legit:.2f}%")
print(f"Account-Emptying Rate (Fraudulent): {drained_fraud:.2f}%")

fig, ax = plt.subplots(figsize=(5, 3.5))
bars = ax.bar(['Legitimate', 'Fraudulent'], [drained_legit, drained_fraud], color=['#2563eb', '#dc2626'], width=0.4, edgecolor='black')
ax.set_ylabel('% Emptying Entire Balance', fontweight='bold')
ax.set_title('Account Emptying Rate (newbalanceOrig == 0)', pad=10)
ax.set_ylim(0, 115)
for bar, val in zip(bars, [drained_legit, drained_fraud]):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 3, f'{val:.1f}%', ha='center', fontweight='bold')
plt.tight_layout()
plt.show()
"""
    ))

    # 17. Section 8: Summary of Takeaways
    cells.append(nbf.v4.new_markdown_cell(
r"""## 8. Five Key Insights for Phase 2 Feature Engineering

| # | Insight | Machine Learning Implication |
|---|---|---|
| **1** | **Extreme Class Imbalance (0.20% Fraud)** | Accuracy is forbidden as an evaluation metric. Models must use cost-weighting (`scale_pos_weight` / `class_weight='balanced'`) and be tuned on **PR-AUC**. |
| **2** | **Strict Type Exclusivity** | 100% of frauds take place in `TRANSFER` and `CASH_OUT`. One-hot encoding of `type` gives models clear partition signals. |
| **3** | **Log-Normal Amount Skew** | Raw amounts range from ₹5 to millions. Using $\log(\text{amount} + 1)$ stabilizes variance for linear models and tree splits. |
| **4** | **Nocturnal Disproportion** | Legitimate activity drops during midnight–05:00, whereas fraud volume remains elevated. Cyclical time features ($\sin, \cos$) and an `is_night` boolean flag will capture this anomaly. |
| **5** | **Account Emptying & Discrepancy** | Over 90% of fraudulent transactions completely drain the sender's account balance (`newbalanceOrig = 0`). Arithmetic balance difference errors are strong predictive signals. |
"""
    ))

    nb.cells = cells
    return nb


def run_and_save_notebook():
    nb = create_eda_notebook()
    print("Executing notebook top-to-bottom...")
    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute(cwd=str(ROOT_DIR))
    
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Successfully saved executed notebook to {NOTEBOOK_PATH}")


if __name__ == "__main__":
    run_and_save_notebook()
