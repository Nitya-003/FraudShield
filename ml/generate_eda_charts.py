from __future__ import annotations

from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ml.config import FIGURES_DIR
from ml.data import load_raw_data

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "figure.titlesize": 14,
    "figure.dpi": 200,
})

COLORS = {
    "legit": "#2563eb",   
    "fraud": "#dc2626",   
    "accent": "#059669",  
    "neutral": "#64748b", 
}


def plot_class_imbalance(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))

    counts = df["isFraud"].value_counts()
    percentages = df["isFraud"].value_counts(normalize=True) * 100
    labels = ["Legitimate (Class 0)", "Fraudulent (Class 1)"]
    palette = [COLORS["legit"], COLORS["fraud"]]

    bars = ax.bar(labels, counts, color=palette, width=0.45, edgecolor="black", linewidth=0.8)

    ax.set_yscale("log")
    ax.set_ylabel("Number of Transactions (Log Scale)", fontweight="bold")
    ax.set_title("Severe Class Imbalance in Financial Transactions", pad=14)

    for bar, count, pct in zip(bars, counts, percentages):
        height = bar.get_height()
        ax.annotate(
            f"{count:,}\n({pct:.3f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    ax.set_ylim(1, counts.max() * 6)
    plt.tight_layout()
    out_path = out_dir / "01_class_imbalance.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path


def plot_fraud_by_type(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    type_counts = df["type"].value_counts()
    ax1.bar(type_counts.index, type_counts.values, color=COLORS["neutral"], edgecolor="black")
    ax1.set_title("Total Volume by Transaction Type", pad=12)
    ax1.set_ylabel("Transaction Count", fontweight="bold")
    ax1.set_xlabel("Transaction Type", fontweight="bold")
    for x, val in enumerate(type_counts.values):
        ax1.text(x, val + (max(type_counts) * 0.015), f"{val:,}", ha="center", fontsize=9)

    fraud_counts = df[df["isFraud"] == 1]["type"].value_counts()
    types_with_fraud = [t for t in df["type"].cat.categories]
    fraud_vals = [fraud_counts.get(t, 0) for t in types_with_fraud]
    bar_colors = [COLORS["fraud"] if val > 0 else "#94a3b8" for val in fraud_vals]

    ax2.bar(types_with_fraud, fraud_vals, color=bar_colors, edgecolor="black")
    ax2.set_title("Fraud Occurrences by Transaction Type\n(100% in TRANSFER & CASH_OUT)", pad=12)
    ax2.set_ylabel("Fraud Count", fontweight="bold")
    ax2.set_xlabel("Transaction Type", fontweight="bold")
    for x, val in enumerate(fraud_vals):
        ax2.text(x, val + (max(fraud_vals) * 0.02) if max(fraud_vals) > 0 else 1, f"{val:,}", ha="center", fontweight="bold", fontsize=9)

    plt.tight_layout()
    out_path = out_dir / "02_fraud_by_type.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path


def plot_amount_distribution(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))

    legit_amounts = df[df["isFraud"] == 0]["amount"]
    fraud_amounts = df[df["isFraud"] == 1]["amount"]

    log_legit = np.log10(np.clip(legit_amounts, 1.0, None))
    log_fraud = np.log10(np.clip(fraud_amounts, 1.0, None))

    sns.kdeplot(log_legit, ax=ax, label="Legitimate", color=COLORS["legit"], fill=True, alpha=0.35, linewidth=2)
    sns.kdeplot(log_fraud, ax=ax, label="Fraudulent", color=COLORS["fraud"], fill=True, alpha=0.45, linewidth=2)

    ax.axvline(log_legit.median(), color=COLORS["legit"], linestyle="--", alpha=0.7, label=f"Legit Median (₹{10**log_legit.median():,.0f})")
    ax.axvline(log_fraud.median(), color=COLORS["fraud"], linestyle="--", alpha=0.7, label=f"Fraud Median (₹{10**log_fraud.median():,.0f})")

    ax.set_title("Transaction Amount Distribution: Legitimate vs Fraud (Log10 Scale)", pad=12)
    ax.set_xlabel("log10(Amount in ₹)", fontweight="bold")
    ax.set_ylabel("Density", fontweight="bold")
    ax.legend(frameon=True, loc="upper right")

    plt.tight_layout()
    out_path = out_dir / "03_amount_distributions_log.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path


def plot_fraud_by_hour(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.5))

    df_time = df.copy()
    df_time["hour"] = df_time["step"] % 24

    hourly_legit = df_time[df_time["isFraud"] == 0].groupby("hour").size()
    hourly_fraud = df_time[df_time["isFraud"] == 1].groupby("hour").size()

    pct_legit = (hourly_legit / hourly_legit.sum()) * 100
    pct_fraud = (hourly_fraud / hourly_fraud.sum()) * 100

    hours = np.arange(24)
    pct_legit = pct_legit.reindex(hours, fill_value=0.0)
    pct_fraud = pct_fraud.reindex(hours, fill_value=0.0)

    ax.plot(hours, pct_legit, marker="o", label="Legitimate Transactions (%)", color=COLORS["legit"], linewidth=2.2)
    ax.plot(hours, pct_fraud, marker="s", label="Fraudulent Transactions (%)", color=COLORS["fraud"], linewidth=2.2)

    ax.axvspan(0, 5, color="#fef08a", alpha=0.45, label="High Nocturnal Risk Window (00:00 - 05:00)")

    ax.set_xticks(hours)
    ax.set_xlabel("Hour of Day (0 to 23)", fontweight="bold")
    ax.set_ylabel("% of Class Volume per Hour", fontweight="bold")
    ax.set_title("Nocturnal Anomaly: Legitimate Activity Drops at Night While Fraud Persists", pad=12)
    ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    out_path = out_dir / "04_fraud_by_hour_of_day.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path


def plot_balance_emptying(df: pd.DataFrame, out_dir: Path) -> Path:
    """5. Balance Emptying Behavior: Account completely drained."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    drained_legit = ((df[df["isFraud"] == 0]["newbalanceOrig"] == 0) & (df[df["isFraud"] == 0]["amount"] > 0)).mean() * 100
    drained_fraud = ((df[df["isFraud"] == 1]["newbalanceOrig"] == 0) & (df[df["isFraud"] == 1]["amount"] > 0)).mean() * 100

    categories = ["Legitimate", "Fraudulent"]
    percentages = [drained_legit, drained_fraud]
    colors = [COLORS["legit"], COLORS["fraud"]]

    bars = ax1.bar(categories, percentages, color=colors, width=0.45, edgecolor="black")
    ax1.set_ylabel("% of Transactions Emptying Account", fontweight="bold")
    ax1.set_title("Account Emptying Rate (newbalanceOrig = 0)", pad=12)
    ax1.set_ylim(0, 115)
    for bar, val in zip(bars, percentages):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 3, f"{val:.1f}%", ha="center", fontweight="bold", fontsize=11)

    diff_error = df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
    zero_error_legit = (np.isclose(diff_error[df["isFraud"] == 0], 0.0, atol=1.0)).mean() * 100
    zero_error_fraud = (np.isclose(diff_error[df["isFraud"] == 1], 0.0, atol=1.0)).mean() * 100

    bars2 = ax2.bar(categories, [zero_error_legit, zero_error_fraud], color=[COLORS["accent"], "#ea580c"], width=0.45, edgecolor="black")
    ax2.set_ylabel("% Mathematically Exact Balance Deduction", fontweight="bold")
    ax2.set_title("Sender Balance Arithmetic Consistency", pad=12)
    ax2.set_ylim(0, 115)
    for bar, val in zip(bars2, [zero_error_legit, zero_error_fraud]):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 3, f"{val:.1f}%", ha="center", fontweight="bold", fontsize=11)

    plt.tight_layout()
    out_path = out_dir / "05_balance_emptying_pattern.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path


def generate_all_eda_charts() -> list[Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_raw_data(sample_size=1_000_000)
    generated = [
        plot_class_imbalance(df, FIGURES_DIR),
        plot_fraud_by_type(df, FIGURES_DIR),
        plot_amount_distribution(df, FIGURES_DIR),
        plot_fraud_by_hour(df, FIGURES_DIR),
        plot_balance_emptying(df, FIGURES_DIR),
    ]
    print(f"\nGenerated {len(generated)} EDA figures in {FIGURES_DIR}:")
    for p in generated:
        print(f"  - {p.name}")
    return generated


if __name__ == "__main__":
    generate_all_eda_charts()

