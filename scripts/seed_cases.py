"""
Database seeding script for FraudShield Case Management.
Initializes SQLite cases.db and seeds initial triage cases across ALL 4 channels
(transfer_wallet, card, account, upi) and across diverse score ranges (High, Medium, Low).
"""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from ml.config import DATA_DIR, PROCESSED_DATA_DIR

DB_PATH = DATA_DIR / "cases.db"


def seed_cases_db(db_path: Path | None = None, force_recreate: bool = False) -> int:
    """Create cases.db and seed realistic cases across all channels and risk bands."""
    target_path = db_path or DB_PATH
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if force_recreate and target_path.exists():
        target_path.unlink()

    conn = sqlite3.connect(target_path)
    conn.row_factory = sqlite3.Row

    with conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cases (
                case_id TEXT PRIMARY KEY,
                channel TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                counterparty_id TEXT,
                amount REAL NOT NULL,
                risk_score REAL NOT NULL,
                risk_band TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'OPEN',
                analyst_notes TEXT DEFAULT '',
                updated_at TEXT,
                payload_json TEXT
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analyst_logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT NOT NULL,
                action TEXT NOT NULL,
                analyst_name TEXT DEFAULT 'Investigator',
                notes TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (case_id) REFERENCES cases(case_id)
            );
        """)

        # Check existing count
        cursor.execute("SELECT COUNT(*) FROM cases;")
        existing_count = cursor.fetchone()[0]
        if existing_count > 0 and not force_recreate:
            return existing_count

        cases_to_insert = []
        now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        # 1. PaySim / transfer_wallet
        paysim_path = PROCESSED_DATA_DIR / "test_scored_sample.parquet"
        if paysim_path.exists():
            df = pd.read_parquet(paysim_path)
            high_df = df[df["risk_band"] == "High"].head(8)
            med_df = df[df["risk_band"] == "Medium"].head(5)
            low_df = df[df["risk_band"] == "Low"].head(5)

            for subset, band in [(high_df, "High"), (med_df, "Medium"), (low_df, "Low")]:
                for idx, row in subset.iterrows():
                    cid = f"CASE-PAY-{row.get('ID', row.get('step', int(idx) + 1000))}"
                    score = float(row.get("risk_score", 0.50))
                    cases_to_insert.append((
                        cid,
                        "transfer_wallet",
                        str(row.get("nameOrig", f"C{100000+idx}")),
                        str(row.get("nameDest", f"M{900000+idx}")),
                        float(row.get("amount", 1000.0)),
                        score,
                        band,
                        str(row.get("timestamp", now_str)),
                        "OPEN",
                        "",
                        now_str,
                        json.dumps({"type": str(row.get("type", "TRANSFER")), "amount": float(row.get("amount", 0.0))}),
                    ))

        # 2. Card Channel
        card_path = PROCESSED_DATA_DIR / "card_test_sample.parquet"
        if card_path.exists():
            df_card = pd.read_parquet(card_path)
            high_card = df_card[df_card["risk_score"] >= 0.80].head(8)
            med_card = df_card[(df_card["risk_score"] >= 0.20) & (df_card["risk_score"] < 0.80)].head(5)
            low_card = df_card[df_card["risk_score"] < 0.20].head(5)

            for subset, band in [(high_card, "High"), (med_card, "Medium"), (low_card, "Low")]:
                for idx, row in subset.iterrows():
                    cid = f"CASE-CRD-{row.get('ID', f'C{10000+int(idx)}')}"
                    score = float(row.get("risk_score", 0.50))
                    amt = float(row.get("amount", row.get("Amount", 125.0)))
                    cases_to_insert.append((
                        cid,
                        "card",
                        f"CARD-ACCT-{40000 + int(idx)}",
                        "TERMINAL-EU-99",
                        amt,
                        score,
                        band,
                        now_str,
                        "OPEN",
                        "",
                        now_str,
                        json.dumps({"Amount": amt, "V14": float(row.get("V14", 0.0))}),
                    ))

        # 3. Account / BAF Channel
        account_path = PROCESSED_DATA_DIR / "account_test_sample.parquet"
        if account_path.exists():
            df_acc = pd.read_parquet(account_path)
            high_acc = df_acc[df_acc["risk_score"] >= 0.45].head(8)
            med_acc = df_acc[(df_acc["risk_score"] >= 0.08) & (df_acc["risk_score"] < 0.45)].head(6)
            low_acc = df_acc[df_acc["risk_score"] < 0.08].head(5)

            for subset, band in [(high_acc, "High"), (med_acc, "Medium"), (low_acc, "Low")]:
                for idx, row in subset.iterrows():
                    cid = f"CASE-ACC-{row.get('ID', f'A{20000+int(idx)}')}"
                    score = float(row.get("risk_score", 0.10))
                    income = float(row.get("income", 0.5))
                    cases_to_insert.append((
                        cid,
                        "account",
                        f"APPLICANT-{50000 + int(idx)}",
                        "BRANCH-MAIN-01",
                        float(row.get("proposed_credit_limit", 1500.0)),
                        score,
                        band,
                        now_str,
                        "OPEN",
                        "",
                        now_str,
                        json.dumps({"income": income, "customer_age": int(row.get("customer_age", 35))}),
                    ))

        # 4. UPI Channel
        upi_path = PROCESSED_DATA_DIR / "upi_test_sample.parquet"
        if upi_path.exists():
            df_upi = pd.read_parquet(upi_path)
            high_upi = df_upi[df_upi["risk_score"] >= 0.87].head(8)
            med_upi = df_upi[(df_upi["risk_score"] >= 0.15) & (df_upi["risk_score"] < 0.87)].head(6)
            low_upi = df_upi[df_upi["risk_score"] < 0.15].head(5)

            for subset, band in [(high_upi, "High"), (med_upi, "Medium"), (low_upi, "Low")]:
                for idx, row in subset.iterrows():
                    cid = f"CASE-UPI-{row.get('txn_id', f'U{10000+int(idx)}')}"
                    score = float(row.get("risk_score", 0.50))
                    amt = float(row.get("amount", 25000.0))
                    cases_to_insert.append((
                        cid,
                        "upi",
                        str(row.get("payer_vpa", "user@okhdfcbank")),
                        str(row.get("payee_vpa", "merchant@paytm")),
                        amt,
                        score,
                        band,
                        str(row.get("timestamp", now_str)),
                        "OPEN",
                        "",
                        now_str,
                        json.dumps({
                            "scam_type": str(row.get("scam_type", "UNSEEN_DIGITAL_ARREST_EXTORTION")),
                            "amount": amt,
                        }),
                    ))

        cursor.executemany("""
            INSERT OR IGNORE INTO cases (
                case_id, channel, entity_id, counterparty_id, amount,
                risk_score, risk_band, timestamp, status, analyst_notes, updated_at, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, cases_to_insert)
        conn.commit()

        cursor.execute("SELECT COUNT(*) FROM cases;")
        total = cursor.fetchone()[0]
        return total


if __name__ == "__main__":
    count = seed_cases_db(force_recreate=True)
    print(f"Successfully seeded {count} cases across all channels.")
