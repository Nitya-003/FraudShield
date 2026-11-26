from __future__ import annotations

from collections import defaultdict, deque
import threading
from typing import Any
import numpy as np
import pandas as pd

from ml.network import NetworkGraphTracker, DEFAULT_NETWORK_VALUES


class AccountHistoryStore:

    def __init__(self, window_hours: int = 24):
        self.window_hours = window_hours
        self._lock = threading.Lock()

        self.dest_tx_log: dict[str, list[tuple[int, float]]] = defaultdict(list)
        self.dest_cum_count: dict[str, int] = defaultdict(int)
        self.dest_cum_amount: dict[str, float] = defaultdict(float)
        self.dest_last_step: dict[str, int] = {}
    
        self.orig_tx_log: dict[str, list[tuple[int, float, str, float]]] = defaultdict(list)

        self.network_tracker = NetworkGraphTracker(window_hours=window_hours)

    def get_receiver_features(self, name_dest: str, current_step: int) -> dict[str, float | int]:
        with self._lock:
            if not name_dest or name_dest not in self.dest_tx_log:
                return {
                    "dest_past_tx_count": 0,
                    "dest_total_past_amount": 0.0,
                    "dest_time_since_last_tx": 999.0,
                    "dest_tx_last_1h": 0,
                    "dest_tx_last_6h": 0,
                    "dest_tx_last_24h": 0,
                }

            history = self.dest_tx_log[name_dest]
        
            prior_txs = [tx for tx in history if tx[0] < current_step]
            if not prior_txs:
                return {
                    "dest_past_tx_count": 0,
                    "dest_total_past_amount": 0.0,
                    "dest_time_since_last_tx": 999.0,
                    "dest_tx_last_1h": 0,
                    "dest_tx_last_6h": 0,
                    "dest_tx_last_24h": 0,
                }

            past_count = len(prior_txs)
            past_amount = sum(tx[1] for tx in prior_txs)
            last_st = prior_txs[-1][0]
            time_since = float(current_step - last_st)

            c1, c6, c24 = 0, 0, 0
            for st, _ in reversed(prior_txs):
                diff = current_step - st
                if diff <= 1:
                    c1 += 1
                if diff <= 6:
                    c6 += 1
                if diff <= 24:
                    c24 += 1
                else:
                    break

            return {
                "dest_past_tx_count": past_count,
                "dest_total_past_amount": round(past_amount, 2),
                "dest_time_since_last_tx": time_since,
                "dest_tx_last_1h": c1,
                "dest_tx_last_6h": c6,
                "dest_tx_last_24h": c24,
            }

    def get_network_features(self, payload: dict[str, Any], current_step: int | None = None) -> dict[str, float | int]:
   
        with self._lock:
            return self.network_tracker.extract_single(payload, cur_step=current_step)

    def get_sender_sequence_features(
        self,
        name_orig: str,
        current_step: int,
        current_amount: float,
        current_balance: float,
        name_dest: str = "",
    ) -> dict[str, float | int]:
    
        with self._lock:
            prior_orig = [tx for tx in self.orig_tx_log.get(name_orig, []) if tx[0] < current_step]

            txs_1h = [tx for tx in prior_orig if current_step - tx[0] <= 1]
            outflow_1h = sum(tx[1] for tx in txs_1h)
            start_bal_1h = txs_1h[0][3] if txs_1h else (current_balance + outflow_1h)
            outflow_1h_ratio = float(min((current_amount + outflow_1h) / max(start_bal_1h, 1.0), 10.0))

            txs_6h = [tx for tx in prior_orig if current_step - tx[0] <= 6]
            outflow_6h = sum(tx[1] for tx in txs_6h)
            start_bal_6h = txs_6h[0][3] if txs_6h else (current_balance + outflow_6h)
            outflow_6h_ratio = float(min((current_amount + outflow_6h) / max(start_bal_6h, 1.0), 10.0))

            receivers_6h = set(tx[2] for tx in txs_6h if tx[2])
            if name_dest:
                receivers_6h.add(name_dest)
            new_receivers_count = len(receivers_6h)

            prior_dest = [tx for tx in self.dest_tx_log.get(name_dest, []) if tx[0] < current_step]
            dest_amt_6h = sum(tx[1] for tx in prior_dest if current_step - tx[0] <= 6)
            dest_amt_1h = sum(tx[1] for tx in prior_dest if current_step - tx[0] <= 1)

            total_dest_6h = dest_amt_6h + current_amount
            total_dest_1h = dest_amt_1h + current_amount
            concentration = float(total_dest_1h / max(total_dest_6h, 1.0)) if total_dest_6h > 0 else 1.0

            return {
                "sender_outflow_1h_ratio": round(outflow_1h_ratio, 4),
                "sender_outflow_6h_ratio": round(outflow_6h_ratio, 4),
                "sender_new_receivers_count_6h": new_receivers_count,
                "receiver_inflow_concentration": round(concentration, 4),
            }

    def get_all_live_features(self, transaction_dict: dict[str, Any]) -> dict[str, float | int]:
        name_orig = str(transaction_dict.get("nameOrig", ""))
        name_dest = str(transaction_dict.get("nameDest", ""))
        step = int(transaction_dict.get("step", 12))
        amt = float(transaction_dict.get("amount", 0.0))
        bal = float(transaction_dict.get("oldbalanceOrg", 0.0))

        rec_feats = self.get_receiver_features(name_dest, step)
        net_feats = self.get_network_features(transaction_dict, current_step=step)
        seq_feats = self.get_sender_sequence_features(
            name_orig=name_orig,
            current_step=step,
            current_amount=amt,
            current_balance=bal,
            name_dest=name_dest,
        )

        merged = dict(rec_feats)
        merged.update(net_feats)
        merged.update(seq_feats)
        return merged

    def record_transaction(self, transaction_dict: dict[str, Any]) -> None:
        orig = str(transaction_dict.get("nameOrig", ""))
        dest = str(transaction_dict.get("nameDest", ""))
        step = int(transaction_dict.get("step", 12))
        amt = float(transaction_dict.get("amount", 0.0))
        bal = float(transaction_dict.get("oldbalanceOrg", 0.0))

        with self._lock:
            if orig:
                self.orig_tx_log[orig].append((step, amt, dest, bal))

            if dest:
                self.dest_tx_log[dest].append((step, amt))
                self.dest_cum_count[dest] += 1
                self.dest_cum_amount[dest] += amt
                self.dest_last_step[dest] = step

            if orig and dest:
                self.network_tracker.update_single(transaction_dict, cur_step=step)

    def seed_from_dataframe(self, df: pd.DataFrame, max_rows: int | None = 50000) -> int:
        sub_df = df.iloc[:max_rows] if max_rows is not None else df
        count = 0
        for _, row in sub_df.iterrows():
            orig = str(row.get("nameOrig", ""))
            dest = str(row.get("nameDest", ""))
            step = int(row.get("step", 0))
            amt = float(row.get("amount", 0.0))
            bal = float(row.get("oldbalanceOrg", 0.0))

            if orig:
                self.orig_tx_log[orig].append((step, amt, dest, bal))

            if dest:
                self.dest_tx_log[dest].append((step, amt))
                self.dest_cum_count[dest] += 1
                self.dest_cum_amount[dest] += amt
                self.dest_last_step[dest] = step

            if orig and dest:
                self.network_tracker.update_single(row.to_dict(), cur_step=step)
            count += 1
        return count

    def reset(self) -> None:
        with self._lock:
            self.dest_tx_log.clear()
            self.dest_cum_count.clear()
            self.dest_cum_amount.clear()
            self.dest_last_step.clear()
            self.network_tracker = NetworkGraphTracker(window_hours=self.window_hours)


_GLOBAL_HISTORY_STORE: AccountHistoryStore | None = None
_STORE_LOCK = threading.Lock()


def get_history_store() -> AccountHistoryStore:
    global _GLOBAL_HISTORY_STORE
    with _STORE_LOCK:
        if _GLOBAL_HISTORY_STORE is None:
            _GLOBAL_HISTORY_STORE = AccountHistoryStore(window_hours=24)
        return _GLOBAL_HISTORY_STORE


def reset_history_store() -> None:
    global _GLOBAL_HISTORY_STORE
    with _STORE_LOCK:
        if _GLOBAL_HISTORY_STORE is not None:
            _GLOBAL_HISTORY_STORE.reset()
