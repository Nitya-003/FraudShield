from __future__ import annotations

import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd

NETWORK_FEATURE_COLUMNS = [
    "dest_distinct_senders_24h",
    "dest_distinct_senders_total",
    "dest_fan_in_24h",
    "dest_fan_out_24h",
    "orig_in_to_out_gap",
    "dest_out_to_in_gap",
    "repeat_pair_count",
    "is_2hop_flow",
]

DEFAULT_NETWORK_VALUES = {
    "dest_distinct_senders_24h": 0,
    "dest_distinct_senders_total": 0,
    "dest_fan_in_24h": 0,
    "dest_fan_out_24h": 0,
    "orig_in_to_out_gap": 999.0,
    "dest_out_to_in_gap": 999.0,
    "repeat_pair_count": 0,
    "is_2hop_flow": 0,
}


class NetworkGraphTracker:

    def __init__(self, window_hours: int = 24):
        self.window_hours = window_hours
        self.dest_last_received_step: dict[str, int] = {}
        self.orig_last_sent_step: dict[str, int] = {}
        self.all_senders_seen: dict[str, set[str]] = defaultdict(set)
        self.senders_in_window: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.window_incoming: dict[str, deque[tuple[int, str]]] = defaultdict(deque)
        self.window_outgoing: dict[str, deque[int]] = defaultdict(deque)
        self.pair_counts: dict[tuple[str, str], int] = defaultdict(int)

    def extract_single(
        self,
        payload: dict[str, Any],
        cur_step: int | None = None,
    ) -> dict[str, float | int]:
        orig = str(payload.get("nameOrig", ""))
        dest = str(payload.get("nameDest", ""))
        step = int(payload.get("step", cur_step if cur_step is not None else 12))

        repeat_count = self.pair_counts.get((orig, dest), 0)

        distinct_total = len(self.all_senders_seen.get(dest, ()))
        distinct_24h = len(self.senders_in_window.get(dest, {}))
        fan_in_24h = len(self.window_incoming.get(dest, ()))

        fan_out_24h = len(self.window_outgoing.get(dest, ()))

        if orig in self.dest_last_received_step:
            orig_in_gap = float(max(0, step - self.dest_last_received_step[orig]))
        else:
            orig_in_gap = 999.0

        if dest in self.orig_last_sent_step:
            dest_out_gap = float(max(0, step - self.orig_last_sent_step[dest]))
        else:
            dest_out_gap = 999.0

        has_orig_incoming_24h = orig in self.window_incoming and len(self.window_incoming[orig]) > 0
        has_dest_outgoing_24h = dest in self.window_outgoing and len(self.window_outgoing[dest]) > 0
        is_2hop = 1 if (has_orig_incoming_24h or has_dest_outgoing_24h) else 0

        return {
            "dest_distinct_senders_24h": distinct_24h,
            "dest_distinct_senders_total": distinct_total,
            "dest_fan_in_24h": fan_in_24h,
            "dest_fan_out_24h": fan_out_24h,
            "orig_in_to_out_gap": orig_in_gap,
            "dest_out_to_in_gap": dest_out_gap,
            "repeat_pair_count": repeat_count,
            "is_2hop_flow": is_2hop,
        }

    def update_single(self, payload: dict[str, Any], cur_step: int | None = None) -> None:

        orig = str(payload.get("nameOrig", ""))
        dest = str(payload.get("nameDest", ""))
        step = int(payload.get("step", cur_step if cur_step is not None else 12))

        self.pair_counts[(orig, dest)] += 1
        self.all_senders_seen[dest].add(orig)
        self.senders_in_window[dest][orig] += 1
        self.window_incoming[dest].append((step, orig))

        self.dest_last_received_step[dest] = step
        self.orig_last_sent_step[orig] = step
        self.window_outgoing[orig].append(step)

        cutoff = step - self.window_hours
        q_in = self.window_incoming[dest]
        while q_in and q_in[0][0] < cutoff:
            old_st, old_sender = q_in.popleft()
            cnt_dict = self.senders_in_window[dest]
            cnt_dict[old_sender] -= 1
            if cnt_dict[old_sender] <= 0:
                del cnt_dict[old_sender]

        q_out = self.window_outgoing[orig]
        while q_out and q_out[0] < cutoff:
            q_out.popleft()


def compute_network_features(
    df: pd.DataFrame,
    window_hours: int = 24,
) -> dict[str, np.ndarray]:

    n = len(df)
    steps = df["step"].to_numpy(dtype=np.int32)
    origs = df["nameOrig"].astype(str).to_numpy()
    dests = df["nameDest"].astype(str).to_numpy()

    dest_distinct_senders_24h = np.zeros(n, dtype=np.int16)
    dest_distinct_senders_total = np.zeros(n, dtype=np.int32)
    dest_fan_in_24h = np.zeros(n, dtype=np.int16)
    dest_fan_out_24h = np.zeros(n, dtype=np.int16)
    orig_in_to_out_gap = np.full(n, 999.0, dtype=np.float32)
    dest_out_to_in_gap = np.full(n, 999.0, dtype=np.float32)
    repeat_pair_count = np.zeros(n, dtype=np.int16)
    is_2hop_flow = np.zeros(n, dtype=np.int8)

    dest_last_received_step: dict[str, int] = {}
    orig_last_sent_step: dict[str, int] = {}
    all_senders_seen: dict[str, set[str]] = defaultdict(set)
    senders_in_window: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    window_incoming: dict[str, deque[tuple[int, str]]] = defaultdict(deque)
    window_outgoing: dict[str, deque[int]] = defaultdict(deque)
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)

    i = 0
    while i < n:
        cur_step = steps[i]
        j = i
        while j < n and steps[j] == cur_step:
            j += 1

        for k in range(i, j):
            o = origs[k]
            d = dests[k]

            repeat_pair_count[k] = pair_counts.get((o, d), 0)

            if d in all_senders_seen:
                dest_distinct_senders_total[k] = len(all_senders_seen[d])
                dest_distinct_senders_24h[k] = len(senders_in_window[d])
                dest_fan_in_24h[k] = len(window_incoming[d])

            if d in window_outgoing:
                dest_fan_out_24h[k] = len(window_outgoing[d])

            if o in dest_last_received_step:
                orig_in_to_out_gap[k] = float(cur_step - dest_last_received_step[o])

            if d in orig_last_sent_step:
                dest_out_to_in_gap[k] = float(cur_step - orig_last_sent_step[d])

            has_orig_incoming_24h = o in window_incoming and len(window_incoming[o]) > 0
            has_dest_outgoing_24h = d in window_outgoing and len(window_outgoing[d]) > 0
            if has_orig_incoming_24h or has_dest_outgoing_24h:
                is_2hop_flow[k] = 1

        for k in range(i, j):
            o = origs[k]
            d = dests[k]

            pair_counts[(o, d)] += 1
            all_senders_seen[d].add(o)
            senders_in_window[d][o] += 1
            window_incoming[d].append((cur_step, o))

            dest_last_received_step[d] = cur_step
            orig_last_sent_step[o] = cur_step
            window_outgoing[o].append(cur_step)

        cutoff = cur_step - window_hours
        for k in range(i, j):
            o = origs[k]
            d = dests[k]

            q_in = window_incoming[d]
            while q_in and q_in[0][0] < cutoff:
                old_st, old_sender = q_in.popleft()
                cnt_dict = senders_in_window[d]
                cnt_dict[old_sender] -= 1
                if cnt_dict[old_sender] <= 0:
                    del cnt_dict[old_sender]

            q_out = window_outgoing[o]
            while q_out and q_out[0] < cutoff:
                q_out.popleft()

        i = j

    return {
        "dest_distinct_senders_24h": dest_distinct_senders_24h,
        "dest_distinct_senders_total": dest_distinct_senders_total,
        "dest_fan_in_24h": dest_fan_in_24h,
        "dest_fan_out_24h": dest_fan_out_24h,
        "orig_in_to_out_gap": orig_in_to_out_gap,
        "dest_out_to_in_gap": dest_out_to_in_gap,
        "repeat_pair_count": repeat_pair_count,
        "is_2hop_flow": is_2hop_flow,
    }

