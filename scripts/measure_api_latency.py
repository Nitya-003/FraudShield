"""
Single-Transaction Real-Time API Latency Benchmark.
Measures wall-clock latency across 1,000 real HTTP calls to the FastAPI /score microservice
across all four payment channels. Writes exact empirical p50 and p95 metrics to reports/metrics.json
and models/metadata.json.
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from pathlib import Path
import sys

# Ensure repository root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np

# Suppress console warnings during high-throughput benchmarking
warnings.filterwarnings("ignore")
logging.getLogger("uvicorn").setLevel(logging.ERROR)
logging.getLogger("fastapi").setLevel(logging.ERROR)

from fastapi.testclient import TestClient
from ml.api import app
from ml.config import METADATA_FILE, METRICS_FILE, REPORTS_DIR

CHANNELS_PAYLOADS = {
    "transfer_wallet": {
        "channel": "transfer_wallet",
        "payload": {
            "step": 500,
            "type": "TRANSFER",
            "amount": 250000.0,
            "oldbalanceOrg": 250000.0,
            "oldbalanceDest": 0.0,
            "nameOrig": "C_LAT_01",
            "nameDest": "M_LAT_01",
        },
        "top_k": 3,
    },
    "card": {
        "channel": "card",
        "payload": {
            "Amount": 120.0,
            "Time": 10000.0,
            "V1": -1.0,
            "V2": 0.5,
            "V3": -0.8,
            "V14": -3.5,
        },
        "top_k": 3,
    },
    "account": {
        "channel": "account",
        "payload": {
            "income": 0.8,
            "credit_risk_score": 150,
            "current_address_months_count": 24,
            "velocity_6h": 2,
            "customer_age": 30,
        },
        "top_k": 3,
    },
    "upi": {
        "channel": "upi",
        "payload": {
            "amount": 45000.0,
            "step": 120,
            "is_new_device": 1,
            "hour_of_day": 23,
            "dest_is_new_payee": 1,
        },
        "top_k": 3,
    },
}


def run_latency_benchmark(num_calls: int = 1000, save_to_file: bool = True) -> dict[str, any]:
    print("=" * 70)
    print(f"  MEASURING REAL SINGLE-TRANSACTION API INFERENCE LATENCY ({num_calls:,} CALLS)")
    print("=" * 70)

    client = TestClient(app)

    # 1. Warm-up (50 requests across channels)
    print("1. Executing 50 warmup requests to prime JIT / model caches...")
    ch_keys = list(CHANNELS_PAYLOADS.keys())
    for i in range(50):
        ch = ch_keys[i % len(ch_keys)]
        _ = client.post("/score", json=CHANNELS_PAYLOADS[ch])

    # 2. Benchmark 1,000 requests
    print(f"2. Benchmarking {num_calls} real scoring requests through /score...")
    internal_latencies = []
    roundtrip_latencies = []
    per_channel: dict[str, list[float]] = {ch: [] for ch in ch_keys}

    for i in range(num_calls):
        ch = ch_keys[i % len(ch_keys)]
        req_body = CHANNELS_PAYLOADS[ch]

        t0 = time.perf_counter()
        resp = client.post("/score", json=req_body)
        t1 = time.perf_counter()

        assert resp.status_code == 200, f"Call {i} failed: {resp.text}"
        data = resp.json()

        server_ms = float(data["latency_ms"])
        rt_ms = float((t1 - t0) * 1000.0)

        internal_latencies.append(server_ms)
        roundtrip_latencies.append(rt_ms)
        per_channel[ch].append(server_ms)

    # Compute distribution percentiles
    latency_stats = {
        "sample_size": num_calls,
        "p50_ms": round(float(np.percentile(internal_latencies, 50)), 2),
        "p90_ms": round(float(np.percentile(internal_latencies, 90)), 2),
        "p95_ms": round(float(np.percentile(internal_latencies, 95)), 2),
        "p99_ms": round(float(np.percentile(internal_latencies, 99)), 2),
        "mean_ms": round(float(np.mean(internal_latencies)), 2),
        "min_ms": round(float(np.min(internal_latencies)), 2),
        "max_ms": round(float(np.max(internal_latencies)), 2),
        "roundtrip_p50_ms": round(float(np.percentile(roundtrip_latencies, 50)), 2),
        "roundtrip_p95_ms": round(float(np.percentile(roundtrip_latencies, 95)), 2),
        "by_channel_p50": {
            ch: round(float(np.percentile(times, 50)), 2)
            for ch, times in per_channel.items()
        },
        "by_channel_p95": {
            ch: round(float(np.percentile(times, 95)), 2)
            for ch, times in per_channel.items()
        },
    }

    print("\nEmpirical Latency Benchmark Results:")
    print(f"  • Single-Transaction Server Latency p50: {latency_stats['p50_ms']} ms")
    print(f"  • Single-Transaction Server Latency p95: {latency_stats['p95_ms']} ms")
    print(f"  • Single-Transaction Server Latency p99: {latency_stats['p99_ms']} ms")
    print(f"  • Single-Transaction Client Roundtrip p50: {latency_stats['roundtrip_p50_ms']} ms")
    print(f"  • Single-Transaction Client Roundtrip p95: {latency_stats['roundtrip_p95_ms']} ms")
    for ch in ch_keys:
        print(f"    - Channel {ch:16s} | p50: {latency_stats['by_channel_p50'][ch]:.2f} ms | p95: {latency_stats['by_channel_p95'][ch]:.2f} ms")

    # Update reports/metrics.json and models/metadata.json only if requested
    if save_to_file:
        if METRICS_FILE.exists():
            with open(METRICS_FILE, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
            metrics_data["api_latency"] = latency_stats
            with open(METRICS_FILE, "w", encoding="utf-8") as f:
                json.dump(metrics_data, f, indent=2)
            print(f"\nSaved 'api_latency' to {METRICS_FILE}")

        # Update models/metadata.json
        if METADATA_FILE.exists():
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                meta_data = json.load(f)
            meta_data["api_latency"] = latency_stats
            with open(METADATA_FILE, "w", encoding="utf-8") as f:
                json.dump(meta_data, f, indent=2)
            print(f"Saved 'api_latency' to {METADATA_FILE}")

    return latency_stats


if __name__ == "__main__":
    run_latency_benchmark(1000)
