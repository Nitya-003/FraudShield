"""
FraudShield API Hardening & Operational Endpoint Tests.

Tests:
1. API-key authentication (valid key, invalid key -> 401, bearer auth).
2. Rate limiting (sliding window threshold -> 429 with Retry-After).
3. Request logging and response headers (X-Model-Version, X-Process-Time-Ms).
4. Readable input validation error formatting (HTTP 422).
5. Comprehensive /health and /version endpoints.
6. Real stream replay endpoints (/api/stream, /api/stream/tick, /api/stream/stats).
7. BAF demographic fairness analysis (/api/fairness/baf).
8. Standardized reason-code catalog (/api/reason-codes).
"""

from __future__ import annotations

import os
import time
from starlette.testclient import TestClient

from ml.api import app, _RATE_LIMIT_BUCKETS, RATE_LIMIT_MAX_PER_MINUTE

client = TestClient(app)


def test_health_endpoint() -> None:
    """Verify /health returns rich operational telemetry."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "uptime_seconds" in data
    assert "channels_available" in data
    assert "stream_replay" in data
    assert data["stream_replay"]["events_in_buffer"] >= 0
    assert data["platform_version"] == "2.0.0"


def test_version_endpoint() -> None:
    """Verify /version returns platform and champion model versions."""
    resp = client.get("/version")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["platform_version"] == "2.0.0"
    assert "champions" in data
    champs = data["champions"]
    for ch in ["transfer_wallet", "card", "account", "upi"]:
        assert ch in champs
        assert "version" in champs[ch]
        assert "model_name" in champs[ch]


def test_response_headers_metadata() -> None:
    """Verify standard security and performance headers on API responses."""
    resp = client.get("/health")
    assert "x-model-version" in resp.headers or "X-Model-Version" in resp.headers
    assert resp.headers.get("X-Model-Version") == "2.0.0"
    assert "X-Process-Time-Ms" in resp.headers
    proc_time = float(resp.headers["X-Process-Time-Ms"])
    assert proc_time >= 0.0


def test_api_key_authentication() -> None:
    """Verify API-key authentication behavior."""
    # 1. Valid key via X-API-Key
    resp_valid = client.get("/api/auth/verify", headers={"X-API-Key": "fs_live_test_key_2026"})
    assert resp_valid.status_code == 200
    assert resp_valid.json()["authenticated"] is True

    # 2. Valid key via Bearer token
    resp_bearer = client.get("/api/auth/verify", headers={"Authorization": "Bearer fs_live_test_key_2026"})
    assert resp_bearer.status_code == 200
    assert resp_bearer.json()["authenticated"] is True

    # 3. Invalid key returns 401 Unauthorized
    resp_invalid = client.get("/api/auth/verify", headers={"X-API-Key": "invalid_spoofed_key_999"})
    assert resp_invalid.status_code == 401
    assert "Invalid API Key" in resp_invalid.json()["detail"]


def test_readable_validation_errors() -> None:
    """Verify that pydantic validation errors are transformed into clean human-readable JSON."""
    # Send empty payload to /score (missing required channel and payload)
    resp = client.post("/score", json={})
    assert resp.status_code == 422
    data = resp.json()
    assert data["status"] == "error"
    assert data["error_type"] == "validation_error"
    assert "Input validation failed" in data["message"]
    assert "channel" in data["message"] or "payload" in data["message"]
    assert isinstance(data["details"], list)
    assert len(data["details"]) > 0


def test_rate_limiting_enforcement() -> None:
    """Verify that requests exceeding quota return HTTP 429 with Retry-After header."""
    test_ip = "192.168.1.99"
    now = time.time()
    # Pre-fill bucket to exceed limit
    _RATE_LIMIT_BUCKETS[test_ip] = [now] * (RATE_LIMIT_MAX_PER_MINUTE + 5)

    # Make request simulating this client IP
    resp = client.get("/api/overview", headers={"x-forwarded-for": test_ip})
    # Note: TestClient client.host defaults to 'testclient', let's simulate with mock client IP
    _RATE_LIMIT_BUCKETS["testclient"] = [now] * (RATE_LIMIT_MAX_PER_MINUTE + 5)
    resp_limited = client.get("/api/overview")
    assert resp_limited.status_code == 429
    assert resp_limited.headers.get("Retry-After") == "60"
    data = resp_limited.json()
    assert data["status"] == "error"
    assert data["error_type"] == "rate_limit_exceeded"

    # Reset bucket so subsequent tests are unaffected
    _RATE_LIMIT_BUCKETS.clear()


def test_stream_replay_endpoints() -> None:
    """Verify real stream replay polling, tick, and stats endpoints."""
    # 1. Stream polling
    resp_stream = client.get("/api/stream?limit=10")
    assert resp_stream.status_code == 200
    data = resp_stream.json()
    assert data["status"] == "success"
    assert data["mode"] == "test_set_replay"
    assert isinstance(data["items"], list)
    assert len(data["items"]) > 0
    assert "stats" in data

    # 2. Stream tick
    resp_tick = client.post("/api/stream/tick")
    assert resp_tick.status_code == 200
    tick_data = resp_tick.json()
    assert tick_data["status"] == "success"
    assert "decision" in tick_data
    assert "risk_score" in tick_data["decision"]

    # 3. Stream stats
    resp_stats = client.get("/api/stream/stats")
    assert resp_stats.status_code == 200
    stats = resp_stats.json()["stats"]
    assert "total_events" in stats
    assert "loss_averted_inr" in stats
    assert "latency_p50_ms" in stats


def test_fairness_baf_endpoint() -> None:
    """Verify /api/fairness/baf returns audited disparity and mitigation tradeoff."""
    resp = client.get("/api/fairness/baf")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["demographic_attribute"] == "customer_age"
    # Documented 3.06x disparity at 1% FPR
    assert data["disparity_at_1pct_target_fpr"]["disparity_ratio"] == 3.06
    # Mitigation equalizes to 1.0x disparity
    assert data["equalized_fpr_mitigation"]["mitigated_disparity_ratio"] == 1.0
    assert data["equalized_fpr_mitigation"]["recall_tradeoff_delta_pct"] == -3.43


def test_reason_codes_catalog_endpoint() -> None:
    """Verify /api/reason-codes returns standardized catalog."""
    resp = client.get("/api/reason-codes")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    catalog = data["catalog"]
    assert isinstance(catalog, list)
    assert len(catalog) >= 10
    codes = [c["code"] for c in catalog]
    assert "RC_AMOUNT_HIGH_DEVIATION" in codes
    assert "RC_ORIGIN_BALANCE_DRAINED" in codes
    assert "RC_REMOTE_ACCESS_ACTIVE" in codes

