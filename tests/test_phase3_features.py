"""
Unit Tests for Phase 3 Features:
1. Attack Lab Evasion Simulation Engine (ml/attack_lab.py)
2. Customer Step-Up Verification Messaging (ml/step_up.py)
3. Entity View 1-2 Hop Graph, Flow Breakdown & Sparse Fallback (ml/entity_view.py)
4. Case Management Filters, Audit Trail & Feedback Dataset Export (ml/case_management.py)
5. Printable Case Report Plain Wording & Neutrality (ml/case_report.py)
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import pandas as pd

from ml.attack_lab import (
    get_attack_history,
    get_attack_presets,
    simulate_attack_scenario,
)
from ml.case_management import (
    export_feedback_dataset,
    feed_confirmed_labels_to_feedback_loop,
    get_case_audit_logs,
    get_cases,
    update_case_status,
)
from ml.case_report import generate_case_report_html
from ml.entity_view import (
    DEFAULT_MULE_HUB,
    build_entity_network_graph,
    build_flow_breakdown_chart,
    get_default_flagged_entities,
    get_entity_profile,
)
from ml.step_up import generate_step_up_message


# =============================================================================
# 1. ATTACK LAB TESTS
# =============================================================================

def test_attack_lab_presets() -> None:
    presets = get_attack_presets()
    assert len(presets) >= 3
    channels = {p["channel"] for p in presets}
    assert "transfer_wallet" in channels
    assert "upi" in channels
    for p in presets:
        assert "payload" in p
        assert "amount" in p["payload"]


def test_attack_lab_split_transfers_simulation() -> None:
    presets = get_attack_presets()
    p = presets[0]  # PaySim full drain
    res = simulate_attack_scenario(
        channel=p["channel"],
        base_payload=p["payload"],
        attack_type="split_transfers",
        params={"k_splits": 3, "split_mode": "equal", "time_gap_hours": 0, "amount_fraction": 1.0},
    )

    assert "base_score" in res
    assert "peak_attack_score" in res
    assert "verdict" in res
    assert res["verdict"] in ["BLOCKED", "STEP_UP_CHALLENGED", "EVADED"]
    assert len(res["transfers"]) == 3
    total_split_amt = sum(t["amount"] for t in res["transfers"])
    assert pytest.approx(total_split_amt, 0.01) == p["payload"]["amount"]


def test_attack_lab_timing_delay_simulation() -> None:
    presets = get_attack_presets()
    p = presets[0]
    res = simulate_attack_scenario(
        channel=p["channel"],
        base_payload=p["payload"],
        attack_type="change_timing",
        params={"k_splits": 2, "split_mode": "equal", "time_gap_hours": 4, "amount_fraction": 1.0},
    )
    assert len(res["transfers"]) == 2
    step1 = res["transfers"][0]["step"]
    step2 = res["transfers"][1]["step"]
    assert step2 - step1 == 4  # 4 hour offset


def test_attack_lab_partial_drain_simulation() -> None:
    presets = get_attack_presets()
    p = presets[0]
    res = simulate_attack_scenario(
        channel=p["channel"],
        base_payload=p["payload"],
        attack_type="reduce_amount",
        params={"k_splits": 1, "amount_fraction": 0.15},
    )
    assert len(res["transfers"]) == 1
    assert pytest.approx(res["transfers"][0]["amount"], 0.01) == p["payload"]["amount"] * 0.15


def test_attack_lab_history_logging() -> None:
    hist = get_attack_history()
    assert isinstance(hist, pd.DataFrame)
    assert len(hist) > 0
    assert "verdict" in hist.columns
    assert "base_score" in hist.columns


# =============================================================================
# 2. STEP-UP MESSAGING TESTS
# =============================================================================

def test_step_up_messaging_bilingual() -> None:
    payload = {
        "amount": 75000.0,
        "nameDest": "C99887766",
        "oldbalanceOrg": 80000.0,
        "type": "TRANSFER",
    }
    msg = generate_step_up_message(channel="transfer_wallet", payload=payload)
    assert "english_message" in msg
    assert "hindi_message" in msg
    assert "otp_code" in msg
    assert "₹75,000.00" in msg["english_message"]
    assert "₹75,000.00" in msg["hindi_message"]
    assert msg["otp_code"] in msg["english_message"]
    assert msg["otp_code"] in msg["hindi_message"]


def test_step_up_upi_collect_scam_warning() -> None:
    payload = {
        "amount": 25000.0,
        "payee_vpa": "fake_refund@okaxis",
        "type": "COLLECT",
        "is_collect_request": True,
    }
    msg = generate_step_up_message(channel="upi", payload=payload)
    assert msg["is_collect_request"] is True
    assert msg["has_scam_warning"] is True
    assert "CRITICAL WARNING: You are SENDING money, NOT receiving money" in msg["english_message"]
    assert "महत्वपूर्ण चेतावनी: आप पैसे भेज रहे हैं, प्राप्त नहीं कर रहे हैं" in msg["hindi_message"]


# =============================================================================
# 3. ENTITY VIEW TESTS
# =============================================================================

def test_entity_view_default_mule_hub() -> None:
    default_entities = get_default_flagged_entities("transfer_wallet")
    assert DEFAULT_MULE_HUB in default_entities
    profile = get_entity_profile(DEFAULT_MULE_HUB, "transfer_wallet")
    assert profile["entity_id"] == DEFAULT_MULE_HUB
    assert profile["linked_accounts_count"] >= 5
    assert profile["transaction_count"] >= 10
    assert profile["is_sparse"] is False


def test_entity_view_graph_1_and_2_hop() -> None:
    fig1 = build_entity_network_graph(DEFAULT_MULE_HUB, hops=1)
    assert fig1 is not None
    assert len(fig1.data) == 2  # edge trace + node trace

    fig2 = build_entity_network_graph(DEFAULT_MULE_HUB, hops=2)
    assert fig2 is not None
    assert len(fig2.data) == 2


def test_entity_view_flow_chart() -> None:
    fig_flow = build_flow_breakdown_chart(DEFAULT_MULE_HUB)
    assert fig_flow is not None
    assert len(fig_flow.data) >= 1


def test_entity_view_sparse_account_handling() -> None:
    sparse_id = "C824064320"
    profile = get_entity_profile(sparse_id, "transfer_wallet")
    assert profile["is_sparse"] is True
    assert "Sparse Entity" in profile["sparse_message"] or "Sparse" in profile["sparse_message"]
    # Ensure graph generation does not crash on sparse entity
    fig_sparse = build_entity_network_graph(sparse_id, hops=1)
    assert fig_sparse is not None


# =============================================================================
# 4. CASE MANAGEMENT TESTS
# =============================================================================

def test_case_management_band_filtering() -> None:
    all_cases = get_cases(limit=100)
    assert not all_cases.empty

    high_cases = get_cases(band="High", limit=100)
    if not high_cases.empty:
        assert (high_cases["risk_band"] == "High").all()

    med_cases = get_cases(band="Medium", limit=100)
    if not med_cases.empty:
        assert (med_cases["risk_band"] == "Medium").all()


def test_case_management_status_and_audit_trail() -> None:
    cases = get_cases(limit=10)
    assert not cases.empty
    test_case_id = cases.iloc[0]["case_id"]

    test_note = "Investigation note: verified via customer call."
    update_case_status(
        case_id=test_case_id,
        new_status="CONFIRMED_FRAUD",
        notes=test_note,
        analyst_name="Test Analyst",
    )

    updated_cases = get_cases(limit=100)
    row = updated_cases[updated_cases["case_id"] == test_case_id].iloc[0]
    assert row["status"] == "CONFIRMED_FRAUD"
    assert row["analyst_notes"] == test_note

    # Verify audit logs
    audit_trail = get_case_audit_logs(test_case_id)
    assert len(audit_trail) >= 1
    latest_log = audit_trail[0]
    assert latest_log["action"] == "STATUS_CHANGE_CONFIRMED_FRAUD"
    assert latest_log["analyst_name"] == "Test Analyst"
    assert latest_log["notes"] == test_note


def test_case_management_feedback_export() -> None:
    df_fb = export_feedback_dataset()
    assert isinstance(df_fb, pd.DataFrame)
    if not df_fb.empty:
        assert "case_id" in df_fb.columns
        assert "channel" in df_fb.columns
        assert "true_label" in df_fb.columns
        assert set(df_fb["true_label"].unique()).issubset({0, 1})


def test_case_management_feedback_loop_trigger() -> None:
    res = feed_confirmed_labels_to_feedback_loop()
    assert "status" in res
    assert "updated_unseen_recall_pct" in res
    assert res["updated_unseen_recall_pct"] >= 17.50


# =============================================================================
# 5. CASE REPORT NEUTRAL WORDING TESTS
# =============================================================================

def test_case_report_neutral_wording() -> None:
    payload = {"amount": 50000.0, "type": "TRANSFER"}
    res = {
        "risk_score": 0.88,
        "band": "Medium",
        "action": "Step-Up (OTP)",
        "top_reasons": ["Unusual transfer velocity"],
    }
    html_out = generate_case_report_html(
        channel="transfer_wallet",
        payload=payload,
        result=res,
        report_id="FS-TEST-001",
    )

    assert "FS-TEST-001" in html_out
    assert "FRAUDSHIELD TRANSACTION DECISION REPORT" in html_out

    # Forbidden words test
    forbidden_terms = [
        "audit-grade",
        "audit-ready",
        "production-grade",
        "forensic",
        "surveillance",
        "exfiltration",
        "defense-in-depth",
        "mission-critical",
        "regulatory compliance",
    ]
    html_lower = html_out.lower()
    for term in forbidden_terms:
        assert term not in html_lower, f"Forbidden hype term '{term}' found in case report!"

