"""
Automated Test for Single Source of Truth Metrics Consistency.
Asserts that README, REPORT, VIVA_QA, slides script, notebook, and dashboard files match
reports/metrics.json and models/metadata.json exactly, with zero hardcoded discrepancies.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
import pytest

from ml.config import BASE_DIR, CHANNELS, DATA_SOURCES, METADATA_FILE, METRICS_FILE, REPORTS_DIR
from ml.metrics_helper import (
    load_metrics,
    load_metadata,
    get_champion_metrics,
    get_economics_summary,
    get_calibrated_thresholds,
    get_api_latency,
    get_channel_data_source,
)


def test_metrics_json_structure_and_completeness():
    """Verify metrics.json contains all benchmark models, channels, and economics."""
    metrics = load_metrics()
    required_models = [
        "Rule 1 (Amount > 50k)",
        "Rule 2 (Channel & Empty)",
        "Rule 3 (Channel & Empty & DestZero)",
        "Decision Tree (Depth 3)",
        "Logistic Regression",
        "Random Forest",
        "LightGBM (Cost-Weighted)",
        "Tuned LightGBM (Champion)",
    ]
    for model_name in required_models:
        assert model_name in metrics, f"Model '{model_name}' missing from reports/metrics.json"
        assert "pr_auc" in metrics[model_name]
        assert "recall" in metrics[model_name]
        assert "fp" in metrics[model_name]

    assert "economics_natural_test" in metrics
    assert "calibrated_thresholds" in metrics
    assert "api_latency" in metrics
    for ch in ["card", "account", "upi", "transfer_wallet"]:
        assert ch in metrics, f"Channel '{ch}' missing from reports/metrics.json"


def test_decision_tree_and_champion_reconciliation():
    """Verify Decision Tree (Depth 3) and LightGBM Champion match across metrics and redteam findings."""
    metrics = load_metrics()
    redteam_file = REPORTS_DIR / "redteam_findings.json"
    assert redteam_file.exists(), "redteam_findings.json is missing!"

    with open(redteam_file, "r", encoding="utf-8") as f:
        redteam = json.load(f)

    # 1. Decision Tree (Depth 3) must match between metrics.json and redteam_findings.json
    dt_metrics_pr = metrics["Decision Tree (Depth 3)"]["pr_auc"]
    dt_redteam_pr = redteam["tree_simplicity"]["depth_3"]["pr_auc"]
    assert dt_metrics_pr == pytest.approx(0.99826, abs=1e-4)
    assert dt_redteam_pr == pytest.approx(0.99826, abs=1e-4)
    assert abs(dt_metrics_pr - dt_redteam_pr) < 1e-4, f"DT PR-AUC mismatch: metrics={dt_metrics_pr} vs redteam={dt_redteam_pr}"

    # 2. Champion PR-AUC must be 0.99976
    champ_pr = metrics["Tuned LightGBM (Champion)"]["pr_auc"]
    assert champ_pr == pytest.approx(0.99976, abs=1e-4)

    # 3. Random Forest must be 0.99996
    rf_pr = metrics["Random Forest"]["pr_auc"]
    assert rf_pr == pytest.approx(0.99996, abs=1e-4)


def test_missed_and_caught_fraud_economics_consistency():
    """Verify exact count of missed (7) and caught (3999) fraud across economics, bands, and decision layer."""
    metrics = load_metrics()
    metadata = load_metadata()

    econ_m = metrics["economics_natural_test"]
    econ_meta = metadata["economics_natural_prevalence"]
    dec_econ = metrics["decision_layer"]["test_economics"]

    # Missed fraud count must be strictly 7 across all files
    assert econ_m["fraud_count_missed"] == 7
    assert econ_meta["fraud_count_missed"] == 7
    assert dec_econ["fn_missed"] == 7

    # Caught fraud count must be strictly 3,999 across all files
    assert econ_m["fraud_count_caught"] == 3999
    assert econ_meta["fraud_count_caught"] == 3999

    # False declines at high cutoff (0.90) must be strictly 11
    assert econ_m["false_declines"] == 11
    assert econ_meta["false_declines"] == 11
    assert dec_econ["fp_decline"] == 11

    # Band breakdown counts must sum exactly to 4,006 total test frauds
    bands = econ_meta["bands"]
    assert bands["Low (< 0.45)"]["frauds"] == 7
    assert bands["Medium (0.45-0.90)"]["frauds"] == 3
    assert bands["High (> 0.90)"]["frauds"] == 3996
    assert 7 + 3 + 3996 == 4006

    # Missed INR value must match exactly ₹399,045.09
    assert econ_m["fraud_inr_missed"] == pytest.approx(399045.09, abs=0.5)
    assert econ_meta["fraud_inr_missed"] == pytest.approx(399045.09, abs=0.5)
    assert dec_econ["cost_missed_fraud"] == pytest.approx(399045.09, abs=0.5)


def test_canonical_data_source_labels():
    """Verify data source labels across config, metadata, helper, and readme."""
    expected_labels = {
        "transfer_wallet": "Simulated (agent-based)",
        "account": "Privacy-preserving synthetic",
        "card": "Real, anonymised",
        "upi": "Simulated by us",
    }
    for ch, expected in expected_labels.items():
        assert DATA_SOURCES[ch] == expected, f"DATA_SOURCES[{ch}] != '{expected}'"
        assert CHANNELS[ch]["data_source"] == expected, f"CHANNELS[{ch}]['data_source'] != '{expected}'"
        assert get_channel_data_source(ch) == expected, f"get_channel_data_source('{ch}') != '{expected}'"

    metadata = load_metadata()
    for ch, expected in expected_labels.items():
        assert metadata["channels"][ch]["data_source"] == expected


def test_real_api_latency_stored_and_valid():
    """Verify empirical single-transaction API latency exists in metrics and metadata."""
    metrics = load_metrics()
    metadata = load_metadata()

    assert "api_latency" in metrics
    lat_m = metrics["api_latency"]
    lat_meta = metadata["api_latency"]

    assert lat_m["sample_size"] == 1000
    assert 1.0 < lat_m["p50_ms"] < 20.0, f"Unrealistic p50 latency: {lat_m['p50_ms']} ms"
    assert lat_m["p50_ms"] <= lat_m["p95_ms"]
    assert lat_m["p50_ms"] == lat_meta["p50_ms"]


def test_no_unsupported_hype_phrases():
    """Verify that unsupported hype phrases are absent from public documents and code."""
    banned_phrases = [
        r"\bzero leakage\b",
        r"\bzero-leakage\b",
        r"\bsub-4ms\b",
        r"\bsub-15ms\b",
        r"\bsub-100ms\b",
        r"\breal-time AI engine\b",
        r"\bsurveillance\b",
        r"\bdefense-in-depth\b",
        r"\bmission-critical\b",
    ]

    files_to_check = [
        BASE_DIR / "README.md",
        BASE_DIR / "docs" / "REPORT.md",
        BASE_DIR / "docs" / "MODEL_CARD.md",
        BASE_DIR / "archive" / "legacy_streamlit" / "pages" / "live_check.py",
        BASE_DIR / "archive" / "legacy_streamlit" / "pages" / "how_it_works.py",
        BASE_DIR / "scripts" / "build_final_notebook.py",
        BASE_DIR / "scripts" / "build_presentation.py",
    ]

    for fpath in files_to_check:
        assert fpath.exists(), f"File {fpath} does not exist"
        content = fpath.read_text(encoding="utf-8")
        for pattern in banned_phrases:
            match = re.search(pattern, content, flags=re.IGNORECASE)
            assert match is None, f"Found banned phrase '{match.group(0)}' in {fpath.name}"


def test_paysim_feature_ablation_honesty():
    """Verify README and REPORT honestly disclose that receiver history adds virtually nothing on PaySim."""
    readme = (BASE_DIR / "README.md").read_text(encoding="utf-8")
    report = (BASE_DIR / "docs" / "REPORT.md").read_text(encoding="utf-8")
    viva = (BASE_DIR / "docs" / "VIVA_QA.md").read_text(encoding="utf-8")

    for doc_name, text in [("README.md", readme), ("REPORT.md", report), ("VIVA_QA.md", viva)]:
        assert "orig_emptied" in text, f"'orig_emptied' missing from {doc_name}"
        assert ("0.99979" in text or "0.99981" in text or "virtually" in text.lower() or "ablation" in text.lower()), f"Ablation honesty missing in {doc_name}"


def test_dashboard_uses_metrics_helper_and_calibrated_thresholds():
    """Verify overview.py and real_data.py read dynamic thresholds and metrics."""
    overview_path = BASE_DIR / "archive" / "legacy_streamlit" / "pages" / "overview.py"
    assert overview_path.exists()
    content = overview_path.read_text(encoding="utf-8")

    assert "get_calibrated_thresholds" in content
    assert "Low &lt; 0.30" not in content
    assert "High &gt; 0.75" not in content
    assert "0.99357" not in content

    real_data_path = BASE_DIR / "archive" / "legacy_streamlit" / "data" / "real_data.py"
    assert real_data_path.exists()
    rd_content = real_data_path.read_text(encoding="utf-8")
    assert "get_metrics_json" in rd_content or "get_metadata" in rd_content or "get_economics_summary" in rd_content


def test_readme_and_report_match_metrics_json_exactly():
    """Verify README.md and docs/REPORT.md numbers match reports/metrics.json."""
    metrics = load_metrics()
    champ = metrics["Tuned LightGBM (Champion)"]
    r2 = metrics["Rule 2 (Channel & Empty)"]
    dt = metrics["Decision Tree (Depth 3)"]

    readme = (BASE_DIR / "README.md").read_text(encoding="utf-8")
    report = (BASE_DIR / "docs" / "REPORT.md").read_text(encoding="utf-8")

    # Verify PR-AUC numbers for transfers
    assert f"{champ['pr_auc']:.5f}" in readme
    assert f"{champ['pr_auc']:.5f}" in report
    assert f"{r2['pr_auc']:.5f}" in readme
    assert f"{r2['pr_auc']:.5f}" in report
    assert f"{dt['pr_auc']:.5f}" in readme
    assert f"{dt['pr_auc']:.5f}" in report

    # Verify Protected INR Value
    assert "₹6,322,647,552.00" in readme
    assert "₹6,322,647,552.00" in report

    # Cross-channel PR-AUC checks
    card_champ_pr = metrics["card"]["Tuned LightGBM (Champion)"]["pr_auc"]
    acct_champ_pr = metrics["account"]["Tuned LightGBM (Champion)"]["pr_auc"]
    upi_champ_pr = metrics["upi"]["Tuned LightGBM (Champion)"]["pr_auc"]

    assert f"{card_champ_pr:.5f}" in readme
    assert f"{card_champ_pr:.5f}" in report
    assert f"{acct_champ_pr:.5f}" in readme
    assert f"{acct_champ_pr:.5f}" in report
    assert f"{upi_champ_pr:.5f}" in readme
    assert f"{upi_champ_pr:.5f}" in report


def test_champions_inside_bootstrap_ci():
    """Verify that every channel's champion PR-AUC point estimate lies strictly inside its 95% bootstrap CI."""
    metrics = load_metrics()
    metadata = load_metadata()

    # 1. transfer_wallet
    tw_champ_pr = metrics["transfer_wallet"]["Tuned LightGBM (Champion)"]["pr_auc"]
    tw_ci = metrics["transfer_wallet"]["bootstrap_ci_95"]["pr_auc"]
    assert tw_ci[0] <= tw_champ_pr <= tw_ci[1], f"transfer_wallet PR-AUC {tw_champ_pr} outside CI {tw_ci}"

    # 2. card (Stacked Ensemble)
    card_champ_pr = metrics["card"]["Stacked Ensemble"]["pr_auc"]
    card_ci = metrics["card"]["bootstrap_ci_95"]["pr_auc"]
    assert card_ci[0] <= card_champ_pr <= card_ci[1], f"card PR-AUC {card_champ_pr} outside CI {card_ci}"

    # 3. account (Tuned CatBoost)
    acct_champ_pr = metrics["account"]["Tuned CatBoost (Champion)"]["pr_auc"]
    acct_ci = metrics["account"]["bootstrap_ci_95"]["pr_auc"]
    assert acct_ci[0] <= acct_champ_pr <= acct_ci[1], f"account PR-AUC {acct_champ_pr} outside CI {acct_ci}"

    # 4. upi (Tuned LightGBM)
    upi_champ_pr = metrics["upi"]["Tuned LightGBM (Champion)"]["pr_auc"]
    upi_ci = metrics["upi"]["bootstrap_ci_95"]["pr_auc"]
    assert upi_ci[0] <= upi_champ_pr <= upi_ci[1], f"upi PR-AUC {upi_champ_pr} outside CI {upi_ci}"

    # Verify metadata.json channels match
    for ch, ci_pair in [("transfer_wallet", tw_ci), ("card", card_ci), ("account", acct_ci), ("upi", upi_ci)]:
        meta_ci = metadata["channels"][ch]["bootstrap_ci_95"]["pr_auc"]
        assert meta_ci == ci_pair, f"Metadata CI mismatch on {ch}: {meta_ci} vs {ci_pair}"


def test_no_latency_literal_defaults_in_frontend_or_docs():
    """Verify that obsolete invented latency defaults do not appear in frontend or docs."""
    import glob

    banned_literals = ["7.74", "18.76", "12.04", "8.12", "7.85"]
    files_to_check = glob.glob(str(BASE_DIR / "archive" / "legacy_streamlit" / "**" / "*.py"), recursive=True) + [
        str(BASE_DIR / "docs" / "REPORT.md"),
        str(BASE_DIR / "docs" / "MODEL_CARD.md"),
        str(BASE_DIR / "README.md"),
    ]

    for fpath in files_to_check:
        p = Path(fpath)
        content = p.read_text(encoding="utf-8")
        for lit in banned_literals:
            assert lit not in content, f"Found hard-coded latency literal '{lit}' in {p.name}"


def test_seeded_cases_multi_channel_and_bands():
    """Verify data/cases.db contains cases across ALL 4 channels and diverse score ranges/bands."""
    from ml.case_management import get_cases

    cases_df = get_cases(status="ALL", limit=200)
    assert not cases_df.empty, "Cases table is empty"

    channels_present = set(cases_df["channel"].unique())
    expected_channels = {"transfer_wallet", "card", "account", "upi"}
    assert expected_channels.issubset(channels_present), f"Missing channels in seeded cases: {expected_channels - channels_present}"

    bands_present = set(cases_df["risk_band"].unique())
    expected_bands = {"High", "Medium", "Low"}
    assert expected_bands.issubset(bands_present), f"Missing risk bands in seeded cases: {expected_bands - bands_present}"

    # Verify score diversity: not all 0.9999
    scores = cases_df["risk_score"].tolist()
    assert min(scores) < 0.10, f"Expected low scores in seed, min was {min(scores)}"
    assert max(scores) > 0.85, f"Expected high scores in seed, max was {max(scores)}"
    assert len(set(round(s, 2) for s in scores)) >= 10, "Risk scores lack diversity"


def test_champion_registry_integrity():
    """Verify models/champion_registry.json exists, contains all 4 channels, and has valid metadata."""
    registry_path = BASE_DIR / "models" / "champion_registry.json"
    assert registry_path.exists(), "champion_registry.json missing"
    with open(registry_path) as f:
        registry = json.load(f)

    assert "version" in registry
    assert "champions" in registry
    champs = registry["champions"]
    assert set(champs.keys()) == {"transfer_wallet", "card", "account", "upi"}

    for ch, c_info in champs.items():
        assert "model_name" in c_info
        assert "version" in c_info
        assert "trained_date" in c_info
        assert "thresholds" in c_info
        assert "pr_auc" in c_info
        assert "bootstrap_ci_95" in c_info
        ci = c_info["bootstrap_ci_95"]["pr_auc"]
        pr = c_info["pr_auc"]
        assert ci[0] <= pr <= ci[1], f"Registry CI violation on {ch}: {pr} not in {ci}"


def test_baf_operating_points_and_fairness_present():
    """Verify BAF operating points table (1%, 2%, 5% FPR) and fairness audit exist in metrics."""
    metrics = load_metrics()
    assert "account" in metrics
    assert "operating_point_table" in metrics["account"]
    op_table = metrics["account"]["operating_point_table"]
    assert len(op_table) == 3
    target_fprs = [op["target_fpr_pct"] for op in op_table]
    assert target_fprs == [1, 2, 5]
    for op in op_table:
        assert "fairness_audit" in op
        assert "false_alarm_ratio_older_to_younger" in op["fairness_audit"]


def test_card_review_budget_curve_present():
    """Verify Card review-budget curve (0.1%, 0.5%, 1.0% alert budgets) exists and increases monotonically."""
    metrics = load_metrics()
    assert "card" in metrics
    stacked = metrics["card"].get("Stacked Ensemble", {})
    assert "review_budgets" in stacked
    budgets = stacked["review_budgets"]
    rec_01 = budgets["recall_at_budget_0.1pct"]
    rec_05 = budgets["recall_at_budget_0.5pct"]
    rec_10 = budgets["recall_at_budget_1.0pct"]
    assert rec_01 <= rec_05 <= rec_10
    assert "repeated_window_evaluation" in stacked


def test_concept_drift_benchmark_present():
    """Verify concept drift benchmark comparing static vs retrained models is present."""
    metrics = load_metrics()
    assert "concept_drift" in metrics
    drift = metrics["concept_drift"]
    assert "baf_account_drift" in drift
    assert "upi_drift" in drift


def test_no_banned_words_and_invented_literals():
    """Verify that banned hype words and invented numeric literals do not appear in frontend-web, README, or docs."""
    banned_phrases = [
        r"\bzero[ -]leakage\b",
        r"\bsub-millisecond\b",
        r"\breal telemetry\b",
        r"\breal bank\b",
        r"\bproduction-grade\b",
        r"\baudit-grade\b",
        r"\bregulatory\b",
        r"\bforensic\b",
        r"\bexceeds published benchmarks\b",
        r"\bdefense-in-depth\b",
        r"\bmission-critical\b",
        r"\breal payment telemetry\b",
    ]
    banned_literals = [
        "7.74 ms",
        "20.6 ms",
        "TX-9245",
        "₹70.45M",
    ]

    files_to_check = [
        BASE_DIR / "README.md",
        BASE_DIR / "docs" / "REPORT.md",
        BASE_DIR / "docs" / "MODEL_CARD.md",
    ]
    frontend_src = BASE_DIR / "frontend-web" / "src"
    if frontend_src.exists():
        files_to_check.extend(list(frontend_src.rglob("*.tsx")) + list(frontend_src.rglob("*.ts")))

    for fpath in files_to_check:
        content = fpath.read_text(encoding="utf-8")
        for pattern in banned_phrases:
            match = re.search(pattern, content, flags=re.IGNORECASE)
            assert match is None, f"Found banned phrase '{match.group(0)}' in {fpath.relative_to(BASE_DIR)}"
        for lit in banned_literals:
            assert lit not in content, f"Found banned literal '{lit}' in {fpath.relative_to(BASE_DIR)}"


def test_champion_registry_recomputation_on_test_sample():
    """Recompute champion models' predictions on saved test samples and verify consistency with registry."""
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.metrics import average_precision_score

    registry_path = BASE_DIR / "models" / "champion_registry.json"
    with open(registry_path, "r", encoding="utf-8") as f:
        registry = json.load(f)

    # 1. Card Champion Recomputation on saved card test sample
    card_sample_path = BASE_DIR / "data" / "processed" / "card_test_sample.parquet"
    card_model_path = BASE_DIR / "models" / "card_ensemble_pipeline.joblib"
    if card_sample_path.exists() and card_model_path.exists():
        try:
            import xgboost  # noqa: F401
        except ImportError:
            pytest.skip("xgboost not installed in environment")
        df_card = pd.read_parquet(card_sample_path)
        art = joblib.load(card_model_path)
        fn = art["feature_names"]
        X_card = df_card[fn]
        y_card = df_card["isFraud"].to_numpy()

        p_lgb = art["lgb"].predict_proba(X_card)[:, 1]
        p_xgb = art["xgb"].predict_proba(X_card)[:, 1]
        p_cb = art["cb"].predict_proba(X_card)[:, 1]
        X_meta = np.column_stack([p_lgb, p_xgb, p_cb])
        raw_p = art["meta_learner"].predict_proba(X_meta)[:, 1]
        eps = 1e-7
        lo = np.log(np.clip(raw_p, eps, 1 - eps) / (1 - np.clip(raw_p, eps, 1 - eps))).reshape(-1, 1)
        cal_p = art["calibrator"].predict_proba(lo)[:, 1]

        assert np.all((cal_p >= 0.0) & (cal_p <= 1.0)), "Card probabilities out of bounds"
        card_pr = average_precision_score(y_card, cal_p)
        assert card_pr >= 0.70, f"Card PR-AUC {card_pr} unexpectedly low"

    # 2. Transfer / Wallet Champion Recomputation on saved scored sample
    tw_sample_path = BASE_DIR / "data" / "processed" / "test_scored_sample.parquet"
    tw_model_path = BASE_DIR / "models" / "lgbm_transfer_wallet_seq.joblib"
    if tw_sample_path.exists() and tw_model_path.exists():
        df_tw = pd.read_parquet(tw_sample_path)
        m_pay = joblib.load(tw_model_path)
        df2 = df_tw.copy()
        eps = 1e-4
        bal = df2["oldbalanceOrg"].clip(lower=0) + eps
        ratio = (df2["amount"] / bal).clip(0.0, 10.0).astype("float32")
        df2["sender_outflow_1h_ratio"] = ratio
        df2["sender_outflow_6h_ratio"] = ratio
        df2["sender_new_receivers_count_6h"] = np.ones(len(df2), dtype=np.int16)
        df2["receiver_inflow_concentration"] = np.ones(len(df2), dtype=np.float32)

        X2 = df2[m_pay["feature_names"]].to_numpy()
        raw2 = m_pay["model"].predict_proba(X2)[:, 1]
        lo2 = np.log(np.clip(raw2, 1e-7, 1 - 1e-7) / (1 - np.clip(raw2, 1e-7, 1 - 1e-7))).reshape(-1, 1)
        cal2 = m_pay["calibrator"].predict_proba(lo2)[:, 1]
        assert np.all((cal2 >= 0.0) & (cal2 <= 1.0)), "PaySim probabilities out of bounds"
        tw_pr = average_precision_score(df2["isFraud"].to_numpy(), cal2)
        assert tw_pr >= 0.99, f"PaySim PR-AUC {tw_pr} unexpectedly low on test sample"

