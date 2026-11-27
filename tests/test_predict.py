"""
Unit tests for the live prediction engine and SHAP explainability.
"""

import pytest
from ml.predict import predict


def test_predict_normal_transaction():
    tx = {
        "step": 100,
        "type": "PAYMENT",
        "amount": 500.0,
        "oldbalanceOrg": 10000.0,
        "newbalanceOrig": 9500.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
    }
    res = predict(tx)

    assert "risk_score" in res
    assert "band" in res
    assert "action" in res
    assert "top_reasons" in res
    assert "factors" in res

    assert res["band"] == "Low"
    assert res["action"] in {"Approve", "APPROVE"}
    assert res["risk_score"] < 0.45
    assert len(res["top_reasons"]) == 3
    assert all(isinstance(r, str) and len(r) > 0 for r in res["top_reasons"])


def test_predict_blatant_fraud():
    tx = {
        "step": 450,
        "type": "TRANSFER",
        "amount": 1000000.0,
        "oldbalanceOrg": 1000000.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
    }
    res = predict(tx)

    assert res["band"] == "High"
    assert res["action"] in {"Decline & Alert", "DECLINE_AND_ALERT"}
    assert res["risk_score"] > 0.90
    assert len(res["top_reasons"]) == 3


def test_predict_response_structure():
    tx = {
        "step": 150,
        "type": "CASH_OUT",
        "amount": 25000.0,
        "oldbalanceOrg": 50000.0,
        "newbalanceOrig": 25000.0,
        "oldbalanceDest": 10000.0,
        "newbalanceDest": 35000.0,
    }
    res = predict(tx)

    assert 0.0 <= res["risk_score"] <= 1.0
    assert res["band"] in {"Low", "Medium", "High"}
    assert res["action"] in {"Approve", "Step-Up (OTP)", "Decline & Alert", "APPROVE", "STEP_UP_OTP", "DECLINE_AND_ALERT"}
    for factor in res["factors"]:
        assert "feature" in factor
        assert "shap_value" in factor
        assert "direction" in factor
        assert "human_readable" in factor

