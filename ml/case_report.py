from __future__ import annotations

import html
from datetime import datetime
from typing import Any


def generate_case_report_html(
    channel: str,
    payload: dict[str, Any],
    result: dict[str, Any],
    low_th: float = 0.45,
    high_th: float = 0.90,
    model_version: str = "FraudShield Decision Engine (Champion Model)",
    report_id: str | None = None,
) -> str:
    if report_id is None:
        report_id = f"FS-REP-{abs(hash(str(payload))) % 10000000:07d}"

    timestamp_now = html.escape(str(result.get("timestamp", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"))))
    data_source_label = html.escape(str(result.get("data_source", "Simulated (agent-based)")))
    action = html.escape(str(result.get("action", "Approve")))
    band = str(result.get("band", "Low")).capitalize()
    score = float(result.get("risk_score", 0.0))
    cf_rec = str(result.get("counterfactual", "No parameter adjustment required; within normal authorization limits."))

    amt = float(payload.get("amount", payload.get("Amount", 0.0)))

    band_color = "#dc2626" if band == "High" else ("#d97706" if band == "Medium" else "#16a34a")

    reasons = result.get("top_reasons", ["Normal authorization pattern"])
    if not reasons:
        reasons = ["Normal authorization pattern"]
    reasons_html = "".join([f"<li>{html.escape(r)}</li>" for r in reasons])

    report_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>FraudShield Transaction Report - {report_id}</title>
<style>
  @media print {{
    body {{ background: #fff !important; color: #000 !important; }}
    .no-print {{ display: none !important; }}
  }}
  body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: #0f172a;
    background: #ffffff;
    margin: 0;
    padding: 24px;
    font-size: 13px;
    line-height: 1.5;
  }}
  .card {{
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 24px;
    max-width: 820px;
    margin: 0 auto;
    background: #ffffff;
  }}
  .header {{
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    border-bottom: 2px solid #0f172a;
    padding-bottom: 12px;
    margin-bottom: 20px;
  }}
  .title {{
    margin: 0;
    font-size: 20px;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: #0f172a;
  }}
  .subtitle {{
    font-size: 12px;
    color: #64748b;
    margin-top: 3px;
  }}
  .meta {{
    text-align: right;
    font-size: 11.5px;
    color: #64748b;
  }}
  .grid-2 {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-bottom: 18px;
  }}
  .section-box {{
    background: #f8fafc;
    padding: 14px;
    border-radius: 6px;
    border: 1px solid #e2e8f0;
  }}
  .section-title {{
    font-weight: 700;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #475569;
    margin-bottom: 8px;
  }}
  .field-row {{
    display: flex;
    justify-content: space-between;
    margin-bottom: 4px;
  }}
  .reasons-box {{
    margin-bottom: 18px;
  }}
  .reasons-box ul {{
    margin: 6px 0 0 0;
    padding-left: 20px;
    color: #1e293b;
  }}
  .counterfactual-box {{
    background: #f0fdf4;
    border-left: 4px solid #16a34a;
    padding: 12px 14px;
    border-radius: 4px;
    font-size: 12px;
    color: #166534;
    margin-bottom: 20px;
  }}
  .footer {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 10.5px;
    color: #94a3b8;
    border-top: 1px solid #e2e8f0;
    padding-top: 10px;
  }}
</style>
</head>
<body>
  <div class="card">
    <div class="header">
      <div>
        <h2 class="title">FRAUDSHIELD TRANSACTION DECISION REPORT</h2>
        <div class="subtitle">Pre-Authorization Risk Assessment & Investigation Summary</div>
      </div>
      <div class="meta">
        <div><strong>Report ID:</strong> {report_id}</div>
        <div><strong>Generated:</strong> {timestamp_now}</div>
      </div>
    </div>

    <div class="grid-2">
      <div class="section-box">
        <div class="section-title">Transaction Parameters</div>
        <div class="field-row"><span>Channel:</span> <strong>{html.escape(channel)}</strong></div>
        <div class="field-row"><span>Data Source:</span> <span>{data_source_label}</span></div>
        <div class="field-row"><span>Transaction Amount:</span> <strong>₹{amt:,.2f}</strong></div>
        <div class="field-row"><span>Recommended Action:</span> <strong>{action}</strong></div>
      </div>
      <div class="section-box">
        <div class="section-title">Risk Assessment & Scoring</div>
        <div class="field-row"><span>Calibrated Risk Score:</span> <strong>{score:.4f}</strong></div>
        <div class="field-row"><span>Assigned Band:</span> <strong style="color:{band_color}">{band.upper()}</strong></div>
        <div class="field-row"><span>Threshold Matrix:</span> <span>Low: {low_th:.3f} | High: {high_th:.3f}</span></div>
        <div class="field-row"><span>Decision Engine:</span> <span>{html.escape(model_version)}</span></div>
      </div>
    </div>

    <div class="reasons-box">
      <div class="section-title">Key Attribution Signals (SHAP Log-Odds Contributions)</div>
      <ul>
        {reasons_html}
      </ul>
    </div>

    <div class="counterfactual-box">
      <strong>Actionable Counterfactual Guidance:</strong> {html.escape(cf_rec)}
    </div>

    <div class="footer">
      <span>Audited Machine Learning Pipeline · No Handcrafted Overrides</span>
      <span>FraudShield Decision Engine</span>
    </div>
  </div>
</body>
</html>
"""
    return report_html.strip()
