import os

def update():
    with open('README.md', 'r', encoding='utf-8') as f:
        text = f.read()

    # 1. Update test badge and counts
    text = text.replace('badge/Tests-149%20Passed-success.svg', 'badge/Tests-156%20Passed-success.svg')
    text = text.replace('149 tests', '156 tests')
    text = text.replace('149 / 149 Tests Passing', '156 / 156 Tests Passing')

    # 2. Add academic note callout
    old_intro = 'Developed for **Fundamentals of Machine Learning** coursework  \n**Candidate:** Ayush  \n**Technical Role:** Senior ML Engineer & System Designer\n\n---'
    new_intro = '''Developed for **Fundamentals of Machine Learning** coursework  
**Candidate:** Ayush  
**Technical Role:** Senior ML Engineer & System Designer

> [!NOTE]
> **Academic Project & Dataset Disclosure:** FraudShield is an academic research and engineering project built for machine learning coursework. All underlying datasets are either public academic benchmarks (ULB Credit Card, PaySim agent-based simulation, BAF synthetic dataset) or simulated by our research pipeline (Indian UPI Simulator). The system does not use or connect to private banking infrastructure or live bank networks.

---'''

    if old_intro in text:
        text = text.replace(old_intro, new_intro)

    # 3. Update screenshots table in Section 6
    old_s6 = '''| View | Screenshot | Description |
|---|---|---|
| **Executive Overview** | ![Overview](reports/figures/screens/01_overview.png) | Audited fraud value intercepted (₹6,322.65M on PaySim test set), hourly volume distribution, risk band stacked distribution, and top fraud drivers. |
| **Live Pre-Authorization Check** | ![Live Check](reports/figures/screens/02_live_check.png) | Fast single-digit millisecond scoring (p50: 7.92 ms), custom semicircle SVG risk gauge, TreeSHAP reason codes, counterfactual adjustments, and bilingual customer challenge previews. |
| **Scored Transactions Archive** | ![Transactions](reports/figures/screens/03_transactions.png) | Paged TanStack data table with multi-channel filters, inline risk progress bars, and selection checkboxes. |
| **Investigator Queue & Cases** | ![Cases](reports/figures/screens/04_cases.png) | Alert adjudication workflow supporting `OPEN`, `CONFIRMED_FRAUD`, and `FALSE_POSITIVE` states with audit notes and feedback export. |
| **360-Degree Entity Graph** | ![Entity View](reports/figures/screens/07_entity.png) | Multi-hop topology for mule accounts (defaulting to mule hub `C716083600` with 89 inbound links and ₹62.9M volume). |
| **Model Registry & Governance** | ![Models](reports/figures/screens/06_models.png) | 5 deep-dive tabs for champion benchmarks, split evasion robustness, demographic fairness audits, feature ablation, and feedback loops. |
| **Model Drift Monitoring** | ![Monitoring](reports/figures/screens/08_monitoring.png) | Continuous Population Stability Index (PSI) tracking across all channels with automated shift alerts. |
| **High-Throughput Batch Scoring** | ![Batch Scoring](reports/figures/screens/09_batch.png) | Bulk transaction array scoring engine with summary statistics, score histograms, and downloadable scored exports. |
| **Developer API Reference** | ![API Docs](reports/figures/screens/10_api_docs.png) | Measured latency benchmarks (p50: 7.92 ms, p95: 14.36 ms), endpoint schemas, and copyable cURL snippets. |'''

    new_s6 = '''| View | Screenshot | Description |
|---|---|---|
| **Animated Shield Landing Gate** | ![Landing Gate](reports/figures/screens/00_landing.png) | Aperture vault door opening animation revealing the live decision ledger fed by the real stream replay. |
| **Executive Overview** | ![Overview](reports/figures/screens/01_overview.png) | Audited fraud value intercepted (₹6,322.65M on PaySim test set), hourly volume distribution, risk band distribution, and top TreeSHAP fraud drivers. |
| **Live Pre-Authorization Check** | ![Live Check](reports/figures/screens/02_live_check.png) | Empirical inference latency (p50: 7.92 ms), custom semicircle SVG risk gauge, TreeSHAP reason codes, what-if sliders, and bilingual challenge previews (English + Hindi). |
| **Transaction Ledger & Detail Sheet** | ![Transactions Detail](reports/figures/screens/phase6_01_transactions_row_detail.png) | Slide-over transaction detail modal displaying full transaction telemetry, decision factors, saved view filters, and CSV export. |
| **Investigator Queue & Keyboard Triage** | ![Cases Triage](reports/figures/screens/phase6_02_cases_precision_and_hotkeys.png) | Rapid adjudication queue with single-key shortcuts (`J`/`K`/`Enter`/`C`/`F`) and statistical precision gate (>= 20 reviews). |
| **360-Degree Entity Graph** | ![Entity Graph](reports/figures/screens/phase6_03_entity_graph_interactive.png) | Interactive force-directed network diagram mapping 51 interconnected accounts and mule nodes for flagged hub `C716083600`. |
| **High-Throughput Batch Scoring** | ![Batch Scoring](reports/figures/screens/phase6_04_batch_drag_and_drop.png) | Drag-and-drop file ingestion zone for CSV and JSON with vectorized batch scoring and plain-language validation errors. |
| **Model Registry & Governance** | ![Models](reports/figures/screens/06_models.png) | Champion benchmark tables, 95% bootstrap confidence intervals, split evasion tests, and demographic fairness audits. |
| **Developer API & Live Test Runner** | ![API Docs](reports/figures/screens/phase6_05_apidocs_curl_and_livetest.png) | Interactive reference with in-browser live endpoint testing, measured round-trip latency, and copyable cURL snippets. |
| **Production Drift & PSI Monitoring** | ![Monitoring](reports/figures/screens/phase6_06_monitoring_drift.png) | Real-time Population Stability Index (PSI) drift tracking across channels and longitudinal concept drift benchmarks across BAF Months 3–7. |'''

    if old_s6 in text:
        text = text.replace(old_s6, new_s6)

    # 4. Project Directory Structure
    struct_block = """## 5.1 Project Directory Structure

```
FraudShield-main/
├── api/                   # Production FastAPI routing & Pydantic schemas
├── ml/                    # Core machine learning engine & pipeline
│   ├── channels/          # Channel-specific model implementations (Card, Wallet, Account, UPI)
│   ├── api.py             # FastAPI backend with REST endpoints & stream replay
│   ├── config.py          # Centralized paths, thresholds, and hyperparameters
│   ├── evaluate.py        # Natural prevalence evaluation & bootstrap CI routines
│   ├── features.py        # Pre-authorization feature engineering & leakage checks
│   ├── history_store.py   # In-memory rolling transaction history store
│   ├── pipeline.py        # Master pipeline re-runner (`python -m ml.pipeline --all`)
│   ├── retrain.py         # Retrain & promote champion governance engine
│   ├── router.py          # Unified multi-channel routing & decision layer
│   └── stream_replay.py   # Chronological stream replay engine for live views
├── frontend-web/          # Modern React 18 + TypeScript + Tailwind web application
│   ├── src/
│   │   ├── components/    # Reusable UI components & landing page sections
│   │   ├── pages/         # Route pages (Overview, Live Check, Transactions, Cases, etc.)
│   │   └── api/           # Typed API client generated from FastAPI schema
│   ├── vite.config.ts     # Vite bundler configuration & code splitting
│   └── tailwind.config.ts # "The Ledger" design system palette & typography
├── models/                # Serialized model artifacts, champion registry, & metadata
├── reports/               # Audited metrics.json, drift benchmarks, & evaluation figures
│   └── figures/screens/   # Real high-resolution screenshots of the React application
├── docs/                  # Academic documentation, viva defense prep, and slide outlines
│   ├── REPORT.md          # Comprehensive 10-page academic report
│   ├── VIVA_QA.md         # 30 viva defense questions with direct technical answers
│   └── SLIDES_OUTLINE.md  # 6-slide presentation outline
├── notebooks/             # Executable Jupyter notebooks (EDA, Modeling, Final synthesis)
├── tests/                 # Automated pytest suite (156 tests covering ML, API, E2E, & A11y)
├── archive/               # Preserved legacy artifacts (e.g. legacy Streamlit app)
├── run.py                 # Unified launcher for API server & web interface
├── Dockerfile             # Multi-stage Docker build for API & static web frontend
├── docker-compose.yml     # Container orchestration for reproducible deployment
└── requirements.txt       # Frozen Python dependencies
```

"""
    if '## 5.1 Project Directory Structure' not in text:
        text = text.replace('## 6. Visual Walkthrough', struct_block + '## 6. Visual Walkthrough')

    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(text)
    print('README.md successfully updated!')

if __name__ == '__main__':
    update()

