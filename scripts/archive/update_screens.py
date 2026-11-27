"""
Capture updated screenshots from the active running Streamlit server.
"""

from __future__ import annotations

import time
from pathlib import Path
from playwright.sync_api import sync_playwright

BASE_DIR = Path(__file__).resolve().parent.parent
SCREENS_DIR = BASE_DIR / "reports" / "figures" / "screens"
SCREENS_DIR.mkdir(parents=True, exist_ok=True)

def capture():
    pages = [
        ("00_home.png", "http://127.0.0.1:8501/?theme=light"),
        ("00_home_dark.png", "http://127.0.0.1:8501/?theme=dark"),
        ("01_overview.png", "http://127.0.0.1:8501/overview?theme=light"),
        ("01_overview_dark.png", "http://127.0.0.1:8501/overview?theme=dark"),
        ("02_transactions.png", "http://127.0.0.1:8501/transactions?theme=light"),
        ("02_transactions_dark.png", "http://127.0.0.1:8501/transactions?theme=dark"),
        ("03_model_performance.png", "http://127.0.0.1:8501/performance?theme=light"),
        ("03_model_performance_dark.png", "http://127.0.0.1:8501/performance?theme=dark"),
        ("04_live_fraud_check_blank.png", "http://127.0.0.1:8501/live_check?theme=light"),
        ("05_live_fraud_check_verdict.png", "http://127.0.0.1:8501/live_check?theme=light"),
        ("05_live_fraud_check_verdict_dark.png", "http://127.0.0.1:8501/live_check?theme=dark"),
        ("06_demo_controls.png", "http://127.0.0.1:8501/demo_controls?theme=light"),
        ("06_demo_controls_dark.png", "http://127.0.0.1:8501/demo_controls?theme=dark"),
        ("07_how_it_works.png", "http://127.0.0.1:8501/how_it_works?theme=light"),
        ("07_how_it_works_dark.png", "http://127.0.0.1:8501/how_it_works?theme=dark"),
        ("08_monitoring.png", "http://127.0.0.1:8501/monitoring?theme=light"),
        ("08_monitoring_dark.png", "http://127.0.0.1:8501/monitoring?theme=dark"),
    ]

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=2)

        for filename, url in pages:
            print(f"Capturing {filename}...")
            page.goto(url, wait_until="networkidle")
            page.wait_for_timeout(2500)

            if "05_live_fraud_check_verdict" in filename:
                preset_btn = page.locator("button:has-text('Account Takeover Drain')")
                if preset_btn.count() > 0:
                    preset_btn.first.click()
                    page.wait_for_timeout(1500)
                    submit_btn = page.locator("button:has-text('Evaluate Live Risk Score'), button:has-text('Run Real-Time Risk Assessment')")
                    if submit_btn.count() > 0:
                        submit_btn.first.click()
                        page.wait_for_timeout(2000)

            out_path = SCREENS_DIR / filename
            page.screenshot(path=str(out_path), full_page=True)
            print(f"  ✓ Saved {filename} ({out_path.stat().st_size:,} bytes)")

        browser.close()

if __name__ == "__main__":
    capture()

