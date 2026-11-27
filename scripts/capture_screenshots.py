"""
FraudShield Automated Visual Verification & Screenshot Suite.
Captures full-resolution 1440x900 @ 2x device scale screenshots for Light and Dark modes.
Includes all 13 core pages and interaction verification.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ml.config import SCREENS_DIR


def wait_for_server(url: str = "http://127.0.0.1:8501/_stcore/health", timeout: float = 30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=1.0) as resp:
                if resp.status == 200:
                    print("Streamlit server is ready!")
                    return True
        except (URLError, TimeoutError, OSError):
            time.sleep(0.5)
    raise TimeoutError("Streamlit server failed to start within timeout.")


def capture_all_pages():
    SCREENS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Start Streamlit server headless using venv python
    print("Launching Streamlit in headless mode on port 8501...")
    env = os.environ.copy()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(BASE_DIR / "frontend" / "app.py"),
            "--server.port=8501",
            "--server.headless=true",
            "--browser.gatherUsageStats=false",
            "--server.enableCORS=false",
            "--server.enableXsrfProtection=false",
        ],
        cwd=str(BASE_DIR / "frontend"),
        env=env,
    )

    try:
        wait_for_server()

        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(
                viewport={"width": 1440, "height": 900},
                device_scale_factor=2,
            )

            def check_no_errors():
                exceptions = page.locator(".stException")
                if exceptions.count() > 0:
                    err_text = exceptions.first.inner_text()
                    raise AssertionError(f"Streamlit page encountered exception: {err_text}")

            PAGES_CONFIG = [
                ("00_home", ""),
                ("01_overview", "overview"),
                ("02_transactions", "transactions"),
                ("03_model_performance", "performance"),
                ("06_demo_controls", "demo_controls"),
                ("07_how_it_works", "how_it_works"),
                ("08_monitoring", "monitoring"),
                ("09_case_management", "cases"),
                ("10_batch_scoring", "batch"),
                ("11_entity_view", "entity"),
                ("12_shadow_mode", "shadow_mode"),
                ("13_api_docs", "api"),
            ]

            # =========================================================
            # LIGHT MODE SCREENSHOTS
            # =========================================================
            print("\n--- Capturing LIGHT MODE Screenshots (1440x900 @ 2x) ---")
            for slug, route in PAGES_CONFIG:
                url = f"http://127.0.0.1:8501/{route}?theme=light" if route else "http://127.0.0.1:8501/?theme=light"
                print(f"Capturing {slug}.png from {url}...")
                page.goto(url, wait_until="networkidle")
                page.wait_for_timeout(2500)
                check_no_errors()
                out_path = SCREENS_DIR / f"{slug}.png"
                page.screenshot(path=str(out_path), full_page=True)
                print(f"  ✓ Saved: {out_path.name}")

            # Live Fraud Check Blank & Verdict (Light)
            print("Capturing 04_live_fraud_check_blank.png...")
            page.goto("http://127.0.0.1:8501/live_check?theme=light", wait_until="networkidle")
            page.wait_for_timeout(2500)
            check_no_errors()
            p4 = SCREENS_DIR / "04_live_fraud_check_blank.png"
            page.screenshot(path=str(p4), full_page=True)
            print(f"  ✓ Saved: {p4.name}")

            print("Testing Live Scoring Interaction (Takeover Drain)...")
            try:
                page.evaluate("() => { const btns = Array.from(document.querySelectorAll('button')); const b = btns.find(x => x.textContent.includes('Takeover Drain')); if (b) b.click(); }")
                page.wait_for_timeout(2000)
                page.evaluate("() => { const btns = Array.from(document.querySelectorAll('button')); const b = btns.find(x => x.textContent.includes('Assessment')); if (b) b.click(); }")
                page.wait_for_timeout(2500)
                check_no_errors()
            except (RuntimeError, TimeoutError, OSError) as e:
                print(f"  Note on preset interaction: {e}")

            p5 = SCREENS_DIR / "05_live_fraud_check_verdict.png"
            page.screenshot(path=str(p5), full_page=True)
            print(f"  ✓ Saved: {p5.name}")

            # =========================================================
            # DARK MODE SCREENSHOTS
            # =========================================================
            print("\n--- Capturing DARK MODE Screenshots (1440x900 @ 2x) ---")
            for slug, route in PAGES_CONFIG:
                url = f"http://127.0.0.1:8501/{route}?theme=dark" if route else "http://127.0.0.1:8501/?theme=dark"
                print(f"Capturing {slug}_dark.png from {url}...")
                page.goto(url, wait_until="networkidle")
                page.wait_for_timeout(2500)
                check_no_errors()
                out_path = SCREENS_DIR / f"{slug}_dark.png"
                page.screenshot(path=str(out_path), full_page=True)
                print(f"  ✓ Saved: {out_path.name}")

            # Live Fraud Check Verdict (Dark)
            print("Capturing 05_live_fraud_check_verdict_dark.png...")
            page.goto("http://127.0.0.1:8501/live_check?theme=dark", wait_until="networkidle")
            page.wait_for_timeout(2000)
            try:
                page.evaluate("() => { const btns = Array.from(document.querySelectorAll('button')); const b = btns.find(x => x.textContent.includes('Takeover Drain')); if (b) b.click(); }")
                page.wait_for_timeout(2000)
                page.evaluate("() => { const btns = Array.from(document.querySelectorAll('button')); const b = btns.find(x => x.textContent.includes('Assessment')); if (b) b.click(); }")
                page.wait_for_timeout(2500)
                check_no_errors()
            except (RuntimeError, TimeoutError, OSError) as e:
                print(f"  Note on preset interaction (dark): {e}")
            p5_dark = SCREENS_DIR / "05_live_fraud_check_verdict_dark.png"
            page.screenshot(path=str(p5_dark), full_page=True)
            print(f"  ✓ Saved: {p5_dark.name}")

            browser.close()

    finally:
        print("Stopping Streamlit server...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("Streamlit server stopped.")


if __name__ == "__main__":
    capture_all_pages()
