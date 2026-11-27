from playwright.sync_api import sync_playwright
from pathlib import Path
import time

BASE_DIR = Path(__file__).resolve().parent.parent
SCREENS_DIR = BASE_DIR / "reports" / "figures" / "screens"
SCREENS_DIR.mkdir(parents=True, exist_ok=True)

PAGES = [
    ("02_live_check.png", "02_live_check_dark.png", "http://127.0.0.1:8501/live_check"),
    ("03_transactions.png", "03_transactions_dark.png", "http://127.0.0.1:8501/transactions"),
    ("04_models.png", "04_models_dark.png", "http://127.0.0.1:8501/performance"),
    ("05_monitoring.png", "05_monitoring_dark.png", "http://127.0.0.1:8501/monitoring"),
    ("06_demo_controls.png", "06_demo_controls_dark.png", "http://127.0.0.1:8501/demo_controls"),
]

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=2)
    
    for light_fn, dark_fn, base_url in PAGES:
        for fn, url in [(light_fn, f"{base_url}?theme=light"), (dark_fn, f"{base_url}?theme=dark")]:
            print(f"Navigating to {url}...")
            page.goto(url, wait_until="networkidle", timeout=30000)
            
            # Wait for main page content to finish rendering
            try:
                page.wait_for_selector("h1", timeout=15000)
            except Exception as e:
                print(f"Warning on {fn} waiting for h1: {e}")
            
            # Wait for status widget (RUNNING...) to detach
            try:
                page.wait_for_selector('div[data-testid="stStatusWidget"]', state="detached", timeout=15000)
            except Exception:
                pass
            
            page.wait_for_timeout(1500)
            sh = page.evaluate('() => document.querySelector("section[data-testid=\'stMain\']") ? document.querySelector("section[data-testid=\'stMain\']").scrollHeight : 1200')
            page.set_viewport_size({"width": 1440, "height": max(sh + 60, 1100)})
            page.wait_for_timeout(1000)
            
            out_path = SCREENS_DIR / fn
            page.screenshot(path=str(out_path), full_page=True)
            print(f"Saved {fn} (height: {sh})")

    page.close()
    browser.close()
    print("Done capturing all Phase 5 screens.")
