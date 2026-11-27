import os
import sys
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
SCREENS_DIR = ROOT / "reports" / "figures" / "screens"
SCREENS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_URL = "http://localhost:8000/"

def capture():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        # 1. Desktop Dark - Sequence Step 0 (Features)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(TARGET_URL)
        page.evaluate("() => { localStorage.setItem('fraudshield-theme', 'dark'); document.documentElement.classList.add('dark'); }")
        page.reload()
        page.wait_for_timeout(1500)
        
        # Scroll to pinned section
        page.evaluate("window.scrollTo(0, 600)")
        page.wait_for_timeout(1000)
        p1 = SCREENS_DIR / "phase3_scroll_features_dark.png"
        page.screenshot(path=str(p1))
        print(f"Captured: {p1}")

        # Scroll further to Step 1 (Score)
        page.evaluate("window.scrollTo(0, 1100)")
        page.wait_for_timeout(1000)
        p2 = SCREENS_DIR / "phase3_scroll_score_dark.png"
        page.screenshot(path=str(p2))
        print(f"Captured: {p2}")

        # Scroll further to Step 2 (Action)
        page.evaluate("window.scrollTo(0, 1600)")
        page.wait_for_timeout(1000)
        p3 = SCREENS_DIR / "phase3_scroll_action_dark.png"
        page.screenshot(path=str(p3))
        print(f"Captured: {p3}")

        # Scroll to 4-channel results and footer (Screen 3)
        page.locator('text="Validated performance on held-out test splits."').scroll_into_view_if_needed()
        page.wait_for_timeout(1000)
        p4 = SCREENS_DIR / "phase3_results_footer_dark.png"
        page.screenshot(path=str(p4))
        print(f"Captured: {p4}")
        page.close()

        # 2. Desktop Light - 4-channel results and footer (Screen 3 Light)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(TARGET_URL)
        page.evaluate("() => { localStorage.setItem('fraudshield-theme', 'light'); document.documentElement.classList.remove('dark'); }")
        page.reload()
        page.wait_for_timeout(1500)
        page.locator('text="Validated performance on held-out test splits."').scroll_into_view_if_needed()
        page.wait_for_timeout(1000)
        p5 = SCREENS_DIR / "phase3_results_footer_light.png"
        page.screenshot(path=str(p5))
        print(f"Captured: {p5}")
        page.close()

        # 3. Mobile Dark (390x844) - Screen 3 Results
        page = browser.new_page(viewport={"width": 390, "height": 844})
        page.goto(TARGET_URL)
        page.evaluate("() => { localStorage.setItem('fraudshield-theme', 'dark'); document.documentElement.classList.add('dark'); }")
        page.reload()
        page.wait_for_timeout(1500)
        page.locator('text="Validated performance on held-out test splits."').scroll_into_view_if_needed()
        page.wait_for_timeout(1000)
        p6 = SCREENS_DIR / "phase3_mobile_results_dark.png"
        page.screenshot(path=str(p6))
        print(f"Captured: {p6}")
        page.close()

        browser.close()

if __name__ == "__main__":
    capture()
