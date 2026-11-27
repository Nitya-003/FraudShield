"""
Capture high-resolution screenshots of the React application in Light and Dark mode,
Desktop (1440x900) and Mobile (390x844) for all core pages.
"""

from __future__ import annotations

import time
from pathlib import Path
from playwright.sync_api import sync_playwright

BASE_DIR = Path(__file__).resolve().parent.parent
SCREENS_DIR = BASE_DIR / "reports" / "figures" / "screens"
BASE_URL = "http://localhost:8000"

PAGES = [
    ("00_landing", "landing"),
    ("01_overview", "overview"),
    ("02_live_check", "live_check"),
    ("03_transactions", "transactions"),
    ("04_cases", "cases"),
    ("05_attack_lab", "attack_lab"),
    ("06_models", "models"),
    ("07_entity", "entity"),
    ("08_monitoring", "monitoring"),
    ("09_batch", "batch"),
    ("10_api_docs", "api_docs"),
    ("11_how_it_works", "how_it_works"),
]


def capture_all():
    SCREENS_DIR.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        # 1. Desktop Light & Dark (1440x900 @ 2x)
        print("--- Capturing Desktop Views (1440x900) ---")
        context_desktop = browser.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=2)
        page = context_desktop.new_page()

        # Capture Desktop Light
        for slug, tab in PAGES:
            url = f"{BASE_URL}/" if tab == "landing" else f"{BASE_URL}/?tab={tab}"
            page.goto(url, wait_until="networkidle")
            page.wait_for_timeout(1000)
            out_file = SCREENS_DIR / f"{slug}.png"
            page.screenshot(path=str(out_file), full_page=True)
            print(f"Captured: {out_file.name}")

        # Toggle Dark Mode
        print("Toggling dark mode...")
        theme_btn = page.locator("button[aria-label*='Toggle color theme']")
        if theme_btn.count() > 0:
            theme_btn.click()
            page.wait_for_timeout(500)
        else:
            page.evaluate("() => document.documentElement.classList.add('dark')")

        # Capture Desktop Dark
        for slug, tab in PAGES:
            url = f"{BASE_URL}/" if tab == "landing" else f"{BASE_URL}/?tab={tab}"
            page.goto(url, wait_until="networkidle")
            # Ensure dark mode class is set
            page.evaluate("() => document.documentElement.classList.add('dark')")
            page.wait_for_timeout(1000)
            out_file = SCREENS_DIR / f"{slug}_dark.png"
            page.screenshot(path=str(out_file), full_page=True)
            print(f"Captured: {out_file.name}")

        context_desktop.close()

        # 2. Mobile Light & Dark (390x844 @ 2x)
        print("\n--- Capturing Mobile Views (390x844) ---")
        context_mobile = browser.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=2)
        page_m = context_mobile.new_page()

        # Capture Mobile Light
        for slug, tab in PAGES:
            url = f"{BASE_URL}/" if tab == "landing" else f"{BASE_URL}/?tab={tab}"
            page_m.goto(url, wait_until="networkidle")
            page_m.wait_for_timeout(1000)
            out_file = SCREENS_DIR / f"mobile_{slug}_light.png"
            page_m.screenshot(path=str(out_file), full_page=True)
            print(f"Captured: {out_file.name}")

        # Capture Mobile Dark
        for slug, tab in PAGES:
            url = f"{BASE_URL}/" if tab == "landing" else f"{BASE_URL}/?tab={tab}"
            page_m.goto(url, wait_until="networkidle")
            page_m.evaluate("() => document.documentElement.classList.add('dark')")
            page_m.wait_for_timeout(1000)
            out_file = SCREENS_DIR / f"mobile_{slug}_dark.png"
            page_m.screenshot(path=str(out_file), full_page=True)
            print(f"Captured: {out_file.name}")

        context_mobile.close()
        browser.close()
    print("\nAll screenshots captured successfully into:", SCREENS_DIR)


if __name__ == "__main__":
    capture_all()

