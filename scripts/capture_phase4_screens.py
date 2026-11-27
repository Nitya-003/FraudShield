import os
import sys
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
SCREENS_DIR = ROOT / "reports" / "figures" / "screens"
SCREENS_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "http://localhost:8000"

def capture():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        # 1. Overview Desktop Dark (1440x900)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(f"{BASE_URL}/overview")
        page.evaluate("() => { localStorage.setItem('fraudshield-theme', 'dark'); document.documentElement.classList.add('dark'); }")
        page.reload()
        page.wait_for_timeout(1500)
        p1 = SCREENS_DIR / "phase4_overview_desktop_dark.png"
        page.screenshot(path=str(p1))
        print(f"Captured: {p1}")
        page.close()

        # 2. Overview Desktop Light (1440x900)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(f"{BASE_URL}/overview")
        page.evaluate("() => { localStorage.setItem('fraudshield-theme', 'light'); document.documentElement.classList.remove('dark'); }")
        page.reload()
        page.wait_for_timeout(1500)
        p2 = SCREENS_DIR / "phase4_overview_desktop_light.png"
        page.screenshot(path=str(p2))
        print(f"Captured: {p2}")
        page.close()

        # 3. Overview Mobile Dark (390x844)
        page = browser.new_page(viewport={"width": 390, "height": 844})
        page.goto(f"{BASE_URL}/overview")
        page.evaluate("() => { localStorage.setItem('fraudshield-theme', 'dark'); document.documentElement.classList.add('dark'); }")
        page.reload()
        page.wait_for_timeout(1500)
        p3 = SCREENS_DIR / "phase4_overview_mobile_dark.png"
        page.screenshot(path=str(p3))
        print(f"Captured: {p3}")
        page.close()

        # 4. Live Check Desktop Dark (1440x900)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(f"{BASE_URL}/live-check")
        page.evaluate("() => { localStorage.setItem('fraudshield-theme', 'dark'); document.documentElement.classList.add('dark'); }")
        page.reload()
        page.wait_for_timeout(1500)
        p4 = SCREENS_DIR / "phase4_live_check_desktop_dark.png"
        page.screenshot(path=str(p4))
        print(f"Captured: {p4}")
        page.close()

        # 5. Live Check Desktop Light (1440x900)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(f"{BASE_URL}/live-check")
        page.evaluate("() => { localStorage.setItem('fraudshield-theme', 'light'); document.documentElement.classList.remove('dark'); }")
        page.reload()
        page.wait_for_timeout(1500)
        p5 = SCREENS_DIR / "phase4_live_check_desktop_light.png"
        page.screenshot(path=str(p5))
        print(f"Captured: {p5}")
        page.close()

        # 6. Live Check Mobile Dark (390x844)
        page = browser.new_page(viewport={"width": 390, "height": 844})
        page.goto(f"{BASE_URL}/live-check")
        page.evaluate("() => { localStorage.setItem('fraudshield-theme', 'dark'); document.documentElement.classList.add('dark'); }")
        page.reload()
        page.wait_for_timeout(1500)
        p6 = SCREENS_DIR / "phase4_live_check_mobile_dark.png"
        page.screenshot(path=str(p6))
        print(f"Captured: {p6}")
        page.close()

        browser.close()

if __name__ == "__main__":
    capture()
