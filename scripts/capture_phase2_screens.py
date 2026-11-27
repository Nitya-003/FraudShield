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

        # 1. Desktop Dark (1440x900)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(TARGET_URL)
        page.evaluate("() => { localStorage.setItem('fraudshield-theme', 'dark'); document.documentElement.classList.add('dark'); }")
        page.reload()
        page.wait_for_timeout(1800) # Wait for animation and canvas to settle
        desktop_dark_path = SCREENS_DIR / "phase2_hero_desktop_dark.png"
        page.screenshot(path=str(desktop_dark_path), full_page=False)
        print(f"Captured: {desktop_dark_path}")
        page.close()

        # 2. Desktop Light (1440x900)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(TARGET_URL)
        page.evaluate("() => { localStorage.setItem('fraudshield-theme', 'light'); document.documentElement.classList.remove('dark'); }")
        page.reload()
        page.wait_for_timeout(1800)
        desktop_light_path = SCREENS_DIR / "phase2_hero_desktop_light.png"
        page.screenshot(path=str(desktop_light_path), full_page=False)
        print(f"Captured: {desktop_light_path}")
        page.close()

        # 3. Mobile Dark (390x844)
        page = browser.new_page(viewport={"width": 390, "height": 844})
        page.goto(TARGET_URL)
        page.evaluate("() => { localStorage.setItem('fraudshield-theme', 'dark'); document.documentElement.classList.add('dark'); }")
        page.reload()
        page.wait_for_timeout(1800)
        mobile_dark_path = SCREENS_DIR / "phase2_hero_mobile_dark.png"
        page.screenshot(path=str(mobile_dark_path), full_page=False)
        print(f"Captured: {mobile_dark_path}")
        page.close()

        # 4. Mobile Light (390x844)
        page = browser.new_page(viewport={"width": 390, "height": 844})
        page.goto(TARGET_URL)
        page.evaluate("() => { localStorage.setItem('fraudshield-theme', 'light'); document.documentElement.classList.remove('dark'); }")
        page.reload()
        page.wait_for_timeout(1800)
        mobile_light_path = SCREENS_DIR / "phase2_hero_mobile_light.png"
        page.screenshot(path=str(mobile_light_path), full_page=False)
        print(f"Captured: {mobile_light_path}")
        page.close()

        browser.close()

if __name__ == "__main__":
    capture()
