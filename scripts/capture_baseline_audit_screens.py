import os
import shutil
from pathlib import Path
from playwright.sync_api import sync_playwright

out_dir = Path("reports/figures/baseline_screens")
out_dir.mkdir(parents=True, exist_ok=True)
brain_screens_dir = Path(r"C:\Users\AYUSH\.gemini\antigravity\brain\6e421928-09c0-4b70-9854-1510db5b76c8\baseline_screens")
brain_screens_dir.mkdir(parents=True, exist_ok=True)

BASE_URL = "http://localhost:8000"

PAGES = [
    ("landing", "/"),
    ("overview", "/overview"),
    ("live_check", "/live-check"),
    ("transactions", "/transactions"),
    ("cases", "/cases"),
    ("models", "/models"),
    ("attack_lab", "/attack-lab"),
    ("entity", "/entity"),
    ("batch", "/batch-scoring"),
    ("monitoring", "/monitoring"),
    ("api_docs", "/api-docs"),
]

VIEWPORTS = [
    ("desktop", 1440, 900),
    ("mobile", 390, 844),
]

THEMES = ["dark", "light"]

def run():
    print("=== CAPTURING BASELINE SCREENS BEFORE UI-POLISH-2 ===")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        for vp_name, width, height in VIEWPORTS:
            for theme in THEMES:
                ctx = browser.new_context(
                    viewport={"width": width, "height": height},
                    device_scale_factor=1.5
                )
                if theme == "dark":
                    ctx.add_init_script("localStorage.setItem('fraudshield-theme', 'dark'); document.documentElement.classList.add('dark');")
                else:
                    ctx.add_init_script("localStorage.setItem('fraudshield-theme', 'light'); document.documentElement.classList.remove('dark');")

                page = ctx.new_page()

                for name, route in PAGES:
                    target_url = f"{BASE_URL}{route}"
                    print(f"Capturing: {vp_name} {theme} -> {name} ({target_url})")
                    try:
                        page.goto(target_url, wait_until="networkidle")
                        page.wait_for_timeout(600)
                    except Exception as e:
                        print(f"  Warning loading {target_url}: {e}")

                    fname = f"baseline_{name}_{vp_name}_{theme}.png"
                    save_path = out_dir / fname
                    page.screenshot(path=str(save_path), full_page=(vp_name == "mobile"))
                    shutil.copy2(save_path, brain_screens_dir / fname)

                page.close()
                ctx.close()

        browser.close()
    print("=== BASELINE SCREENS CAPTURED SUCCESSFULLY ===")

if __name__ == "__main__":
    run()
