from playwright.sync_api import sync_playwright
from pathlib import Path

out_dir = Path("reports/figures/screens")
out_dir.mkdir(parents=True, exist_ok=True)

with sync_playwright() as pw:
    browser = pw.chromium.launch(headless=True)
    
    # Desktop 1440x900
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    
    # Overview Light
    page.goto("http://localhost:8000/overview", wait_until="networkidle")
    page.wait_for_timeout(1000)
    page.screenshot(path=str(out_dir / "phase0_overview_light.png"))
    
    # Overview Dark
    page.evaluate("document.documentElement.classList.add('dark')")
    page.wait_for_timeout(500)
    page.screenshot(path=str(out_dir / "phase0_overview_dark.png"))
    
    # Landing Page Dark
    page.goto("http://localhost:8000/", wait_until="networkidle")
    page.evaluate("document.documentElement.classList.add('dark')")
    page.wait_for_timeout(1000)
    page.screenshot(path=str(out_dir / "phase0_landing_dark.png"))

    # Mobile 390x844 Overview
    mobile_page = browser.new_page(viewport={"width": 390, "height": 844})
    mobile_page.goto("http://localhost:8000/overview", wait_until="networkidle")
    mobile_page.evaluate("document.documentElement.classList.add('dark')")
    mobile_page.wait_for_timeout(1000)
    mobile_page.screenshot(path=str(out_dir / "phase0_mobile_overview_dark.png"))
    
    browser.close()
    print("Phase 0 visual captures completed successfully.")

