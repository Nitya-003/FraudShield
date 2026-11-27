import time
import shutil
from pathlib import Path
from playwright.sync_api import sync_playwright

out_dir = Path("reports/figures/screens")
out_dir.mkdir(parents=True, exist_ok=True)
brain_dir = Path(r"C:\Users\AYUSH\.gemini\antigravity\brain\6e421928-09c0-4b70-9854-1510db5b76c8")

def save_both(page, filename: str):
    local_path = out_dir / filename
    page.screenshot(path=str(local_path))
    if brain_dir.exists():
        shutil.copy2(local_path, brain_dir / filename)
    print(f"Saved: {filename}")

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        # 1. Desktop Dark (1440x900)
        ctx_dark = browser.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=1.5)
        ctx_dark.add_init_script("localStorage.setItem('fraudshield-theme', 'dark'); document.documentElement.classList.add('dark');")
        page_d = ctx_dark.new_page()

        # Attack Lab with Simulation Run (showing step player, delta badges & animated mule hop diagram)
        print("Capturing Attack Lab simulation...")
        page_d.goto("http://localhost:8000/attack-lab", wait_until="networkidle")
        page_d.wait_for_timeout(1000)
        sim_btn = page_d.locator("button:has-text('Execute Simulation')").first
        if sim_btn.is_visible():
            sim_btn.click()
            page_d.wait_for_timeout(1500)
        save_both(page_d, "phase5_attack_lab_simulation_dark.png")

        # Entity View with Graph & Side Drawer open
        print("Capturing Entity View topology...")
        page_d.goto("http://localhost:8000/entity?account=C716083600", wait_until="networkidle")
        page_d.wait_for_timeout(1200)
        drawer_btn = page_d.locator("button:has-text('Open Side Drawer')").first
        if drawer_btn.is_visible():
            drawer_btn.click()
            page_d.wait_for_timeout(600)
        save_both(page_d, "phase5_entity_drawer_dark.png")

        # Overview Page Dark
        print("Capturing Overview...")
        page_d.goto("http://localhost:8000/overview", wait_until="networkidle")
        page_d.wait_for_timeout(1000)
        save_both(page_d, "phase5_overview_desktop_dark.png")

        # Live Check Page Dark
        print("Capturing Live Check...")
        page_d.goto("http://localhost:8000/live-check", wait_until="networkidle")
        page_d.wait_for_timeout(1000)
        btn = page_d.locator("button:has-text('UPI Collect Scam')").first
        if btn.is_visible():
            btn.click()
            page_d.wait_for_timeout(400)
        score_btn = page_d.locator("button:has-text('Score Transaction')").first
        if score_btn.is_visible():
            score_btn.click()
            page_d.wait_for_timeout(900)
        save_both(page_d, "phase5_live_check_desktop_dark.png")

        # 2. Desktop Light (1440x900)
        ctx_light = browser.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=1.5)
        ctx_light.add_init_script("localStorage.setItem('fraudshield-theme', 'light'); document.documentElement.classList.remove('dark');")
        page_l = ctx_light.new_page()

        # Attack Lab Light
        page_l.goto("http://localhost:8000/attack-lab", wait_until="networkidle")
        page_l.wait_for_timeout(1000)
        sim_btn = page_l.locator("button:has-text('Execute Simulation')").first
        if sim_btn.is_visible():
            sim_btn.click()
            page_l.wait_for_timeout(1500)
        save_both(page_l, "phase5_attack_lab_simulation_light.png")

        # Entity View Light
        page_l.goto("http://localhost:8000/entity?account=C716083600", wait_until="networkidle")
        page_l.wait_for_timeout(1200)
        drawer_btn = page_l.locator("button:has-text('Open Side Drawer')").first
        if drawer_btn.is_visible():
            drawer_btn.click()
            page_l.wait_for_timeout(600)
        save_both(page_l, "phase5_entity_drawer_light.png")

        # 3. Mobile Dark & Light (390x844)
        for theme in ["dark", "light"]:
            ctx_mob = browser.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=1.5)
            if theme == "dark":
                ctx_mob.add_init_script("localStorage.setItem('fraudshield-theme', 'dark'); document.documentElement.classList.add('dark');")
            else:
                ctx_mob.add_init_script("localStorage.setItem('fraudshield-theme', 'light'); document.documentElement.classList.remove('dark');")
            page_m = ctx_mob.new_page()

            page_m.goto("http://localhost:8000/attack-lab", wait_until="networkidle")
            page_m.wait_for_timeout(1000)
            sim_btn = page_m.locator("button:has-text('Execute Simulation')").first
            if sim_btn.is_visible():
                sim_btn.click()
                page_m.wait_for_timeout(1500)
            save_both(page_m, f"phase5_attack_lab_mobile_{theme}.png")

        browser.close()
        print("All Phase 5 screenshots captured and saved successfully!")

if __name__ == "__main__":
    run()