import time
import shutil
from pathlib import Path
from playwright.sync_api import sync_playwright

out_dir = Path("reports/figures/screens")
out_dir.mkdir(parents=True, exist_ok=True)
brain_screens_dir = Path(r"C:\Users\AYUSH\.gemini\antigravity\brain\6e421928-09c0-4b70-9854-1510db5b76c8")
brain_screens_dir.mkdir(parents=True, exist_ok=True)
BASE_URL = "http://localhost:8000"

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        # 1. Desktop Dark 1440x900
        ctx_dark = browser.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=1.5)
        ctx_dark.add_init_script("localStorage.setItem('fraudshield-theme', 'dark'); document.documentElement.classList.add('dark');")
        page_d = ctx_dark.new_page()

        # Command palette open
        page_d.goto(f"{BASE_URL}/overview", wait_until="networkidle")
        page_d.wait_for_timeout(800)
        page_d.keyboard.press("Control+k")
        page_d.wait_for_timeout(500)
        cmd_img = out_dir / "phase6_command_palette_dark.png"
        page_d.screenshot(path=str(cmd_img))
        shutil.copy2(cmd_img, brain_screens_dir / "phase6_command_palette_dark.png")

        # Cases triage drawer
        page_d.keyboard.press("Escape")
        page_d.wait_for_timeout(300)
        page_d.goto(f"{BASE_URL}/cases", wait_until="networkidle")
        page_d.wait_for_timeout(800)
        first_row = page_d.locator("tbody tr").first
        if first_row.is_visible():
            first_row.click()
            page_d.wait_for_timeout(600)
        case_img = out_dir / "phase6_cases_drawer_dark.png"
        page_d.screenshot(path=str(case_img))
        shutil.copy2(case_img, brain_screens_dir / "phase6_cases_drawer_dark.png")

        # 2. Tablet Portrait (768x1024)
        ctx_tab = browser.new_context(viewport={"width": 768, "height": 1024}, device_scale_factor=1.5)
        ctx_tab.add_init_script("localStorage.setItem('fraudshield-theme', 'dark'); document.documentElement.classList.add('dark');")
        page_t = ctx_tab.new_page()
        page_t.goto(f"{BASE_URL}/overview", wait_until="networkidle")
        page_t.wait_for_timeout(800)
        tab_img = out_dir / "phase6_overview_tablet_dark.png"
        page_t.screenshot(path=str(tab_img))
        shutil.copy2(tab_img, brain_screens_dir / "phase6_overview_tablet_dark.png")

        # 3. Mobile Light (390x844)
        ctx_mob = browser.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=1.5)
        ctx_mob.add_init_script("localStorage.setItem('fraudshield-theme', 'light'); document.documentElement.classList.remove('dark');")
        page_m = ctx_mob.new_page()
        page_m.goto(f"{BASE_URL}/live-check", wait_until="networkidle")
        page_m.wait_for_timeout(800)
        mob_img = out_dir / "phase6_live_check_mobile_light.png"
        page_m.screenshot(path=str(mob_img))
        shutil.copy2(mob_img, brain_screens_dir / "phase6_live_check_mobile_light.png")

        # 4. Mobile Dark Transactions (390x844)
        ctx_mob_dark = browser.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=1.5)
        ctx_mob_dark.add_init_script("localStorage.setItem('fraudshield-theme', 'dark'); document.documentElement.classList.add('dark');")
        page_md = ctx_mob_dark.new_page()
        page_md.goto(f"{BASE_URL}/transactions", wait_until="networkidle")
        page_md.wait_for_timeout(800)
        mob_tx_img = out_dir / "phase6_transactions_mobile_dark.png"
        page_md.screenshot(path=str(mob_tx_img))
        shutil.copy2(mob_tx_img, brain_screens_dir / "phase6_transactions_mobile_dark.png")

        browser.close()
        print("Phase 6 screenshots captured and copied successfully!")

if __name__ == "__main__":
    run()
