import shutil
from pathlib import Path
from playwright.sync_api import sync_playwright

out_dir = Path("reports/figures/screens")
out_dir.mkdir(parents=True, exist_ok=True)
brain_screens_dir = Path(r"C:\Users\AYUSH\.gemini\antigravity\brain\6e421928-09c0-4b70-9854-1510db5b76c8")
brain_screens_dir.mkdir(parents=True, exist_ok=True)

BASE_URL = "http://localhost:8000"

def run():
    print("=== CAPTURING PHASE 1 NAVBAR SCREENSHOTS ===")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        # 1. Desktop Dark 1440x900 - Overview with new navbar
        ctx_dark = browser.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=1.5)
        ctx_dark.add_init_script("localStorage.setItem('fraudshield-theme', 'dark'); document.documentElement.classList.add('dark');")
        page_d = ctx_dark.new_page()
        page_d.goto(f"{BASE_URL}/overview", wait_until="networkidle")
        page_d.wait_for_timeout(800)

        nav_dark_img = out_dir / "phase1_navbar_desktop_dark.png"
        page_d.screenshot(path=str(nav_dark_img))
        shutil.copy2(nav_dark_img, brain_screens_dir / "phase1_navbar_desktop_dark.png")

        # More dropdown open
        more_btn = page_d.locator("button:has-text('More')").first
        if more_btn.is_visible():
            more_btn.click()
            page_d.wait_for_timeout(400)
            more_img = out_dir / "phase1_navbar_more_dropdown_dark.png"
            page_d.screenshot(path=str(more_img))
            shutil.copy2(more_img, brain_screens_dir / "phase1_navbar_more_dropdown_dark.png")
            page_d.keyboard.press("Escape")
            page_d.wait_for_timeout(200)

        # Channel dropdown open
        chan_btn = page_d.locator("button[aria-label*='Current channel']").first
        if chan_btn.is_visible():
            chan_btn.click()
            page_d.wait_for_timeout(400)
            chan_img = out_dir / "phase1_navbar_channel_dropdown_dark.png"
            page_d.screenshot(path=str(chan_img))
            shutil.copy2(chan_img, brain_screens_dir / "phase1_navbar_channel_dropdown_dark.png")
            page_d.keyboard.press("Escape")
            page_d.wait_for_timeout(200)

        # 2. Desktop Light 1440x900
        ctx_light = browser.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=1.5)
        ctx_light.add_init_script("localStorage.setItem('fraudshield-theme', 'light'); document.documentElement.classList.remove('dark');")
        page_l = ctx_light.new_page()
        page_l.goto(f"{BASE_URL}/overview", wait_until="networkidle")
        page_l.wait_for_timeout(800)

        nav_light_img = out_dir / "phase1_navbar_desktop_light.png"
        page_l.screenshot(path=str(nav_light_img))
        shutil.copy2(nav_light_img, brain_screens_dir / "phase1_navbar_desktop_light.png")

        # 3. Mobile Sheet Dark 390x844
        ctx_mob = browser.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=1.5)
        ctx_mob.add_init_script("localStorage.setItem('fraudshield-theme', 'dark'); document.documentElement.classList.add('dark');")
        page_m = ctx_mob.new_page()
        page_m.goto(f"{BASE_URL}/overview", wait_until="networkidle")
        page_m.wait_for_timeout(800)

        # Open mobile sheet
        menu_btn = page_m.locator("button[aria-label='Open navigation menu']").first
        if menu_btn.is_visible():
            menu_btn.click()
            page_m.wait_for_timeout(500)
            mob_sheet_img = out_dir / "phase1_navbar_mobile_sheet_dark.png"
            page_m.screenshot(path=str(mob_sheet_img))
            shutil.copy2(mob_sheet_img, brain_screens_dir / "phase1_navbar_mobile_sheet_dark.png")

        browser.close()
    print("=== PHASE 1 SCREENSHOTS CAPTURED SUCCESSFULLY ===")

if __name__ == "__main__":
    run()
