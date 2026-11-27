from pathlib import Path
from playwright.sync_api import sync_playwright

screens_dir = Path('reports/figures/screens')
screens_dir.mkdir(parents=True, exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    
    # 1. Desktop Light (1440x900)
    ctx_light = browser.new_context(viewport={'width': 1440, 'height': 900}, device_scale_factor=2)
    page = ctx_light.new_page()
    page.add_init_script("sessionStorage.setItem('fraudshield_shield_opened', 'true'); localStorage.setItem('fraudshield-theme', 'light');")
    page.goto('http://localhost:8000/', wait_until='networkidle')
    page.wait_for_timeout(800)
    page.screenshot(path=str(screens_dir / '00_landing.png'), full_page=True)
    page.close()
    ctx_light.close()

    # 2. Desktop Dark (1440x900)
    ctx_dark = browser.new_context(viewport={'width': 1440, 'height': 900}, device_scale_factor=2)
    page = ctx_dark.new_page()
    page.add_init_script("sessionStorage.setItem('fraudshield_shield_opened', 'true'); localStorage.setItem('fraudshield-theme', 'dark');")
    page.goto('http://localhost:8000/', wait_until='networkidle')
    page.evaluate("() => document.documentElement.classList.add('dark')")
    page.wait_for_timeout(800)
    page.screenshot(path=str(screens_dir / '00_landing_dark.png'), full_page=True)
    page.close()
    ctx_dark.close()

    # 3. Mobile Light (390x844)
    ctx_mob_light = browser.new_context(viewport={'width': 390, 'height': 844}, device_scale_factor=2)
    page = ctx_mob_light.new_page()
    page.add_init_script("sessionStorage.setItem('fraudshield_shield_opened', 'true'); localStorage.setItem('fraudshield-theme', 'light');")
    page.goto('http://localhost:8000/', wait_until='networkidle')
    page.wait_for_timeout(800)
    page.screenshot(path=str(screens_dir / 'mobile_00_landing_light.png'), full_page=True)
    page.close()
    ctx_mob_light.close()

    # 4. Mobile Dark (390x844)
    ctx_mob_dark = browser.new_context(viewport={'width': 390, 'height': 844}, device_scale_factor=2)
    page = ctx_mob_dark.new_page()
    page.add_init_script("sessionStorage.setItem('fraudshield_shield_opened', 'true'); localStorage.setItem('fraudshield-theme', 'dark');")
    page.goto('http://localhost:8000/', wait_until='networkidle')
    page.evaluate("() => document.documentElement.classList.add('dark')")
    page.wait_for_timeout(800)
    page.screenshot(path=str(screens_dir / 'mobile_00_landing_dark.png'), full_page=True)
    page.close()
    ctx_mob_dark.close()

    browser.close()
print('All landing page screens updated successfully!')

