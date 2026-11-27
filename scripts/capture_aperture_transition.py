import time
from pathlib import Path
from playwright.sync_api import sync_playwright

out_dir = Path("reports/figures/screens")
out_dir.mkdir(parents=True, exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch(
        headless=True,
        args=["--use-gl=angle", "--use-angle=swiftshader"]
    )
    
    # 1. Dark Mode Context
    ctx_dark = browser.new_context(
        viewport={"width": 1440, "height": 900},
        device_scale_factor=2,
        color_scheme="dark"
    )
    page = ctx_dark.new_page()
    page.add_init_script("""
        localStorage.setItem('fraudshield-theme', 'dark');
        document.documentElement.classList.add('dark');
    """)
    
    page.goto("http://localhost:8000/", wait_until="domcontentloaded")
    # Immediate pulse capture
    page.wait_for_timeout(80)
    page.screenshot(path=str(out_dir / "transition_01_aperture_pulse.png"))
    print("Captured transition_01_aperture_pulse.png")

    # Mid-split capture (approx 350ms)
    page.wait_for_timeout(250)
    page.screenshot(path=str(out_dir / "transition_02_shutter_split.png"))
    print("Captured transition_02_shutter_split.png")

    # Settle and reveal clean hero in dark mode
    page.wait_for_timeout(1000)
    page.screenshot(path=str(out_dir / "transition_03_hero_revealed_dark.png"))
    print("Captured transition_03_hero_revealed_dark.png")

    # 2. Light Mode Context
    ctx_light = browser.new_context(
        viewport={"width": 1440, "height": 900},
        device_scale_factor=2,
        color_scheme="light"
    )
    page_light = ctx_light.new_page()
    page_light.add_init_script("""
        localStorage.setItem('fraudshield-theme', 'light');
        document.documentElement.classList.remove('dark');
    """)
    page_light.goto("http://localhost:8000/", wait_until="domcontentloaded")
    page_light.wait_for_timeout(1400)
    page_light.screenshot(path=str(out_dir / "transition_04_hero_revealed_light.png"))
    print("Captured transition_04_hero_revealed_light.png")

    # 3. Mobile Viewport (390x844)
    ctx_mobile = browser.new_context(
        viewport={"width": 390, "height": 844},
        device_scale_factor=2,
        color_scheme="dark"
    )
    page_mobile = ctx_mobile.new_page()
    page_mobile.add_init_script("""
        localStorage.setItem('fraudshield-theme', 'dark');
        document.documentElement.classList.add('dark');
    """)
    page_mobile.goto("http://localhost:8000/", wait_until="domcontentloaded")
    page_mobile.wait_for_timeout(1400)
    page_mobile.screenshot(path=str(out_dir / "transition_05_mobile_hero.png"))
    print("Captured transition_05_mobile_hero.png")

    browser.close()
    print("All transition screenshots updated successfully.")
