import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

async def capture():
    out_dir = Path("reports/figures/screens")
    out_dir.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch()

        # Desktop 1440x900
        page = await browser.new_page(viewport={"width": 1440, "height": 900})
        await page.goto("http://127.0.0.1:8000/overview")
        await page.evaluate("() => { sessionStorage.setItem('fraudshield-intro-seen', 'true'); document.documentElement.classList.remove('dark'); }")
        await page.reload()
        await page.wait_for_timeout(800)

        # 1. Desktop Light: Channel Dropdown Open
        channel_btn = page.locator("button[title*='Select Payment Channel']").first
        await channel_btn.click()
        await page.wait_for_timeout(300)
        await page.screenshot(path=str(out_dir / "phase3_01_channel_light.png"))

        # Close popover and open Command Palette (Light)
        await page.keyboard.press("Escape")
        await page.wait_for_timeout(200)
        await page.keyboard.press("Control+k")
        await page.wait_for_timeout(400)
        await page.screenshot(path=str(out_dir / "phase3_02_cmdk_light.png"))

        # Close cmdk and open About Dialog (Light)
        await page.keyboard.press("Escape")
        await page.wait_for_timeout(200)
        about_btn = page.locator("button[title*='About FraudShield']").first
        await about_btn.click()
        await page.wait_for_timeout(400)
        await page.screenshot(path=str(out_dir / "phase3_03_about_light.png"))

        # 2. Desktop Dark Mode
        await page.keyboard.press("Escape")
        await page.wait_for_timeout(200)
        await page.evaluate("() => document.documentElement.classList.add('dark')")
        await page.wait_for_timeout(300)

        # Channel Dropdown Open (Dark)
        await channel_btn.click()
        await page.wait_for_timeout(300)
        await page.screenshot(path=str(out_dir / "phase3_04_channel_dark.png"))

        # Command Palette (Dark)
        await page.keyboard.press("Escape")
        await page.wait_for_timeout(200)
        await page.keyboard.press("Control+k")
        await page.wait_for_timeout(400)
        await page.screenshot(path=str(out_dir / "phase3_05_cmdk_dark.png"))

        # About Dialog (Dark)
        await page.keyboard.press("Escape")
        await page.wait_for_timeout(200)
        await about_btn.click()
        await page.wait_for_timeout(400)
        await page.screenshot(path=str(out_dir / "phase3_06_about_dark.png"))

        # 3. Mobile 390x844: Navigation Menu
        page_m = await browser.new_page(viewport={"width": 390, "height": 844})
        await page_m.goto("http://127.0.0.1:8000/overview")
        await page_m.evaluate("() => { sessionStorage.setItem('fraudshield-intro-seen', 'true'); document.documentElement.classList.remove('dark'); }")
        await page_m.reload()
        await page_m.wait_for_timeout(800)

        # Open Mobile Hamburger Menu (Light)
        menu_btn = page_m.locator("button[aria-label='Open mobile navigation menu']").first
        await menu_btn.click()
        await page_m.wait_for_timeout(300)
        await page_m.screenshot(path=str(out_dir / "phase3_07_mobile_menu_light.png"))

        # Dark Mobile
        await page_m.evaluate("() => document.documentElement.classList.add('dark')")
        await page_m.wait_for_timeout(300)
        await page_m.screenshot(path=str(out_dir / "phase3_08_mobile_menu_dark.png"))

        await browser.close()
        print("Phase 3 Shell screenshots captured successfully.")

if __name__ == "__main__":
    asyncio.run(capture())

