import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

async def run():
    out_dir = Path("reports/figures/screens")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        
        # 1. Desktop Light (Intro Overlay Capture)
        page = await browser.new_page(viewport={"width": 1440, "height": 900})
        await page.goto("http://127.0.0.1:8000")
        # Ensure sessionStorage is clean so intro triggers
        await page.evaluate("() => sessionStorage.clear()")
        await page.reload()
        # Wait 400ms for intro door opening animation
        await page.wait_for_timeout(400)
        await page.screenshot(path=str(out_dir / "intro_01_split_opening.png"))
        
        # Wait 1000ms for core reveal
        await page.wait_for_timeout(900)
        await page.screenshot(path=str(out_dir / "intro_02_core_revealed.png"))
        
        # Wait for intro to complete and reveal landing page
        await page.wait_for_timeout(2000)
        await page.screenshot(path=str(out_dir / "landing_desktop_light.png"))
        
        # 2. Desktop Dark
        await page.evaluate("() => document.documentElement.classList.add('dark')")
        await page.wait_for_timeout(500)
        await page.screenshot(path=str(out_dir / "landing_desktop_dark.png"))
        
        # 3. Mobile Viewport Light & Dark
        page_m = await browser.new_page(viewport={"width": 390, "height": 844})
        await page_m.goto("http://127.0.0.1:8000")
        await page_m.evaluate("() => { sessionStorage.setItem('fraudshield-intro-seen', 'true'); document.documentElement.classList.remove('dark'); }")
        await page_m.reload()
        await page_m.wait_for_timeout(1000)
        await page_m.screenshot(path=str(out_dir / "landing_mobile_light.png"))
        
        await page_m.evaluate("() => document.documentElement.classList.add('dark')")
        await page_m.wait_for_timeout(500)
        await page_m.screenshot(path=str(out_dir / "landing_mobile_dark.png"))
        
        await browser.close()
        print("Captured all intro & landing screenshots successfully.")

if __name__ == "__main__":
    asyncio.run(run())

