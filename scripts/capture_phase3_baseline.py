import time
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT_DIR = Path(r"C:\Users\ARNAV ARYA\.gemini\antigravity\brain\9d48f82a-9772-420e-bc78-fb7ea758cd3d\screens\phase3_baseline")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PAGES = [
    ("transactions", "transactions"),
    ("cases", "cases"),
    ("models", "models"),
    ("overview", "overview"),
    ("livecheck", "live-check"),
]

def capture():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        for mode in ["light", "dark"]:
            for name, path in PAGES:
                # Desktop
                ctx = browser.new_context(viewport={"width": 1440, "height": 900})
                page = ctx.new_page()
                page.goto(f"http://localhost:5173/{path}", wait_until="domcontentloaded")
                time.sleep(1.0)
                if mode == "dark":
                    page.evaluate("document.documentElement.classList.add('dark'); localStorage.setItem('fraudshield-theme', 'dark');")
                else:
                    page.evaluate("document.documentElement.classList.remove('dark'); localStorage.setItem('fraudshield-theme', 'light');")
                time.sleep(0.5)
                page.screenshot(path=str(OUT_DIR / f"{name}_desktop_{mode}.png"))
                ctx.close()

                # Mobile
                ctx = browser.new_context(viewport={"width": 390, "height": 844})
                page = ctx.new_page()
                page.goto(f"http://localhost:5173/{path}", wait_until="domcontentloaded")
                time.sleep(1.0)
                if mode == "dark":
                    page.evaluate("document.documentElement.classList.add('dark'); localStorage.setItem('fraudshield-theme', 'dark');")
                else:
                    page.evaluate("document.documentElement.classList.remove('dark'); localStorage.setItem('fraudshield-theme', 'light');")
                time.sleep(0.5)
                page.screenshot(path=str(OUT_DIR / f"{name}_mobile_{mode}.png"))
                ctx.close()

        browser.close()
        print("Captured phase 3 baseline screens successfully.")

if __name__ == "__main__":
    capture()

