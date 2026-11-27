import time
from pathlib import Path
from playwright.sync_api import sync_playwright

SCREENS_DIR = Path("reports/figures/screens")
SCREENS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Clean up stale screenshots (keep gifs/mp4s if any)
for f in SCREENS_DIR.glob("*.png"):
    if f.name.startswith("phase") or f.name.startswith("mobile_") or (f.name[:2].isdigit() and f.name[2] == "_") or f.name.startswith("frame_") or f.name.startswith("transition_") or f.name.startswith("intro_"):
        try:
            f.unlink()
            print(f"Deleted stale: {f.name}")
        except Exception as e:
            print(f"Could not delete {f.name}: {e}")

PAGES = [
    ("landing", ""),
    ("overview", "overview"),
    ("live_check", "live-check"),
    ("transactions", "transactions"),
    ("cases", "cases"),
    ("models", "models"),
    ("entity", "entity"),
    ("monitoring", "monitoring"),
    ("batch", "batch"),
    ("api_docs", "api-docs"),
    ("attack_lab", "attack-lab"),
]

def capture_all():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        for mode in ["dark", "light"]:
            # Desktop 1440x900
            for name, path in PAGES:
                ctx = browser.new_context(viewport={"width": 1440, "height": 900})
                page = ctx.new_page()
                page.goto(f"http://localhost:8000/{path}", wait_until="networkidle")
                time.sleep(0.5)
                if mode == "dark":
                    page.evaluate("document.documentElement.classList.add('dark'); localStorage.setItem('fraudshield-theme', 'dark');")
                else:
                    page.evaluate("document.documentElement.classList.remove('dark'); localStorage.setItem('fraudshield-theme', 'light');")
                time.sleep(0.4)
                out_path = SCREENS_DIR / f"final_{name}_desktop_{mode}.png"
                page.screenshot(path=str(out_path))
                print(f"Captured: {out_path.name}")
                ctx.close()

            # Mobile 390x844 (core pages)
            for name, path in PAGES[:6]:
                ctx = browser.new_context(viewport={"width": 390, "height": 844})
                page = ctx.new_page()
                page.goto(f"http://localhost:8000/{path}", wait_until="networkidle")
                time.sleep(0.5)
                if mode == "dark":
                    page.evaluate("document.documentElement.classList.add('dark'); localStorage.setItem('fraudshield-theme', 'dark');")
                else:
                    page.evaluate("document.documentElement.classList.remove('dark'); localStorage.setItem('fraudshield-theme', 'light');")
                time.sleep(0.4)
                out_path = SCREENS_DIR / f"final_{name}_mobile_{mode}.png"
                page.screenshot(path=str(out_path))
                print(f"Captured: {out_path.name}")
                ctx.close()

        # Capture command palette
        ctx = browser.new_context(viewport={"width": 1440, "height": 900})
        page = ctx.new_page()
        page.goto("http://localhost:8000/overview", wait_until="networkidle")
        page.evaluate("document.documentElement.classList.add('dark'); localStorage.setItem('fraudshield-theme', 'dark');")
        time.sleep(0.5)
        page.keyboard.press("Control+k")
        time.sleep(0.5)
        page.screenshot(path=str(SCREENS_DIR / "final_command_palette_dark.png"))
        print("Captured: final_command_palette_dark.png")
        ctx.close()

        browser.close()
        print("All final screenshots regenerated cleanly.")

if __name__ == "__main__":
    capture_all()
