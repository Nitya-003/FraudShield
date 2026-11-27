import os
import shutil
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

SCREENS_DIR = Path("reports/figures/screens")
SCREENS_DIR.mkdir(parents=True, exist_ok=True)

ARTIFACT_DIR = Path("C:/Users/ARNAV ARYA/.gemini/antigravity/brain/9d48f82a-9772-420e-bc78-fb7ea758cd3d/screens/final")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Clean up old phase folders and stale temporary screenshots
for sub in SCREENS_DIR.iterdir():
    if sub.is_dir() and sub.name.startswith("phase"):
        try:
            shutil.rmtree(sub)
            print(f"Deleted phase dir: {sub.name}")
        except Exception as e:
            print(f"Error removing {sub}: {e}")
    elif sub.is_file() and sub.suffix == ".png":
        # Delete old temporary files
        if (
            sub.name.startswith("phase")
            or sub.name.startswith("mobile_")
            or (sub.name[:2].isdigit() and sub.name[2] == "_")
            or sub.name.startswith("frame_")
            or sub.name.startswith("transition_")
            or sub.name.startswith("intro_")
        ):
            try:
                sub.unlink()
                print(f"Deleted stale file: {sub.name}")
            except Exception as e:
                print(f"Error unlinking {sub}: {e}")

ROUTES = [
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

VIEWPORTS = [
    ("desktop", {"width": 1440, "height": 900}),
    ("mobile", {"width": 390, "height": 844}),
]

def capture_all():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        for mode in ["dark", "light"]:
            for vp_name, vp in VIEWPORTS:
                ctx = browser.new_context(viewport=vp)
                page = ctx.new_page()

                for name, path in ROUTES:
                    page.goto(f"http://localhost:8000/{path}", wait_until="networkidle")
                    page.evaluate("() => sessionStorage.setItem('fraudshield-intro-seen', 'true')")
                    if mode == "dark":
                        page.evaluate("() => { document.documentElement.classList.add('dark'); localStorage.setItem('fraudshield-theme', 'dark'); }")
                    else:
                        page.evaluate("() => { document.documentElement.classList.remove('dark'); localStorage.setItem('fraudshield-theme', 'light'); }")
                    page.wait_for_timeout(500)

                    out_path = SCREENS_DIR / f"final_{name}_{vp_name}_{mode}.png"
                    page.screenshot(path=str(out_path))
                    # Copy to artifact dir
                    shutil.copy2(str(out_path), str(ARTIFACT_DIR / out_path.name))
                    print(f"Captured: {out_path.name}")

                ctx.close()

        # Capture command palette
        ctx = browser.new_context(viewport={"width": 1440, "height": 900})
        page = ctx.new_page()
        page.goto("http://localhost:8000/overview", wait_until="networkidle")
        page.evaluate("() => sessionStorage.setItem('fraudshield-intro-seen', 'true')")
        page.evaluate("() => { document.documentElement.classList.add('dark'); localStorage.setItem('fraudshield-theme', 'dark'); }")
        page.wait_for_timeout(500)
        page.keyboard.press("Control+k")
        page.wait_for_timeout(500)
        cmd_path = SCREENS_DIR / "final_command_palette_dark.png"
        page.screenshot(path=str(cmd_path))
        shutil.copy2(str(cmd_path), str(ARTIFACT_DIR / cmd_path.name))
        print("Captured: final_command_palette_dark.png")
        ctx.close()

        # Generate copies for presentation compatibility if needed
        shutil.copy2(str(SCREENS_DIR / "final_overview_desktop_dark.png"), str(SCREENS_DIR / "01_overview.png"))
        shutil.copy2(str(SCREENS_DIR / "final_live_check_desktop_dark.png"), str(SCREENS_DIR / "02_live_check.png"))

        browser.close()
    print("Done generating clean final screenshot set.")

if __name__ == "__main__":
    capture_all()

