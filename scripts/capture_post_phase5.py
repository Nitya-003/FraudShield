import os
import shutil
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

out_dir = Path("reports/figures/screens/phase5_post")
out_dir.mkdir(parents=True, exist_ok=True)

artifact_dir = Path("C:/Users/ARNAV ARYA/.gemini/antigravity/brain/9d48f82a-9772-420e-bc78-fb7ea758cd3d/screens/phase5_post")
artifact_dir.mkdir(parents=True, exist_ok=True)

routes = [
    ("/", "landing"),
    ("/overview", "overview"),
    ("/live-check", "live_check"),
    ("/transactions", "transactions"),
    ("/cases", "cases"),
    ("/models", "models"),
    ("/entity", "entity"),
    ("/attack-lab", "attack_lab"),
    ("/monitoring", "monitoring"),
    ("/batch", "batch"),
    ("/api-docs", "api_docs"),
]

viewports = [
    ("desktop", {"width": 1440, "height": 900}),
    ("mobile", {"width": 390, "height": 844}),
]

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        for vp_name, vp in viewports:
            for theme in ["light", "dark"]:
                page = browser.new_page(viewport=vp)
                for path, name in routes:
                    try:
                        page.goto(f"http://localhost:8000{path}", wait_until="networkidle")
                        page.evaluate("() => sessionStorage.setItem('fraudshield-intro-seen', 'true')")
                        if theme == "dark":
                            page.evaluate("() => { document.documentElement.classList.add('dark'); localStorage.setItem('fraudshield-theme', 'dark'); }")
                        else:
                            page.evaluate("() => { document.documentElement.classList.remove('dark'); localStorage.setItem('fraudshield-theme', 'light'); }")
                        page.wait_for_timeout(450)
                        out_path = out_dir / f"{name}_{vp_name}_{theme}.png"
                        page.screenshot(path=str(out_path))
                        # Copy to artifact dir as well
                        shutil.copy2(str(out_path), str(artifact_dir / out_path.name))
                        print(f"Captured {out_path.name}")
                    except Exception as e:
                        print(f"Error capturing {path} ({vp_name}, {theme}): {e}")
                page.close()
        browser.close()
    print("Done capturing post-Phase 5 screenshots.")

if __name__ == "__main__":
    run()
