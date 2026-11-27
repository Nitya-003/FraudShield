import os
import shutil
import time
import zipfile
from pathlib import Path
from playwright.sync_api import sync_playwright
from PIL import Image

out_dir = Path("reports/figures/screens")
out_dir.mkdir(parents=True, exist_ok=True)
brain_screens_dir = Path(r"C:\Users\AYUSH\.gemini\antigravity\brain\6e421928-09c0-4b70-9854-1510db5b76c8")
brain_screens_dir.mkdir(parents=True, exist_ok=True)
BASE_URL = "http://localhost:8000"

PAGES = [
    ("landing", "/"),
    ("overview", "/overview"),
    ("live_check", "/live-check"),
    ("transactions", "/transactions"),
    ("cases", "/cases"),
    ("models", "/models"),
    ("attack_lab", "/attack-lab"),
    ("entity", "/entity"),
    ("batch", "/batch-scoring"),
    ("monitoring", "/monitoring"),
    ("api_docs", "/api-docs"),
]

def run():
    print("=== STARTING PHASE 7 DELIVERY ASSET REGENERATION ===")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        # 1. Desktop Dark (1440x900)
        ctx_dark = browser.new_context(
            viewport={"width": 1440, "height": 900},
            device_scale_factor=1.5,
            record_video_dir=str(out_dir),
            record_video_size={"width": 1440, "height": 900}
        )
        ctx_dark.add_init_script("localStorage.setItem('fraudshield-theme', 'dark'); document.documentElement.classList.add('dark');")
        page_d = ctx_dark.new_page()

        landing_frames = []

        for name, route in PAGES:
            print(f"Capturing Desktop Dark: {name} -> {route}")
            page_d.goto(f"{BASE_URL}{route}", wait_until="networkidle")
            page_d.wait_for_timeout(1000)

            if name == "landing":
                # Collect frames for animated GIF
                for step in range(8):
                    page_d.mouse.move(720 + (100 if step % 2 == 0 else -100), 450 + (30 if step % 4 == 0 else -30))
                    page_d.wait_for_timeout(250)
                    frame_path = out_dir / f"radar_frame_{step}.png"
                    page_d.screenshot(path=str(frame_path))
                    landing_frames.append(Image.open(frame_path).resize((640, 400), Image.Resampling.LANCZOS))
            elif name == "live_check":
                btn = page_d.locator("button:has-text('UPI Collect Scam')").first
                if btn.is_visible():
                    btn.click()
                    page_d.wait_for_timeout(300)
                score_btn = page_d.locator("button:has-text('Score Transaction')").first
                if score_btn.is_visible():
                    score_btn.click()
                    page_d.wait_for_timeout(800)
            elif name == "cases":
                first_row = page_d.locator("tbody tr").first
                if first_row.is_visible():
                    first_row.click()
                    page_d.wait_for_timeout(500)

            img_path = out_dir / f"final_{name}_desktop_dark.png"
            page_d.screenshot(path=str(img_path))
            shutil.copy2(img_path, brain_screens_dir / f"final_{name}_desktop_dark.png")

        video_path = page_d.video.path() if page_d.video else None
        page_d.close()
        ctx_dark.close()

        if video_path and os.path.exists(video_path):
            mp4_target = out_dir / "fraud_radar_hero.mp4"
            shutil.copy2(video_path, mp4_target)
            shutil.copy2(video_path, brain_screens_dir / "fraud_radar_hero.mp4")
            print(f"Copied landing video to {mp4_target}")

        # Assemble GIF from captured frames
        if landing_frames:
            gif_path = out_dir / "fraud_radar_hero_10s.gif"
            landing_frames[0].save(
                gif_path,
                save_all=True,
                append_images=landing_frames[1:],
                duration=350,
                loop=0,
                optimize=True
            )
            shutil.copy2(gif_path, brain_screens_dir / "fraud_radar_hero_10s.gif")
            print(f"Generated animated landing GIF at {gif_path}")

        # 2. Desktop Light (1440x900)
        ctx_light = browser.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=1.5)
        ctx_light.add_init_script("localStorage.setItem('fraudshield-theme', 'light'); document.documentElement.classList.remove('dark');")
        page_l = ctx_light.new_page()

        for name, route in PAGES:
            print(f"Capturing Desktop Light: {name} -> {route}")
            page_l.goto(f"{BASE_URL}{route}", wait_until="networkidle")
            page_l.wait_for_timeout(1000)

            if name == "live_check":
                btn = page_l.locator("button:has-text('UPI Collect Scam')").first
                if btn.is_visible():
                    btn.click()
                    page_l.wait_for_timeout(300)
                score_btn = page_l.locator("button:has-text('Score Transaction')").first
                if score_btn.is_visible():
                    score_btn.click()
                    page_l.wait_for_timeout(800)

            img_path = out_dir / f"final_{name}_desktop_light.png"
            page_l.screenshot(path=str(img_path))
            shutil.copy2(img_path, brain_screens_dir / f"final_{name}_desktop_light.png")

        page_l.close()
        ctx_light.close()

        # 3. Mobile Dark & Light (390x844)
        for theme in ["dark", "light"]:
            ctx_mob = browser.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=1.5)
            if theme == "dark":
                ctx_mob.add_init_script("localStorage.setItem('fraudshield-theme', 'dark'); document.documentElement.classList.add('dark');")
            else:
                ctx_mob.add_init_script("localStorage.setItem('fraudshield-theme', 'light'); document.documentElement.classList.remove('dark');")
            page_m = ctx_mob.new_page()

            for name, route in [("landing", "/"), ("overview", "/overview"), ("live_check", "/live-check"), ("transactions", "/transactions")]:
                print(f"Capturing Mobile {theme}: {name}")
                page_m.goto(f"{BASE_URL}{route}", wait_until="networkidle")
                page_m.wait_for_timeout(800)
                img_path = out_dir / f"final_{name}_mobile_{theme}.png"
                page_m.screenshot(path=str(img_path))
                shutil.copy2(img_path, brain_screens_dir / f"final_{name}_mobile_{theme}.png")

            page_m.close()
            ctx_mob.close()

        browser.close()

    # 4. Update PPTX slides with the NEW UI screenshots
    pptx_path = Path("FraudShield.pptx")
    if pptx_path.exists():
        print("Updating FraudShield.pptx with new UI screenshots...")
        temp_pptx = Path("FraudShield_updated.pptx")
        with zipfile.ZipFile(pptx_path, "r") as zin:
            with zipfile.ZipFile(temp_pptx, "w", compression=zipfile.ZIP_DEFLATED) as zout:
                for item in zin.infolist():
                    data = zin.read(item.filename)
                    if item.filename == "ppt/media/image1.png":
                        with open(out_dir / "final_overview_desktop_dark.png", "rb") as f_new:
                            data = f_new.read()
                        print("  Replaced ppt/media/image1.png with final_overview_desktop_dark.png")
                    elif item.filename == "ppt/media/image2.png":
                        with open(out_dir / "final_live_check_desktop_dark.png", "rb") as f_new:
                            data = f_new.read()
                        print("  Replaced ppt/media/image2.png with final_live_check_desktop_dark.png")
                    zout.writestr(item, data)
        shutil.move(str(temp_pptx), str(pptx_path))
        print("Successfully updated FraudShield.pptx with redesigned UI slides!")

    print("=== PHASE 7 ASSET DELIVERY COMPLETE ===")

if __name__ == "__main__":
    run()

