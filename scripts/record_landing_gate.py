import time
from pathlib import Path
from playwright.sync_api import sync_playwright
from PIL import Image

out_dir = Path("reports/figures/screens")
out_dir.mkdir(parents=True, exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch(
        headless=True,
        args=["--use-gl=angle", "--use-angle=swiftshader"]
    )
    
    # Record video context (1440x900)
    ctx = browser.new_context(
        viewport={"width": 1440, "height": 900},
        device_scale_factor=2,
        record_video_dir=str(out_dir),
        record_video_size={"width": 1440, "height": 900}
    )
    page = ctx.new_page()

    # 1. Load Landing Page (Dark)
    page.goto("http://localhost:8000/", wait_until="networkidle")
    page.evaluate("document.documentElement.classList.add('dark')")
    page.wait_for_timeout(1500)

    # Capture Hero Screen (Dark)
    page.screenshot(path=str(out_dir / "phase2_01_hero_the_gate_dark.png"))
    print("Captured phase2_01_hero_the_gate_dark.png")

    # Capture Hero Screen (Light)
    page.evaluate("document.documentElement.classList.remove('dark')")
    page.wait_for_timeout(800)
    page.screenshot(path=str(out_dir / "phase2_02_hero_the_gate_light.png"))
    print("Captured phase2_02_hero_the_gate_light.png")

    page.evaluate("document.documentElement.classList.add('dark')")
    page.wait_for_timeout(500)

    # 2. Record 10-second frame sequence of streaming tokens and mouse parallax
    frames = []
    print("Recording 10s hero & scroll sequence...")
    start_time = time.time()
    step = 0
    while time.time() - start_time < 10.0:
        elapsed = time.time() - start_time
        # Gentle cursor parallax movement
        target_x = 720 + 200 * (1 if step % 2 == 0 else -1)
        target_y = 450 + 50 * (1 if step % 4 == 0 else -1)
        page.mouse.move(target_x, target_y)

        # Smooth scroll downwards after 4 seconds to show the 3-layer unfold & 4-channel stats
        if elapsed > 4.5 and elapsed < 8.0:
            scroll_target = int((elapsed - 4.5) * 350)
            page.evaluate(f"window.scrollTo(0, {scroll_target})")
        elif elapsed >= 8.0:
            page.evaluate("window.scrollTo(0, 1400)")

        # Capture animation frame for GIF
        frame_bytes = page.screenshot()
        # Save snapshot at key intervals
        if step == 0:
            with open(out_dir / "phase2_03_frame_0s_gate.png", "wb") as f:
                f.write(frame_bytes)
        elif step == 10:
            with open(out_dir / "phase2_04_frame_5s_unfold.png", "wb") as f:
                f.write(frame_bytes)
        elif step == 18:
            with open(out_dir / "phase2_05_frame_9s_stats.png", "wb") as f:
                f.write(frame_bytes)

        img = Image.open(out_dir / "phase2_01_hero_the_gate_dark.png").resize((480, 300), Image.Resampling.LANCZOS)
        frames.append(img)
        page.wait_for_timeout(450)
        step += 1

    # Capture 4-channel stats row screenshot
    page.evaluate("window.scrollTo(0, 1600)")
    page.wait_for_timeout(800)
    page.screenshot(path=str(out_dir / "phase2_06_channel_stats_row.png"))
    print("Captured phase2_06_channel_stats_row.png")

    # Capture "What this can't do" dialog
    page.click("button:has-text(\"What this can't do\")")
    page.wait_for_timeout(600)
    page.screenshot(path=str(out_dir / "phase2_07_limitation_dialog.png"))
    print("Captured phase2_07_limitation_dialog.png")
    page.keyboard.press("Escape")
    page.wait_for_timeout(400)

    # 3. Mobile Viewport (390x844)
    ctx_m = browser.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=2)
    page_m = ctx_m.new_page()
    page_m.goto("http://localhost:8000/", wait_until="networkidle")
    page_m.evaluate("document.documentElement.classList.add('dark')")
    page_m.wait_for_timeout(1000)
    page_m.screenshot(path=str(out_dir / "phase2_08_mobile_landing_390x844.png"))
    print("Captured phase2_08_mobile_landing_390x844.png")

    ctx.close()
    browser.close()

# Save animated GIF from sequence
if frames:
    gif_path = out_dir / "the_gate_hero_10s.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=450,
        loop=0,
        optimize=True
    )
    print(f"Saved animated GIF: {gif_path}")

print("Phase 2 screen capture and verification complete.")

