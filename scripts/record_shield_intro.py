"""
Record the animated shield opening intro sequence.
Captures:
1. Video (.webm / .mp4) in reports/figures/videos/
2. Animated GIF in reports/figures/screens/shield_opens_intro.gif
3. High-resolution frames at key intervals (t=0.0s, t=0.6s, t=1.2s, t=1.8s, t=2.4s)
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from PIL import Image
from playwright.sync_api import sync_playwright

BASE_DIR = Path(__file__).resolve().parent.parent
SCREENS_DIR = BASE_DIR / "reports" / "figures" / "screens"
VIDEOS_DIR = BASE_DIR / "reports" / "figures" / "videos"
SCREENS_DIR.mkdir(parents=True, exist_ok=True)
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("--- Recording Shield Opening Animation ---")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        # Create context with video recording
        context = browser.new_context(
            viewport={"width": 1280, "height": 720},
            record_video_dir=str(VIDEOS_DIR),
            record_video_size={"width": 1280, "height": 720},
            device_scale_factor=1.5,
        )

        page = context.new_page()

        # Clear sessionStorage to ensure fresh intro animation
        page.add_init_script("""
            sessionStorage.removeItem('fraudshield_shield_opened');
        """)

        frames: list[Image.Image] = []

        # Navigate to homepage
        page.goto("http://localhost:8000/", wait_until="domcontentloaded")

        start_time = time.time()
        sample_count = 0
        while time.time() - start_time < 2.6:
            # Capture frame
            png_bytes = page.screenshot()
            img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
            # Resize slightly for optimized GIF output
            img_thumb = img.resize((640, 360), Image.Resampling.LANCZOS)
            frames.append(img_thumb)

            # Save key milestone frames
            t_elapsed = time.time() - start_time
            if sample_count in (0, 5, 10, 15, 20):
                stage_name = f"frame_{sample_count:02d}_{t_elapsed:.1f}s.png"
                img.save(SCREENS_DIR / stage_name)
                print(f"Captured key frame: {stage_name}")

            sample_count += 1
            time.sleep(0.08)

        # Save animated GIF
        gif_path = SCREENS_DIR / "shield_opens_intro.gif"
        if frames:
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                optimize=True,
                duration=90,
                loop=0,
            )
            print(f"Saved animated GIF: {gif_path.name} ({len(frames)} frames)")

        # Close page and context to finalize video writing
        page.close()
        video = page.video
        video_path = video.path() if video else None
        context.close()
        browser.close()

        if video_path and Path(video_path).exists():
            target_video = VIDEOS_DIR / "shield_opens_intro.webm"
            if Path(video_path) != target_video:
                if target_video.exists():
                    target_video.unlink()
                Path(video_path).rename(target_video)
            print(f"Saved video to: {target_video}")


if __name__ == "__main__":
    main()

