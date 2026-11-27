import io
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
    
    ctx = browser.new_context(
        viewport={"width": 1000, "height": 625},
        device_scale_factor=1.5,
        color_scheme="dark"
    )
    page = ctx.new_page()
    page.add_init_script("""
        localStorage.setItem('fraudshield-theme', 'dark');
        document.documentElement.classList.add('dark');
    """)

    # Record frames across 1.4 seconds of initial opening
    frames = []
    page.goto("http://localhost:8000/", wait_until="domcontentloaded")
    
    # 20 frames sampled across the first 1.2 seconds
    for _ in range(16):
        buf = page.screenshot()
        frames.append(Image.open(io.BytesIO(buf)))
        page.wait_for_timeout(75)

    if frames:
        frames[0].save(
            out_dir / "aperture_opening_transition.gif",
            save_all=True,
            append_images=frames[1:],
            duration=75,
            loop=0
        )
        print("Generated aperture_opening_transition.gif successfully.")

    browser.close()
