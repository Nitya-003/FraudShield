"""
Download self-hosted WOFF2 font files for FraudShield:
- Newsreader (Editorial serif display face for titles and headers)
- Inter (Clean UI face)
- JetBrains Mono (Tabular numerals and code)
Saves directly into frontend/public/fonts/.
"""

import os
import re
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
FONTS_DIR = BASE_DIR / "frontend" / "public" / "fonts"
FONTS_DIR.mkdir(parents=True, exist_ok=True)

CSS_URL = (
    "https://fonts.googleapis.com/css2?"
    "family=Inter:wght@400;500;600;700&"
    "family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;0,6..72,700;1,6..72,400&"
    "family=JetBrains+Mono:wght@400;500;600&display=swap"
)

headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

print("Fetching font definitions from Google Fonts...")
req = urllib.request.Request(CSS_URL, headers=headers)
with urllib.request.urlopen(req) as resp:
    css_content = resp.read().decode("utf-8")

font_blocks = re.findall(r"@font-face\s*\{([^}]+)\}", css_content)
downloaded = []

for block in font_blocks:
    family_match = re.search(r"font-family:\s*['\"]([^'\"]+)['\"]", block)
    weight_match = re.search(r"font-weight:\s*(\d+)", block)
    style_match = re.search(r"font-style:\s*(\w+)", block)
    url_match = re.search(r"src:\s*url\((https://[^\)]+\.woff2)\)", block)

    if family_match and url_match:
        family = family_match.group(1).replace(" ", "")
        weight = weight_match.group(1) if weight_match else "400"
        style = style_match.group(1) if style_match else "normal"
        url = url_match.group(1)

        filename = f"{family}-{weight}-{style}.woff2"
        target_path = FONTS_DIR / filename

        if not target_path.exists():
            f_req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(f_req) as f_resp:
                target_path.write_bytes(f_resp.read())
            downloaded.append(filename)

print(f"Downloaded {len(downloaded)} new font files.")
all_fonts = list(FONTS_DIR.glob("*.woff2"))
print(f"Total self-hosted fonts in {FONTS_DIR}: {len(all_fonts)} files")
for f in all_fonts:
    print(f" - {f.name} ({f.stat().st_size:,} bytes)")

