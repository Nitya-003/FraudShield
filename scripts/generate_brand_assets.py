"""
Generates brand assets for FraudShield Phase 1:
- favicon.ico, favicon-32x32.png, apple-touch-icon.png, icon-192.png, icon-512.png
- Open Graph social share image (1200x630 og-image.png)
"""
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import math

public_dir = Path("frontend-web/public")
public_dir.mkdir(parents=True, exist_ok=True)

def draw_shield(draw, cx, cy, radius, bg_col, stroke_col, emerald_col):
    # Shield points
    top_y = cy - radius
    bot_y = cy + radius * 0.95
    left_x = cx - radius * 0.78
    right_x = cx + radius * 0.78
    mid_y = cy - radius * 0.35
    
    # Outer polygon
    shield_pts = [
        (cx, top_y),
        (right_x, top_y + radius * 0.3),
        (right_x, mid_y + radius * 0.5),
        (cx + radius * 0.35, bot_y - radius * 0.15),
        (cx, bot_y),
        (cx - radius * 0.35, bot_y - radius * 0.15),
        (left_x, mid_y + radius * 0.5),
        (left_x, top_y + radius * 0.3),
    ]
    draw.polygon(shield_pts, fill=bg_col, outline=stroke_col, width=max(2, int(radius * 0.04)))
    
    # Facets
    draw.line([(cx, top_y), (cx, cy - radius * 0.25)], fill=stroke_col, width=max(1, int(radius * 0.03)))
    draw.line([(cx, cy + radius * 0.25), (cx, bot_y)], fill=stroke_col, width=max(1, int(radius * 0.03)))
    draw.line([(left_x, mid_y + radius * 0.3), (cx - radius * 0.25, cy)], fill=stroke_col, width=max(1, int(radius * 0.03)))
    draw.line([(right_x, mid_y + radius * 0.3), (cx + radius * 0.25, cy)], fill=stroke_col, width=max(1, int(radius * 0.03)))
    
    # Central Aperture Hexagon
    r_hex = radius * 0.32
    hex_pts = []
    for i in range(6):
        angle = math.radians(60 * i - 30)
        hex_pts.append((cx + r_hex * math.cos(angle), cy + r_hex * math.sin(angle)))
    draw.polygon(hex_pts, fill="#0B0F14", outline=emerald_col, width=max(2, int(radius * 0.04)))
    
    # Core emerald dot
    r_dot = radius * 0.08
    draw.ellipse([(cx - r_dot, cy - r_dot), (cx + r_dot, cy + r_dot)], fill=emerald_col)

# 1. Generate 512x512 app icon
img_512 = Image.new("RGBA", (512, 512), (11, 15, 20, 255))
d512 = ImageDraw.Draw(img_512)
# Rounded container outline
d512.rounded_rectangle([(8, 8), (504, 504)], radius=112, fill="#0B0F14", outline="#1E2630", width=3)
draw_shield(d512, 256, 256, 170, bg_col="#11171E", stroke_col="#1E2630", emerald_col="#2DD4A0")
img_512.save(public_dir / "icon-512.png")

# 2. Resize to 192x192, 180x180 (apple touch), 32x32, 16x16
img_192 = img_512.resize((192, 192), Image.Resampling.LANCZOS)
img_192.save(public_dir / "icon-192.png")

img_apple = img_512.resize((180, 180), Image.Resampling.LANCZOS)
img_apple.save(public_dir / "apple-touch-icon.png")

img_32 = img_512.resize((32, 32), Image.Resampling.LANCZOS)
img_32.save(public_dir / "favicon-32x32.png")

img_16 = img_512.resize((16, 16), Image.Resampling.LANCZOS)
img_16.save(public_dir / "favicon-16x16.png")

# Save favicon.ico with multi-size icon
img_512.save(public_dir / "favicon.ico", format="ICO", sizes=[(16, 16), (32, 32), (48, 48), (64, 64)])

# 3. Generate OpenGraph Image (1200 x 630)
og_img = Image.new("RGB", (1200, 630), (11, 15, 20))
d_og = ImageDraw.Draw(og_img)

# Subtle background grid & radial glow
for x in range(0, 1200, 48):
    d_og.line([(x, 0), (x, 630)], fill="#11171E", width=1)
for y in range(0, 630, 48):
    d_og.line([(0, y), (1200, y)], fill="#11171E", width=1)

# Outer frame border
d_og.rectangle([(24, 24), (1176, 606)], outline="#1E2630", width=2)

# Aperture Shield on left
draw_shield(d_og, 240, 315, 150, bg_col="#11171E", stroke_col="#1E2630", emerald_col="#2DD4A0")

# Text labels on right
# Load default or standard font
try:
    font_badge = ImageFont.truetype("arial.ttf", 16)
    font_title = ImageFont.truetype("arialbd.ttf", 64)
    font_sub = ImageFont.truetype("arial.ttf", 26)
    font_meta = ImageFont.truetype("consola.ttf", 18)
except Exception:
    font_badge = font_title = font_sub = font_meta = ImageFont.load_default()

# Pill badge
d_og.rounded_rectangle([(440, 150), (690, 185)], radius=6, fill="#11171E", outline="#1E2630", width=1)
d_og.text((455, 158), "DECISION GATE v2.0", fill="#2DD4A0", font=font_badge)

# Main Title
d_og.text((440, 205), "FraudShield", fill="#E8EDF2", font=font_title)

# Tagline
d_og.text((440, 290), "Calibrated Multi-Channel Payment Risk Gate", fill="#8A96A3", font=font_sub)
d_og.text((440, 335), "Decide in milliseconds whether a payment should go through.", fill="#64748B", font=font_sub)

# Metric summary pill row
d_og.rounded_rectangle([(440, 410), (1120, 480)], radius=8, fill="#11171E", outline="#1E2630", width=1)
d_og.text((465, 435), "P50 LATENCY: 7.92 ms   |   RECALL: 99.8%   |   FALSE ALARMS / 100k: 1.2", fill="#E8EDF2", font=font_meta)

og_img.save(public_dir / "og-image.png")
print("Brand assets generated successfully in frontend-web/public/")

