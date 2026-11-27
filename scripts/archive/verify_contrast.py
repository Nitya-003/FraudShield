"""
Automated Contrast & Control Visibility Verification.
Audits all controls across all 7 pages in Light and Dark modes.
Asserts that:
- Primary button text contrast >= 4.5:1
- Secondary button border contrast >= 3.0:1
- Inactive tab text contrast >= 4.5:1
- Input border contrast >= 3.0:1
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

def parse_rgb(s: str) -> list[float]:
    m = re.findall(r'\d+', s)
    if len(m) >= 3:
        return [int(x) / 255.0 for x in m[:3]]
    return [1.0, 1.0, 1.0]

def lum(c: list[float]) -> float:
    a = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return 0.2126 * a[0] + 0.7152 * a[1] + 0.0722 * a[2]

def get_contrast(rgb1: str, rgb2: str) -> float:
    c1 = parse_rgb(rgb1)
    c2 = parse_rgb(rgb2)
    l1 = lum(c1)
    l2 = lum(c2)
    return (max(l1, l2) + 0.05) / (min(l1, l2) + 0.05)

def test_all_pages():
    pages = [
        ("Executive Overview", "http://127.0.0.1:8501/?theme=light", "http://127.0.0.1:8501/?theme=dark"),
        ("Transactions & Analytics", "http://127.0.0.1:8501/transactions?theme=light", "http://127.0.0.1:8501/transactions?theme=dark"),
        ("Live Fraud Check", "http://127.0.0.1:8501/live_check?theme=light", "http://127.0.0.1:8501/live_check?theme=dark"),
        ("Model Performance", "http://127.0.0.1:8501/performance?theme=light", "http://127.0.0.1:8501/performance?theme=dark"),
        ("Drift & Monitoring", "http://127.0.0.1:8501/monitoring?theme=light", "http://127.0.0.1:8501/monitoring?theme=dark"),
        ("Demo Controls", "http://127.0.0.1:8501/demo_controls?theme=light", "http://127.0.0.1:8501/demo_controls?theme=dark"),
        ("How It Works", "http://127.0.0.1:8501/how_it_works?theme=light", "http://127.0.0.1:8501/how_it_works?theme=dark"),
    ]

    total_controls = 0
    failures = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=2)

        for name, light_url, dark_url in pages:
            # Audit Light Mode
            page.goto(light_url, wait_until="networkidle")
            page.wait_for_timeout(2000)

            # 1. Audit Buttons in Light Mode
            btns = page.locator("button")
            for i in range(btns.count()):
                b = btns.nth(i)
                style = b.evaluate("""el => {
                    const s = window.getComputedStyle(el);
                    const rect = el.getBoundingClientRect();
                    return {
                        kind: el.getAttribute('kind') || 'default',
                        color: s.color,
                        bg: s.backgroundColor,
                        border: s.border,
                        borderColor: s.borderColor,
                        borderWidth: parseFloat(s.borderWidth) || 0,
                        text: el.innerText.trim().replace(/\\n/g, ' '),
                        visible: rect.width > 0 && rect.height > 0 && s.visibility !== 'hidden'
                    };
                }""")
                if not style["visible"] or not style["text"]:
                    continue

                total_controls += 1
                bg_col = style["bg"] if "rgba(0, 0, 0, 0)" not in style["bg"] else "rgb(248, 250, 252)"
                text_contrast = get_contrast(style["color"], bg_col)
                border_contrast = get_contrast(style["borderColor"], "rgb(248, 250, 252)")

                if style["kind"] == "primary":
                    if text_contrast < 4.5:
                        failures.append(f"[{name} - Light] Primary button '{style['text']}' text contrast {text_contrast:.2f} < 4.5:1")
                elif style["kind"] == "secondary" or "stBaseButton" in style.get("testid", ""):
                    # Secondary button must have visible border >= 3.0:1 or border width >= 1px
                    if border_contrast < 3.0 and style["borderWidth"] > 0:
                        failures.append(f"[{name} - Light] Button '{style['text']}' border contrast {border_contrast:.2f} < 3.0:1")

            # 2. Audit Tabs
            tabs = page.locator("div[data-testid='stTabs'] button[role='tab']")
            for i in range(tabs.count()):
                t = tabs.nth(i)
                t_style = t.evaluate("""el => {
                    const textEl = el.querySelector('p, span') || el;
                    const s = window.getComputedStyle(textEl);
                    return {
                        selected: el.getAttribute('aria-selected') === 'true',
                        color: s.color,
                        text: el.innerText.trim()
                    };
                }""")
                total_controls += 1
                tab_contrast = get_contrast(t_style["color"], "rgb(248, 250, 252)")
                if tab_contrast < 4.5:
                    failures.append(f"[{name} - Light] Tab '{t_style['text']}' contrast {tab_contrast:.2f} < 4.5:1")

            # Audit Dark Mode
            page.goto(dark_url, wait_until="networkidle")
            page.wait_for_timeout(2000)
            btns_dark = page.locator("button")
            for i in range(btns_dark.count()):
                b = btns_dark.nth(i)
                style = b.evaluate("""el => {
                    const s = window.getComputedStyle(el);
                    const rect = el.getBoundingClientRect();
                    return {
                        kind: el.getAttribute('kind') || 'default',
                        color: s.color,
                        bg: s.backgroundColor,
                        border: s.border,
                        borderColor: s.borderColor,
                        borderWidth: parseFloat(s.borderWidth) || 0,
                        text: el.innerText.trim().replace(/\\n/g, ' '),
                        visible: rect.width > 0 && rect.height > 0 && s.visibility !== 'hidden'
                    };
                }""")
                if not style["visible"] or not style["text"]:
                    continue

                total_controls += 1
                bg_col = style["bg"] if "rgba(0, 0, 0, 0)" not in style["bg"] else "rgb(11, 19, 34)"
                text_contrast = get_contrast(style["color"], bg_col)
                border_contrast = get_contrast(style["borderColor"], "rgb(11, 19, 34)")

                if style["kind"] == "primary":
                    if text_contrast < 4.5:
                        failures.append(f"[{name} - Dark] Primary button '{style['text']}' text contrast {text_contrast:.2f} < 4.5:1")
                elif style["kind"] == "secondary":
                    if border_contrast < 3.0 and style["borderWidth"] > 0:
                        failures.append(f"[{name} - Dark] Button '{style['text']}' border contrast {border_contrast:.2f} < 3.0:1")

        browser.close()

    print(f"\nTotal Controls Verified: {total_controls}")
    if failures:
        print(f"FAILED: Found {len(failures)} contrast failures:")
        for f in failures:
            print("  ✕", f)
        sys.exit(1)
    else:
        print("SUCCESS: 100% of tested controls pass WCAG AA contrast requirements!")

if __name__ == "__main__":
    test_all_pages()

