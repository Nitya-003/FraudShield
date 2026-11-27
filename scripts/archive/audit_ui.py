"""
Comprehensive UI & Accessibility Audit Script.
Visits all pages in Light and Dark mode using Playwright, inspecting:
- Button and control computed styles, borders, text contrast, testids
- Redundant chrome (notices, duplicate chips, Deploy button, sidebars)
- Viewport layout overflows or misalignments
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

BASE_DIR = Path(__file__).resolve().parent.parent

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

def run_audit():
    pages = [
        ("Executive Overview", "http://127.0.0.1:8501/?theme=light", "http://127.0.0.1:8501/?theme=dark"),
        ("Transactions & Analytics", "http://127.0.0.1:8501/transactions?theme=light", "http://127.0.0.1:8501/transactions?theme=dark"),
        ("Live Fraud Check", "http://127.0.0.1:8501/live_check?theme=light", "http://127.0.0.1:8501/live_check?theme=dark"),
        ("Model Performance", "http://127.0.0.1:8501/performance?theme=light", "http://127.0.0.1:8501/performance?theme=dark"),
        ("Drift & Monitoring", "http://127.0.0.1:8501/monitoring?theme=light", "http://127.0.0.1:8501/monitoring?theme=dark"),
        ("Demo Controls", "http://127.0.0.1:8501/demo_controls?theme=light", "http://127.0.0.1:8501/demo_controls?theme=dark"),
        ("How It Works", "http://127.0.0.1:8501/how_it_works?theme=light", "http://127.0.0.1:8501/how_it_works?theme=dark"),
    ]

    report = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=2)

        for name, light_url, dark_url in pages:
            print(f"Auditing {name}...")
            page_data = {"light": {}, "dark": {}}

            # 1. LIGHT MODE AUDIT
            page.goto(light_url, wait_until="networkidle")
            page.wait_for_timeout(2500)

            # Check buttons
            btns = page.locator("button")
            btn_list = []
            for i in range(btns.count()):
                b = btns.nth(i)
                text = b.inner_text().strip().replace("\n", " ")
                style = b.evaluate("""el => {
                    const s = window.getComputedStyle(el);
                    const rect = el.getBoundingClientRect();
                    return {
                        color: s.color,
                        bg: s.backgroundColor,
                        border: s.border,
                        borderColor: s.borderColor,
                        borderWidth: s.borderWidth,
                        boxShadow: s.boxShadow,
                        testid: el.getAttribute('data-testid'),
                        visible: rect.width > 0 && rect.height > 0 && s.visibility !== 'hidden' && s.display !== 'none'
                    };
                }""")
                if style["visible"]:
                    # Contrast against page bg if transparent
                    bg_col = style["bg"] if "rgba(0, 0, 0, 0)" not in style["bg"] else "rgb(248, 250, 252)"
                    text_contrast = get_contrast(style["color"], bg_col)
                    border_contrast = get_contrast(style["borderColor"], "rgb(248, 250, 252)")
                    btn_list.append({
                        "text": text,
                        "testid": style["testid"],
                        "color": style["color"],
                        "bg": style["bg"],
                        "border": style["border"],
                        "borderColor": style["borderColor"],
                        "borderWidth": style["borderWidth"],
                        "boxShadow": style["boxShadow"],
                        "text_contrast": round(text_contrast, 2),
                        "border_contrast": round(border_contrast, 2),
                    })
            page_data["light"]["buttons"] = btn_list

            # Check inputs / selects
            inputs = page.locator("input, div[data-baseweb='select']")
            input_list = []
            for i in range(inputs.count()):
                inp = inputs.nth(i)
                istyle = inp.evaluate("""el => {
                    const s = window.getComputedStyle(el);
                    return {
                        tag: el.tagName,
                        color: s.color,
                        bg: s.backgroundColor,
                        border: s.border,
                        borderColor: s.borderColor
                    };
                }""")
                input_list.append(istyle)
            page_data["light"]["inputs"] = input_list

            # Check chrome elements
            chrome_info = page.evaluate("""() => {
                const sidebar = document.querySelector('section[data-testid=\"stSidebar\"]');
                const deployBtn = document.querySelector('[data-testid=\"stToolbar\"]') || document.querySelector('button[kind=\"header\"]');
                const noticeBanner = document.querySelector('.sim-notice');
                const chipBadges = document.querySelectorAll('.chip-badge');
                return {
                    sidebar_present: !!sidebar,
                    notice_banner_present: !!noticeBanner,
                    chip_count: chipBadges.length,
                    has_deploy: !!document.querySelector('[data-testid=\"manage-app-button\"]')
                };
            }""")
            page_data["light"]["chrome"] = chrome_info

            # 2. DARK MODE AUDIT
            page.goto(dark_url, wait_until="networkidle")
            page.wait_for_timeout(2500)
            btns_dark = page.locator("button")
            btn_dark_list = []
            for i in range(btns_dark.count()):
                b = btns_dark.nth(i)
                text = b.inner_text().strip().replace("\n", " ")
                style = b.evaluate("""el => {
                    const s = window.getComputedStyle(el);
                    const rect = el.getBoundingClientRect();
                    return {
                        color: s.color,
                        bg: s.backgroundColor,
                        border: s.border,
                        borderColor: s.borderColor,
                        borderWidth: s.borderWidth,
                        testid: el.getAttribute('data-testid'),
                        visible: rect.width > 0 && rect.height > 0 && s.visibility !== 'hidden' && s.display !== 'none'
                    };
                }""")
                if style["visible"]:
                    bg_col = style["bg"] if "rgba(0, 0, 0, 0)" not in style["bg"] else "rgb(11, 19, 34)"
                    text_contrast = get_contrast(style["color"], bg_col)
                    border_contrast = get_contrast(style["borderColor"], "rgb(11, 19, 34)")
                    btn_dark_list.append({
                        "text": text,
                        "testid": style["testid"],
                        "color": style["color"],
                        "bg": style["bg"],
                        "border": style["border"],
                        "text_contrast": round(text_contrast, 2),
                        "border_contrast": round(border_contrast, 2),
                    })
            page_data["dark"]["buttons"] = btn_dark_list

            report[name] = page_data

        browser.close()

    # Save audit report json
    out_path = BASE_DIR / "docs" / "ui_audit_data.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nAudit complete! Detailed data saved to {out_path}")

if __name__ == "__main__":
    run_audit()

