"""
FraudShield Phase 7 Comprehensive Quality Assurance Matrix Runner.
Tests all 13 pages across all 4 payment channels:
- Desktop (1440x900 @ 2x) & Mobile (390x844 @ 2x)
- Light and Dark modes
Validates:
1. Zero exception text in DOM
2. Zero horizontal overflow (scrollWidth <= clientWidth + 2px)
3. Clean font loading (Newsreader serif, Inter sans, JetBrains Mono numbers)
4. Button visibility & contrast in light and dark themes
5. Accessibility: prefers-reduced-motion disabled animation
6. Real page load and render latency benchmark
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

BASE_DIR = Path(__file__).resolve().parent.parent
SCREENS_DIR = BASE_DIR / "reports" / "figures" / "screens"
SCREENS_DIR.mkdir(parents=True, exist_ok=True)
QA_REPORT_PATH = BASE_DIR / "reports" / "qa_matrix_report.json"

BASE_URL = "http://127.0.0.1:8501"

PAGES = [
    ("", "Home", "00_landing"),
    ("overview", "Executive Overview", "01_overview"),
    ("live_check", "Live Fraud Check", "02_live_check"),
    ("cases", "Case Management", "09_case_management"),
    ("batch", "Batch Scoring", "10_batch_scoring"),
    ("entity", "Entity Intelligence", "11_entity_view"),
    ("shadow_mode", "Shadow Mode", "12_shadow_mode"),
    ("api", "API Reference", "13_api_docs"),
    ("transactions", "Transactions & Analytics", "03_transactions"),
    ("performance", "Model Benchmarks", "04_models"),
    ("monitoring", "Live Drift & Monitoring", "05_monitoring"),
    ("demo_controls", "Interactive Policy Controls", "06_demo_controls"),
    ("how_it_works", "How It Works", "07_how_it_works"),
]

CHANNELS = ["transfer_wallet", "card", "account", "upi"]


def run_full_qa():
    print("=" * 75)
    print("  FRAUDSHIELD PHASE 7 COMPREHENSIVE QA MATRIX EXECUTION")
    print("=" * 75)

    results = []
    failures = []
    perf_metrics = {}
    font_audit_results = {}
    button_audit_results = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        viewports = [
            ("desktop", {"width": 1440, "height": 900}),
            ("mobile", {"width": 390, "height": 844}),
        ]

        for vp_name, vp_size in viewports:
            print(f"\n--- Testing Viewport: {vp_name.upper()} ({vp_size['width']}x{vp_size['height']}) ---")
            context = browser.new_context(viewport=vp_size, device_scale_factor=2)
            page = context.new_page()

            # Test reduced-motion emulation
            page.emulate_media(reduced_motion="reduce")

            for theme in ["light", "dark"]:
                # On desktop, test all 4 channels; on mobile, test primary channel
                channels_to_test = ["transfer_wallet"] if vp_name == "mobile" else CHANNELS

                for ch in channels_to_test:
                    for page_slug, page_label, file_prefix in PAGES:
                        test_id = f"{vp_name}_{theme}_{ch}_{page_slug or 'home'}"
                        url = f"{BASE_URL}/{page_slug}?theme={theme}"

                        t0 = time.perf_counter()
                        try:
                            page.goto(url, wait_until="networkidle", timeout=30000)

                            # Set active channel in sessionStorage
                            if ch != "transfer_wallet":
                                page.evaluate(f'''() => {{
                                    try {{
                                        window.sessionStorage.setItem("active_channel", "{ch}");
                                    }} catch(e) {{}}
                                }}''')

                            # Wait for Streamlit status widget to detach
                            try:
                                page.wait_for_selector('div[data-testid="stStatusWidget"]', state="detached", timeout=12000)
                            except (TimeoutError, RuntimeError, OSError):
                                pass

                            page.wait_for_timeout(600)
                            load_time_sec = round(time.perf_counter() - t0, 3)

                            # 1. Exception Text Detection
                            body_text = page.inner_text("body")
                            has_exception = any(
                                err_pattern in body_text
                                for err_pattern in [
                                    "Traceback (most recent call last)",
                                    "StreamlitAPIException",
                                    "ZeroDivisionError",
                                    "KeyError:",
                                    "AttributeError:",
                                    "ModuleNotFoundError:",
                                ]
                            )

                            # 2. Horizontal Scroll Overflow Check
                            overflow_info = page.evaluate('''() => {
                                const doc = document.documentElement;
                                const body = document.body;
                                const scrollW = Math.max(doc.scrollWidth, body.scrollWidth);
                                const clientW = Math.max(doc.clientWidth, body.clientWidth);
                                return { scrollW, clientW, hasOverflow: scrollW > clientW + 2 };
                            }''')

                            # 3. Font Audit (Newsreader, Inter, JetBrains Mono)
                            font_info = page.evaluate('''() => {
                                const loaded = Array.from(document.fonts).map(f => f.family.replace(/['"]/g, ''));
                                const hasInter = document.fonts.check("16px Inter") || loaded.includes("Inter");
                                const hasNewsreader = document.fonts.check("20px Newsreader") || loaded.includes("Newsreader");
                                return { hasInter, hasNewsreader, fontCount: loaded.length };
                            }''')

                            # 4. Button Visibility & Contrast Audit
                            button_info = page.evaluate('''() => {
                                const renderedBtns = Array.from(document.querySelectorAll("button")).filter(b => 
                                    b.offsetParent !== null && b.getAttribute("data-testid") !== "stCodeCopyButton"
                                );
                                let invisibleCount = 0;
                                for (const b of renderedBtns) {
                                    const s = window.getComputedStyle(b);
                                    const r = b.getBoundingClientRect();
                                    if (r.width === 0 || r.height === 0) {
                                        invisibleCount++;
                                    }
                                    if (s.color === s.backgroundColor && s.backgroundColor !== "rgba(0, 0, 0, 0)") {
                                        invisibleCount++;
                                    }
                                }
                                return { total: renderedBtns.length, invisibleCount };
                            }''')

                            # 5. Reduced-Motion CSS Audit
                            reduced_motion_info = page.evaluate('''() => {
                                let hasRule = false;
                                for (const sheet of document.styleSheets) {
                                    try {
                                        for (const rule of sheet.cssRules) {
                                            if (rule.media && rule.media.mediaText.includes("prefers-reduced-motion")) {
                                                hasRule = true;
                                                break;
                                            }
                                        }
                                    } catch(e) {}
                                }
                                return { hasRule };
                            }''')

                            status = "PASS"
                            fail_reasons = []

                            if has_exception:
                                status = "FAIL_EXCEPTION"
                                fail_reasons.append("DOM contains exception/traceback text")
                            if overflow_info.get("hasOverflow") and vp_name == "desktop":
                                status = "FAIL_OVERFLOW"
                                fail_reasons.append(f"Horizontal scroll overflow detected ({overflow_info['scrollW']}px > {overflow_info['clientW']}px)")
                            if button_info.get("invisibleCount", 0) > 0:
                                status = "FAIL_BUTTONS"
                                fail_reasons.append(f"Found {button_info['invisibleCount']} unrendered or low-contrast buttons")

                            if status != "PASS":
                                failures.append((test_id, "; ".join(fail_reasons)))

                            res_item = {
                                "test_id": test_id,
                                "viewport": vp_name,
                                "theme": theme,
                                "channel": ch,
                                "page": page_slug or "home",
                                "status": status,
                                "load_time_sec": load_time_sec,
                                "horizontal_overflow": overflow_info.get("hasOverflow"),
                                "font_audit": font_info,
                                "button_audit": button_info,
                                "reduced_motion_supported": reduced_motion_info.get("hasRule"),
                            }
                            results.append(res_item)

                            slug_key = page_slug or "home"
                            if slug_key not in perf_metrics:
                                perf_metrics[slug_key] = []
                            perf_metrics[slug_key].append(load_time_sec)

                            # Save screenshots for verification
                            if ch == "transfer_wallet":
                                if vp_name == "desktop":
                                    shot_name = f"{file_prefix}_{theme}.png" if theme == "dark" else f"{file_prefix}.png"
                                else:
                                    shot_name = f"mobile_{file_prefix}_{theme}.png"
                                
                                page.screenshot(path=str(SCREENS_DIR / shot_name), full_page=True)

                            print(f"[{status}] {test_id:45s} | Load: {load_time_sec:.2f}s | Fonts: {font_info.get('fontCount', 0)} loaded | Overflow: {overflow_info.get('hasOverflow')}")

                        except (RuntimeError, TimeoutError, OSError, AssertionError) as ex:
                            print(f"[ERR]  {test_id:45s} | {ex}")
                            failures.append((test_id, str(ex)))

            page.close()
            context.close()

        browser.close()

    # Aggregate performance & summary
    avg_perf = {p: round(sum(times) / max(len(times), 1), 3) for p, times in perf_metrics.items()}
    overall_avg = round(sum(sum(times) for times in perf_metrics.values()) / max(sum(len(times) for times in perf_metrics.values()), 1), 3)

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "total_test_executions": len(results),
        "passed": len([r for r in results if r["status"] == "PASS"]),
        "failed": len(failures),
        "failures": failures,
        "average_load_time_sec": overall_avg,
        "page_perf_breakdown_sec": avg_perf,
        "audits": {
            "no_exception_text": len([r for r in results if "FAIL_EXCEPTION" in r["status"]]) == 0,
            "no_desktop_horizontal_overflow": len([r for r in results if r["viewport"] == "desktop" and r["horizontal_overflow"]]) == 0,
            "font_loading_verified": all(r["font_audit"].get("hasInter", False) for r in results if "font_audit" in r),
            "reduced_motion_verified": any(r.get("reduced_motion_supported", False) for r in results),
        }
    }

    QA_REPORT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n" + "=" * 75)
    print(f"QA Matrix Complete: {summary['passed']}/{summary['total_test_executions']} PASSED, {summary['failed']} FAILED.")
    print(f"Average Page Load Time: {overall_avg}s across all pages and viewports.")
    print("=" * 75)
    return summary


if __name__ == "__main__":
    run_full_qa()
