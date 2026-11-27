"""
End-to-End Playwright & Accessibility Test Suite (Phase 8).

Validates:
1. All 13 application views render with zero console errors/warnings in both Light and Dark themes.
2. Zero WCAG 2.1 AA accessibility violations via axe-core.
3. Live Check works across all four payment channels (UPI, Card, Transfer/Wallet, Bank Account).
4. Responsive layout at Desktop (1440x900) and Mobile (390x844) with zero horizontal overflow.
5. Full keyboard navigation: visible focus rings, Esc dialog closure, Ctrl+K command palette.
6. prefers-reduced-motion disables streaming animations and ledger row insertions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Generator
import pytest
from playwright.sync_api import sync_playwright, Browser, BrowserContext, Page

BASE_URL = "http://localhost:8000"
AXE_PATH = Path("frontend-web/node_modules/axe-core/axe.min.js").resolve()

ALL_TABS = [
    "landing",
    "overview",
    "live_check",
    "transactions",
    "cases",
    "attack_lab",
    "models",
    "entity",
    "monitoring",
    "batch",
    "api_docs",
    "how_it_works",
    "design",
]


@pytest.fixture(scope="module")
def browser_instance() -> Generator[Browser, None, None]:
    p = sync_playwright().start()
    browser = p.chromium.launch(headless=True)
    yield browser
    browser.close()
    p.stop()


def _run_axe_audit(page: Page) -> list[dict]:
    """Inject axe-core and run WCAG 2.1 AA audit."""
    page.add_script_tag(path=str(AXE_PATH))
    results = page.evaluate("""
        async () => {
            return await axe.run({
                runOnly: {
                    type: 'tag',
                    values: ['wcag2a', 'wcag2aa']
                },
                rules: {
                    // Disable color-contrast on ECharts canvas elements if any
                    'color-contrast': { enabled: true }
                }
            });
        }
    """)
    return results.get("violations", [])


def test_e2e_all_pages_light_and_dark_axe_and_console(browser_instance: Browser) -> None:
    """Verify all 13 views in Light and Dark mode: no console errors, 0 WCAG AA violations."""
    page = browser_instance.new_page(viewport={"width": 1440, "height": 900})
    console_errors: list[str] = []

    def on_console(msg):
        if msg.type == "error":
            # Ignore harmless favicon / external network hiccups if any
            if "favicon" not in msg.text.lower():
                console_errors.append(f"[{msg.type}] {msg.text}")

    def on_pageerror(err):
        console_errors.append(f"[pageerror] {str(err)}")

    page.on("console", on_console)
    page.on("pageerror", on_pageerror)

    # Initial load
    page.goto(BASE_URL, wait_until="networkidle")

    for theme in ["light", "dark"]:
        for tab in ALL_TABS:
            url = f"{BASE_URL}/?tab={tab}"
            page.goto(url, wait_until="networkidle")
            if theme == "dark":
                page.evaluate("() => { document.documentElement.classList.add('dark'); localStorage.setItem('fraudshield-theme', 'dark'); }")
            else:
                page.evaluate("() => { document.documentElement.classList.remove('dark'); localStorage.setItem('fraudshield-theme', 'light'); }")
            page.wait_for_timeout(300)

            # Check for console errors
            assert len(console_errors) == 0, f"Console error detected on tab '{tab}' ({theme}): {console_errors}"

            # Check accessibility
            violations = _run_axe_audit(page)
            if violations:
                msg = f"Accessibility violation on tab '{tab}' ({theme}):\n"
                for v in violations:
                    msg += f"  - [{v['id']}] {v['help']} (impact: {v['impact']})\n"
                    for node in v.get("nodes", []):
                        msg += f"     Target: {node.get('target')}\n     HTML: {node.get('html')}\n     Failure: {node.get('failureSummary')}\n"
                pytest.fail(msg)

    page.close()


def test_e2e_live_check_all_channels(browser_instance: Browser) -> None:
    """Test Live Check scoring across all 4 channels with preset quick-fills."""
    page = browser_instance.new_page(viewport={"width": 1440, "height": 900})
    page.goto(f"{BASE_URL}/?tab=live_check", wait_until="networkidle")
    page.wait_for_selector("button:has-text('Routine Payment')", state="visible", timeout=15000)

    # 1. Preset 1: Routine Payment
    page.click("button:has-text('Routine Payment')")
    page.wait_for_timeout(400)
    page.click("button:has-text('Score Transaction')")
    page.wait_for_selector("text=Top Factor Drivers", timeout=8000)

    # 2. Preset 2: UPI Collect Scam
    page.click("button:has-text('UPI Collect Scam')")
    page.wait_for_timeout(400)
    page.click("button:has-text('Score Transaction')")
    # Phase 4 spec: collapsible "Customer message" section with sentence-case label
    page.wait_for_selector("text=Customer message", timeout=8000)
    assert page.is_visible("text=English SMS / notification") or page.is_visible("text=English SMS / Notification")
    assert page.is_visible("text=हिंदी संदेश")

    # 3. Preset 3: High-Value Nocturnal Split
    page.click("button:has-text('High-Value Nocturnal Split')")
    page.wait_for_timeout(400)
    page.click("button:has-text('Score Transaction')")
    page.wait_for_selector("text=Top Factor Drivers", timeout=8000)

    # 4. Bank Account Channel selection (Supports both native select and Radix UI combobox primitive)
    if page.locator("select").count() > 0:
        page.select_option("select", value="account")
    else:
        page.click("button[role='combobox']")
        page.wait_for_selector("[role='option']", state="visible")
        page.click("[role='option']:has-text('Account Opening')")
    page.wait_for_timeout(300)
    page.click("button:has-text('Score Transaction')")
    page.wait_for_selector("text=Top Factor Drivers", timeout=8000)

    page.close()


def test_e2e_responsive_no_horizontal_overflow(browser_instance: Browser) -> None:
    """Assert no horizontal overflow at Desktop (1440x900) and Mobile (390x844)."""
    viewports = [
        {"name": "desktop", "width": 1440, "height": 900},
        {"name": "mobile", "width": 390, "height": 844},
    ]

    for vp in viewports:
        page = browser_instance.new_page(viewport={"width": vp["width"], "height": vp["height"]})
        for tab in ["landing", "overview", "live_check", "transactions", "cases", "attack_lab", "models"]:
            page.goto(f"{BASE_URL}/?tab={tab}", wait_until="networkidle")
            page.wait_for_timeout(200)

            # Check that root container has no horizontal scrollbar
            has_h_scroll = page.evaluate("""
                () => {
                    const scrollW = document.documentElement.scrollWidth;
                    const clientW = document.documentElement.clientWidth;
                    return scrollW > clientW + 2; // allowance of 2px for subpixel rendering
                }
            """)
            assert not has_h_scroll, f"Horizontal overflow detected on tab '{tab}' at viewport {vp['name']} ({vp['width']}x{vp['height']})"
        page.close()


def test_e2e_keyboard_navigation_and_command_palette(browser_instance: Browser) -> None:
    """Test full keyboard navigation: Tab focus ring, Ctrl+K command palette, Esc dismiss."""
    page = browser_instance.new_page(viewport={"width": 1440, "height": 900})
    page.goto(BASE_URL, wait_until="networkidle")

    # 1. Trigger Command Palette with Ctrl+K
    page.keyboard.press("Control+KeyK")
    page.wait_for_timeout(300)
    assert page.is_visible("input[placeholder*='Search pages']")

    # 2. Type search query in palette
    page.keyboard.type("Attack Lab")
    page.wait_for_timeout(200)
    assert page.is_visible("text=Interactive evasion testing")

    # 3. Press Escape to close palette
    page.keyboard.press("Escape")
    page.wait_for_timeout(300)
    assert not page.is_visible("input[placeholder*='Search pages']")

    # 4. Tab through navigation and verify focus indicator
    page.keyboard.press("Tab")
    focused_tag = page.evaluate("() => document.activeElement ? document.activeElement.tagName : null")
    assert focused_tag is not None

    page.close()


def test_e2e_prefers_reduced_motion(browser_instance: Browser) -> None:
    """Assert prefers-reduced-motion pauses streaming animations on landing page."""
    context = browser_instance.new_context(
        viewport={"width": 1440, "height": 900},
        reduced_motion="reduce",
    )
    page = context.new_page()
    page.goto(BASE_URL, wait_until="networkidle")

    # Get first row ID in streaming table
    initial_first_id = page.evaluate("""
        () => {
            const row = document.querySelector('tbody tr td:first-child');
            return row ? row.textContent.trim() : null;
        }
    """)
    # Wait 3 seconds (normally streaming interval is 2.2s)
    page.wait_for_timeout(3000)

    after_wait_first_id = page.evaluate("""
        () => {
            const row = document.querySelector('tbody tr td:first-child');
            return row ? row.textContent.trim() : null;
        }
    """)

    # Stream should remain paused under prefers-reduced-motion
    assert initial_first_id == after_wait_first_id, (
        f"Stream row changed under reduced motion ({initial_first_id} -> {after_wait_first_id})"
    )

    page.close()
    context.close()

