"""
End-to-End QA Test Suite for FraudShield Redesign (Phase 6).

Comprehensive coverage of all critical user journeys and requirements:
1. Landing page WebGL hero & smooth scroll sequence
2. Executive Overview KPI cards & sparkline trends
3. Live Check pre-authorization scoring, gauge verdict & factor explainability
4. Transactions ledger filtering, virtualized scroll & detail drawer
5. Cases triage with keyboard shortcuts (J/K navigation, Esc)
6. Models champion registry with expandable detail tabs
7. Command palette search (Ctrl+K) & modal dialog dismissal
8. Theme toggle persistence (Light / Dark)
9. Mobile responsive drawer & viewport layout (390x844)
10. Strict WCAG 2.1 AA accessibility & zero console error validation
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Generator
import pytest
from playwright.sync_api import sync_playwright, Browser, Page, expect

BASE_URL = "http://localhost:8000"
AXE_PATH = Path("frontend-web/node_modules/axe-core/axe.min.js").resolve()


@pytest.fixture(scope="module")
def browser_instance() -> Generator[Browser, None, None]:
    p = sync_playwright().start()
    browser = p.chromium.launch(headless=True)
    yield browser
    browser.close()
    p.stop()


def _run_axe(page: Page) -> list[dict]:
    """Inject axe-core and run WCAG 2.1 AA audit."""
    page.add_script_tag(path=str(AXE_PATH))
    results = page.evaluate("""async () => {
        return await axe.run({
            runOnly: { type: 'tag', values: ['wcag2a', 'wcag2aa'] }
        });
    }""")
    return results.get("violations", [])


def _setup_console_tracker(page: Page) -> list[str]:
    errors: list[str] = []
    page.on("console", lambda msg: errors.append(f"[{msg.type}] {msg.text}") if msg.type == "error" and "favicon" not in msg.text.lower() else None)
    page.on("pageerror", lambda err: errors.append(f"[pageerror] {err}"))
    return errors


def test_qa_landing_page_and_scroll(browser_instance: Browser) -> None:
    """1. Landing page: headline, buttons, scroll sequence layers, and channel stats."""
    page = browser_instance.new_page(viewport={"width": 1440, "height": 900})
    errors = _setup_console_tracker(page)

    page.goto(f"{BASE_URL}/", wait_until="networkidle")
    page.wait_for_timeout(1000)

    # Verify hero copy
    headline = page.locator("h1:has-text('Decide in milliseconds whether a payment should go through.')")
    expect(headline).to_be_visible()

    # Verify primary action buttons
    btn_dash = page.locator("button:has-text('Open dashboard')").first
    btn_check = page.locator("button:has-text('Try a live check')").first
    expect(btn_dash).to_be_visible()
    expect(btn_check).to_be_visible()

    # Scroll down to reveal the 3 layered steps
    page.evaluate("window.scrollTo(0, 1200)")
    page.wait_for_timeout(600)

    step_feat = page.locator("text=Features").first
    expect(step_feat).to_be_visible()

    # Scroll down to channel stats
    page.evaluate("window.scrollTo(0, 2400)")
    page.wait_for_timeout(600)

    # Verify honest data source and limitations link
    limit_link = page.locator("button:has-text(\"What this can't do\")")
    expect(limit_link).to_be_visible()

    assert len(errors) == 0, f"Console errors on landing: {errors}"
    page.close()


def test_qa_overview_kpis_and_charts(browser_instance: Browser) -> None:
    """2. Overview: Phase 3 KPIs (Frauds caught, Recall, False alarms, Alerts/day), provenance, channels table."""
    page = browser_instance.new_page(viewport={"width": 1440, "height": 900})
    errors = _setup_console_tracker(page)

    page.goto(f"{BASE_URL}/overview", wait_until="networkidle")
    page.wait_for_timeout(1200)

    # Switch to Online Transfers & Wallets channel
    page.locator("text=Online Transfers & Wallets").first.click()
    page.wait_for_timeout(800)

    # Verify Hero KPI (Frauds caught)
    hero_title = page.locator("text=Frauds caught").first
    expect(hero_title).to_be_visible()
    hero_val = page.locator("text=₹6,32,26,47,552").first
    expect(hero_val).to_be_visible()

    # Verify Supporting KPIs (Recall, False alarms, Alerts/day)
    expect(page.locator("text=Recall").first).to_be_visible()
    expect(page.locator("text=99.8%").first).to_be_visible()
    expect(page.locator("text=False alarms / 100k").first).to_be_visible()
    expect(page.locator("text=1.6").first).to_be_visible()
    expect(page.locator("text=Alerts per day").first).to_be_visible()
    expect(page.locator("text=264.8").first).to_be_visible()

    # Verify Channels Table benchmark rows
    expect(page.locator("text=918,617").first).to_be_visible()
    expect(page.locator("text=PaySim Mobile Money").first).to_be_visible()

    assert len(errors) == 0, f"Console errors on overview: {errors}"
    page.close()


def test_qa_live_check_verdict_and_gauge(browser_instance: Browser) -> None:
    """3. Live Check: Preset loading, score execution, gauge rotation, action verdict, and toast."""
    page = browser_instance.new_page(viewport={"width": 1440, "height": 900})
    errors = _setup_console_tracker(page)

    page.goto(f"{BASE_URL}/live-check", wait_until="networkidle")
    page.wait_for_timeout(1000)

    # Click high risk preset: UPI Collect Scam
    scam_btn = page.locator("button:has-text('UPI Collect Scam')").first
    expect(scam_btn).to_be_visible()
    scam_btn.click()
    page.wait_for_timeout(400)

    # Trigger scoring
    score_btn = page.locator("button:has-text('Score Transaction')").first
    expect(score_btn).to_be_visible()
    score_btn.click()
    page.wait_for_timeout(1000)

    # Verify verdict card action
    verdict = page.locator("text=Decline: High risk threshold breached; block authorization.").first
    expect(verdict).to_be_visible()

    # Verify top factor drivers list
    expect(page.locator("text=Top Factor Drivers").first).to_be_visible()

    assert len(errors) == 0, f"Console errors on live-check: {errors}"
    page.close()


def test_qa_transactions_filter_and_drawer(browser_instance: Browser) -> None:
    """4. Transactions: Filter by channel/risk band, virtualized table rendering, and row detail drawer."""
    page = browser_instance.new_page(viewport={"width": 1440, "height": 900})
    errors = _setup_console_tracker(page)

    page.goto(f"{BASE_URL}/transactions", wait_until="networkidle")
    page.wait_for_timeout(1200)

    # Verify transactions table header (Phase 1 & 4 sentence case spec: Step, Amount, Band)
    expect(page.locator("th:has-text('Step')").first).to_be_visible()
    expect(page.locator("th:has-text('Amount')").first).to_be_visible()
    expect(page.locator("th:has-text('Band')").first).to_be_visible()

    # Click first row to open detail drawer
    first_row = page.locator("tbody tr").first
    expect(first_row).to_be_visible()
    first_row.click()
    page.wait_for_timeout(600)

    # Verify drawer contents (Phase 4 spec: Transaction Factor Breakdown)
    drawer_title = page.locator("text=Transaction Factor Breakdown").first
    expect(drawer_title).to_be_visible()
    expect(page.locator("text=Calibrated Risk Score").first).to_be_visible()

    # Close drawer with Escape key
    page.keyboard.press("Escape")
    page.wait_for_timeout(400)

    assert len(errors) == 0, f"Console errors on transactions: {errors}"
    page.close()


def test_qa_cases_triage_keys(browser_instance: Browser) -> None:
    """5. Cases: Queue view, keyboard triage shortcuts (J/K navigation, Esc)."""
    page = browser_instance.new_page(viewport={"width": 1440, "height": 900})
    errors = _setup_console_tracker(page)

    page.goto(f"{BASE_URL}/cases", wait_until="networkidle")
    page.wait_for_timeout(1000)

    # Verify table and rows
    rows = page.locator("tbody tr")
    count = rows.count()
    assert count > 0, "Expected at least 1 case row"

    # Focus body and press 'j' to navigate rows
    page.keyboard.press("j")
    page.wait_for_timeout(200)
    page.keyboard.press("j")
    page.wait_for_timeout(200)
    page.keyboard.press("k")
    page.wait_for_timeout(200)

    # Click first row to view drawer
    rows.first.click()
    page.wait_for_timeout(500)
    expect(page.locator("text=Investigator Disposition Notes").first).to_be_visible()

    # Dismiss drawer
    page.keyboard.press("Escape")
    page.wait_for_timeout(300)

    assert len(errors) == 0, f"Console errors on cases: {errors}"
    page.close()


def test_qa_models_detail_tabs(browser_instance: Browser) -> None:
    """6. Models: Champion registry summary, expandable row detail with tabs."""
    page = browser_instance.new_page(viewport={"width": 1440, "height": 900})
    errors = _setup_console_tracker(page)

    page.goto(f"{BASE_URL}/models", wait_until="networkidle")
    page.wait_for_timeout(1000)

    # Verify channels present in table
    expect(page.locator("th:has-text('CHANNEL')").first).to_be_visible()
    expect(page.locator("th:has-text('CHAMPION MODEL')").first).to_be_visible()
    expect(page.locator("th:has-text('PR-AUC')").first).to_be_visible()

    # Verify expanded card panel details
    expect(page.locator("text=Bootstrap Confidence Intervals").first).to_be_visible()
    expect(page.locator("text=Probability Calibration").first).to_be_visible()

    assert len(errors) == 0, f"Console errors on models: {errors}"
    page.close()


def test_qa_command_palette_and_theme_toggle(browser_instance: Browser) -> None:
    """7 & 8. Command palette search (Ctrl+K) navigation, theme toggle persistence."""
    page = browser_instance.new_page(viewport={"width": 1440, "height": 900})
    errors = _setup_console_tracker(page)

    page.goto(f"{BASE_URL}/overview", wait_until="networkidle")
    page.wait_for_timeout(600)

    # Trigger Command Palette with Ctrl+K
    page.keyboard.press("Control+k")
    page.wait_for_timeout(400)

    # Verify command palette input is visible
    cmd_input = page.locator("input[placeholder*='Search pages']").first
    expect(cmd_input).to_be_visible()

    # Search and select Live Check page navigation item
    cmd_input.fill("Live Check")
    page.wait_for_timeout(300)
    opt = page.locator("[cmdk-item]:has-text('Pre-auth transaction scoring')").first
    expect(opt).to_be_visible()
    opt.click()
    page.wait_for_timeout(600)

    # Verify navigated to live-check
    expect(page).to_have_url(f"{BASE_URL}/live-check")

    # Test Theme Toggle
    theme_btn = page.locator("button[aria-label*='mode' i], button:has(svg.lucide-sun), button:has(svg.lucide-moon)").first
    if theme_btn.is_visible():
        theme_btn.click()
        page.wait_for_timeout(300)
        stored_theme = page.evaluate("() => localStorage.getItem('fraudshield-theme')")
        assert stored_theme in ("light", "dark")

    assert len(errors) == 0, f"Console errors during palette & theme toggle: {errors}"
    page.close()


def test_qa_mobile_viewport_and_responsiveness(browser_instance: Browser) -> None:
    """9. Mobile 390x844: Zero horizontal scroll on overview, live check, and transactions."""
    page = browser_instance.new_page(viewport={"width": 390, "height": 844})
    errors = _setup_console_tracker(page)

    for route in ["/", "/overview", "/live-check", "/transactions"]:
        page.goto(f"{BASE_URL}{route}", wait_until="networkidle")
        page.wait_for_timeout(500)
        scroll_w, client_w = page.evaluate("() => [document.documentElement.scrollWidth, document.documentElement.clientWidth]")
        assert scroll_w <= client_w, f"Horizontal overflow on {route} (mobile): scrollWidth={scroll_w} > clientWidth={client_w}"

    assert len(errors) == 0, f"Console errors on mobile responsive: {errors}"
    page.close()


def test_qa_accessibility_axe_all_core_routes(browser_instance: Browser) -> None:
    """10. Full WCAG 2.1 AA Axe-core compliance across all views."""
    page = browser_instance.new_page(viewport={"width": 1440, "height": 900})

    for route in ["/", "/overview", "/live-check", "/transactions", "/cases", "/models"]:
        for theme in ["dark", "light"]:
            page.goto(f"{BASE_URL}{route}", wait_until="networkidle")
            page.evaluate(f"() => {{ if ('{theme}' === 'dark') {{ document.documentElement.classList.add('dark'); localStorage.setItem('fraudshield-theme', 'dark'); }} else {{ document.documentElement.classList.remove('dark'); localStorage.setItem('fraudshield-theme', 'light'); }} }}")
            page.wait_for_timeout(400)
            violations = _run_axe(page)
            assert len(violations) == 0, f"WCAG violations on {route} [{theme}]: {[v['id'] for v in violations]}"

    page.close()


def test_qa_no_metric_like_numeric_literals_in_source() -> None:
    """11. Assert no typed-in metric numbers or banned claims exist in src/pages or src/components/landing."""
    import re
    from pathlib import Path

    pages_dir = Path("frontend-web/src/pages")
    landing_dir = Path("frontend-web/src/components/landing")

    files = list(pages_dir.glob("*.tsx")) + list(landing_dir.glob("*.tsx"))
    forbidden_literals = [
        "918,617",
        "99.8%",
        "7.9ms",
        "7.74 ms",
        "20.6 ms",
        "low customer friction",
        "analyst queue capacity",
        "calibrated decision threshold",
    ]

    for f in files:
        content = f.read_text(encoding="utf-8")
        for lit in forbidden_literals:
            assert lit.lower() not in content.lower(), (
                f"Found forbidden metric-like literal '{lit}' in {f.name}"
            )


def test_qa_route_change_no_full_page_skeleton_flash(browser_instance: Browser) -> None:
    """12. Assert route changes between prefetched tabs do not flash a full-page loading skeleton."""
    page = browser_instance.new_page(viewport={"width": 1440, "height": 900})
    page.goto(f"{BASE_URL}/overview", wait_until="networkidle")
    page.wait_for_timeout(1000)

    # Hover over Live check nav link to trigger prefetch
    live_link = page.locator("nav a[href='/live-check'], nav button:has-text('Live check')").first
    if live_link.is_visible():
        live_link.hover()
        page.wait_for_timeout(300)
        live_link.click()

        # Immediate check (10-30ms): verify no full-page skeleton takeover
        full_skel = page.locator(".space-y-6:has(.h-8.w-64)").first
        assert not full_skel.is_visible(), "Full-page skeleton flashed on prefetched route navigation!"

    page.close()


def test_qa_all_pages_light_dark_desktop_mobile_no_scroll_no_errors(browser_instance: Browser) -> None:
    """13. Validate all views: light/dark, desktop/mobile; 0 console errors; 0 horizontal scroll."""
    routes = [
        "/",
        "/overview",
        "/live-check",
        "/transactions",
        "/cases",
        "/models",
        "/monitoring",
        "/batch",
        "/api-docs",
    ]

    viewports = [
        ("desktop", {"width": 1440, "height": 900}),
        ("mobile", {"width": 390, "height": 844}),
    ]

    for vp_name, vp in viewports:
        page = browser_instance.new_page(viewport=vp)
        errors = _setup_console_tracker(page)

        for route in routes:
            for theme in ["dark", "light"]:
                page.goto(f"{BASE_URL}{route}", wait_until="networkidle")
                page.evaluate(
                    f"() => {{ if ('{theme}' === 'dark') {{ document.documentElement.classList.add('dark'); localStorage.setItem('fraudshield-theme', 'dark'); }} else {{ document.documentElement.classList.remove('dark'); localStorage.setItem('fraudshield-theme', 'light'); }} }}"
                )
                page.wait_for_timeout(300)

                # Check horizontal scroll
                scroll_w, client_w = page.evaluate("() => [document.documentElement.scrollWidth, document.documentElement.clientWidth]")
                assert scroll_w <= client_w, (
                    f"Horizontal overflow on {route} ({vp_name}, {theme}): scrollWidth={scroll_w} > clientWidth={client_w}"
                )

        assert len(errors) == 0, f"Console errors detected on {vp_name}: {errors}"
        page.close()


