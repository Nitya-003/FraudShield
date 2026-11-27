import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

AXE_PATH = Path("frontend-web/node_modules/axe-core/axe.min.js").resolve()
BASE_URL = "http://localhost:8000"

VIEWPORTS = [
    ("desktop_wide", 1440, 900),
    ("desktop_std", 1024, 768),
    ("tablet", 768, 1024),
    ("mobile", 390, 844),
]

ROUTES = [
    ("/", "Landing"),
    ("/overview", "Overview"),
    ("/live-check", "Live Check"),
    ("/transactions", "Transactions"),
    ("/cases", "Cases"),
    ("/models", "Models"),
    ("/attack-lab", "Attack Lab"),
    ("/entity", "Entity View"),
    ("/batch-scoring", "Batch Scoring"),
    ("/monitoring", "Monitoring"),
    ("/api-docs", "API Docs"),
]

def audit():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context()
        page = ctx.new_page()

        console_issues = []
        page.on("console", lambda msg: console_issues.append(f"[{msg.type}] {msg.text}") if msg.type in ("error", "warning") and "favicon" not in msg.text.lower() else None)
        page.on("pageerror", lambda err: console_issues.append(f"[pageerror] {err}"))

        print("=== 1. AUDITING RESPONSIVE HORIZONTAL OVERFLOW ===")
        overflow_issues = []
        for vp_name, width, height in VIEWPORTS:
            page.set_viewport_size({"width": width, "height": height})
            for route, name in ROUTES:
                page.goto(f"{BASE_URL}{route}", wait_until="networkidle")
                page.wait_for_timeout(400)
                scroll_w, client_w = page.evaluate("() => [document.documentElement.scrollWidth, document.documentElement.clientWidth]")
                if scroll_w > client_w:
                    overflow_issues.append(f"{name} ({route}) at {vp_name} ({width}x{height}): scrollWidth={scroll_w} > clientWidth={client_w}")
                else:
                    print(f"  OK: {name} at {vp_name} ({width}x{height})")

        print("\n=== 2. AUDITING ACCESSIBILITY (AXE-CORE WCAG AA) ===")
        axe_violations = []
        page.set_viewport_size({"width": 1440, "height": 900})
        for theme in ["dark", "light"]:
            if theme == "dark":
                page.evaluate("() => { document.documentElement.classList.add('dark'); localStorage.setItem('fraudshield-theme', 'dark'); }")
            else:
                page.evaluate("() => { document.documentElement.classList.remove('dark'); localStorage.setItem('fraudshield-theme', 'light'); }")

            for route, name in ROUTES:
                page.goto(f"{BASE_URL}{route}", wait_until="networkidle")
                page.wait_for_timeout(500)
                page.add_script_tag(path=str(AXE_PATH))
                res = page.evaluate("""async () => {
                    return await axe.run({
                        runOnly: { type: 'tag', values: ['wcag2a', 'wcag2aa'] }
                    });
                }""")
                violations = res.get("violations", [])
                if violations:
                    for v in violations:
                        axe_violations.append(f"[{theme}] {name} ({route}): {v['id']} - {v['description']} (impact: {v['impact']})")
                else:
                    print(f"  OK: {name} [{theme}] - 0 violations")

        browser.close()

        print("\n=== AUDIT SUMMARY ===")
        print(f"Console Issues: {len(console_issues)}")
        for ci in console_issues[:10]:
            print(f"  {ci}")

        print(f"Overflow Issues: {len(overflow_issues)}")
        for oi in overflow_issues:
            print(f"  {oi}")

        print(f"Axe Violations: {len(axe_violations)}")
        for av in axe_violations:
            print(f"  {av}")

if __name__ == "__main__":
    audit()

