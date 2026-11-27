from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.goto("http://localhost:8000/overview", wait_until="networkidle")
    page.evaluate("document.documentElement.classList.add('dark')")
    page.wait_for_timeout(600)
    page.click("button:has-text('Search')")
    page.wait_for_timeout(600)
    page.screenshot(path="reports/figures/screens/phase1_04_command_palette_cmdk.png")
    browser.close()
    print("Updated phase1_04_command_palette_cmdk.png successfully.")

