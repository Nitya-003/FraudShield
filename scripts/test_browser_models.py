from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.on('console', lambda msg: print('BROWSER CONSOLE:', msg.type, msg.text))
    page.on('pageerror', lambda err: print('BROWSER ERROR:', err))
    page.goto('http://127.0.0.1:8000')
    page.wait_for_timeout(1000)
    page.get_by_role('button', name='Models').click()
    page.wait_for_timeout(2000)

