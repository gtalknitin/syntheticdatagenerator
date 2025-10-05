"""
Test TradingView chart with Playwright
"""

from playwright.sync_api import sync_playwright
import os
import time

def test_chart():
    chart_file = os.path.abspath('/Users/nitindhawan/NikAlgoBulls/zerodha_strategy/data/synthetic/src/charts/test_tv_api.html')

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        # Enable console logging
        page.on('console', lambda msg: print(f'CONSOLE: {msg.text}'))
        page.on('pageerror', lambda err: print(f'PAGE ERROR: {err}'))

        # Load the chart
        page.goto(f'file://{chart_file}')

        # Wait for chart to load
        time.sleep(2)

        # Get info div content
        info_content = page.locator('#info').inner_html()
        print("\n" + "="*60)
        print("Chart Test Results:")
        print("="*60)
        print(info_content.replace('<br>', '\n').replace('<span style="color: #26a69a;">', '').replace('<span style="color: #f23645;">', '').replace('</span>', ''))
        print("="*60)

        # Take screenshot
        screenshot_path = '/Users/nitindhawan/NikAlgoBulls/zerodha_strategy/data/synthetic/src/charts/test_result.png'
        page.screenshot(path=screenshot_path)
        print(f"\nScreenshot saved to: {screenshot_path}")

        time.sleep(5)
        browser.close()

if __name__ == '__main__':
    test_chart()
