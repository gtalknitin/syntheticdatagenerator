"""
Test latest generated chart with Playwright
"""

from playwright.sync_api import sync_playwright
import time
import glob
import os

def test_latest_chart():
    # Find latest chart file
    chart_files = glob.glob('/Users/nitindhawan/NikAlgoBulls/zerodha_strategy/data/synthetic/src/charts/output/tradingview_chart_*.html')
    if not chart_files:
        print("No chart files found!")
        return

    latest_chart = max(chart_files, key=os.path.getctime)
    print(f"Testing chart: {latest_chart}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        # Capture console messages
        console_messages = []
        page.on('console', lambda msg: console_messages.append(f'{msg.type}: {msg.text}'))
        page.on('pageerror', lambda err: print(f'PAGE ERROR: {err}'))

        # Load the chart
        page.goto(f'file://{latest_chart}')

        # Wait for chart to initialize
        time.sleep(3)

        # Print console messages
        print("\n" + "="*60)
        print("Console Output:")
        print("="*60)
        for msg in console_messages:
            print(msg)
        print("="*60)

        # Take screenshot
        screenshot_path = '/Users/nitindhawan/NikAlgoBulls/zerodha_strategy/data/synthetic/src/charts/test_screenshot.png'
        page.screenshot(path=screenshot_path, full_page=True)
        print(f"\nScreenshot saved to: {screenshot_path}")

        # Check if chart rendered
        try:
            # Check for canvas element (charts render to canvas)
            canvas_count = page.locator('canvas').count()
            print(f"\nCanvas elements found: {canvas_count}")

            if canvas_count > 0:
                print("✅ Chart appears to be rendering!")
            else:
                print("❌ No canvas elements found - chart may not be rendering")

        except Exception as e:
            print(f"Error checking chart: {e}")

        print("\nKeeping browser open for 10 seconds for visual inspection...")
        time.sleep(10)
        browser.close()

if __name__ == '__main__':
    test_latest_chart()
