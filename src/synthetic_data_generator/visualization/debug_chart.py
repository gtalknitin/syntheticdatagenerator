#!/usr/bin/env python3
"""
Debug chart with Playwright - capture console errors
"""

from playwright.sync_api import sync_playwright
import time

def debug_chart():
    chart_path = '/Users/nitindhawan/NikAlgoBulls/zerodha_strategy/data/synthetic/src/charts/output/tradingview_chart_20251004_181150.html'

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        console_logs = []
        errors = []

        # Capture all console messages
        page.on('console', lambda msg: console_logs.append(f'[{msg.type}] {msg.text}'))
        page.on('pageerror', lambda err: errors.append(f'ERROR: {err}'))

        print("Loading chart...")
        page.goto(f'file://{chart_path}')

        # Wait for page to load
        time.sleep(3)

        print("\n" + "="*70)
        print("CONSOLE LOGS:")
        print("="*70)
        for log in console_logs:
            print(log)

        if errors:
            print("\n" + "="*70)
            print("JAVASCRIPT ERRORS:")
            print("="*70)
            for error in errors:
                print(error)

        # Check for canvas elements
        canvas_count = page.locator('canvas').count()
        print("\n" + "="*70)
        print(f"Canvas elements found: {canvas_count}")
        print("="*70)

        # Check what's available in LightweightCharts object
        result = page.evaluate("""
            () => {
                const lwc = window.LightweightCharts || {};
                return {
                    hasCreateChart: typeof createChart !== 'undefined',
                    hasLightweightCharts: typeof LightweightCharts !== 'undefined',
                    lightweightChartsKeys: Object.keys(lwc).sort(),
                    hasCandlestickSeries: typeof lwc.CandlestickSeries !== 'undefined',
                    candlestickCount: typeof candlestickData !== 'undefined' ? candlestickData.length : 0,
                    volumeCount: typeof volumeData !== 'undefined' ? volumeData.length : 0
                }
            }
        """)

        print("\nJavaScript Environment Check:")
        print(f"  createChart function: {result['hasCreateChart']}")
        print(f"  LightweightCharts object: {result['hasLightweightCharts']}")
        print(f"  CandlestickSeries available: {result['hasCandlestickSeries']}")
        print(f"  LightweightCharts exports: {', '.join(result['lightweightChartsKeys'][:20])}")
        print(f"  Candlestick data points: {result['candlestickCount']}")
        print(f"  Volume data points: {result['volumeCount']}")

        # Take screenshot
        screenshot_path = '/Users/nitindhawan/NikAlgoBulls/zerodha_strategy/data/synthetic/src/charts/debug_screenshot.png'
        page.screenshot(path=screenshot_path, full_page=True)
        print(f"\nScreenshot saved: {screenshot_path}")

        print("\nKeeping browser open for 30 seconds...")
        time.sleep(30)
        browser.close()

if __name__ == '__main__':
    debug_chart()
