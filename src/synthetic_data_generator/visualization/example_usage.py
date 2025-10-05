"""
Example Usage: How to customize chart generation
"""

import chart_config as config

# Example 1: Generate chart for all data with default settings
# Just run: python generate_ohlc_chart.py


# Example 2: Filter by specific strike and option type
def filter_by_strike_and_type():
    config.FILTER_CONFIG['strike'] = 24150
    config.FILTER_CONFIG['option_type'] = 'CE'  # Call options only
    config.FILTER_CONFIG['expiry_type'] = 'weekly'


# Example 3: Filter by date range
def filter_by_date_range():
    config.FILTER_CONFIG['date_range']['start'] = '2025-07-01'
    config.FILTER_CONFIG['date_range']['end'] = '2025-07-31'


# Example 4: Change candle interval
def change_candle_interval():
    config.CHART_CONFIG['resample_interval'] = '30T'  # 30-minute candles
    # Other options: '15T', '1H', '2H', '1D'


# Example 5: Customize chart appearance
def customize_chart():
    config.CHART_CONFIG['chart_title'] = 'NIFTY 24150 CE - Hourly Analysis'
    config.CHART_CONFIG['width'] = 1920
    config.CHART_CONFIG['height'] = 1080
    config.CHART_CONFIG['show_volume'] = True
    config.CHART_CONFIG['show_underlying'] = True
    config.CHART_CONFIG['show_vix'] = True


# Example 6: Change output settings
def change_output_settings():
    config.OUTPUT_CONFIG['save_html'] = True
    config.OUTPUT_CONFIG['save_png'] = True  # Requires: pip install kaleido
    config.OUTPUT_CONFIG['auto_open'] = True


if __name__ == '__main__':
    # Uncomment the configuration you want to apply
    # filter_by_strike_and_type()
    # filter_by_date_range()
    # change_candle_interval()
    # customize_chart()
    # change_output_settings()

    # Then run the main chart generator
    from generate_ohlc_chart import main
    main()
