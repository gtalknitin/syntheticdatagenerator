"""
Configuration for OHLC Chart Generation
"""

import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'hourly_v9_balanced')  # V9 balanced data (1-hour candles)
OUTPUT_DIR = os.path.join(BASE_DIR, 'src', 'charts', 'output')

# Data source configuration
DATA_CONFIG = {
    'data_directory': DATA_DIR,
    'file_pattern': 'NIFTY_OPTIONS_1H_*.csv',  # V9 1-hour data
    'date_format': '%Y-%m-%d %H:%M:%S'
}

# Chart configuration
CHART_CONFIG = {
    'resample_interval': '1h',  # Already 1-hour data, no resampling needed
    'chart_title': 'NIFTY Options - V9 Balanced Data (1H OHLC)',
    'width': 1400,
    'height': 800,
    'show_volume': True,
    'show_underlying': True,
    'show_vix': True
}

# Filter configuration (optional)
FILTER_CONFIG = {
    'strike': 25400,  # V9 data has strikes from 22,000 to 30,000
    'option_type': 'CE',  # None for all, 'CE' for calls, 'PE' for puts
    'expiry_type': 'weekly',  # None for all, 'weekly' or 'monthly'
    'date_range': {
        'start': '2025-06-09',  # V9 data: June 9 - September 26, 2025
        'end': '2025-09-26'     # None or 'YYYY-MM-DD'
    }
}

# Output configuration
OUTPUT_CONFIG = {
    'output_directory': OUTPUT_DIR,
    'save_html': True,
    'save_png': False,  # Requires kaleido package
    'auto_open': True
}
