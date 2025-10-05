"""
TradingView-style Chart Generator for NIFTY Options Data
Generates interactive charts with scrolling, panning, and zoom using TradingView Lightweight Charts
"""

import os
import glob
import pandas as pd
import json
from datetime import datetime
import chart_config as config


def load_data_files(data_dir, file_pattern):
    """Load all CSV files matching the pattern"""
    file_path = os.path.join(data_dir, file_pattern)
    files = sorted(glob.glob(file_path))

    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {file_path}")

    print(f"Found {len(files)} data files")
    return files


def read_and_combine_data(files):
    """Read and combine all CSV files"""
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined_df)} rows of data")
    return combined_df


def apply_filters(df, filter_config):
    """Apply filters based on configuration"""
    filtered_df = df.copy()

    if filter_config['strike'] is not None:
        filtered_df = filtered_df[filtered_df['strike'] == filter_config['strike']]
        print(f"Filtered by strike {filter_config['strike']}: {len(filtered_df)} rows")

    if filter_config['option_type'] is not None:
        filtered_df = filtered_df[filtered_df['option_type'] == filter_config['option_type']]
        print(f"Filtered by option_type {filter_config['option_type']}: {len(filtered_df)} rows")

    if filter_config['expiry_type'] is not None:
        filtered_df = filtered_df[filtered_df['expiry_type'] == filter_config['expiry_type']]
        print(f"Filtered by expiry_type {filter_config['expiry_type']}: {len(filtered_df)} rows")

    if filter_config['date_range']['start'] is not None:
        filtered_df = filtered_df[filtered_df['timestamp'] >= filter_config['date_range']['start']]

    if filter_config['date_range']['end'] is not None:
        filtered_df = filtered_df[filtered_df['timestamp'] <= filter_config['date_range']['end']]

    return filtered_df


def resample_to_hourly(df, resample_interval='1h'):
    """Resample 5-minute data to hourly candles"""
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'oi': 'last',
        'underlying_price': 'last',
        'vix': 'last'
    }

    # Group by symbol and strike to handle multiple options
    if 'strike' in df.columns and 'option_type' in df.columns:
        grouped = df.groupby(['symbol', 'strike', 'option_type'])
        resampled_dfs = []

        for name, group in grouped:
            resampled = group.resample(resample_interval).agg(ohlc_dict).dropna()
            resampled['strike'] = name[1]
            resampled['option_type'] = name[2]
            resampled_dfs.append(resampled)

        result = pd.concat(resampled_dfs)
    else:
        result = df.resample(resample_interval).agg(ohlc_dict).dropna()

    print(f"Resampled to {resample_interval} candles: {len(result)} rows")
    return result.reset_index()


def prepare_tradingview_data(df):
    """Convert DataFrame to TradingView format"""
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Convert to Unix timestamp (seconds)
    candlestick_data = []
    volume_data = []
    underlying_data = []
    vix_data = []

    for _, row in df.iterrows():
        # Skip rows with NaN values in OHLC
        if pd.isna(row['open']) or pd.isna(row['high']) or pd.isna(row['low']) or pd.isna(row['close']):
            continue

        ts = int(row['timestamp'].timestamp())

        # Candlestick data
        candlestick_data.append({
            'time': ts,
            'open': round(float(row['open']), 2),
            'high': round(float(row['high']), 2),
            'low': round(float(row['low']), 2),
            'close': round(float(row['close']), 2)
        })

        # Volume data
        volume_data.append({
            'time': ts,
            'value': int(row['volume']),
            'color': '#26a69a' if row['close'] >= row['open'] else '#ef5350'
        })

        # Underlying price data
        if pd.notna(row['underlying_price']):
            underlying_data.append({
                'time': ts,
                'value': round(float(row['underlying_price']), 2)
            })

        # VIX data
        if pd.notna(row['vix']):
            vix_data.append({
                'time': ts,
                'value': round(float(row['vix']), 2)
            })

    print(f"Prepared {len(candlestick_data)} candlesticks, {len(volume_data)} volume bars")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    if len(candlestick_data) > 0:
        print(f"First candle: {candlestick_data[0]}")
        print(f"Last candle: {candlestick_data[-1]}")

    return {
        'candlestick': candlestick_data,
        'volume': volume_data,
        'underlying': underlying_data,
        'vix': vix_data
    }


def create_html_chart(data, chart_config, filter_info):
    """Create HTML file with TradingView chart"""

    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background-color: #0a0e1a;
            color: #d1d4dc;
        }}
        .container {{
            padding: 20px;
            max-width: 100%;
        }}
        .header {{
            margin-bottom: 20px;
        }}
        .header h1 {{
            font-size: 24px;
            margin-bottom: 10px;
        }}
        .filter-info {{
            background: #131722;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }}
        .filter-item {{
            display: flex;
            flex-direction: column;
        }}
        .filter-label {{
            font-size: 12px;
            color: #787b86;
            margin-bottom: 4px;
        }}
        .filter-value {{
            font-size: 14px;
            font-weight: 600;
        }}
        #chart-container {{
            position: relative;
            background: #131722;
            border-radius: 8px;
            overflow: hidden;
        }}
        #main-chart {{
            width: 100%;
            height: 500px;
            min-height: 500px;
        }}
        #volume-chart {{
            width: 100%;
            height: 150px;
            min-height: 150px;
        }}
        #indicator-chart {{
            width: 100%;
            height: 150px;
            min-height: 150px;
        }}
        .controls {{
            margin-top: 20px;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }}
        .control-button {{
            background: #2962ff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.2s;
        }}
        .control-button:hover {{
            background: #1e53e5;
        }}
        .stats {{
            margin-top: 20px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .stat-card {{
            background: #131722;
            padding: 15px;
            border-radius: 8px;
        }}
        .stat-label {{
            font-size: 12px;
            color: #787b86;
            margin-bottom: 5px;
        }}
        .stat-value {{
            font-size: 20px;
            font-weight: 600;
        }}
        .legend {{
            display: flex;
            gap: 20px;
            margin-bottom: 10px;
            padding: 10px;
            background: #131722;
            border-radius: 4px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 14px;
        }}
        .legend-color {{
            width: 16px;
            height: 3px;
            border-radius: 2px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{title}</h1>
        </div>

        <div class="filter-info">
            {filter_info_html}
        </div>

        <div id="chart-container">
            <div id="main-chart"></div>
            {volume_chart}
            {indicator_chart}
        </div>

        <div class="legend">
            <div class="legend-item">
                <div class="legend-color" style="background: #2962ff;"></div>
                <span>Underlying Price</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #f23645;"></div>
                <span>VIX</span>
            </div>
        </div>

        <div class="controls">
            <button class="control-button" onclick="chart.timeScale().fitContent()">Fit Content</button>
            <button class="control-button" onclick="chart.timeScale().scrollToRealTime()">Scroll to End</button>
            <button class="control-button" onclick="resetZoom()">Reset Zoom</button>
        </div>

        <div class="stats" id="stats">
            <!-- Stats will be populated by JavaScript -->
        </div>
    </div>

    <script>
        // Chart data
        const candlestickData = {candlestick_data};
        const volumeData = {volume_data};
        const underlyingData = {underlying_data};
        const vixData = {vix_data};

        // Main initialization function
        function initChart() {{
            // Debug: Log data
            console.log('Candlestick data points:', candlestickData.length);
            console.log('Volume data points:', volumeData.length);
            if (candlestickData.length > 0) {{
                console.log('First candlestick:', candlestickData[0]);
                console.log('Last candlestick:', candlestickData[candlestickData.length - 1]);
            }}

            // Validate TradingView library
            if (typeof LightweightCharts === 'undefined') {{
                console.error('TradingView Lightweight Charts library not loaded!');
                alert('Error: Chart library failed to load. Check your internet connection.');
                return;
            }}

            // Validate chart container
            const chartContainer = document.getElementById('main-chart');
            if (!chartContainer) {{
                console.error('Chart container not found!');
                return;
            }}
            console.log('Chart container dimensions:', chartContainer.offsetWidth, 'x', chartContainer.offsetHeight);

            try {{
                // Create main chart
                const chart = LightweightCharts.createChart(chartContainer, {{
            layout: {{
                background: {{ color: '#131722' }},
                textColor: '#d1d4dc',
            }},
            grid: {{
                vertLines: {{ color: '#1e222d' }},
                horzLines: {{ color: '#1e222d' }},
            }},
            crosshair: {{
                mode: LightweightCharts.CrosshairMode.Normal,
            }},
            timeScale: {{
                borderColor: '#2B2B43',
                timeVisible: true,
                secondsVisible: false,
            }},
            rightPriceScale: {{
                borderColor: '#2B2B43',
            }},
        }});

                // Add candlestick series (v5 API)
                const candlestickSeries = chart.addSeries(LightweightCharts.CandlestickSeries, {{
                    upColor: '#26a69a',
                    downColor: '#ef5350',
                    borderVisible: false,
                    wickUpColor: '#26a69a',
                    wickDownColor: '#ef5350',
                }});

                if (candlestickData.length > 0) {{
                    candlestickSeries.setData(candlestickData);
                    console.log('Candlestick data set successfully');
                }} else {{
                    console.error('No candlestick data available!');
                }}

                // Add underlying price line (on same chart)
                const underlyingLine = chart.addSeries(LightweightCharts.LineSeries, {{
                    color: '#2962ff',
                    lineWidth: 2,
                    priceScaleId: 'left',
                }});
                underlyingLine.setData(underlyingData);

            {volume_chart_js}
            {indicator_chart_js}

        // Sync crosshair across charts
        function syncCrosshair(chart, series, point) {{
            if (point) {{
                chart.setCrosshairPosition(point.value, point.time, series);
            }} else {{
                chart.clearCrosshairPosition();
            }}
        }}

        // Reset zoom function
        function resetZoom() {{
            chart.timeScale().fitContent();
        }}

        // Calculate and display statistics
        function calculateStats() {{
            if (candlestickData.length === 0) return;

            const prices = candlestickData.map(d => d.close);
            const high = Math.max(...prices);
            const low = Math.min(...prices);
            const current = prices[prices.length - 1];
            const open = candlestickData[0].open;
            const change = ((current - open) / open * 100).toFixed(2);

            const statsHtml = `
                <div class="stat-card">
                    <div class="stat-label">Current Price</div>
                    <div class="stat-value">₹${{current.toFixed(2)}}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">High</div>
                    <div class="stat-value">₹${{high.toFixed(2)}}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Low</div>
                    <div class="stat-value">₹${{low.toFixed(2)}}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Change</div>
                    <div class="stat-value" style="color: ${{change >= 0 ? '#26a69a' : '#ef5350'}}">${{change}}%</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Total Candles</div>
                    <div class="stat-value">${{candlestickData.length}}</div>
                </div>
            `;

            document.getElementById('stats').innerHTML = statsHtml;
        }}

        calculateStats();

        // Fit content to show all data
        setTimeout(() => {{
            chart.timeScale().fitContent();
            console.log('Chart fitted to content');
        }}, 100);

                // Responsive chart
                window.addEventListener('resize', () => {{
                    chart.applyOptions({{ width: document.getElementById('main-chart').clientWidth }});
                    {resize_volume_chart}
                    {resize_indicator_chart}
                }});

            }} catch (error) {{
                console.error('Error creating chart:', error);
                alert('Error creating chart: ' + error.message);
                document.getElementById('chart-container').innerHTML =
                    '<div style="padding: 40px; text-align: center; color: #f23645;">' +
                    '<h2>Chart Error</h2>' +
                    '<p>' + error.message + '</p>' +
                    '<p>Check browser console for details (F12)</p>' +
                    '</div>';
            }}
        }}

        // Initialize chart when page loads
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', initChart);
        }} else {{
            initChart();
        }}
    </script>
</body>
</html>"""

    # Prepare filter info HTML
    filter_info_items = []
    if filter_info.get('strike'):
        filter_info_items.append(f'<div class="filter-item"><div class="filter-label">Strike</div><div class="filter-value">{filter_info["strike"]}</div></div>')
    if filter_info.get('option_type'):
        filter_info_items.append(f'<div class="filter-item"><div class="filter-label">Type</div><div class="filter-value">{filter_info["option_type"]}</div></div>')
    if filter_info.get('expiry_type'):
        filter_info_items.append(f'<div class="filter-item"><div class="filter-label">Expiry</div><div class="filter-value">{filter_info["expiry_type"]}</div></div>')
    filter_info_items.append(f'<div class="filter-item"><div class="filter-label">Interval</div><div class="filter-value">{chart_config["resample_interval"]}</div></div>')

    # Prepare volume chart HTML and JS if enabled
    volume_chart_html = '<div id="volume-chart"></div>' if chart_config['show_volume'] else ''
    volume_chart_js = ''
    resize_volume_chart = ''

    if chart_config['show_volume']:
        volume_chart_js = """
            // Create volume chart
            const volumeChart = LightweightCharts.createChart(document.getElementById('volume-chart'), {
            layout: {
                background: { color: '#131722' },
                textColor: '#d1d4dc',
            },
            grid: {
                vertLines: { color: '#1e222d' },
                horzLines: { color: '#1e222d' },
            },
            timeScale: {
                borderColor: '#2B2B43',
                timeVisible: true,
                secondsVisible: false,
            },
            rightPriceScale: {
                borderColor: '#2B2B43',
            },
        });

            const volumeSeries = volumeChart.addSeries(LightweightCharts.HistogramSeries, {
                color: '#26a69a',
                priceFormat: {
                    type: 'volume',
                },
                priceScaleId: '',
            });
            volumeSeries.setData(volumeData);

        // Sync time scales
        chart.timeScale().subscribeVisibleLogicalRangeChange(timeRange => {
            volumeChart.timeScale().setVisibleLogicalRange(timeRange);
        });
        volumeChart.timeScale().subscribeVisibleLogicalRangeChange(timeRange => {
            chart.timeScale().setVisibleLogicalRange(timeRange);
        });
        """
        resize_volume_chart = 'volumeChart.applyOptions({ width: document.getElementById("volume-chart").clientWidth });'

    # Prepare indicator chart HTML and JS if enabled
    indicator_chart_html = '<div id="indicator-chart"></div>' if (chart_config['show_underlying'] or chart_config['show_vix']) else ''
    indicator_chart_js = ''
    resize_indicator_chart = ''

    if chart_config['show_vix']:
        indicator_chart_js = """
            // Create VIX chart
            const indicatorChart = LightweightCharts.createChart(document.getElementById('indicator-chart'), {
            layout: {
                background: { color: '#131722' },
                textColor: '#d1d4dc',
            },
            grid: {
                vertLines: { color: '#1e222d' },
                horzLines: { color: '#1e222d' },
            },
            timeScale: {
                borderColor: '#2B2B43',
                timeVisible: true,
                secondsVisible: false,
            },
            rightPriceScale: {
                borderColor: '#2B2B43',
            },
        });

            const vixSeries = indicatorChart.addSeries(LightweightCharts.LineSeries, {
                color: '#f23645',
                lineWidth: 2,
            });
            vixSeries.setData(vixData);

        // Sync time scales
        chart.timeScale().subscribeVisibleLogicalRangeChange(timeRange => {
            indicatorChart.timeScale().setVisibleLogicalRange(timeRange);
        });
        indicatorChart.timeScale().subscribeVisibleLogicalRangeChange(timeRange => {
            chart.timeScale().setVisibleLogicalRange(timeRange);
        });
        """
        resize_indicator_chart = 'indicatorChart.applyOptions({ width: document.getElementById("indicator-chart").clientWidth });'

    html_content = html_template.format(
        title=chart_config['chart_title'],
        filter_info_html=''.join(filter_info_items),
        candlestick_data=json.dumps(data['candlestick']),
        volume_data=json.dumps(data['volume']),
        underlying_data=json.dumps(data['underlying']),
        vix_data=json.dumps(data['vix']),
        volume_chart=volume_chart_html,
        volume_chart_js=volume_chart_js,
        resize_volume_chart=resize_volume_chart,
        indicator_chart=indicator_chart_html,
        indicator_chart_js=indicator_chart_js,
        resize_indicator_chart=resize_indicator_chart
    )

    return html_content


def save_chart(html_content, output_config, filename_prefix='tradingview_chart'):
    """Save chart to HTML file"""
    os.makedirs(output_config['output_directory'], exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    html_file = os.path.join(
        output_config['output_directory'],
        f"{filename_prefix}_{timestamp}.html"
    )

    with open(html_file, 'w') as f:
        f.write(html_content)

    print(f"Chart saved to: {html_file}")

    if output_config['auto_open']:
        import webbrowser
        webbrowser.open('file://' + os.path.abspath(html_file))

    return html_file


def main():
    """Main execution function"""
    print("=" * 60)
    print("TradingView Chart Generator")
    print("=" * 60)

    # Load data files
    files = load_data_files(
        config.DATA_CONFIG['data_directory'],
        config.DATA_CONFIG['file_pattern']
    )

    # Read and combine data
    df = read_and_combine_data(files)

    # Apply filters
    df_filtered = apply_filters(df, config.FILTER_CONFIG)

    if len(df_filtered) == 0:
        print("No data after applying filters!")
        return

    # Store filter info for display
    filter_info = {
        'strike': config.FILTER_CONFIG['strike'],
        'option_type': config.FILTER_CONFIG['option_type'],
        'expiry_type': config.FILTER_CONFIG['expiry_type']
    }

    # Resample to hourly
    df_hourly = resample_to_hourly(
        df_filtered,
        config.CHART_CONFIG['resample_interval']
    )

    # Prepare data for TradingView
    print("\nPreparing chart data...")
    chart_data = prepare_tradingview_data(df_hourly)

    # Create HTML chart
    print("Generating TradingView chart...")
    html_content = create_html_chart(chart_data, config.CHART_CONFIG, filter_info)

    # Save chart
    save_chart(html_content, config.OUTPUT_CONFIG)

    print("\n" + "=" * 60)
    print("Chart generation complete!")
    print("Features: Scrolling, Panning, Zoom, Time Navigation")
    print("=" * 60)


if __name__ == '__main__':
    main()
