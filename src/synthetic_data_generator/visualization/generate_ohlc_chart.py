"""
OHLC Candle Chart Generator for NIFTY Options Data
Generates 1-hour candle charts from 5-minute intraday data
"""

import os
import glob
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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


def resample_to_hourly(df, resample_interval='1H'):
    """Resample 5-minute data to hourly candles"""
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # For OHLC data
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


def create_ohlc_chart(df, chart_config):
    """Create interactive OHLC candlestick chart"""

    # Determine number of subplots
    num_subplots = 1
    subplot_titles = [chart_config['chart_title']]
    row_heights = [0.6]

    if chart_config['show_volume']:
        num_subplots += 1
        subplot_titles.append('Volume')
        row_heights.append(0.2)

    if chart_config['show_underlying'] or chart_config['show_vix']:
        num_subplots += 1
        subplot_titles.append('Underlying Price & VIX')
        row_heights.append(0.2)

    # Create subplots
    fig = make_subplots(
        rows=num_subplots,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=subplot_titles,
        row_heights=row_heights,
        specs=[[{"secondary_y": False}]] * num_subplots
    )

    # Main OHLC candlestick chart
    candlestick = go.Candlestick(
        x=df['timestamp'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='OHLC',
        increasing_line_color='green',
        decreasing_line_color='red'
    )
    fig.add_trace(candlestick, row=1, col=1)

    current_row = 2

    # Add volume chart
    if chart_config['show_volume']:
        colors = ['green' if close >= open else 'red'
                  for close, open in zip(df['close'], df['open'])]
        volume_bar = go.Bar(
            x=df['timestamp'],
            y=df['volume'],
            name='Volume',
            marker_color=colors,
            showlegend=False
        )
        fig.add_trace(volume_bar, row=current_row, col=1)
        current_row += 1

    # Add underlying price and VIX
    if chart_config['show_underlying'] or chart_config['show_vix']:
        if chart_config['show_underlying']:
            underlying = go.Scatter(
                x=df['timestamp'],
                y=df['underlying_price'],
                name='Underlying Price',
                line=dict(color='blue', width=2),
                yaxis='y3'
            )
            fig.add_trace(underlying, row=current_row, col=1)

        if chart_config['show_vix']:
            vix = go.Scatter(
                x=df['timestamp'],
                y=df['vix'],
                name='VIX',
                line=dict(color='orange', width=2),
                yaxis='y4'
            )
            fig.add_trace(vix, row=current_row, col=1)

    # Update layout
    fig.update_layout(
        title=chart_config['chart_title'],
        xaxis_title='Date',
        yaxis_title='Price',
        width=chart_config['width'],
        height=chart_config['height'],
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        template='plotly_dark'
    )

    # Update x-axes
    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]),  # Hide weekends
        ]
    )

    return fig


def save_chart(fig, output_config, filename_prefix='ohlc_chart'):
    """Save chart to file"""
    os.makedirs(output_config['output_directory'], exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if output_config['save_html']:
        html_file = os.path.join(
            output_config['output_directory'],
            f"{filename_prefix}_{timestamp}.html"
        )
        fig.write_html(html_file)
        print(f"Chart saved to: {html_file}")

        if output_config['auto_open']:
            import webbrowser
            webbrowser.open('file://' + os.path.abspath(html_file))

    if output_config['save_png']:
        png_file = os.path.join(
            output_config['output_directory'],
            f"{filename_prefix}_{timestamp}.png"
        )
        try:
            fig.write_image(png_file)
            print(f"PNG saved to: {png_file}")
        except Exception as e:
            print(f"Could not save PNG (install kaleido package): {e}")


def main():
    """Main execution function"""
    print("=" * 60)
    print("OHLC Chart Generator")
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

    # Resample to hourly
    df_hourly = resample_to_hourly(
        df_filtered,
        config.CHART_CONFIG['resample_interval']
    )

    # Create chart
    print("\nGenerating chart...")
    fig = create_ohlc_chart(df_hourly, config.CHART_CONFIG)

    # Save chart
    save_chart(fig, config.OUTPUT_CONFIG)

    print("\n" + "=" * 60)
    print("Chart generation complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
