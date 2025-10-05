#!/usr/bin/env python3
"""
Create OHLC candlestick chart for underlying price movement
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import glob

# Read all data files
data_dir = Path("/Users/nitindhawan/NikAlgoBulls/zerodha_strategy/data/synthetic/intraday_v8_extended")
csv_files = sorted(glob.glob(str(data_dir / "NIFTY_OPTIONS_5MIN_*.csv")))

print(f"Processing {len(csv_files)} files...")

# Extract underlying prices aggregated to 1-hour candles
hourly_data = []

for file in csv_files:
    df = pd.read_csv(file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Get unique underlying prices per timestamp
    underlying = df.groupby('timestamp')['underlying_price'].first().reset_index()
    underlying.set_index('timestamp', inplace=True)

    # Resample to 1-hour candles
    hourly = underlying.resample('1H').agg({
        'underlying_price': ['first', 'max', 'min', 'last']
    })
    hourly.columns = ['open', 'high', 'low', 'close']
    hourly = hourly.dropna()

    hourly_data.append(hourly)

# Combine all data
all_hourly = pd.concat(hourly_data)
all_hourly = all_hourly.sort_index()

print(f"Generated {len(all_hourly)} hourly candles")

# Create candlestick chart
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10),
                                gridspec_kw={'height_ratios': [3, 1]})

# Plot candlesticks
width = 0.0007  # Width of candlestick in days
for idx, (timestamp, row) in enumerate(all_hourly.iterrows()):
    color = 'green' if row['close'] >= row['open'] else 'red'

    # Draw the wick (high-low line)
    ax1.plot([idx, idx], [row['low'], row['high']],
             color='black', linewidth=0.5, zorder=1)

    # Draw the body (open-close rectangle)
    height = abs(row['close'] - row['open'])
    bottom = min(row['open'], row['close'])

    rect = Rectangle((idx - width*500, bottom), width*1000, height,
                     facecolor=color, edgecolor='black', linewidth=0.5, zorder=2)
    ax1.add_patch(rect)

# Format chart
ax1.set_ylabel('NIFTY Price', fontsize=12, fontweight='bold')
ax1.set_title('NIFTY Underlying Price Movement (1-Hour Candles)\nJune 16 - Sep 30, 2025',
              fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-1, len(all_hourly))

# Set x-axis labels (show dates)
step = max(1, len(all_hourly) // 20)  # Show ~20 labels
tick_positions = range(0, len(all_hourly), step)
tick_labels = [all_hourly.index[i].strftime('%m/%d') for i in tick_positions]
ax1.set_xticks(tick_positions)
ax1.set_xticklabels(tick_labels, rotation=45)

# Calculate and plot daily returns
daily_close = all_hourly.resample('D').last()['close']
daily_returns = daily_close.pct_change() * 100  # Percentage

# Plot returns as bar chart
colors = ['green' if x >= 0 else 'red' for x in daily_returns]
x_positions = [all_hourly.index.get_loc(date, method='nearest')
               for date in daily_returns.index if date in all_hourly.index]

# Create bar positions aligned with candlesticks
bar_positions = []
bar_values = []
for i, (date, ret) in enumerate(daily_returns.items()):
    if not pd.isna(ret):
        # Find closest hourly candle
        closest_idx = np.argmin(np.abs(all_hourly.index - date))
        bar_positions.append(closest_idx)
        bar_values.append(ret)

ax2.bar(bar_positions, bar_values,
        color=['green' if x >= 0 else 'red' for x in bar_values],
        alpha=0.6, width=10)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax2.set_ylabel('Daily Return (%)', fontsize=10, fontweight='bold')
ax2.set_xlabel('Date', fontsize=10, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_xlim(-1, len(all_hourly))
ax2.set_xticks(tick_positions)
ax2.set_xticklabels(tick_labels, rotation=45)

# Add statistics
total_return = ((all_hourly['close'].iloc[-1] - all_hourly['close'].iloc[0]) /
                all_hourly['close'].iloc[0] * 100)
avg_daily_return = daily_returns.mean()
volatility = daily_returns.std()

stats_text = f"Total Return: {total_return:+.2f}%  |  " \
             f"Avg Daily: {avg_daily_return:+.3f}%  |  " \
             f"Volatility: {volatility:.2f}%"
fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout(rect=[0, 0.03, 1, 1])

# Save chart
output_file = data_dir / 'underlying_price_chart.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✓ Chart saved: {output_file}")

# Also create trend analysis
print("\n" + "="*80)
print("TREND ANALYSIS")
print("="*80)

# Weekly trend
weekly = all_hourly.resample('W').agg({
    'open': 'first',
    'close': 'last'
})
weekly['change'] = weekly['close'] - weekly['open']
weekly['change_pct'] = (weekly['change'] / weekly['open']) * 100

print("\nWeekly Movement:")
for date, row in weekly.iterrows():
    direction = "📈 UP" if row['change'] > 0 else "📉 DOWN" if row['change'] < 0 else "➡️  FLAT"
    print(f"  {date.strftime('%Y-%m-%d')}: {direction:8} {row['change']:+7.2f} pts ({row['change_pct']:+.2f}%)")

# Count up vs down weeks
up_weeks = (weekly['change'] > 0).sum()
down_weeks = (weekly['change'] < 0).sum()
print(f"\n  Up weeks: {up_weeks} ({up_weeks/len(weekly)*100:.1f}%)")
print(f"  Down weeks: {down_weeks} ({down_weeks/len(weekly)*100:.1f}%)")

# Monthly summary
monthly = all_hourly.resample('M').agg({
    'open': 'first',
    'close': 'last',
    'high': 'max',
    'low': 'min'
})
monthly['change'] = monthly['close'] - monthly['open']
monthly['change_pct'] = (monthly['change'] / monthly['open']) * 100
monthly['range'] = monthly['high'] - monthly['low']

print("\nMonthly Summary:")
for date, row in monthly.iterrows():
    direction = "BULLISH" if row['change'] > 0 else "BEARISH"
    print(f"  {date.strftime('%B %Y')}:")
    print(f"    Open: {row['open']:.2f}, Close: {row['close']:.2f}")
    print(f"    Change: {row['change']:+.2f} pts ({row['change_pct']:+.2f}%) - {direction}")
    print(f"    Range: {row['range']:.2f} pts (High: {row['high']:.2f}, Low: {row['low']:.2f})")

plt.show()
