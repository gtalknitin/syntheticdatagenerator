#!/usr/bin/env python3
"""
Analyze synthetic data for bullish/bearish bias
"""
import pandas as pd
import glob
from pathlib import Path

data_dir = Path("/Users/nitindhawan/NikAlgoBulls/zerodha_strategy/data/synthetic/intraday_v8_extended")

# Get all CSV files
csv_files = sorted(glob.glob(str(data_dir / "NIFTY_OPTIONS_5MIN_*.csv")))

print(f"Total files: {len(csv_files)}\n")

# Analyze daily price movements
daily_stats = []

for file in csv_files:
    df = pd.read_csv(file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Get date from filename
    date = Path(file).stem.split('_')[-1]

    # Get unique underlying prices for the day (should be same for all options at same time)
    underlying_prices = df.groupby('timestamp')['underlying_price'].first()

    open_price = underlying_prices.iloc[0]
    close_price = underlying_prices.iloc[-1]
    high_price = underlying_prices.max()
    low_price = underlying_prices.min()

    daily_change = close_price - open_price
    daily_change_pct = (daily_change / open_price) * 100

    daily_stats.append({
        'date': date,
        'open': open_price,
        'close': close_price,
        'high': high_price,
        'low': low_price,
        'change': daily_change,
        'change_pct': daily_change_pct,
        'direction': 'UP' if daily_change > 0 else 'DOWN' if daily_change < 0 else 'FLAT'
    })

stats_df = pd.DataFrame(daily_stats)

print("=" * 80)
print("DAILY PRICE MOVEMENT ANALYSIS")
print("=" * 80)

# Overall statistics
print(f"\nTotal trading days: {len(stats_df)}")
print(f"\nUp days: {(stats_df['direction'] == 'UP').sum()} ({(stats_df['direction'] == 'UP').sum()/len(stats_df)*100:.1f}%)")
print(f"Down days: {(stats_df['direction'] == 'DOWN').sum()} ({(stats_df['direction'] == 'DOWN').sum()/len(stats_df)*100:.1f}%)")
print(f"Flat days: {(stats_df['direction'] == 'FLAT').sum()} ({(stats_df['direction'] == 'FLAT').sum()/len(stats_df)*100:.1f}%)")

print(f"\nAverage daily change: {stats_df['change'].mean():.2f} points ({stats_df['change_pct'].mean():.3f}%)")
print(f"Median daily change: {stats_df['change'].median():.2f} points ({stats_df['change_pct'].median():.3f}%)")

print(f"\nOverall price movement:")
print(f"  Start price (Jun 16): {stats_df.iloc[0]['open']:.2f}")
print(f"  End price (Sep 30): {stats_df.iloc[-1]['close']:.2f}")
print(f"  Total change: {stats_df.iloc[-1]['close'] - stats_df.iloc[0]['open']:.2f} points")
print(f"  Total change %: {((stats_df.iloc[-1]['close'] - stats_df.iloc[0]['open']) / stats_df.iloc[0]['open'] * 100):.2f}%")

# Weekly/Monthly trends
print("\n" + "=" * 80)
print("WEEKLY TRENDS (for strategy direction determination)")
print("=" * 80)

# Simulate weekly closes (Wednesdays for strategy entry)
wednesdays = stats_df[pd.to_datetime(stats_df['date'], format='%Y%m%d').dt.dayofweek == 2].copy()
print(f"\nWednesday closing prices (strategy entry days):")
print(wednesdays[['date', 'close', 'change_pct']].to_string(index=False))

# Check if there's consistent upward bias
print("\n" + "=" * 80)
print("BIAS ANALYSIS")
print("=" * 80)

if stats_df['change'].mean() > 10:
    print("\n⚠️  WARNING: BULLISH BIAS DETECTED!")
    print(f"   Average daily gain: {stats_df['change'].mean():.2f} points")
    print(f"   This could cause strategy to always show bullish view")
elif stats_df['change'].mean() < -10:
    print("\n⚠️  WARNING: BEARISH BIAS DETECTED!")
    print(f"   Average daily loss: {stats_df['change'].mean():.2f} points")
    print(f"   This could cause strategy to always show bearish view")
else:
    print("\n✓ Data appears balanced with mixed up/down days")

# Month-by-month analysis
print("\n" + "=" * 80)
print("MONTHLY ANALYSIS")
print("=" * 80)

stats_df['month'] = pd.to_datetime(stats_df['date'], format='%Y%m%d').dt.to_period('M')
monthly = stats_df.groupby('month').agg({
    'change': ['sum', 'mean'],
    'direction': lambda x: (x == 'UP').sum()
}).round(2)

print("\nMonthly statistics:")
for month in stats_df['month'].unique():
    month_data = stats_df[stats_df['month'] == month]
    total_change = month_data.iloc[-1]['close'] - month_data.iloc[0]['open']
    up_days = (month_data['direction'] == 'UP').sum()
    total_days = len(month_data)

    print(f"\n{month}:")
    print(f"  Total change: {total_change:.2f} points")
    print(f"  Up days: {up_days}/{total_days} ({up_days/total_days*100:.1f}%)")
    print(f"  Avg daily change: {month_data['change'].mean():.2f} points")

print("\n" + "=" * 80)
