#!/usr/bin/env python3
"""
Analyze underlying price trend without plotting
"""
import pandas as pd
import numpy as np
from pathlib import Path
import glob

# Read all data files
data_dir = Path("/Users/nitindhawan/NikAlgoBulls/zerodha_strategy/data/synthetic/intraday_v8_extended")
csv_files = sorted(glob.glob(str(data_dir / "NIFTY_OPTIONS_5MIN_*.csv")))

print(f"Processing {len(csv_files)} files...")
print("=" * 80)

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

print(f"\n✓ Generated {len(all_hourly)} hourly candles")
print(f"  Period: {all_hourly.index[0]} to {all_hourly.index[-1]}")

# Overall statistics
print("\n" + "="*80)
print("OVERALL PRICE MOVEMENT")
print("="*80)

start_price = all_hourly['open'].iloc[0]
end_price = all_hourly['close'].iloc[-1]
total_change = end_price - start_price
total_change_pct = (total_change / start_price) * 100

print(f"\nStart Price (Jun 16): {start_price:.2f}")
print(f"End Price (Sep 30):   {end_price:.2f}")
print(f"Total Change:         {total_change:+.2f} points ({total_change_pct:+.2f}%)")
print(f"High:                 {all_hourly['high'].max():.2f}")
print(f"Low:                  {all_hourly['low'].min():.2f}")
print(f"Range:                {all_hourly['high'].max() - all_hourly['low'].min():.2f}")

# Hourly candle analysis
hourly_changes = all_hourly['close'] - all_hourly['open']
up_candles = (hourly_changes > 0).sum()
down_candles = (hourly_changes < 0).sum()
flat_candles = (hourly_changes == 0).sum()

print(f"\nHourly Candles:")
print(f"  Up:   {up_candles} ({up_candles/len(all_hourly)*100:.1f}%)")
print(f"  Down: {down_candles} ({down_candles/len(all_hourly)*100:.1f}%)")
print(f"  Flat: {flat_candles} ({flat_candles/len(all_hourly)*100:.1f}%)")

if up_candles > down_candles * 1.2:
    print("\n⚠️  BULLISH BIAS DETECTED in hourly candles!")
elif down_candles > up_candles * 1.2:
    print("\n⚠️  BEARISH BIAS DETECTED in hourly candles!")
else:
    print("\n✓ Hourly candles appear balanced")

# Daily analysis
daily = all_hourly.resample('D').agg({
    'open': 'first',
    'close': 'last',
    'high': 'max',
    'low': 'min'
})
daily['change'] = daily['close'] - daily['open']
daily['change_pct'] = (daily['change'] / daily['open']) * 100

print("\n" + "="*80)
print("DAILY MOVEMENT ANALYSIS")
print("="*80)

up_days = (daily['change'] > 0).sum()
down_days = (daily['change'] < 0).sum()

print(f"\nDaily Candles:")
print(f"  Up days:   {up_days} ({up_days/len(daily)*100:.1f}%)")
print(f"  Down days: {down_days} ({down_days/len(daily)*100:.1f}%)")
print(f"  Avg change: {daily['change'].mean():+.2f} points")
print(f"  Avg change %: {daily['change_pct'].mean():+.3f}%")

# Weekly trend
print("\n" + "="*80)
print("WEEKLY MOVEMENT")
print("="*80)

weekly = all_hourly.resample('W').agg({
    'open': 'first',
    'close': 'last',
    'high': 'max',
    'low': 'min'
})
weekly['change'] = weekly['close'] - weekly['open']
weekly['change_pct'] = (weekly['change'] / weekly['open']) * 100

print("\nWeek-by-Week:")
for i, (date, row) in enumerate(weekly.iterrows(), 1):
    direction = "📈 BULLISH" if row['change'] > 0 else "📉 BEARISH" if row['change'] < 0 else "➡️  NEUTRAL"
    print(f"  Week {i:2} ({date.strftime('%m/%d')}): {direction:12} "
          f"{row['change']:+7.2f} pts ({row['change_pct']:+6.2f}%) "
          f"[{row['open']:.0f} → {row['close']:.0f}]")

up_weeks = (weekly['change'] > 0).sum()
down_weeks = (weekly['change'] < 0).sum()
print(f"\nWeekly Summary:")
print(f"  Up weeks:   {up_weeks} ({up_weeks/len(weekly)*100:.1f}%)")
print(f"  Down weeks: {down_weeks} ({down_weeks/len(weekly)*100:.1f}%)")

# Monthly analysis
print("\n" + "="*80)
print("MONTHLY SUMMARY")
print("="*80)

monthly = all_hourly.resample('M').agg({
    'open': 'first',
    'close': 'last',
    'high': 'max',
    'low': 'min'
})
monthly['change'] = monthly['close'] - monthly['open']
monthly['change_pct'] = (monthly['change'] / monthly['open']) * 100
monthly['range'] = monthly['high'] - monthly['low']

for date, row in monthly.iterrows():
    direction = "BULLISH ↗" if row['change'] > 0 else "BEARISH ↘"
    print(f"\n{date.strftime('%B %Y')}:")
    print(f"  Direction: {direction}")
    print(f"  Open:  {row['open']:.2f}")
    print(f"  Close: {row['close']:.2f}")
    print(f"  High:  {row['high']:.2f}")
    print(f"  Low:   {row['low']:.2f}")
    print(f"  Change: {row['change']:+.2f} pts ({row['change_pct']:+.2f}%)")
    print(f"  Range:  {row['range']:.2f} pts")

# Trend persistence analysis
print("\n" + "="*80)
print("TREND PERSISTENCE ANALYSIS")
print("="*80)

# Check for sustained trends (5+ consecutive days same direction)
daily_directions = np.sign(daily['change'])
consecutive = []
current_count = 1
current_direction = daily_directions.iloc[0]

for i in range(1, len(daily_directions)):
    if daily_directions.iloc[i] == current_direction and current_direction != 0:
        current_count += 1
    else:
        if current_count >= 5:
            consecutive.append((current_direction, current_count))
        current_count = 1
        current_direction = daily_directions.iloc[i]

if consecutive:
    print("\nSustained Trends (5+ consecutive days):")
    for direction, count in consecutive:
        trend_type = "BULLISH" if direction > 0 else "BEARISH"
        print(f"  {count} consecutive {trend_type} days")
else:
    print("\n✓ No sustained unidirectional trends (5+ days)")

# Calculate streaks
max_up_streak = 0
max_down_streak = 0
current_up = 0
current_down = 0

for change in daily['change']:
    if change > 0:
        current_up += 1
        current_down = 0
        max_up_streak = max(max_up_streak, current_up)
    elif change < 0:
        current_down += 1
        current_up = 0
        max_down_streak = max(max_down_streak, current_down)
    else:
        current_up = 0
        current_down = 0

print(f"\nMax consecutive up days: {max_up_streak}")
print(f"Max consecutive down days: {max_down_streak}")

if max_up_streak > 7 or max_down_streak > 7:
    print("⚠️  WARNING: Extended unidirectional trends detected!")

# Final assessment
print("\n" + "="*80)
print("BIAS ASSESSMENT")
print("="*80)

bias_score = 0
checks = []

# Check 1: Overall direction
if total_change_pct > 2:
    bias_score += 2
    checks.append("❌ Overall strongly bullish (+2.93%)")
elif total_change_pct < -2:
    bias_score -= 2
    checks.append("✓ Overall bearish")
else:
    checks.append("✓ Overall balanced")

# Check 2: Daily ratio
if up_days / len(daily) > 0.6:
    bias_score += 2
    checks.append("❌ Too many up days (>60%)")
elif down_days / len(daily) > 0.6:
    bias_score -= 2
    checks.append("✓ Slight bearish bias in days")
else:
    checks.append("✓ Daily ratio balanced")

# Check 3: Hourly ratio
if up_candles / len(all_hourly) > 0.55:
    bias_score += 1
    checks.append(f"❌ Hourly candles bullish ({up_candles/len(all_hourly)*100:.1f}% up)")
elif down_candles / len(all_hourly) > 0.55:
    bias_score -= 1
    checks.append("✓ Hourly candles balanced")
else:
    checks.append("✓ Hourly candles balanced")

# Check 4: Trend streaks
if max_up_streak > max_down_streak + 3:
    bias_score += 1
    checks.append(f"❌ Longer bullish streaks ({max_up_streak} vs {max_down_streak})")
elif max_down_streak > max_up_streak + 3:
    bias_score -= 1
    checks.append("✓ No bullish bias in streaks")
else:
    checks.append("✓ Balanced streaks")

print("\nChecks:")
for check in checks:
    print(f"  {check}")

print(f"\nBias Score: {bias_score:+d}")
print("\nConclusion:")
if bias_score >= 3:
    print("  🔴 STRONG BULLISH BIAS - Data will cause strategy to enter only bullish positions!")
elif bias_score >= 1:
    print("  🟡 MODERATE BULLISH BIAS - May favor bullish signals")
elif bias_score <= -3:
    print("  🔵 STRONG BEARISH BIAS")
elif bias_score <= -1:
    print("  🟡 MODERATE BEARISH BIAS")
else:
    print("  🟢 DATA APPEARS BALANCED")

print("\n" + "="*80)
