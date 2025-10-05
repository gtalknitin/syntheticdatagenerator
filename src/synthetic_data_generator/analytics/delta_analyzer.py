#!/usr/bin/env python3
"""
Analyze delta vs strike distance for weekly vs monthly options
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Read one day of data
data_file = Path("/Users/nitindhawan/NikAlgoBulls/zerodha_strategy/data/synthetic/intraday_v8_extended/NIFTY_OPTIONS_5MIN_20250616.csv")

print("Analyzing Delta vs Strike Distance...")
print("=" * 80)

df = pd.read_csv(data_file)

# Get first timestamp only
df = df[df['timestamp'] == df['timestamp'].iloc[0]]

spot = df['underlying_price'].iloc[0]
print(f"\nSpot Price: {spot:.2f}")

# Analyze weekly vs monthly separately
for expiry_type in ['weekly', 'monthly']:
    print(f"\n{expiry_type.upper()} OPTIONS:")
    print("-" * 80)

    exp_df = df[df['expiry_type'] == expiry_type]

    if len(exp_df) == 0:
        print(f"No {expiry_type} options found")
        continue

    expiry_date = exp_df['expiry'].iloc[0]
    print(f"Expiry: {expiry_date}")

    # Calculate TTE
    from datetime import datetime
    exp_date = datetime.strptime(expiry_date, '%Y-%m-%d')
    current_date = datetime.strptime(df['timestamp'].iloc[0], '%Y-%m-%d %H:%M:%S')
    tte_days = (exp_date - current_date).days
    print(f"Days to Expiry: {tte_days}")

    # Analyze CE options (for bullish monthly / bearish weekly)
    print("\n📈 CALL OPTIONS (CE):")
    ce_df = exp_df[exp_df['option_type'] == 'CE'].copy()
    ce_df['distance_from_atm'] = ce_df['strike'] - spot
    ce_df['abs_delta'] = ce_df['delta'].abs()

    # Find strikes at different delta levels
    for target_delta in [0.5, 0.3, 0.2, 0.15, 0.1, 0.05]:
        # Find closest strike to target delta
        ce_df['delta_diff'] = abs(ce_df['abs_delta'] - target_delta)
        closest = ce_df.loc[ce_df['delta_diff'].idxmin()]

        print(f"  {target_delta:.2f} delta → Strike: {closest['strike']:.0f} "
              f"({closest['distance_from_atm']:+.0f} pts from ATM), "
              f"Actual Δ: {closest['abs_delta']:.3f}, "
              f"Premium: ₹{closest['close']:.2f}")

    # Analyze PE options (for bearish monthly / bullish weekly)
    print("\n📉 PUT OPTIONS (PE):")
    pe_df = exp_df[exp_df['option_type'] == 'PE'].copy()
    pe_df['distance_from_atm'] = spot - pe_df['strike']
    pe_df['abs_delta'] = pe_df['delta'].abs()

    for target_delta in [0.5, 0.3, 0.2, 0.15, 0.1, 0.05]:
        pe_df['delta_diff'] = abs(pe_df['abs_delta'] - target_delta)
        closest = pe_df.loc[pe_df['delta_diff'].idxmin()]

        print(f"  {target_delta:.2f} delta → Strike: {closest['strike']:.0f} "
              f"({closest['distance_from_atm']:+.0f} pts from ATM), "
              f"Actual Δ: {closest['abs_delta']:.3f}, "
              f"Premium: ₹{closest['close']:.2f}")

print("\n" + "=" * 80)
print("KEY FINDINGS:")
print("=" * 80)

# Summary analysis
weekly = df[df['expiry_type'] == 'weekly']
monthly = df[df['expiry_type'] == 'monthly']

if len(weekly) > 0 and len(monthly) > 0:
    # Find 0.1 delta strikes
    weekly_ce = weekly[(weekly['option_type'] == 'CE') & (weekly['delta'].abs() >= 0.08) & (weekly['delta'].abs() <= 0.12)]
    monthly_ce = monthly[(monthly['option_type'] == 'CE') & (monthly['delta'].abs() >= 0.08) & (monthly['delta'].abs() <= 0.12)]

    if len(weekly_ce) > 0 and len(monthly_ce) > 0:
        weekly_dist = weekly_ce.iloc[0]['strike'] - spot
        monthly_dist = monthly_ce.iloc[0]['strike'] - spot

        print(f"\n0.1 Delta CE Distance from ATM:")
        print(f"  Weekly:  ~{weekly_dist:+.0f} points")
        print(f"  Monthly: ~{monthly_dist:+.0f} points")
        print(f"  Ratio: {monthly_dist/weekly_dist:.2f}x")

        # Premium analysis
        weekly_prem = weekly_ce.iloc[0]['close']
        monthly_prem = monthly_ce.iloc[0]['close']
        print(f"\n0.1 Delta CE Premiums:")
        print(f"  Weekly:  ₹{weekly_prem:.2f}")
        print(f"  Monthly: ₹{monthly_prem:.2f}")
        print(f"  Ratio: {monthly_prem/weekly_prem:.2f}x")

    print("\n⚠️  ANALYSIS:")
    print("  1. Same strike range used for both weekly and monthly")
    print("  2. Black-Scholes adjusts prices based on TTE, but...")
    print("  3. Weekly hedges need strikes closer to ATM for same delta")
    print("  4. If strategy uses fixed point distances, deltas will be wrong!")
