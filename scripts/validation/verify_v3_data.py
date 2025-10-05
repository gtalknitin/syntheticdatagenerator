#!/usr/bin/env python3
"""Verify that v3 synthetic data has proper monthly expiry availability"""

import pandas as pd
import os
from datetime import datetime

def verify_monthly_expiries():
    """Verify monthly expiries are available from day 1 of each month"""

    data_dir = "../data/synthetic/intraday_jul_sep_v3"

    # Check July 1st specifically
    july1_file = os.path.join(data_dir, "NIFTY_OPTIONS_5MIN_20250701.csv")

    if os.path.exists(july1_file):
        df = pd.read_csv(july1_file)
        unique_expiries = df[['expiry', 'expiry_type']].drop_duplicates().sort_values('expiry')

        print("="*60)
        print("VERIFICATION: July 1st, 2025 Data")
        print("="*60)
        print("\nAvailable expiries on July 1st:")
        print("-"*40)

        july_monthly_found = False
        for _, row in unique_expiries.iterrows():
            expiry_date = pd.to_datetime(row['expiry'])
            print(f"  {expiry_date.strftime('%Y-%m-%d')} - {row['expiry_type']}")

            if row['expiry_type'] == 'monthly' and expiry_date.month == 7:
                july_monthly_found = True

        print("\n" + "="*60)
        if july_monthly_found:
            print("✅ SUCCESS: July 31st monthly expiry IS available on July 1st!")
            print("   This fixes the critical bug from v2 data.")
            print("   The strategy can now properly create monthly spreads from day 1.")
        else:
            print("❌ FAILURE: July 31st monthly expiry NOT found on July 1st")
            print("   The bug persists - monthly expiries are still not available early enough.")

        # Check data integrity
        print("\n" + "="*60)
        print("DATA INTEGRITY CHECK:")
        print("-"*40)
        print(f"Total rows: {len(df):,}")
        print(f"Unique strikes: {df['strike'].nunique()}")
        print(f"Unique expiries: {len(unique_expiries)}")
        print(f"Strike range: {df['strike'].min()} - {df['strike'].max()}")

        # Verify credit spread feasibility
        july31_data = df[df['expiry'] == '2025-07-31']
        if not july31_data.empty:
            atm_strike = 25000  # Approximate ATM
            call_spreads = july31_data[(july31_data['option_type'] == 'CE') &
                                      (july31_data['strike'].isin([atm_strike, atm_strike + 200]))]

            if len(call_spreads['strike'].unique()) == 2:
                print("\n✅ Bull call spread (25000/25200) data available for July 31 expiry")
            else:
                print("\n⚠️ Incomplete data for bull call spread")

    else:
        print("⚠️ July 1st data file not found yet. Generation may still be in progress.")

    # Check overall generation status
    print("\n" + "="*60)
    print("GENERATION STATUS:")
    print("-"*40)

    if os.path.exists(data_dir):
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        print(f"Files generated so far: {len(csv_files)}")

        if csv_files:
            dates = sorted([f.replace('NIFTY_OPTIONS_5MIN_', '').replace('.csv', '') for f in csv_files])
            print(f"Date range: {dates[0]} to {dates[-1]}")
    else:
        print("Output directory not found.")

if __name__ == "__main__":
    verify_monthly_expiries()