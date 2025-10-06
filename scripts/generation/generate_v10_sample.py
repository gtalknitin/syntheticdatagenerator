#!/usr/bin/env python3
"""
Generate V10 SAMPLE data (first 7 days) for quick validation

This allows us to validate the approach before running full 438-day generation
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.synthetic_data_generator.generators.v10.generator import V10RealEnhancedGenerator
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    print("="*80)
    print("V10 SAMPLE GENERATION (First 7 Days)")
    print("="*80)

    # Initialize generator
    generator = V10RealEnhancedGenerator()

    # Override output path for sample
    generator.base_path = Path('data/generated/v10_sample/hourly')
    generator.base_path.mkdir(parents=True, exist_ok=True)

    # Get first 7 trading dates
    trading_dates = sorted(generator.nifty_hourly['trading_date'].unique())[:7]

    print(f"\nGenerating sample: {len(trading_dates)} days")
    print(f"Dates: {trading_dates[0]} to {trading_dates[-1]}")
    print()

    # Generate for each day
    for i, date in enumerate(trading_dates, 1):
        print(f"  [{i}/{len(trading_dates)}] {date}", end=' ... ')

        # Generate
        df = generator.generate_day_data(date)

        if df.empty:
            print("⊘ No data")
            continue

        # Validate
        validation = generator.validate_day_data(df)

        # Save
        filename = f"NIFTY_OPTIONS_1H_{date.strftime('%Y%m%d')}.csv"
        filepath = generator.base_path / filename
        df.to_csv(filepath, index=False)

        # Update stats
        generator.stats['files_created'] += 1
        generator.stats['total_rows'] += len(df)
        generator.stats['dates_processed'].append(str(date))

        # Print
        vix = df['vix'].iloc[0]
        strikes = df['strike'].nunique()
        status = "✓" if validation['valid'] else "⚠"
        print(f"{status} ({len(df):,} rows, {strikes} strikes, VIX: {vix:.1f})")

    # Save metadata
    generator.save_metadata()

    # Print summary
    print("\n" + "="*80)
    print("✅ SAMPLE GENERATION COMPLETE!")
    print("="*80)
    print(f"\nFiles: {generator.stats['files_created']}")
    print(f"Total rows: {generator.stats['total_rows']:,}")
    print(f"\nGreeks Validation:")

    gv = generator.stats['greeks_validation']
    if gv['total_checks'] > 0:
        print(f"  Deep ITM: {gv['deep_itm_passed']}/{gv['total_checks']}")
        print(f"  Deep OTM: {gv['deep_otm_passed']}/{gv['total_checks']}")
        print(f"  ATM: {gv['atm_passed']}/{gv['total_checks']}")

    print(f"\nLocation: {generator.base_path}")
    print("="*80)

    return generator.stats


if __name__ == '__main__':
    main()
