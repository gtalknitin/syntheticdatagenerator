#!/usr/bin/env python3
"""
Resume synthetic data generation from Aug 7 to Sep 30, 2025
"""

import sys
import os
sys.path.append('/Users/nitindhawan/NikAlgoBulls/zerodha_strategy/data/synthetic')

from generate_synthetic_data_v2 import NiftyOptionsDataGeneratorV2

def main():
    print("="*60)
    print("RESUMING NIFTY OPTIONS DATASET GENERATION")
    print("Period: August 7 - September 30, 2025")
    print("="*60)

    # Initialize generator
    generator = NiftyOptionsDataGeneratorV2()

    # Output directory (same as v3)
    output_dir = "/Users/nitindhawan/NikAlgoBulls/zerodha_strategy/data/synthetic/intraday_jul_sep_v3"

    print(f"Resuming generation from August 7, 2025...")
    print(f"Output directory: {output_dir}")

    try:
        # Generate remaining data using the built-in method
        generator.generate_period_data(
            start_date="2025-08-07",  # Start from where we left off
            end_date="2025-09-30",
            initial_spot=24624.22,    # Continuing from last known spot
            output_dir=output_dir
        )

        print("\n" + "="*60)
        print("✅ DATASET GENERATION COMPLETE!")
        print("Full period: July 1 - September 30, 2025")
        print("="*60)

        return True

    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()

    # Quick verification
    output_dir = "/Users/nitindhawan/NikAlgoBulls/zerodha_strategy/data/synthetic/intraday_jul_sep_v3"
    csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')] if os.path.exists(output_dir) else []

    print(f"\nFinal dataset: {len(csv_files)} files")
    if csv_files:
        dates = sorted([f.replace('NIFTY_OPTIONS_5MIN_', '').replace('.csv', '') for f in csv_files])
        print(f"Date range: {dates[0]} to {dates[-1]}")

    if len(csv_files) >= 60:
        print("🎉 Complete dataset ready for backtesting!")
    else:
        print("⚠️  Dataset may be incomplete")

    sys.exit(0 if success else 1)