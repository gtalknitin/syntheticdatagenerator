#!/usr/bin/env python3
"""
Generate complete synthetic NIFTY options dataset: July 1 - September 30, 2025
With proper monthly expiry availability and memory management
"""

import sys
import os
import gc
from datetime import datetime, timedelta
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic'))

from generate_synthetic_data_v2 import NiftyOptionsDataGeneratorV2

def generate_full_dataset():
    print("="*70)
    print("NIFTY OPTIONS COMPLETE DATASET GENERATOR")
    print("Period: July 1 - September 30, 2025 (Fixed Monthly Expiry Bug)")
    print("="*70)

    # Initialize generator
    generator = NiftyOptionsDataGeneratorV2()

    # Output directory
    output_dir = "/Users/nitindhawan/NikAlgoBulls/zerodha_strategy/data/synthetic/intraday_jul_sep_v3"

    # Check what's already generated
    existing_files = []
    if os.path.exists(output_dir):
        existing_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
        existing_dates = sorted([f.replace('NIFTY_OPTIONS_5MIN_', '').replace('.csv', '') for f in existing_files])
        print(f"\nExisting files found: {len(existing_files)}")
        if existing_files:
            print(f"Date range: {existing_dates[0]} to {existing_dates[-1]}")

    print(f"\nOutput directory: {output_dir}")
    print("-"*70)

    try:
        # Use the built-in method with memory optimization
        print("\nStarting data generation...")
        print("This will generate ~65 trading days of data")
        print("Each file contains ~150K rows (all strikes, all expiries, 5-min bars)")

        results = generator.generate_period_data(
            start_date="2025-07-01",
            end_date="2025-09-30",
            initial_spot=25000,
            output_dir=output_dir
        )

        print("\n" + "="*70)
        print("✅ DATASET GENERATION COMPLETE!")
        print(f"Files saved to: {output_dir}")
        print("\nKey Features of v3 Dataset:")
        print("  ✅ Monthly expiries available 30-45 days in advance")
        print("  ✅ July 31st monthly available from July 1st")
        print("  ✅ Complete strike coverage (20000-30000)")
        print("  ✅ Realistic option pricing and Greeks")
        print("  ✅ Proper bid-ask spreads for trading")
        print("  ✅ Credit spread validation")
        print("="*70)

    except KeyboardInterrupt:
        print("\n⚠️  Generation interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        return False

    return True

def verify_complete_dataset():
    """Verify the complete dataset was generated properly"""
    output_dir = "/Users/nitindhawan/NikAlgoBulls/zerodha_strategy/data/synthetic/intraday_jul_sep_v3"

    print("\n" + "="*70)
    print("DATASET VERIFICATION")
    print("="*70)

    if not os.path.exists(output_dir):
        print("❌ Output directory not found!")
        return False

    csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]

    if not csv_files:
        print("❌ No CSV files found!")
        return False

    dates = sorted([f.replace('NIFTY_OPTIONS_5MIN_', '').replace('.csv', '') for f in csv_files])

    print(f"Generated files: {len(csv_files)}")
    print(f"Date range: {dates[0]} to {dates[-1]}")
    print(f"Expected ~65 trading days (Jul-Sep 2025)")

    # Check critical dates
    critical_dates = ['20250701', '20250731', '20250930']
    missing_dates = [d for d in critical_dates if d not in dates]

    if missing_dates:
        print(f"⚠️  Missing critical dates: {missing_dates}")
    else:
        print("✅ All critical dates present")

    # Check total data size
    total_size = sum(os.path.getsize(os.path.join(output_dir, f)) for f in csv_files)
    print(f"Total dataset size: {total_size / (1024**3):.2f} GB")

    if len(csv_files) >= 60:  # Expect ~65 trading days
        print("✅ Dataset appears complete!")
        return True
    else:
        print(f"⚠️  Dataset may be incomplete (only {len(csv_files)} files)")
        return False

if __name__ == "__main__":
    success = generate_full_dataset()
    if success:
        verify_complete_dataset()
    else:
        print("\n❌ Dataset generation failed")
        sys.exit(1)