#!/usr/bin/env python3
"""
Regenerate synthetic NIFTY options data with proper monthly expiry availability
Version 3.0 - Fixes the monthly expiry availability issue
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic'))

from generate_synthetic_data_v2 import NiftyOptionsDataGeneratorV2
from datetime import datetime

def main():
    print("="*60)
    print("NIFTY Options Synthetic Data Generator v3.0")
    print("Generating data with proper monthly expiry availability")
    print("="*60)

    # Initialize generator with fixed logic
    generator = NiftyOptionsDataGeneratorV2()

    # Output directory for v3 data
    output_dir = "../data/synthetic/intraday_jul_sep_v3"

    print("\nGenerating synthetic data for July-September 2025...")
    print(f"Output directory: {output_dir}")
    print("-"*60)

    # Generate data
    results = generator.generate_period_data(
        start_date="2025-07-01",
        end_date="2025-09-30",
        initial_spot=25000,
        output_dir=output_dir
    )

    print("\n" + "="*60)
    print("Data generation complete!")
    print(f"Files saved to: {output_dir}")
    print("\nKey improvements in v3:")
    print("  ✅ Monthly expiries available 30-45 days in advance")
    print("  ✅ July 31st monthly available from July 1st")
    print("  ✅ Realistic option chain availability")
    print("  ✅ Proper overlap between weekly and monthly options")
    print("="*60)

if __name__ == "__main__":
    main()