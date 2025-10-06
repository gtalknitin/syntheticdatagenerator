#!/usr/bin/env python3
"""
V10 Data Quality Validator

Checks for critical data quality issues:
1. Duplicate rows (timestamp, strike, option_type, expiry)
2. Missing values
3. Invalid OHLC relationships
4. Invalid bid/ask spreads
5. Greeks bounds
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def validate_file(filepath: Path) -> dict:
    """Validate a single CSV file"""
    df = pd.read_csv(filepath)

    issues = []

    # Check 1: Duplicates (CRITICAL!)
    duplicates = df.groupby(['timestamp', 'strike', 'option_type', 'expiry']).size()
    dups = duplicates[duplicates > 1]

    if len(dups) > 0:
        issues.append(f"CRITICAL: {len(dups)} duplicate row groups found!")
        # Show sample
        for key in list(dups.index)[:3]:
            issues.append(f"  Duplicate: {key}")

    # Check 2: Missing values
    missing = df.isnull().sum()
    if missing.any():
        issues.append(f"Missing values: {missing[missing > 0].to_dict()}")

    # Check 3: OHLC validity
    invalid_ohlc = df[
        (df['high'] < df['low']) |
        (df['high'] < df['open']) |
        (df['high'] < df['close']) |
        (df['low'] > df['open']) |
        (df['low'] > df['close'])
    ]
    if len(invalid_ohlc) > 0:
        issues.append(f"Invalid OHLC in {len(invalid_ohlc)} rows")

    # Check 4: Bid/Ask validity
    invalid_spread = df[(df['bid'] >= df['ask']) | (df['bid'] > df['close'])]
    if len(invalid_spread) > 0:
        issues.append(f"Invalid bid/ask in {len(invalid_spread)} rows")

    # Check 5: Greeks bounds
    invalid_delta = df[(df['delta'] < -1) | (df['delta'] > 1)]
    if len(invalid_delta) > 0:
        issues.append(f"Delta out of bounds in {len(invalid_delta)} rows")

    invalid_gamma = df[df['gamma'] < 0]
    if len(invalid_gamma) > 0:
        issues.append(f"Negative gamma in {len(invalid_gamma)} rows")

    invalid_vega = df[df['vega'] < 0]
    if len(invalid_vega) > 0:
        issues.append(f"Negative vega in {len(invalid_vega)} rows")

    return {
        'file': filepath.name,
        'rows': len(df),
        'valid': len(issues) == 0,
        'issues': issues
    }


def main():
    print("="*80)
    print("V10 DATA QUALITY VALIDATION")
    print("="*80)

    data_path = Path('data/generated/v10_real_enhanced/hourly')
    files = sorted(data_path.glob('*.csv'))

    print(f"\nValidating {len(files)} files...")

    all_valid = True
    total_issues = 0
    critical_files = []

    for i, file in enumerate(files, 1):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(files)}")

        result = validate_file(file)

        if not result['valid']:
            all_valid = False
            total_issues += len(result['issues'])

            # Check for critical issues (duplicates)
            has_critical = any('CRITICAL' in issue for issue in result['issues'])
            if has_critical:
                critical_files.append(result)

    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)

    if all_valid:
        print("\n✅ ALL FILES PASSED VALIDATION!")
    else:
        print(f"\n❌ VALIDATION FAILED!")
        print(f"  Files with issues: {len([f for f in files if not validate_file(f)['valid']])}")
        print(f"  Total issues: {total_issues}")

        if critical_files:
            print(f"\n🚨 CRITICAL: {len(critical_files)} files have DUPLICATE ROWS!")
            print("\nSample critical files:")
            for result in critical_files[:5]:
                print(f"\n  {result['file']}:")
                for issue in result['issues']:
                    print(f"    - {issue}")

    print("="*80)

    return 0 if all_valid else 1


if __name__ == '__main__':
    sys.exit(main())
