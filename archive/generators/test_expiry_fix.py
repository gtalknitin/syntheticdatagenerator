#!/usr/bin/env python3
"""Test script to verify the fixed expiry generation logic"""

from datetime import datetime
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_synthetic_data_v2 import NiftyOptionsDataGeneratorV2

def test_expiry_availability():
    """Test that monthly expiries are available 30-45 days in advance"""

    generator = NiftyOptionsDataGeneratorV2()

    # Test for July 1st - should see July 31st monthly expiry
    test_date = datetime(2025, 7, 1)
    expiries = generator.get_active_expiries(test_date)

    print(f"Expiries available on July 1st, 2025:")
    print("-" * 50)

    monthly_found = False
    for expiry_date, expiry_type in expiries:
        days_to_expiry = (expiry_date - test_date).days
        print(f"  {expiry_date.strftime('%Y-%m-%d')} ({expiry_type:7s}) - {days_to_expiry} days away")

        if expiry_type == 'monthly' and expiry_date.month == 7:
            monthly_found = True
            print(f"    ✓ July monthly expiry FOUND! ({days_to_expiry} days in advance)")

    print("\n" + "="*50)
    if monthly_found:
        print("✅ SUCCESS: July 31st monthly expiry is available on July 1st!")
        print("   This matches real market behavior (30-45 day availability)")
    else:
        print("❌ FAILURE: July 31st monthly expiry NOT available on July 1st")
        print("   This does not match real market behavior")

    # Also test for mid-month
    print("\n" + "="*50)
    test_date2 = datetime(2025, 7, 15)
    expiries2 = generator.get_active_expiries(test_date2)

    print(f"\nExpiries available on July 15th, 2025:")
    print("-" * 50)

    for expiry_date, expiry_type in expiries2:
        days_to_expiry = (expiry_date - test_date2).days
        print(f"  {expiry_date.strftime('%Y-%m-%d')} ({expiry_type:7s}) - {days_to_expiry} days away")

    # Check for both July and August monthly
    july_monthly = any(e[0].month == 7 and e[1] == 'monthly' for e in expiries2)
    aug_monthly = any(e[0].month == 8 and e[1] == 'monthly' for e in expiries2)

    print("\n" + "="*50)
    print("Summary for July 15th:")
    print(f"  July monthly available: {'✅ Yes' if july_monthly else '❌ No'}")
    print(f"  August monthly available: {'✅ Yes' if aug_monthly else '❌ No'}")

    if july_monthly and aug_monthly:
        print("\n✅ EXCELLENT: Both current and next month's monthly expiries available!")

    return monthly_found

if __name__ == "__main__":
    success = test_expiry_availability()
    sys.exit(0 if success else 1)