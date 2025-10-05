#!/usr/bin/env python3
"""
Resume v4 generation from where it stopped
"""

import os
import pandas as pd
from datetime import datetime

# Import the generator
from generate_v4_optimized import OptimizedV4Generator

def resume_generation():
    """Resume generation from last completed file"""
    
    output_dir = "/Users/nitindhawan/NikAlgoBulls/zerodha_strategy/data/synthetic/intraday_jul_sep_v4"
    
    # Get list of already generated files
    existing_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.csv')])
    print(f"\nFound {len(existing_files)} existing files")
    
    if existing_files:
        # Get last generated date
        last_file = existing_files[-1]
        last_date_str = last_file.replace('NIFTY_OPTIONS_5MIN_', '').replace('.csv', '')
        last_date = pd.to_datetime(last_date_str, format='%Y%m%d')
        print(f"Last generated: {last_date.strftime('%Y-%m-%d')}")
        
        # Get last closing spot
        df_last = pd.read_csv(os.path.join(output_dir, last_file))
        last_spot = df_last['underlying_price'].iloc[-1]
        print(f"Last closing spot: {last_spot:.2f}")
    else:
        last_date = pd.Timestamp('2025-06-30')
        last_spot = 25000
    
    # Create generator
    generator = OptimizedV4Generator()
    
    # Get remaining days
    remaining_days = [d for d in generator.trading_days if d > last_date]
    print(f"\nRemaining days to generate: {len(remaining_days)}")
    
    if not remaining_days:
        print("✅ All data already generated!")
        return
    
    print("\n" + "="*60)
    print("Resuming v4 Generation")
    print("="*60 + "\n")
    
    start_time = datetime.now()
    current_spot = last_spot
    
    # Process remaining days
    for idx, date in enumerate(remaining_days):
        try:
            closing_spot = generator.generate_day_batch(date, current_spot)
            current_spot = closing_spot
            
            # Progress
            if (idx + 1) % 5 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                avg_time = elapsed / (idx + 1)
                remaining = len(remaining_days) - (idx + 1)
                eta = remaining * avg_time
                print(f"\nProgress: {idx + 1}/{len(remaining_days)} days")
                print(f"ETA: {int(eta//60)}m {int(eta%60)}s\n")
        
        except Exception as e:
            print(f"\n❌ Error generating {date}: {e}")
            print(f"Stopped at {date}. You can resume again from this point.")
            break
    
    # Final count
    final_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.csv')])
    print(f"\n✅ Total files generated: {len(final_files)}")
    
    # Check if complete
    expected_days = len(generator.trading_days)
    if len(final_files) == expected_days:
        print(f"✅ Generation COMPLETE! All {expected_days} trading days generated.")
        
        # Create validation script if not exists
        val_script = os.path.join(output_dir, 'validate_v4_full.py')
        if not os.path.exists(val_script):
            from generate_synthetic_data_v4_full import SyntheticDataGeneratorV4Full
            temp_gen = SyntheticDataGeneratorV4Full('2025-07-01', '2025-09-30')
            temp_gen.output_dir = output_dir
            temp_gen._create_validation_script()
        
        # Create README
        generator.output_dir = output_dir
        generator.create_readme()
        
        print("\nNext steps:")
        print("1. Run validation: python validate_v4_full.py")
        print("2. Check data quality")
        print("3. Run backtests")
    else:
        print(f"⏸️  Generation paused. {expected_days - len(final_files)} days remaining.")
        print("Run this script again to continue.")


if __name__ == "__main__":
    resume_generation()