#!/usr/bin/env python3
"""
Batch generator for v4 synthetic data - Optimized for speed
"""

import os
import sys
import pandas as pd
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from generate_synthetic_data_v4 import SyntheticDataGeneratorV4

def generate_single_day(args):
    """Generate data for a single day (for parallel processing)"""
    trading_date, output_dir, base_spot_price = args
    
    # Create a generator instance for this day
    generator = SyntheticDataGeneratorV4("2025-07-01", "2025-09-30")
    generator.output_dir = output_dir
    generator.base_spot = base_spot_price
    
    try:
        generator.generate_daily_file(trading_date)
        return f"✓ {trading_date.strftime('%Y-%m-%d')}", True
    except Exception as e:
        return f"✗ {trading_date.strftime('%Y-%m-%d')}: {str(e)}", False

def main():
    print("\n" + "="*60)
    print("NIFTY Options Synthetic Data v4.0 - Batch Generator")
    print("="*60)
    
    # Initialize generator to get trading days
    generator = SyntheticDataGeneratorV4("2025-07-01", "2025-09-30")
    trading_days = generator.trading_days
    output_dir = generator.output_dir
    
    print(f"\nOutput directory: {output_dir}")
    print(f"Total trading days: {len(trading_days)}")
    print(f"Period: {trading_days[0].strftime('%Y-%m-%d')} to {trading_days[-1].strftime('%Y-%m-%d')}")
    
    # Generate in batches of 10 days
    batch_size = 10
    base_spot = 25000
    
    print(f"\nGenerating data in batches of {batch_size} days...")
    print("-" * 40)
    
    completed = 0
    failed = []
    
    for i in range(0, len(trading_days), batch_size):
        batch_days = trading_days[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(trading_days) + batch_size - 1) // batch_size
        
        print(f"\nBatch {batch_num}/{total_batches} ({len(batch_days)} days)")
        
        # Sequential processing for this batch (to avoid memory issues)
        for day in batch_days:
            result, success = generate_single_day((day, output_dir, base_spot))
            if success:
                completed += 1
                print(f"  {result}")
            else:
                failed.append(result)
                print(f"  {result}")
            
            # Update base spot for next day (simplified)
            base_spot *= (1 + pd.np.random.uniform(-0.01, 0.01))
    
    # Summary
    print("\n" + "="*60)
    print("GENERATION COMPLETE")
    print("="*60)
    print(f"\n✅ Successfully generated: {completed}/{len(trading_days)} days")
    
    if failed:
        print(f"\n❌ Failed: {len(failed)} days")
        for f in failed:
            print(f"  {f}")
    
    print(f"\nTotal files in output directory: {len(os.listdir(output_dir))}")
    print(f"Output location: {output_dir}")
    
    # Create summary file
    summary_path = os.path.join(output_dir, "GENERATION_SUMMARY.txt")
    with open(summary_path, 'w') as f:
        f.write(f"NIFTY Options Synthetic Data v4.0 Generation Summary\n")
        f.write(f"{'='*50}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Period: {trading_days[0].strftime('%Y-%m-%d')} to {trading_days[-1].strftime('%Y-%m-%d')}\n")
        f.write(f"Trading days: {len(trading_days)}\n")
        f.write(f"Successfully generated: {completed}\n")
        f.write(f"Failed: {len(failed)}\n")
        f.write(f"\nKey improvements in v4:\n")
        f.write(f"- No binary price collapse\n")
        f.write(f"- Proper theta decay curves\n")
        f.write(f"- Realistic bid-ask spreads\n")
        f.write(f"- Volatility smile implementation\n")
        f.write(f"- Proper expiry day behavior\n")
    
    print(f"\n📄 Summary saved to: {summary_path}")

if __name__ == "__main__":
    main()