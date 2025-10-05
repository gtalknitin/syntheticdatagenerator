#!/usr/bin/env python3
"""
V6 Full Dataset Generator - July to September 2025
Generates complete 3-month dataset with proper time series evolution
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import math

# Set random seed for reproducibility
np.random.seed(42)


class V6FullDatasetGenerator:
    """Generate complete July-September 2025 dataset"""

    def __init__(self):
        self.output_dir = '/Users/nitindhawan/NikAlgoBulls/zerodha_strategy/data/synthetic/intraday_v6_corrected'
        self.base_prices = {}  # Store prices for continuity
        self.underlying_price = 25000

        # NSE holidays 2025
        self.holidays = ['2025-08-15']  # Independence Day

    def generate_full_dataset(self):
        """Generate complete July-September 2025 dataset"""
        print("="*70)
        print("V6 FULL DATASET GENERATOR - JULY TO SEPTEMBER 2025")
        print("="*70)

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/metadata", exist_ok=True)

        # Define all expiries
        self.expiries = self._get_expiry_calendar()

        # Get all trading days
        trading_days = self._get_trading_days()

        print(f"\nGenerating {len(trading_days)} trading days...")
        print("-" * 50)

        files_generated = 0
        total_rows = 0

        for i, date in enumerate(trading_days, 1):
            print(f"[{i:02d}/{len(trading_days)}] Generating {date}...", end='', flush=True)

            try:
                df = self._generate_day_data(date)

                # Save file
                filename = f"NIFTY_OPTIONS_5MIN_{date.replace('-', '')}.csv"
                filepath = os.path.join(self.output_dir, filename)
                df.to_csv(filepath, index=False)

                files_generated += 1
                total_rows += len(df)
                print(f" ✅ ({len(df):,} rows)")

            except Exception as e:
                print(f" ❌ Error: {e}")

        # Save metadata
        self._save_metadata(files_generated, total_rows)

        print("\n" + "="*70)
        print(f"GENERATION COMPLETE!")
        print(f"Files Generated: {files_generated}")
        print(f"Total Data Rows: {total_rows:,}")
        print(f"Location: {self.output_dir}/")
        print("="*70)

    def _get_expiry_calendar(self):
        """Get all expiries for Jul-Sep 2025"""
        return [
            # July expiries
            ('2025-07-03', 'weekly'),
            ('2025-07-10', 'weekly'),
            ('2025-07-17', 'weekly'),
            ('2025-07-24', 'weekly'),
            ('2025-07-31', 'monthly'),
            # August expiries
            ('2025-08-07', 'weekly'),
            ('2025-08-14', 'weekly'),
            ('2025-08-21', 'weekly'),
            ('2025-08-28', 'monthly'),
            # September expiries (Wednesday from September)
            ('2025-09-03', 'weekly'),
            ('2025-09-10', 'weekly'),
            ('2025-09-17', 'weekly'),
            ('2025-09-25', 'monthly'),
        ]

    def _get_trading_days(self):
        """Get all trading days for Jul-Sep 2025"""
        trading_days = []

        # Generate all weekdays
        start_date = pd.Timestamp('2025-07-01')
        end_date = pd.Timestamp('2025-09-30')

        current = start_date
        while current <= end_date:
            # Skip weekends and holidays
            if current.weekday() < 5 and current.strftime('%Y-%m-%d') not in self.holidays:
                trading_days.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=1)

        return trading_days

    def _generate_day_data(self, date_str):
        """Generate data for one trading day"""
        all_data = []

        # Generate 79 timestamps (09:15 to 15:30)
        timestamps = pd.date_range(f'{date_str} 09:15:00', f'{date_str} 15:30:00', freq='5min')[:79]

        # Get active expiries (next 6 that haven't expired)
        active_expiries = [
            exp for exp in self.expiries
            if pd.Timestamp(exp[0]) >= pd.Timestamp(date_str)
        ][:6]

        # Generate strikes (reduced range for efficiency)
        strikes = self._generate_strikes()

        # Process each timestamp
        for timestamp in timestamps:
            # Evolve underlying price slightly
            self.underlying_price = self._evolve_underlying()

            # Generate options data for this timestamp
            for expiry_date, expiry_type in active_expiries:
                # Calculate days to expiry
                tte_days = (pd.Timestamp(expiry_date) - pd.Timestamp(timestamp)).days + 1
                if tte_days <= 0:
                    continue

                for strike in strikes:
                    for opt_type in ['CE', 'PE']:
                        option_data = self._generate_option_data(
                            timestamp, strike, opt_type,
                            expiry_date, expiry_type, tte_days
                        )
                        all_data.append(option_data)

        return pd.DataFrame(all_data)

    def _generate_strikes(self):
        """Generate strikes around ATM"""
        atm = round(self.underlying_price / 50) * 50
        # Generate strikes from -2000 to +2000 around ATM, 50 point intervals
        return list(range(atm - 2000, atm + 2001, 50))

    def _evolve_underlying(self):
        """Evolve underlying price with small random walk"""
        # Small random movement (0.02% std dev)
        change = np.random.normal(0, self.underlying_price * 0.0002)

        # Add mean reversion to base price
        month_bases = {7: 25000, 8: 25200, 9: 25400}
        target = 25000  # Default

        # Simple mean reversion
        mean_reversion = (target - self.underlying_price) * 0.001

        new_price = self.underlying_price + change + mean_reversion

        # Bound checking
        new_price = max(23000, min(27000, new_price))

        return new_price

    def _generate_option_data(self, timestamp, strike, opt_type,
                              expiry_date, expiry_type, tte_days):
        """Generate single option data point"""

        # Create unique key for price tracking
        key = (strike, opt_type, expiry_date)

        # Calculate base price if doesn't exist
        if key not in self.base_prices:
            self.base_prices[key] = self._calculate_initial_price(
                strike, opt_type, tte_days
            )

        # Evolve price from base
        current_price = self._evolve_option_price(
            self.base_prices[key], strike, opt_type, tte_days
        )

        # Update base for next iteration
        self.base_prices[key] = current_price

        # Calculate Greeks
        greeks = self._calculate_greeks(strike, opt_type, tte_days, current_price)

        # Calculate bid-ask spread
        spread = self._calculate_spread(current_price, strike)

        return {
            'timestamp': timestamp,
            'symbol': 'NIFTY',
            'strike': strike,
            'option_type': opt_type,
            'expiry': expiry_date,
            'expiry_type': expiry_type,
            'open': round(current_price * 0.99, 2),
            'high': round(current_price * 1.002, 2),
            'low': round(current_price * 0.998, 2),
            'close': round(current_price, 2),
            'volume': np.random.randint(100, 5000),
            'oi': np.random.randint(1000, 50000),
            'bid': round(max(current_price - spread/2, 0.05), 2),
            'ask': round(current_price + spread/2, 2),
            'iv': greeks['iv'],
            'delta': greeks['delta'],
            'gamma': greeks['gamma'],
            'theta': greeks['theta'],
            'vega': greeks['vega'],
            'underlying_price': round(self.underlying_price, 2)
        }

    def _calculate_initial_price(self, strike, opt_type, tte_days):
        """Calculate initial option price using simplified Black-Scholes"""
        S = self.underlying_price
        K = strike
        T = tte_days / 365.25
        r = 0.065  # Risk-free rate
        sigma = 0.15  # Implied volatility

        if T <= 0:
            if opt_type == 'CE':
                return max(S - K, 0.05)
            else:
                return max(K - S, 0.05)

        # Simplified Black-Scholes
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)

        if opt_type == 'CE':
            price = S * self._norm_cdf(d1) - K * np.exp(-r*T) * self._norm_cdf(d2)
        else:
            price = K * np.exp(-r*T) * self._norm_cdf(-d2) - S * self._norm_cdf(-d1)

        return max(price, 0.05)

    def _evolve_option_price(self, base_price, strike, opt_type, tte_days):
        """Evolve option price with controlled randomness"""

        # Calculate moneyness
        moneyness = self.underlying_price / strike

        # Volatility based on moneyness
        if abs(moneyness - 1) < 0.02:  # ATM
            vol = 0.003
        elif abs(moneyness - 1) < 0.10:  # Near ATM
            vol = 0.002
        else:  # Far OTM/ITM
            vol = 0.001

        # Add theta decay
        theta_decay = base_price * 0.0005 / max(tte_days, 1)

        # Random component
        random_change = base_price * np.random.normal(0, vol)

        # Calculate new price
        new_price = base_price - theta_decay + random_change

        # Ensure bounds
        new_price = max(new_price, 0.05)
        new_price = min(new_price, base_price * 1.05)  # Max 5% increase
        new_price = max(new_price, base_price * 0.95)  # Max 5% decrease

        return new_price

    def _calculate_greeks(self, strike, opt_type, tte_days, price):
        """Calculate simplified Greeks"""
        moneyness = self.underlying_price / strike

        # Delta
        if opt_type == 'CE':
            if moneyness > 1.1:
                delta = 0.9
            elif moneyness < 0.9:
                delta = 0.1
            else:
                delta = 0.5 + (moneyness - 1) * 2
            delta = max(0.01, min(0.99, delta))
        else:
            if moneyness < 0.9:
                delta = -0.9
            elif moneyness > 1.1:
                delta = -0.1
            else:
                delta = -0.5 - (1 - moneyness) * 2
            delta = max(-0.99, min(-0.01, delta))

        # Gamma (peaks at ATM)
        gamma = 0.001 * np.exp(-((moneyness - 1) ** 2) / 0.02)

        # Theta (time decay, always negative)
        theta = -price / max(tte_days, 1) * 0.5

        # Vega (sensitivity to IV)
        vega = price * 0.1 * np.exp(-((moneyness - 1) ** 2) / 0.02)

        # IV (implied volatility)
        iv = 0.15 * (1 + 0.5 * abs(moneyness - 1))

        return {
            'iv': round(iv, 4),
            'delta': round(delta, 4),
            'gamma': round(max(gamma, 0.0001), 6),
            'theta': round(min(theta, -0.01), 4),
            'vega': round(max(vega, 0.1), 2)
        }

    def _calculate_spread(self, price, strike):
        """Calculate bid-ask spread"""
        moneyness = self.underlying_price / strike

        if abs(moneyness - 1) < 0.05:  # ATM
            spread_pct = 0.01  # 1%
        elif abs(moneyness - 1) < 0.15:  # Near ATM
            spread_pct = 0.02  # 2%
        else:  # Far OTM/ITM
            spread_pct = 0.05  # 5%

        return price * spread_pct

    def _norm_cdf(self, x):
        """Cumulative normal distribution function"""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def _save_metadata(self, files_generated, total_rows):
        """Save generation metadata"""
        metadata = {
            'version': '6.0',
            'generation_date': datetime.now().isoformat(),
            'period': 'July-September 2025',
            'files_generated': files_generated,
            'total_rows': total_rows,
            'expiries': self.expiries,
            'holidays_excluded': self.holidays
        }

        # Save as JSON
        import json
        metadata_path = os.path.join(self.output_dir, 'metadata', 'generation_info.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"\nMetadata saved: {metadata_path}")


def main():
    """Run the full dataset generator"""
    print("\nStarting V6 Full Dataset Generation...")
    print("This will generate the complete July-September 2025 dataset")
    print("-" * 50)

    generator = V6FullDatasetGenerator()
    generator.generate_full_dataset()

    print("\n✅ Full dataset generation complete!")


if __name__ == "__main__":
    main()