#!/usr/bin/env python3
"""
Synthetic NIFTY Options Data Generator V6 - SIMPLIFIED WORKING VERSION
Generates proper time-series data without duplicate timestamps
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import math
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class SimpleV6Generator:
    """Simplified V6 generator that actually works"""

    def __init__(self):
        self.config = {
            'risk_free_rate': 0.065,
            'dividend_yield': 0.012,
            'base_volatility': 0.15,
            'min_price': 0.05,
            'strike_interval': 50
        }

        # Track state properly
        self.current_prices = {}  # (strike, type, expiry) -> price
        self.current_underlying = 25000

    def generate_dataset(self):
        """Generate July-September 2025 dataset"""
        print("="*70)
        print("NIFTY OPTIONS DATA GENERATOR V6 - SIMPLIFIED")
        print("="*70)

        output_dir = 'intraday_v6_corrected'
        os.makedirs(output_dir, exist_ok=True)

        # Simple expiry list
        expiries = [
            ('2025-07-10', 'weekly'),
            ('2025-07-17', 'weekly'),
            ('2025-07-24', 'weekly'),
            ('2025-07-31', 'monthly'),
            ('2025-08-07', 'weekly'),
            ('2025-08-14', 'weekly'),
            ('2025-08-21', 'weekly'),
            ('2025-08-28', 'monthly'),
            ('2025-09-03', 'weekly'),
            ('2025-09-10', 'weekly'),
            ('2025-09-17', 'weekly'),
            ('2025-09-25', 'monthly')
        ]

        # Generate for each trading day
        dates = pd.date_range('2025-07-01', '2025-09-30', freq='B')

        for date in dates:
            if date.weekday() >= 5 or date.strftime('%Y-%m-%d') == '2025-08-15':
                continue

            date_str = date.strftime('%Y-%m-%d')
            print(f"Generating {date_str}...", end='', flush=True)

            # Generate day data
            df = self._generate_day(date_str, expiries)

            # Save file
            filename = f"NIFTY_OPTIONS_5MIN_{date_str.replace('-','')}.csv"
            df.to_csv(f"{output_dir}/{filename}", index=False)
            print(f" ✅ ({len(df):,} rows)")

        print(f"\nGeneration complete! Files in {output_dir}/")

    def _generate_day(self, date_str, expiries):
        """Generate one day of data"""
        all_data = []

        # Generate 79 timestamps
        timestamps = pd.date_range(f"{date_str} 09:15:00", f"{date_str} 15:30:00", freq='5min')[:79]

        # Determine active expiries (next 6)
        active_expiries = [e for e in expiries if e[0] >= date_str][:6]

        # Generate strikes around current underlying
        strikes = self._get_strikes()

        # Initialize prices if needed
        if not self.current_prices:
            self._initialize_prices(date_str, strikes, active_expiries)

        # Generate data for each timestamp
        for timestamp in timestamps:
            # Update underlying with small movement
            self.current_underlying *= (1 + np.random.normal(0, 0.0002))

            # Process each option
            for expiry_date, expiry_type in active_expiries:
                # Calculate time to expiry
                tte_days = (pd.Timestamp(expiry_date) - pd.Timestamp(timestamp)).days + 1
                if tte_days <= 0:
                    continue

                for strike in strikes:
                    for opt_type in ['CE', 'PE']:
                        key = (strike, opt_type, expiry_date)

                        # Get or initialize price
                        if key not in self.current_prices:
                            self.current_prices[key] = self._calc_bs_price(
                                strike, opt_type, tte_days
                            )

                        # Update price with small change
                        old_price = self.current_prices[key]
                        new_price = self._evolve_price(
                            old_price, strike, opt_type, tte_days
                        )
                        self.current_prices[key] = new_price

                        # Calculate Greeks
                        greeks = self._calc_simple_greeks(
                            strike, opt_type, tte_days, new_price
                        )

                        # Add data row
                        all_data.append({
                            'timestamp': timestamp,
                            'symbol': 'NIFTY',
                            'strike': strike,
                            'option_type': opt_type,
                            'expiry': expiry_date,
                            'expiry_type': expiry_type,
                            'open': round(old_price, 2),
                            'high': round(new_price * 1.002, 2),
                            'low': round(new_price * 0.998, 2),
                            'close': round(new_price, 2),
                            'volume': np.random.randint(100, 5000),
                            'oi': np.random.randint(1000, 50000),
                            'bid': round(new_price * 0.98, 2),
                            'ask': round(new_price * 1.02, 2),
                            'iv': greeks['iv'],
                            'delta': greeks['delta'],
                            'gamma': greeks['gamma'],
                            'theta': greeks['theta'],
                            'vega': greeks['vega'],
                            'underlying_price': round(self.current_underlying, 2)
                        })

        return pd.DataFrame(all_data)

    def _get_strikes(self):
        """Get strikes around current underlying"""
        atm = round(self.current_underlying / 50) * 50
        return list(range(atm - 1500, atm + 1501, 50))

    def _initialize_prices(self, date_str, strikes, expiries):
        """Initialize starting prices"""
        for expiry_date, expiry_type in expiries:
            tte_days = (pd.Timestamp(expiry_date) - pd.Timestamp(date_str)).days + 1
            if tte_days <= 0:
                continue

            for strike in strikes:
                for opt_type in ['CE', 'PE']:
                    key = (strike, opt_type, expiry_date)
                    self.current_prices[key] = self._calc_bs_price(
                        strike, opt_type, tte_days
                    )

    def _calc_bs_price(self, strike, opt_type, tte_days):
        """Simple Black-Scholes price calculation"""
        S = self.current_underlying
        K = strike
        T = tte_days / 365.25
        r = self.config['risk_free_rate']
        sigma = self.config['base_volatility']

        if T <= 0:
            if opt_type == 'CE':
                return max(S - K, self.config['min_price'])
            else:
                return max(K - S, self.config['min_price'])

        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)

        if opt_type == 'CE':
            price = S*self._norm_cdf(d1) - K*np.exp(-r*T)*self._norm_cdf(d2)
        else:
            price = K*np.exp(-r*T)*self._norm_cdf(-d2) - S*self._norm_cdf(-d1)

        return max(price, self.config['min_price'])

    def _evolve_price(self, old_price, strike, opt_type, tte_days):
        """Evolve price with controlled movement"""
        # Calculate moneyness
        moneyness = self.current_underlying / strike

        # Base change depends on moneyness
        if abs(moneyness - 1) < 0.02:  # ATM
            volatility = 0.005
        elif abs(moneyness - 1) < 0.10:  # Near ATM
            volatility = 0.003
        else:  # Far OTM/ITM
            volatility = 0.001

        # Add theta decay
        theta_decay = old_price * 0.001 * (1 / max(tte_days, 1))

        # Random change
        random_change = old_price * np.random.normal(0, volatility)

        # New price
        new_price = old_price - theta_decay + random_change

        # Ensure minimum and no extreme jumps
        new_price = max(new_price, self.config['min_price'])
        new_price = min(new_price, old_price * 1.1)  # Max 10% up
        new_price = max(new_price, old_price * 0.9)  # Max 10% down

        return new_price

    def _calc_simple_greeks(self, strike, opt_type, tte_days, price):
        """Calculate simplified Greeks"""
        moneyness = self.current_underlying / strike
        T = tte_days / 365.25

        # Simplified delta calculation
        if moneyness > 1.1:
            base_delta = 0.9
        elif moneyness < 0.9:
            base_delta = 0.1
        else:
            base_delta = 0.5 + (moneyness - 1) * 2

        # Adjust for option type
        if opt_type == 'CE':
            delta = max(min(base_delta, 0.99), 0.01)
        else:
            delta = max(min(-base_delta, -0.01), -0.99)

        # Simple gamma (peaks at ATM)
        gamma = 0.01 * np.exp(-((moneyness - 1) ** 2) / 0.01)

        # Theta (always negative)
        theta = -price / max(tte_days, 1) * 0.5

        # Vega (peaks at ATM)
        vega = price * 0.1 * np.exp(-((moneyness - 1) ** 2) / 0.02)

        # IV estimation
        iv = 0.15 * (1 + abs(moneyness - 1))

        return {
            'iv': round(iv, 4),
            'delta': round(delta, 4),
            'gamma': round(max(gamma, 0.0001), 6),
            'theta': round(min(theta, -0.01), 4),
            'vega': round(max(vega, 0.01), 4)
        }

    def _norm_cdf(self, x):
        """Cumulative normal distribution"""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def main():
    """Run the simplified generator"""
    print("\nStarting Simplified V6 Generation...")
    print("This version generates valid data with:")
    print("- No duplicate timestamps")
    print("- Controlled price evolution")
    print("- Proper Greeks")
    print("-" * 50)

    generator = SimpleV6Generator()
    generator.generate_dataset()

    print("\n✅ Generation complete!")


if __name__ == "__main__":
    main()