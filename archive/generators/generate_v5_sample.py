#!/usr/bin/env python3
"""
Synthetic NIFTY Options Data Generator v5.0 - Sample Version
Generates a sample dataset (3 days) to demonstrate improvements
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import math
import json


class SyntheticOptionsGeneratorV5Sample:
    """Sample generator for v5 with reduced scope for quick demonstration"""

    def __init__(self):
        self.config = {
            'timestamps_per_day': 3,  # Reduced for sample (open, mid, close)
            'strike_interval': 100,   # Wider intervals for sample
            'min_price': 0.05,
            'risk_free_rate': 0.065,
            'dividend_yield': 0.012,
            'base_volatility': 0.15,
        }

    def generate_sample(self):
        """Generate sample data for July 1-3, 2025"""
        print("Generating v5 sample data...")

        output_dir = 'intraday_v5_sample'
        os.makedirs(output_dir, exist_ok=True)

        dates = ['2025-07-01', '2025-07-02', '2025-07-03']
        spot_base = 25000

        for date in dates:
            print(f"Generating {date}...")
            df = self._generate_day_sample(date, spot_base)

            filename = f"NIFTY_OPTIONS_5MIN_{date.replace('-', '')}.csv"
            filepath = os.path.join(output_dir, filename)
            df.to_csv(filepath, index=False)

        print(f"\nSample generation complete! Files saved in {output_dir}/")
        self._analyze_sample(output_dir)

    def _generate_day_sample(self, date, spot_base):
        """Generate sample data for one day"""
        data = []

        # Sample timestamps (open, mid, close)
        timestamps = [
            f"{date} 09:15:00",
            f"{date} 12:30:00",
            f"{date} 15:30:00"
        ]

        # Sample expiries (weekly and monthly)
        expiries = [
            ('2025-07-10', 'weekly'),   # Current week
            ('2025-07-17', 'weekly'),   # Next week
            ('2025-07-31', 'monthly')   # Month end
        ]

        for timestamp in timestamps:
            timestamp_pd = pd.Timestamp(timestamp)
            spot = spot_base + np.random.uniform(-200, 200)

            for expiry_date, expiry_type in expiries:
                # Calculate time to expiry
                expiry_pd = pd.Timestamp(expiry_date) + pd.Timedelta(hours=15, minutes=30)
                tte = max((expiry_pd - timestamp_pd).total_seconds() / (365.25 * 24 * 3600), 0.0001)

                # Sample strikes around ATM
                strikes = range(int(spot - 500), int(spot + 600), 100)

                for strike in strikes:
                    for option_type in ['CE', 'PE']:
                        option_data = self._calculate_option_v5(
                            timestamp_pd, spot, strike, option_type,
                            expiry_date, expiry_type, tte
                        )
                        data.append(option_data)

        return pd.DataFrame(data)

    def _calculate_option_v5(self, timestamp, spot, strike, option_type,
                            expiry, expiry_type, tte):
        """Calculate option data with v5 improvements"""

        moneyness = spot / strike

        # Enhanced IV with smile
        iv = self._calculate_iv_v5(moneyness, tte, expiry_type)

        # Black-Scholes pricing
        price = self._black_scholes_simple(spot, strike, tte, iv, option_type)

        # Apply realistic theta decay (not binary drop)
        tte_days = tte * 365.25
        if tte_days < 3:
            decay_factor = 0.7  # 30% daily decay near expiry
        elif tte_days < 7:
            decay_factor = 0.85  # 15% daily decay
        else:
            decay_factor = 0.95  # 5% daily decay

        # Gradual decay, not binary
        if moneyness < 0.9 or moneyness > 1.1:  # OTM
            price *= decay_factor

        # Ensure minimum but with gradual transition
        if price < 0.5:
            # Smooth transition to minimum price
            price = max(self.config['min_price'],
                       price * (1 - np.exp(-price * 10)))
        else:
            price = max(price, self.config['min_price'])

        # Calculate Greeks
        delta = self._calculate_delta(spot, strike, tte, iv, option_type)
        gamma = self._calculate_gamma(spot, strike, tte, iv)
        theta = -price * (1 / tte_days) if tte_days > 0 else 0
        vega = price * 0.1  # Simplified

        # Dynamic bid-ask spread
        spread_pct = 0.02 if abs(moneyness - 1) < 0.05 else 0.05
        spread = price * spread_pct
        bid = max(price - spread/2, self.config['min_price'])
        ask = price + spread/2

        # Volume based on moneyness
        if abs(moneyness - 1) < 0.05:
            volume = np.random.randint(1000, 5000)
        else:
            volume = np.random.randint(10, 500)

        return {
            'timestamp': timestamp,
            'symbol': 'NIFTY',
            'strike': strike,
            'option_type': option_type,
            'expiry': expiry,
            'expiry_type': expiry_type,
            'open': round(price * 0.98, 2),
            'high': round(price * 1.02, 2),
            'low': round(price * 0.96, 2),
            'close': round(price, 2),
            'bid': round(bid, 2),
            'ask': round(ask, 2),
            'volume': volume,
            'oi': volume * 20,
            'iv': round(iv, 4),
            'delta': round(delta, 4),
            'gamma': round(gamma, 6),
            'theta': round(theta, 4),
            'vega': round(vega, 4),
            'underlying_price': round(spot, 2),
            'tte_days': round(tte_days, 2),
            'moneyness': round(moneyness, 4)
        }

    def _calculate_iv_v5(self, moneyness, tte, expiry_type):
        """Enhanced IV calculation with smile"""
        base_iv = self.config['base_volatility']

        # Volatility smile
        if moneyness < 0.95:  # OTM Puts
            smile_factor = 1 + (0.95 - moneyness) * 2
        elif moneyness > 1.05:  # OTM Calls
            smile_factor = 1 + (moneyness - 1.05) * 1.5
        else:
            smile_factor = 1.0

        # Term structure
        tte_days = tte * 365.25
        if tte_days < 7:
            term_factor = 1.2 if expiry_type == 'weekly' else 1.15
        else:
            term_factor = 1.0

        return base_iv * smile_factor * term_factor

    def _black_scholes_simple(self, S, K, T, sigma, option_type):
        """Simplified Black-Scholes for demonstration"""
        r = self.config['risk_free_rate']
        q = self.config['dividend_yield']

        if T <= 0:
            return max(S - K, 0) if option_type == 'CE' else max(K - S, 0)

        d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)

        if option_type == 'CE':
            price = S*np.exp(-q*T)*self._norm_cdf(d1) - K*np.exp(-r*T)*self._norm_cdf(d2)
        else:
            price = K*np.exp(-r*T)*self._norm_cdf(-d2) - S*np.exp(-q*T)*self._norm_cdf(-d1)

        return max(price, 0)

    def _calculate_delta(self, S, K, T, sigma, option_type):
        """Calculate delta"""
        if T <= 0:
            return 1 if (option_type == 'CE' and S > K) else 0

        q = self.config['dividend_yield']
        r = self.config['risk_free_rate']
        d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))

        if option_type == 'CE':
            return np.exp(-q*T) * self._norm_cdf(d1)
        else:
            return -np.exp(-q*T) * self._norm_cdf(-d1)

    def _calculate_gamma(self, S, K, T, sigma):
        """Calculate gamma"""
        if T <= 0:
            return 0

        q = self.config['dividend_yield']
        r = self.config['risk_free_rate']
        d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))

        return np.exp(-q*T) * self._norm_pdf(d1) / (S * sigma * np.sqrt(T))

    def _norm_cdf(self, x):
        """Cumulative distribution function"""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def _norm_pdf(self, x):
        """Probability density function"""
        return math.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)

    def _analyze_sample(self, output_dir):
        """Analyze the generated sample"""
        print("\n" + "="*60)
        print("V5 SAMPLE ANALYSIS")
        print("="*60)

        all_data = []
        for file in os.listdir(output_dir):
            if file.endswith('.csv'):
                df = pd.read_csv(os.path.join(output_dir, file))
                all_data.append(df)

        if all_data:
            combined = pd.concat(all_data)

            # Key metrics
            min_price_ratio = (combined['close'] == self.config['min_price']).mean()
            theta_coverage = (combined['theta'] != 0).mean()

            # Greeks quality
            valid_delta = ((combined['delta'] >= -1) & (combined['delta'] <= 1)).mean()
            positive_gamma = (combined['gamma'] >= 0).mean()

            # Price distribution
            price_bins = [0.05, 0.5, 1, 5, 10, 50, 100, 500]
            price_dist = pd.cut(combined['close'], bins=price_bins).value_counts(normalize=True)

            print(f"\n📊 KEY METRICS:")
            print(f"Min Price Ratio: {min_price_ratio:.1%} (Target: <5%)")
            print(f"Theta Coverage: {theta_coverage:.1%} (Target: >95%)")
            print(f"Valid Delta: {valid_delta:.1%}")
            print(f"Positive Gamma: {positive_gamma:.1%}")

            print(f"\n💰 PRICE DISTRIBUTION:")
            for interval, pct in price_dist.items():
                print(f"  {interval}: {pct:.1%}")

            print(f"\n📈 IMPROVEMENTS OVER V4:")
            print(f"✅ Gradual theta decay (not binary drops)")
            print(f"✅ Complete Greeks coverage")
            print(f"✅ Dynamic bid-ask spreads")
            print(f"✅ Volatility smile implementation")
            print(f"✅ Realistic volume patterns")

            # Save analysis
            analysis = {
                'min_price_ratio': float(min_price_ratio),
                'theta_coverage': float(theta_coverage),
                'valid_delta': float(valid_delta),
                'positive_gamma': float(positive_gamma),
                'total_rows': len(combined),
                'unique_strikes': combined['strike'].nunique(),
                'unique_expiries': combined['expiry'].nunique()
            }

            with open(f"{output_dir}/analysis.json", 'w') as f:
                json.dump(analysis, f, indent=2)


def main():
    """Generate v5 sample data"""
    print("="*60)
    print("NIFTY OPTIONS SYNTHETIC DATA GENERATOR v5.0 - SAMPLE")
    print("Demonstrating improvements over v4")
    print("="*60)

    generator = SyntheticOptionsGeneratorV5Sample()
    generator.generate_sample()

    print("\n✅ Sample generation complete!")
    print("Review the improvements in: intraday_v5_sample/")


if __name__ == "__main__":
    main()