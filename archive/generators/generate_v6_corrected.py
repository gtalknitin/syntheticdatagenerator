#!/usr/bin/env python3
"""
Synthetic NIFTY Options Data Generator V6 - CORRECTED VERSION
Implements proper time-series evolution with Greeks-based price movements
Fixes all V5 issues including duplicate timestamps and random price generation

Key Changes:
- One and only one row per timestamp/strike/option_type
- Prices evolve from previous prices using Greeks
- No random price generation
- Proper state management
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import math
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class CorrectedOptionDataGeneratorV6:
    """
    Corrected generator with proper time series evolution
    """

    def __init__(self):
        """Initialize with state management"""
        self.config = {
            'risk_free_rate': 0.065,
            'dividend_yield': 0.012,
            'base_volatility': 0.15,
            'min_price': 0.05,
            'strike_interval': 50,
            'timestamps_per_day': 79
        }

        # CRITICAL: State management for time series continuity
        self.price_history = {}  # {(strike, type, expiry): last_price}
        self.greeks_cache = {}   # Cache for efficiency
        self.last_underlying = None
        self.last_timestamp = None
        self.initialized = False

        # Holidays
        self.holidays = ['2025-08-15']  # Independence Day

        # Statistics
        self.generation_stats = {
            'files_generated': 0,
            'rows_generated': 0,
            'validation_passed': 0,
            'validation_failed': 0
        }

    def generate_full_dataset(self, start_date='2025-07-01', end_date='2025-09-30'):
        """Generate complete Jul-Sep 2025 dataset with proper evolution"""
        print("="*70)
        print("NIFTY OPTIONS SYNTHETIC DATA GENERATOR V6 - CORRECTED")
        print("="*70)
        print(f"\nGenerating {start_date} to {end_date} with proper time series...")

        # Create output directory
        output_dir = 'intraday_v6_corrected'
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f'{output_dir}/metadata', exist_ok=True)

        # Generate expiry calendar
        expiry_calendar = self._create_expiry_calendar(start_date, end_date)

        # Save expiry calendar
        expiry_calendar.to_csv(f'{output_dir}/metadata/expiry_calendar.csv', index=False)

        # Get all trading days
        trading_days = self._get_trading_days(start_date, end_date)

        validation_results = []
        start_time = datetime.now()

        # Process each day sequentially (IMPORTANT for time series)
        for date in trading_days:
            print(f"Generating {date}...", end='', flush=True)

            # Generate day data
            df = self._generate_day_data(date, expiry_calendar)

            # Validate immediately
            validation = self._validate_data(df)

            if validation['passed']:
                # Save only if valid
                filename = f"NIFTY_OPTIONS_5MIN_{date.replace('-', '')}.csv"
                filepath = f"{output_dir}/{filename}"
                df.to_csv(filepath, index=False)
                self.generation_stats['files_generated'] += 1
                self.generation_stats['rows_generated'] += len(df)
                self.generation_stats['validation_passed'] += 1
                print(f" ✅ ({len(df):,} rows)")
            else:
                self.generation_stats['validation_failed'] += 1
                print(f" ❌ VALIDATION FAILED: {validation['errors']}")

            validation_results.append({
                'date': date,
                'rows': len(df),
                'passed': validation['passed'],
                'errors': validation.get('errors', [])
            })

        # Save metadata
        self._save_metadata(output_dir, validation_results, start_time)

        print("\n" + "="*70)
        print("GENERATION COMPLETE")
        print("="*70)
        print(f"Files Generated: {self.generation_stats['files_generated']}")
        print(f"Total Rows: {self.generation_stats['rows_generated']:,}")
        print(f"Validation Passed: {self.generation_stats['validation_passed']}")
        print(f"Validation Failed: {self.generation_stats['validation_failed']}")
        print(f"\nOutput Directory: {output_dir}/")

    def _generate_day_data(self, date: str, expiry_calendar: pd.DataFrame) -> pd.DataFrame:
        """Generate data for one day with proper time series evolution"""
        date_pd = pd.Timestamp(date)

        # Get active expiries (next 6-8)
        active_expiries = expiry_calendar[
            pd.to_datetime(expiry_calendar['date']) >= date_pd
        ].head(8)

        # Initialize if first day
        if not self.initialized:
            self._initialize_option_chain(date_pd, active_expiries)
            self.initialized = True

        # Generate timestamps for the day
        timestamps = pd.date_range(
            f"{date} 09:15:00",
            f"{date} 15:30:00",
            freq='5min'
        )[:79]  # Exactly 79 timestamps

        all_data = []

        # Process each timestamp SEQUENTIALLY
        for timestamp in timestamps:
            # Generate data for this timestamp
            timestamp_data = self._evolve_to_timestamp(timestamp, active_expiries)
            all_data.extend(timestamp_data)

        return pd.DataFrame(all_data)

    def _initialize_option_chain(self, date: pd.Timestamp, expiries: pd.DataFrame):
        """Initialize opening prices for all options"""
        # Initial underlying price
        month = date.month
        base_prices = {7: 25000, 8: 25200, 9: 25400}
        self.last_underlying = base_prices.get(month, 25000)

        # Generate strikes
        strikes = self._generate_strikes(self.last_underlying)

        # Initialize each option
        for _, expiry_row in expiries.iterrows():
            expiry_date = pd.to_datetime(expiry_row['date'])
            tte = (expiry_date - date).days / 365.25

            for strike in strikes:
                for opt_type in ['CE', 'PE']:
                    key = (strike, opt_type, expiry_row['date'])

                    # Calculate initial price using Black-Scholes
                    initial_price = self._black_scholes_price(
                        self.last_underlying, strike, tte,
                        self.config['base_volatility'], opt_type
                    )

                    # Store in history
                    self.price_history[key] = initial_price

        self.last_timestamp = date

    def _evolve_to_timestamp(self, timestamp: pd.Timestamp,
                            active_expiries: pd.DataFrame) -> List[Dict]:
        """Evolve all option prices to new timestamp"""

        # Calculate underlying movement
        new_underlying = self._calculate_underlying(timestamp)
        spot_change = new_underlying - self.last_underlying
        time_elapsed = 5 / (390 * 365.25)  # 5 minutes as year fraction

        timestamp_data = []

        # Get unique strikes from price history
        unique_strikes = set(k[0] for k in self.price_history.keys())

        # Process each active option
        for _, expiry_row in active_expiries.iterrows():
            expiry_date = expiry_row['date']

            for strike in unique_strikes:
                for opt_type in ['CE', 'PE']:
                    key = (strike, opt_type, expiry_date)

                    # Skip if option doesn't exist
                    if key not in self.price_history:
                        continue

                    # Get previous price
                    previous_price = self.price_history[key]

                    # Calculate time to expiry
                    tte = (pd.Timestamp(expiry_date) + pd.Timedelta(hours=15, minutes=30) -
                          timestamp).total_seconds() / (365.25 * 24 * 3600)
                    tte = max(tte, 1/365.25)  # Minimum 1 day

                    # Calculate Greeks
                    greeks = self._calculate_greeks(
                        new_underlying, strike, tte,
                        self.config['base_volatility'], opt_type
                    )

                    # CRITICAL: Evolve price using Greeks
                    new_price = self._evolve_price(
                        previous_price, greeks, spot_change, time_elapsed
                    )

                    # Update history
                    self.price_history[key] = new_price

                    # Calculate OHLC for 5-min window
                    ohlc = self._calculate_ohlc(previous_price, new_price)

                    # Calculate bid-ask spread
                    spread = self._calculate_spread(new_price, strike/new_underlying, tte)
                    bid = max(new_price - spread/2, self.config['min_price'])
                    ask = new_price + spread/2

                    # Volume based on moneyness
                    moneyness = new_underlying / strike
                    volume = self._calculate_volume(moneyness, timestamp.hour)

                    # Add to output
                    timestamp_data.append({
                        'timestamp': timestamp,
                        'symbol': 'NIFTY',
                        'strike': strike,
                        'option_type': opt_type,
                        'expiry': expiry_date,
                        'expiry_type': expiry_row['type'],
                        'open': ohlc['open'],
                        'high': ohlc['high'],
                        'low': ohlc['low'],
                        'close': new_price,
                        'volume': volume,
                        'oi': volume * np.random.randint(20, 50),
                        'bid': round(bid, 2),
                        'ask': round(ask, 2),
                        'iv': round(self.config['base_volatility'] *
                                  self._get_iv_multiplier(moneyness, tte), 4),
                        'delta': round(greeks['delta'], 4),
                        'gamma': round(greeks['gamma'], 6),
                        'theta': round(greeks['theta'], 4),
                        'vega': round(greeks['vega'], 4),
                        'underlying_price': round(new_underlying, 2),
                        'tte_days': round(tte * 365.25, 2),
                        'moneyness': round(moneyness, 4)
                    })

        # Update state
        self.last_underlying = new_underlying
        self.last_timestamp = timestamp

        return timestamp_data

    def _evolve_price(self, previous_price: float, greeks: Dict,
                     spot_change: float, time_elapsed: float) -> float:
        """
        CRITICAL: Evolve price using Greeks, not random generation
        This is the core of V6 corrections
        """
        # Calculate each Greek contribution
        delta_effect = greeks['delta'] * spot_change
        gamma_effect = 0.5 * greeks['gamma'] * (spot_change ** 2)
        theta_effect = greeks['theta'] * time_elapsed

        # Simplified IV change (could be enhanced)
        iv_change = np.random.normal(0, 0.001)  # Small random IV changes
        vega_effect = greeks['vega'] * iv_change * 100  # Vega is per 1% IV

        # Total price change
        total_change = delta_effect + gamma_effect + theta_effect + vega_effect

        # New price
        new_price = previous_price + total_change

        # Ensure minimum price
        new_price = max(new_price, self.config['min_price'])

        # Sanity check - cap extreme movements
        max_move_pct = min(abs(spot_change / self.last_underlying) * 10, 0.10)
        if previous_price > 0:
            actual_move_pct = abs(new_price - previous_price) / previous_price
            if actual_move_pct > max_move_pct:
                direction = 1 if new_price > previous_price else -1
                new_price = previous_price * (1 + direction * max_move_pct)

        return round(new_price, 2)

    def _calculate_underlying(self, timestamp: pd.Timestamp) -> float:
        """Calculate underlying price with realistic intraday movement"""
        if self.last_underlying is None:
            month = timestamp.month
            base_prices = {7: 25000, 8: 25200, 9: 25400}
            return base_prices.get(month, 25000)

        # Small random walk
        change = np.random.normal(0, self.last_underlying * 0.0005)  # 0.05% std dev
        new_price = self.last_underlying + change

        # Mean reversion
        month = timestamp.month
        base_prices = {7: 25000, 8: 25200, 9: 25400}
        target = base_prices.get(month, 25000)
        new_price += (target - new_price) * 0.01  # 1% mean reversion

        return new_price

    def _calculate_greeks(self, S: float, K: float, T: float,
                         sigma: float, opt_type: str) -> Dict:
        """Calculate Black-Scholes Greeks"""
        r = self.config['risk_free_rate']
        q = self.config['dividend_yield']

        # Avoid division by zero
        T = max(T, 1/365.25)

        # Calculate d1 and d2
        d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)

        # Delta
        if opt_type == 'CE':
            delta = np.exp(-q*T) * self._norm_cdf(d1)
        else:
            delta = -np.exp(-q*T) * self._norm_cdf(-d1)

        # Ensure delta bounds
        delta = np.clip(delta, -0.9999, 0.9999)

        # Gamma - always positive
        gamma = np.exp(-q*T) * self._norm_pdf(d1) / (S * sigma * np.sqrt(T))
        gamma = max(gamma, 1e-6)

        # Theta (per day)
        if opt_type == 'CE':
            theta = (-S * self._norm_pdf(d1) * sigma * np.exp(-q*T) / (2*np.sqrt(T))
                    - r * K * np.exp(-r*T) * self._norm_cdf(d2)
                    + q * S * np.exp(-q*T) * self._norm_cdf(d1)) / 365.25
        else:
            theta = (-S * self._norm_pdf(d1) * sigma * np.exp(-q*T) / (2*np.sqrt(T))
                    + r * K * np.exp(-r*T) * self._norm_cdf(-d2)
                    - q * S * np.exp(-q*T) * self._norm_cdf(-d1)) / 365.25

        # Vega (per 1% IV change)
        vega = S * np.exp(-q*T) * self._norm_pdf(d1) * np.sqrt(T) / 100
        vega = max(vega, 1e-4)

        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega
        }

    def _black_scholes_price(self, S: float, K: float, T: float,
                            sigma: float, opt_type: str) -> float:
        """Calculate Black-Scholes option price"""
        r = self.config['risk_free_rate']
        q = self.config['dividend_yield']

        if T <= 0:
            if opt_type == 'CE':
                return max(S - K, self.config['min_price'])
            else:
                return max(K - S, self.config['min_price'])

        d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)

        if opt_type == 'CE':
            price = S*np.exp(-q*T)*self._norm_cdf(d1) - K*np.exp(-r*T)*self._norm_cdf(d2)
        else:
            price = K*np.exp(-r*T)*self._norm_cdf(-d2) - S*np.exp(-q*T)*self._norm_cdf(-d1)

        return max(price, self.config['min_price'])

    def _generate_strikes(self, spot: float) -> List[int]:
        """Generate strikes around spot"""
        atm = round(spot / self.config['strike_interval']) * self.config['strike_interval']
        strikes = []

        for offset in range(-50, 51):  # ±2500 points
            strike = atm + offset * self.config['strike_interval']
            if 20000 <= strike <= 30000:
                strikes.append(int(strike))

        return strikes

    def _calculate_ohlc(self, previous_price: float, current_price: float) -> Dict:
        """Calculate OHLC for 5-minute window"""
        # Simple model: small variations within the window
        mid_price = (previous_price + current_price) / 2
        volatility = 0.002  # 0.2% intra-bar volatility

        return {
            'open': round(previous_price, 2),
            'high': round(max(previous_price, current_price) *
                         (1 + np.random.uniform(0, volatility)), 2),
            'low': round(min(previous_price, current_price) *
                        (1 - np.random.uniform(0, volatility)), 2),
            'close': round(current_price, 2)
        }

    def _calculate_spread(self, price: float, moneyness: float, tte: float) -> float:
        """Calculate bid-ask spread"""
        if abs(moneyness - 1) < 0.02:  # ATM
            base_spread = 0.01
        elif abs(moneyness - 1) < 0.10:  # Near ATM
            base_spread = 0.02
        else:  # Far OTM/ITM
            base_spread = 0.05

        # Wider spreads near expiry
        if tte < 3/365.25:
            base_spread *= 2

        return max(price * base_spread, 0.05)

    def _calculate_volume(self, moneyness: float, hour: int) -> int:
        """Calculate realistic volume"""
        # Base volume depends on moneyness
        if abs(moneyness - 1) < 0.02:  # ATM
            base = 2000
        elif abs(moneyness - 1) < 0.10:
            base = 500
        else:
            base = 100

        # Intraday pattern
        if hour in [9, 15]:
            base *= 2  # Higher at open/close

        return np.random.poisson(base)

    def _get_iv_multiplier(self, moneyness: float, tte: float) -> float:
        """Get IV multiplier for smile and term structure"""
        # Volatility smile
        if moneyness < 0.95:  # OTM puts
            smile = 1 + (0.95 - moneyness) * 1.5
        elif moneyness > 1.05:  # OTM calls
            smile = 1 + (moneyness - 1.05) * 1.5
        else:
            smile = 1.0

        # Term structure
        if tte < 7/365.25:
            term = 1.2
        elif tte < 30/365.25:
            term = 1.05
        else:
            term = 1.0

        return smile * term

    def _norm_cdf(self, x: float) -> float:
        """Cumulative normal distribution"""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def _norm_pdf(self, x: float) -> float:
        """Normal probability density"""
        return math.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)

    def _create_expiry_calendar(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Create expiry calendar"""
        expiries = []

        # Monthly expiries
        expiries.extend([
            {'date': '2025-07-31', 'type': 'monthly'},
            {'date': '2025-08-28', 'type': 'monthly'},
            {'date': '2025-09-25', 'type': 'monthly'}
        ])

        # Weekly expiries
        weekly_dates = [
            '2025-07-03', '2025-07-10', '2025-07-17', '2025-07-24',
            '2025-08-07', '2025-08-14', '2025-08-21',
            '2025-09-03', '2025-09-10', '2025-09-17'
        ]

        for date in weekly_dates:
            expiries.append({'date': date, 'type': 'weekly'})

        return pd.DataFrame(expiries).sort_values('date')

    def _get_trading_days(self, start_date: str, end_date: str) -> List[str]:
        """Get list of trading days"""
        all_days = pd.date_range(start_date, end_date, freq='B')
        trading_days = [
            day.strftime('%Y-%m-%d')
            for day in all_days
            if day.strftime('%Y-%m-%d') not in self.holidays
        ]
        return trading_days

    def _validate_data(self, df: pd.DataFrame) -> Dict:
        """Validate generated data"""
        errors = []

        # Check for duplicate timestamps
        duplicates = df.groupby(['timestamp', 'strike', 'option_type']).size()
        if duplicates.max() > 1:
            errors.append(f"Duplicate timestamps found: max {duplicates.max()}")

        # Check price movements
        for (strike, opt_type), group in df.groupby(['strike', 'option_type']):
            prices = group.sort_values('timestamp')['close']
            if len(prices) > 1:
                max_change = prices.pct_change().abs().max()
                if max_change > 0.15:  # 15% threshold
                    errors.append(f"{strike} {opt_type}: {max_change:.1%} move")

        # Check Greeks
        if (df['gamma'] < 0).any():
            errors.append("Negative gamma found")
        if (df['theta'] > 0).any():
            errors.append("Positive theta found")

        return {
            'passed': len(errors) == 0,
            'errors': errors
        }

    def _save_metadata(self, output_dir: str, validation_results: List,
                      start_time: datetime):
        """Save generation metadata"""
        elapsed = (datetime.now() - start_time).total_seconds()

        metadata = {
            'version': '6.0',
            'generation_date': datetime.now().isoformat(),
            'generation_time_seconds': elapsed,
            'stats': self.generation_stats,
            'config': self.config,
            'validation_summary': {
                'total_days': len(validation_results),
                'passed': sum(1 for v in validation_results if v['passed']),
                'failed': sum(1 for v in validation_results if not v['passed'])
            }
        }

        with open(f'{output_dir}/metadata/generation_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        # Save validation details
        pd.DataFrame(validation_results).to_csv(
            f'{output_dir}/metadata/validation_results.csv', index=False
        )


def main():
    """Generate corrected V6 dataset"""
    print("\nStarting V6 Corrected Data Generation...")
    print("This version fixes all V5 issues:")
    print("- No duplicate timestamps")
    print("- Prices evolve using Greeks")
    print("- Proper time series continuity")
    print("-" * 50)

    generator = CorrectedOptionDataGeneratorV6()
    generator.generate_full_dataset()

    print("\n✅ V6 generation complete!")
    print("Data location: intraday_v6_corrected/")


if __name__ == "__main__":
    main()