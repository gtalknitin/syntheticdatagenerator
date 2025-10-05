#!/usr/bin/env python3
"""
Synthetic NIFTY Options Data Generator
Based on PRD v1.0.0 - September 16, 2025
Generates realistic 5-minute options data following Black-Scholes model
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
from scipy.stats import norm
import os
import json
from typing import Dict, List, Tuple

class NiftyOptionsDataGenerator:
    """Generate synthetic NIFTY options data with 5-minute granularity"""

    def __init__(self):
        # Market parameters as per PRD
        self.risk_free_rate = 0.065  # 6.5%
        self.dividend_yield = 0.012   # 1.2%

        # Strike configuration
        self.strike_interval = 50
        self.strikes_range = 2000  # ATM ± 2000 points

        # Trading hours
        self.market_open = "09:15"
        self.market_close = "15:30"
        self.bar_duration = 5  # minutes
        self.bars_per_day = 75  # 6.25 hours * 12 bars per hour

        # Market regime probabilities
        self.market_regimes = {
            'bull': {'prob': 0.35, 'daily_return': (0.005, 0.02), 'iv_mult': 0.9},
            'bear': {'prob': 0.25, 'daily_return': (-0.02, -0.005), 'iv_mult': 1.2},
            'sideways': {'prob': 0.30, 'daily_return': (-0.003, 0.003), 'iv_mult': 1.0},
            'high_vol': {'prob': 0.10, 'daily_return': (-0.04, 0.04), 'iv_mult': 1.5}
        }

        # Base IV ranges
        self.base_iv = {
            'normal': (0.12, 0.30),
            'high': (0.30, 0.50),
            'extreme': (0.50, 0.70)
        }

        # Trading calendar (NSE holidays excluded)
        self.holidays = [
            datetime(2025, 8, 15),  # Independence Day
        ]

    def black_scholes_price(self, spot: float, strike: float, time_to_expiry: float,
                           volatility: float, option_type: str) -> float:
        """Calculate Black-Scholes option price"""
        if time_to_expiry <= 0:
            return max(0, spot - strike) if option_type == 'CE' else max(0, strike - spot)

        d1 = (np.log(spot / strike) + (self.risk_free_rate - self.dividend_yield + 0.5 * volatility ** 2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
        d2 = d1 - volatility * np.sqrt(time_to_expiry)

        if option_type == 'CE':
            price = spot * np.exp(-self.dividend_yield * time_to_expiry) * norm.cdf(d1) - strike * np.exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(d2)
        else:  # PE
            price = strike * np.exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(-d2) - spot * np.exp(-self.dividend_yield * time_to_expiry) * norm.cdf(-d1)

        return max(0.05, price)  # Minimum price of 0.05

    def calculate_greeks(self, spot: float, strike: float, time_to_expiry: float,
                        volatility: float, option_type: str) -> Dict[str, float]:
        """Calculate option Greeks"""
        if time_to_expiry <= 0:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}

        d1 = (np.log(spot / strike) + (self.risk_free_rate - self.dividend_yield + 0.5 * volatility ** 2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
        d2 = d1 - volatility * np.sqrt(time_to_expiry)

        # Delta
        if option_type == 'CE':
            delta = np.exp(-self.dividend_yield * time_to_expiry) * norm.cdf(d1)
        else:
            delta = -np.exp(-self.dividend_yield * time_to_expiry) * norm.cdf(-d1)

        # Gamma
        gamma = np.exp(-self.dividend_yield * time_to_expiry) * norm.pdf(d1) / (spot * volatility * np.sqrt(time_to_expiry))

        # Theta (per day)
        if option_type == 'CE':
            theta = (-spot * norm.pdf(d1) * volatility * np.exp(-self.dividend_yield * time_to_expiry) / (2 * np.sqrt(time_to_expiry))
                    - self.risk_free_rate * strike * np.exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(d2)
                    + self.dividend_yield * spot * np.exp(-self.dividend_yield * time_to_expiry) * norm.cdf(d1)) / 365
        else:
            theta = (-spot * norm.pdf(d1) * volatility * np.exp(-self.dividend_yield * time_to_expiry) / (2 * np.sqrt(time_to_expiry))
                    + self.risk_free_rate * strike * np.exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(-d2)
                    - self.dividend_yield * spot * np.exp(-self.dividend_yield * time_to_expiry) * norm.cdf(-d1)) / 365

        # Vega
        vega = spot * np.exp(-self.dividend_yield * time_to_expiry) * norm.pdf(d1) * np.sqrt(time_to_expiry) / 100

        return {
            'delta': round(delta, 4),
            'gamma': round(gamma, 6),
            'theta': round(theta, 2),
            'vega': round(vega, 2)
        }

    def get_trading_days(self, start_date: datetime, end_date: datetime) -> List[datetime]:
        """Get list of trading days (exclude weekends and holidays)"""
        trading_days = []
        current_date = start_date

        while current_date <= end_date:
            # Skip weekends (Saturday=5, Sunday=6)
            if current_date.weekday() < 5 and current_date not in self.holidays:
                trading_days.append(current_date)
            current_date += timedelta(days=1)

        return trading_days

    def get_active_expiries(self, current_date: datetime) -> List[Tuple[datetime, str]]:
        """Get active expiry dates for a given trading day"""
        expiries = []

        # Determine expiry day based on month
        if current_date.month < 9:  # July-August: Thursday expiry
            weekly_day = 3  # Thursday
            monthly_day = 3
        else:  # September onwards: Tuesday expiry
            weekly_day = 1  # Tuesday
            monthly_day = 1

        # Find next 3 weekly expiries
        for week in range(3):
            days_ahead = weekly_day - current_date.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            days_ahead += week * 7

            expiry = current_date + timedelta(days=days_ahead)

            # Check if it's last week of month (monthly expiry)
            last_day_of_month = (expiry.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
            days_to_month_end = (last_day_of_month - expiry).days

            if days_to_month_end < 7:
                expiries.append((expiry, 'monthly'))
            else:
                expiries.append((expiry, 'weekly'))

        # Add next month's monthly if less than 4 expiries
        if len(expiries) < 4:
            next_month = (current_date.replace(day=28) + timedelta(days=4)).replace(day=1)
            days_ahead = monthly_day - next_month.weekday()
            if days_ahead < 0:
                days_ahead += 7

            # Find last occurrence of the day in next month
            first_occurrence = next_month + timedelta(days=days_ahead)
            last_occurrence = first_occurrence
            while (last_occurrence + timedelta(days=7)).month == next_month.month:
                last_occurrence += timedelta(days=7)

            expiries.append((last_occurrence, 'monthly'))

        return expiries[:4]  # Return max 4 expiries

    def generate_intraday_spot_path(self, base_price: float, daily_return: float,
                                   regime: str) -> List[float]:
        """Generate 5-minute spot prices for a trading day"""
        prices = []
        current_price = base_price

        # Intraday volatility patterns
        intraday_vol_mult = {
            '09:15-09:30': 1.5,  # Opening volatility
            '09:30-10:00': 1.2,  # Price discovery
            '10:00-14:30': 1.0,  # Normal trading
            '14:30-15:00': 1.1,  # Position squaring
            '15:00-15:30': 1.3   # Closing volatility
        }

        # Generate 5-minute returns
        total_return = daily_return
        bar_returns = np.random.normal(0, abs(total_return) / 10, self.bars_per_day)
        bar_returns = bar_returns / bar_returns.sum() * total_return  # Normalize to daily return

        for i in range(self.bars_per_day):
            # Determine time slot for volatility multiplier
            hour = 9 + (i * 5) // 60
            minute = (i * 5) % 60
            if hour == 9 and minute < 15:
                minute = 15

            current_time = f"{hour:02d}:{minute:02d}"

            # Apply intraday volatility pattern
            if i < 3:  # First 15 minutes
                vol_mult = intraday_vol_mult['09:15-09:30']
            elif i < 9:  # 09:30-10:00
                vol_mult = intraday_vol_mult['09:30-10:00']
            elif i < 63:  # 10:00-14:30
                vol_mult = intraday_vol_mult['10:00-14:30']
            elif i < 69:  # 14:30-15:00
                vol_mult = intraday_vol_mult['14:30-15:00']
            else:  # 15:00-15:30
                vol_mult = intraday_vol_mult['15:00-15:30']

            # Add some noise
            noise = np.random.normal(0, 0.0002 * vol_mult)
            current_price = current_price * (1 + bar_returns[i] + noise)
            prices.append(round(current_price, 2))

        return prices

    def calculate_spread(self, price: float, moneyness: float, volume: int) -> Tuple[float, float]:
        """Calculate bid-ask spread based on moneyness and volume"""
        # Base spread percentage
        if abs(moneyness) < 0.02:  # ATM
            base_spread = 0.005  # 0.5%
        elif abs(moneyness) < 0.05:  # Near ATM
            base_spread = 0.008  # 0.8%
        elif abs(moneyness) < 0.10:  # Slightly OTM/ITM
            base_spread = 0.012  # 1.2%
        else:  # Deep OTM/ITM
            base_spread = 0.020  # 2.0%

        # Adjust for volume (higher volume = tighter spread)
        volume_factor = max(0.5, min(1.5, 10000 / (volume + 1000)))

        spread = price * base_spread * volume_factor
        bid = round(price - spread/2, 2)
        ask = round(price + spread/2, 2)

        return max(0.05, bid), ask

    def generate_volume_oi(self, strike: float, spot: float, days_to_expiry: int) -> Tuple[int, int]:
        """Generate realistic volume and open interest"""
        moneyness = abs((strike - spot) / spot)

        # Volume distribution based on moneyness
        if moneyness < 0.01:  # ATM
            base_volume = np.random.randint(8000, 15000)
            base_oi = np.random.randint(800000, 1500000)
        elif moneyness < 0.02:  # Near ATM
            base_volume = np.random.randint(5000, 10000)
            base_oi = np.random.randint(500000, 1000000)
        elif moneyness < 0.05:  # ±500 points
            base_volume = np.random.randint(2000, 6000)
            base_oi = np.random.randint(200000, 600000)
        elif moneyness < 0.10:  # ±1000 points
            base_volume = np.random.randint(500, 2500)
            base_oi = np.random.randint(50000, 250000)
        else:  # Deep OTM/ITM
            base_volume = np.random.randint(50, 500)
            base_oi = np.random.randint(5000, 50000)

        # Adjust for days to expiry
        expiry_factor = max(0.3, min(1.5, days_to_expiry / 7))

        volume = int(base_volume * expiry_factor * np.random.uniform(0.8, 1.2))
        oi = int(base_oi * expiry_factor * np.random.uniform(0.9, 1.1))

        return volume, oi

    def generate_day_data(self, trading_date: datetime, base_spot: float,
                         regime: str) -> pd.DataFrame:
        """Generate complete options data for one trading day"""
        all_data = []

        # Determine daily return based on regime
        regime_params = self.market_regimes[regime]
        daily_return = np.random.uniform(*regime_params['daily_return'])

        # Generate intraday spot path
        spot_prices = self.generate_intraday_spot_path(base_spot, daily_return, regime)

        # Get active expiries
        expiries = self.get_active_expiries(trading_date)

        # Generate timestamps for the day
        timestamps = []
        for i in range(self.bars_per_day):
            hour = 9 + (i * 5) // 60
            minute = (i * 5) % 60
            if hour == 9 and minute < 15:
                minute = 15

            ts = trading_date.replace(hour=hour, minute=minute)
            timestamps.append(ts)

        # For each timestamp
        for bar_idx, (timestamp, spot) in enumerate(zip(timestamps, spot_prices)):
            # Round spot to nearest 50 for ATM
            atm_strike = round(spot / self.strike_interval) * self.strike_interval

            # Generate strikes
            strikes = []
            for i in range(-40, 41):  # ±2000 points at 50 intervals
                strike = atm_strike + (i * self.strike_interval)
                strikes.append(strike)

            # For each expiry
            for expiry_date, expiry_type in expiries:
                days_to_expiry = (expiry_date - trading_date).days
                time_to_expiry = days_to_expiry / 365.0

                # Skip if expired
                if time_to_expiry <= 0:
                    continue

                # For each strike
                for strike in strikes:
                    moneyness = (strike - spot) / spot

                    # Determine IV based on regime and moneyness
                    if abs(moneyness) < 0.02:  # ATM
                        base_iv_range = self.base_iv['normal']
                    elif abs(moneyness) < 0.10:
                        base_iv_range = self.base_iv['normal']
                    else:
                        base_iv_range = self.base_iv['high'] if regime == 'high_vol' else self.base_iv['normal']

                    iv = np.random.uniform(*base_iv_range) * regime_params['iv_mult']

                    # For both CE and PE
                    for option_type in ['CE', 'PE']:
                        # Calculate theoretical price
                        theo_price = self.black_scholes_price(spot, strike, time_to_expiry, iv, option_type)

                        # Add market noise
                        noise_factor = np.random.uniform(0.98, 1.02)
                        close_price = round(theo_price * noise_factor, 2)

                        # Generate OHLC
                        volatility = 0.02 if bar_idx < 3 or bar_idx > 69 else 0.01
                        high = round(close_price * (1 + np.random.uniform(0, volatility)), 2)
                        low = round(close_price * (1 - np.random.uniform(0, volatility)), 2)
                        open_price = round(np.random.uniform(low, high), 2)

                        # Calculate Greeks
                        greeks = self.calculate_greeks(spot, strike, time_to_expiry, iv, option_type)

                        # Generate volume and OI
                        volume, oi = self.generate_volume_oi(strike, spot, days_to_expiry)

                        # Calculate spread
                        bid, ask = self.calculate_spread(close_price, moneyness, volume)

                        # Create row
                        row = {
                            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                            'symbol': 'NIFTY',
                            'strike': strike,
                            'option_type': option_type,
                            'expiry': expiry_date.strftime('%Y-%m-%d'),
                            'expiry_type': expiry_type,
                            'open': open_price,
                            'high': high,
                            'low': low,
                            'close': close_price,
                            'volume': volume,
                            'oi': oi,
                            'bid': bid,
                            'ask': ask,
                            'iv': round(iv, 4),
                            'delta': greeks['delta'],
                            'gamma': greeks['gamma'],
                            'theta': greeks['theta'],
                            'vega': greeks['vega'],
                            'underlying_price': spot
                        }

                        all_data.append(row)

        return pd.DataFrame(all_data)

    def generate_period_data(self, start_date: str, end_date: str,
                           output_dir: str, initial_spot: float = 25000):
        """Generate synthetic data for entire period"""
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Get trading days
        trading_days = self.get_trading_days(start, end)

        # Track market regime distribution
        regime_tracker = {regime: 0 for regime in self.market_regimes.keys()}

        # Current spot price
        current_spot = initial_spot

        print(f"Generating synthetic data for {len(trading_days)} trading days...")
        print(f"Period: {start_date} to {end_date}")
        print(f"Output directory: {output_dir}")

        # Generate data for each trading day
        for idx, trading_day in enumerate(trading_days):
            # Select market regime
            regime_probs = [self.market_regimes[r]['prob'] for r in self.market_regimes.keys()]
            regime = np.random.choice(list(self.market_regimes.keys()), p=regime_probs)
            regime_tracker[regime] += 1

            # Generate day's data
            print(f"Generating {trading_day.strftime('%Y-%m-%d')} ({idx+1}/{len(trading_days)}) - Regime: {regime}")

            day_data = self.generate_day_data(trading_day, current_spot, regime)

            # Save to CSV
            filename = f"NIFTY_OPTIONS_5MIN_{trading_day.strftime('%Y%m%d')}.csv"
            filepath = os.path.join(output_dir, filename)
            day_data.to_csv(filepath, index=False)

            # Update spot for next day (use closing spot)
            if len(day_data) > 0:
                current_spot = day_data.iloc[-1]['underlying_price']

        # Generate metadata
        metadata = {
            "generation_metadata": {
                "version": "1.0.0",
                "generated_date": datetime.now().strftime("%Y-%m-%d"),
                "period": {
                    "start": start_date,
                    "end": end_date
                },
                "statistics": {
                    "total_trading_days": len(trading_days),
                    "total_5min_bars": len(trading_days) * self.bars_per_day,
                    "strikes_per_expiry": 81,  # ±40 strikes + ATM
                    "initial_spot": initial_spot,
                    "final_spot": current_spot
                },
                "market_regimes": regime_tracker
            }
        }

        # Save metadata
        metadata_file = os.path.join(output_dir, "generation_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print("\nGeneration complete!")
        print(f"Market regime distribution: {regime_tracker}")
        print(f"Metadata saved to: {metadata_file}")

        return metadata


def main():
    """Main execution function"""
    generator = NiftyOptionsDataGenerator()

    # Generate data as per PRD specifications
    output_directory = "/Users/nitindhawan/NikAlgoBulls/zerodha_strategy/data/synthetic/intraday_jul_sep-16Sep"

    metadata = generator.generate_period_data(
        start_date="2025-07-01",
        end_date="2025-09-16",
        output_dir=output_directory,
        initial_spot=25000
    )

    print("\n✅ Synthetic data generation completed successfully!")
    print(f"📁 Data location: {output_directory}")


if __name__ == "__main__":
    main()