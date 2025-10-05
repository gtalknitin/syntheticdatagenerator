#!/usr/bin/env python3
"""
V8 Extended Synthetic Data Generator with VIX and Broader Strikes
Includes 0.1 delta support for weekly hedges
Period: June 14 - September 30, 2025
Strike Range: ₹22,000 - ₹30,000 (161 strikes)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from pathlib import Path
import json

class V8ExtendedDataGenerator:
    def __init__(self):
        self.base_path = Path('/Users/nitindhawan/NikAlgoBulls/zerodha_strategy/data/synthetic/intraday_v8_extended')
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Market parameters
        self.timestamps_per_day = 79  # 09:15 to 15:30 in 5-min intervals
        self.strike_interval = 50
        self.risk_free_rate = 0.065

        # V8: EXPANDED STRIKE RANGE (±4000 from spot instead of ±2000)
        self.strike_offset_range = 4000  # ₹22k to ₹30k for 0.1 delta support

        # Expiry schedule - EXTENDED to include June
        self.expiry_schedule = {
            'weekly': {
                6: [18, 25],             # June - Wednesdays (transition period)
                7: [3, 10, 17, 24],      # July - Thursdays
                8: [7, 14, 21],          # August - Thursdays
                9: [3, 10, 17, 24]       # September - Wednesdays (3rd), then Tuesdays
            },
            'monthly': {
                6: 26,  # June 26 - Thursday (NEW)
                7: 31,  # July 31 - Thursday
                8: 28,  # August 28 - Thursday
                9: 30   # September 30 - Tuesday
            }
        }

        # Trading holidays
        self.holidays = [
            datetime(2025, 8, 15)  # Independence Day
        ]

        # VIX regime periods - EXTENDED with June baseline
        self.vix_regimes = {
            # June baseline (VIX 14-18)
            (datetime(2025, 6, 16), datetime(2025, 6, 30)): {
                'base': 16, 'range': 2, 'trend': 'stable'
            },
            # Normal period (VIX 12-18)
            (datetime(2025, 7, 1), datetime(2025, 7, 20)): {
                'base': 15, 'range': 3, 'trend': 'stable'
            },
            # Rising volatility (VIX 18-25)
            (datetime(2025, 7, 21), datetime(2025, 7, 31)): {
                'base': 21, 'range': 4, 'trend': 'rising'
            },
            # High volatility event (VIX 28-35) - Early August
            (datetime(2025, 8, 1), datetime(2025, 8, 10)): {
                'base': 32, 'range': 3, 'trend': 'elevated'
            },
            # Cooling down (VIX 22-28)
            (datetime(2025, 8, 11), datetime(2025, 8, 20)): {
                'base': 25, 'range': 3, 'trend': 'declining'
            },
            # Normal period (VIX 14-20)
            (datetime(2025, 8, 21), datetime(2025, 9, 5)): {
                'base': 17, 'range': 3, 'trend': 'stable'
            },
            # Another spike (VIX 30-38) - Mid September
            (datetime(2025, 9, 6), datetime(2025, 9, 15)): {
                'base': 34, 'range': 4, 'trend': 'spike'
            },
            # Recovery (VIX 20-25)
            (datetime(2025, 9, 16), datetime(2025, 9, 30)): {
                'base': 22, 'range': 3, 'trend': 'recovering'
            }
        }

        self.stats = {
            'files_created': 0,
            'total_rows': 0,
            'dates_processed': [],
            'strike_stats': {
                'min': 100000,
                'max': 0,
                'total_unique': 0
            },
            'vix_stats': {
                'min': 100,
                'max': 0,
                'days_above_30': 0,
                'avg': 0
            },
            'delta_coverage': {
                'ce_low_delta_strikes': 0,  # 0.05-0.15 delta
                'pe_low_delta_strikes': 0
            }
        }

    def get_vix_for_date(self, date, timestamp_index=0):
        """Get VIX value for a specific date with intraday variation"""
        for (start_date, end_date), regime in self.vix_regimes.items():
            if start_date <= date <= end_date:
                base_vix = regime['base']
                vix_range = regime['range']
                trend = regime['trend']

                # Add trend component
                days_in_regime = (date - start_date).days
                total_days = (end_date - start_date).days + 1
                progress = days_in_regime / max(1, total_days)

                if trend == 'rising':
                    trend_adjustment = vix_range * progress
                elif trend == 'declining':
                    trend_adjustment = -vix_range * progress
                elif trend == 'spike':
                    # Peak in the middle
                    if progress < 0.5:
                        trend_adjustment = vix_range * 2 * progress
                    else:
                        trend_adjustment = vix_range * 2 * (1 - progress)
                else:
                    trend_adjustment = 0

                # Intraday variation (U-shaped)
                intraday_factor = 1.0
                if timestamp_index < 20:  # Morning
                    intraday_factor = 1.02
                elif timestamp_index > 60:  # Late afternoon
                    intraday_factor = 1.03

                # Random noise
                noise = np.random.normal(0, 0.5)

                vix = base_vix + trend_adjustment + noise
                vix = vix * intraday_factor

                return round(max(10, min(50, vix)), 2)

        # Default VIX if date not in regimes
        return 16.0

    def get_trading_dates(self, start_date, end_date):
        """Get all trading dates between start and end"""
        dates = []
        current = start_date

        while current <= end_date:
            # Skip weekends and holidays
            if current.weekday() < 5 and current not in self.holidays:
                dates.append(current)
            current += timedelta(days=1)

        return dates

    def get_active_expiries(self, current_date):
        """Get all active expiries for a given date"""
        expiries = []

        # Add weekly expiries (next 4 weeks)
        for weeks_ahead in range(4):
            expiry_date = self.get_next_weekly_expiry(current_date + timedelta(weeks=weeks_ahead))
            if expiry_date and expiry_date >= current_date:
                expiries.append((expiry_date, 'weekly'))

        # Add monthly expiries (current and next month)
        for month_offset in range(2):
            month = ((current_date.month - 1 + month_offset) % 12) + 1
            year = current_date.year + ((current_date.month - 1 + month_offset) // 12)

            if year == 2025 and month in self.expiry_schedule['monthly']:
                day = self.expiry_schedule['monthly'][month]
                expiry_date = datetime(year, month, day)

                if expiry_date >= current_date:
                    expiries.append((expiry_date, 'monthly'))

        return sorted(set(expiries))[:6]  # Keep maximum 6 expiries

    def get_next_weekly_expiry(self, from_date):
        """Get next weekly expiry from a given date"""
        month = from_date.month

        if month not in self.expiry_schedule['weekly']:
            return None

        for day in self.expiry_schedule['weekly'][month]:
            expiry_date = datetime(2025, month, day)
            if expiry_date >= from_date:
                return expiry_date

        # Check next month
        next_month = (month % 12) + 1
        if next_month in self.expiry_schedule['weekly'] and self.expiry_schedule['weekly'][next_month]:
            return datetime(2025, next_month, self.expiry_schedule['weekly'][next_month][0])

        return None

    def norm_cdf(self, x):
        """Approximation of normal CDF without scipy"""
        a1 =  0.254829592
        a2 = -0.284496736
        a3 =  1.421413741
        a4 = -1.453152027
        a5 =  1.061405429
        p  =  0.3275911

        sign = 1 if x >= 0 else -1
        x = abs(x) / np.sqrt(2.0)

        t = 1.0 / (1.0 + p * x)
        t2 = t * t
        t3 = t2 * t
        t4 = t3 * t
        t5 = t4 * t
        y = 1.0 - (((((a5*t5 + a4*t4) + a3*t3) + a2*t2) + a1*t) * np.exp(-x*x))

        return 0.5 * (1.0 + sign * y)

    def calculate_base_price(self, spot, strike, tte_days, option_type, vix):
        """Calculate base option price using Black-Scholes with VIX adjustment"""
        if tte_days <= 0:
            # Expired option
            intrinsic = max(0, spot - strike) if option_type == 'CE' else max(0, strike - spot)
            return intrinsic

        T = tte_days / 365.0

        # Use VIX as base volatility (convert to decimal)
        base_vol = vix / 100.0

        # Volatility smile adjusted by VIX level
        moneyness = spot / strike
        if abs(moneyness - 1) > 0.05:
            vol_adjustment = abs(moneyness - 1) * 0.2
            iv = base_vol * (1 + vol_adjustment)
        else:
            iv = base_vol

        # Add term structure
        if tte_days <= 7:
            iv *= 1.15
        elif tte_days <= 30:
            iv *= 1.05

        # Black-Scholes calculation
        d1 = (np.log(spot/strike) + (self.risk_free_rate + 0.5*iv**2)*T) / (iv*np.sqrt(T))
        d2 = d1 - iv*np.sqrt(T)

        if option_type == 'CE':
            price = spot*self.norm_cdf(d1) - strike*np.exp(-self.risk_free_rate*T)*self.norm_cdf(d2)
        else:
            price = strike*np.exp(-self.risk_free_rate*T)*self.norm_cdf(-d2) - spot*self.norm_cdf(-d1)

        return max(0.05, round(price, 2))

    def norm_pdf(self, x):
        """Normal PDF approximation"""
        return np.exp(-x*x/2) / np.sqrt(2*np.pi)

    def calculate_greeks(self, spot, strike, tte_days, iv, option_type):
        """Calculate option Greeks"""
        if tte_days <= 0:
            return {
                'delta': 1.0 if (option_type == 'CE' and spot > strike) else 0.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0
            }

        T = tte_days / 365.0
        sqrt_T = np.sqrt(T)

        d1 = (np.log(spot/strike) + (self.risk_free_rate + 0.5*iv**2)*T) / (iv*sqrt_T)
        d2 = d1 - iv*np.sqrt(T)

        if option_type == 'CE':
            delta = self.norm_cdf(d1)
        else:
            delta = -self.norm_cdf(-d1)

        gamma = self.norm_pdf(d1) / (spot * iv * sqrt_T)
        vega = spot * self.norm_pdf(d1) * sqrt_T / 100

        # Theta calculation
        term1 = -spot * self.norm_pdf(d1) * iv / (2 * sqrt_T)
        if option_type == 'CE':
            term2 = -self.risk_free_rate * strike * np.exp(-self.risk_free_rate*T) * self.norm_cdf(d2)
        else:
            term2 = self.risk_free_rate * strike * np.exp(-self.risk_free_rate*T) * self.norm_cdf(-d2)

        theta = (term1 + term2) / 365

        return {
            'delta': round(delta, 4),
            'gamma': round(max(0, gamma), 6),
            'theta': round(min(0, theta), 2),
            'vega': round(vega, 2)
        }

    def generate_intraday_prices(self, base_price, timestamps, vix):
        """Generate realistic intraday price movements influenced by VIX"""
        prices = [base_price]

        # Higher VIX = more volatility in price movements
        volatility_factor = vix / 15.0  # Normalized to baseline VIX of 15

        for _ in range(1, timestamps):
            # Random walk with VIX-adjusted volatility
            change_pct = np.random.normal(0, 0.015 * volatility_factor)
            change_pct = np.clip(change_pct, -0.05, 0.05)  # Max 5% change

            # Mean reversion
            if len(prices) > 5:
                recent_mean = np.mean(prices[-5:])
                if prices[-1] > recent_mean * 1.1:
                    change_pct -= 0.01
                elif prices[-1] < recent_mean * 0.9:
                    change_pct += 0.01

            new_price = prices[-1] * (1 + change_pct)
            new_price = max(0.05, round(new_price, 2))
            prices.append(new_price)

        return prices

    def generate_day_data(self, date):
        """Generate complete options data for one trading day with VIX and extended strikes"""
        data = []
        day_vix_values = []

        # Generate timestamps
        timestamps = []
        current_time = datetime.combine(date, datetime.min.time().replace(hour=9, minute=15))
        for _ in range(self.timestamps_per_day):
            timestamps.append(current_time)
            current_time += timedelta(minutes=5)

        # Generate VIX values for the day
        vix_values = []
        for i in range(self.timestamps_per_day):
            vix = self.get_vix_for_date(date, i)
            vix_values.append(vix)
            day_vix_values.append(vix)

        # Update VIX stats
        self.stats['vix_stats']['min'] = min(self.stats['vix_stats']['min'], min(vix_values))
        self.stats['vix_stats']['max'] = max(self.stats['vix_stats']['max'], max(vix_values))
        if max(vix_values) > 30:
            self.stats['vix_stats']['days_above_30'] += 1

        # Generate spot price movement (more volatile when VIX is high)
        base_spot = 25000 + (date.toordinal() % 30) * 50
        spot_volatility = np.mean(vix_values) / 100.0
        spot_prices = []
        current_spot = base_spot

        for vix in vix_values:
            # Spot moves more when VIX is high
            spot_change = np.random.normal(0, current_spot * spot_volatility * 0.001)
            current_spot += spot_change
            spot_prices.append(round(current_spot, 2))

        # Get active expiries
        active_expiries = self.get_active_expiries(date)

        # V8: EXPANDED STRIKE RANGE (±4000 instead of ±2000)
        strikes = []
        for offset in range(-self.strike_offset_range, self.strike_offset_range + 1, self.strike_interval):
            strike = int(base_spot + offset)
            strikes.append(strike)

        # Update strike stats
        self.stats['strike_stats']['min'] = min(self.stats['strike_stats']['min'], min(strikes))
        self.stats['strike_stats']['max'] = max(self.stats['strike_stats']['max'], max(strikes))

        # Track delta coverage
        ce_low_delta_count = 0
        pe_low_delta_count = 0

        # Generate data for each strike/expiry/option_type combination
        for expiry_date, expiry_type in active_expiries:
            tte_days = (expiry_date - date).days

            for strike in strikes:
                for option_type in ['CE', 'PE']:
                    # Calculate base price using current VIX
                    base_price = self.calculate_base_price(
                        spot_prices[0], strike, tte_days, option_type, vix_values[0]
                    )

                    # Generate intraday prices with VIX influence
                    prices = self.generate_intraday_prices(
                        base_price, self.timestamps_per_day, np.mean(vix_values)
                    )

                    # Calculate IV based on VIX and moneyness
                    moneyness = spot_prices[0] / strike
                    base_iv = vix_values[0] / 100.0
                    if abs(moneyness - 1) > 0.05:
                        iv = base_iv * (1 + abs(moneyness - 1) * 0.2)
                    else:
                        iv = base_iv

                    # Generate data for each timestamp
                    for i, (timestamp, spot, price, vix) in enumerate(zip(timestamps, spot_prices, prices, vix_values)):
                        # Calculate Greeks
                        greeks = self.calculate_greeks(
                            spot, strike, tte_days - (i/self.timestamps_per_day), iv, option_type
                        )

                        # Track delta coverage (first timestamp only to avoid duplicates)
                        if i == 0:
                            delta_abs = abs(greeks['delta'])
                            if 0.05 <= delta_abs <= 0.15:
                                if option_type == 'CE':
                                    ce_low_delta_count += 1
                                else:
                                    pe_low_delta_count += 1

                        # Generate OHLC (more volatile when VIX is high)
                        vix_factor = vix / 15.0
                        noise = np.random.uniform(-0.02 * vix_factor, 0.02 * vix_factor, 3)
                        high = round(price * (1 + abs(noise[0])), 2)
                        low = round(price * (1 - abs(noise[1])), 2)
                        close = round(price * (1 + noise[2]), 2)

                        # Bid-Ask spread (wider when VIX is high)
                        spread_pct = 0.01 * vix_factor if abs(moneyness - 1) < 0.02 else (0.02 + abs(moneyness - 1) * 0.03) * vix_factor
                        bid = round(price * (1 - spread_pct/2), 2)
                        ask = round(price * (1 + spread_pct/2), 2)

                        # Volume and OI (higher when VIX is elevated)
                        volume_boost = 1 + max(0, (vix - 20) / 10)
                        volume = int(np.random.lognormal(7, 1.5) * volume_boost)
                        oi = int(np.random.lognormal(8, 1.5) * volume_boost)

                        data.append([
                            timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                            'NIFTY',
                            strike,
                            option_type,
                            expiry_date.strftime('%Y-%m-%d'),
                            expiry_type,
                            price,  # open
                            high,
                            low,
                            close,
                            volume,
                            oi,
                            bid,
                            ask,
                            round(iv, 4),
                            greeks['delta'],
                            greeks['gamma'],
                            greeks['theta'],
                            greeks['vega'],
                            round(spot, 2),
                            round(vix, 2)  # Add VIX column
                        ])

        # Update delta coverage stats
        self.stats['delta_coverage']['ce_low_delta_strikes'] = max(
            self.stats['delta_coverage']['ce_low_delta_strikes'], ce_low_delta_count
        )
        self.stats['delta_coverage']['pe_low_delta_strikes'] = max(
            self.stats['delta_coverage']['pe_low_delta_strikes'], pe_low_delta_count
        )

        return pd.DataFrame(data, columns=[
            'timestamp', 'symbol', 'strike', 'option_type', 'expiry', 'expiry_type',
            'open', 'high', 'low', 'close', 'volume', 'oi', 'bid', 'ask',
            'iv', 'delta', 'gamma', 'theta', 'vega', 'underlying_price', 'vix'
        ])

    def generate_all_data(self):
        """Generate data for entire extended period"""
        print("Starting V8 EXTENDED data generation with broader strikes...")
        print("Period: June 14 - September 30, 2025")
        print(f"Strike Range: ₹22,000 - ₹30,000 (161 strikes)")
        print(f"Output directory: {self.base_path}")
        print("-" * 60)

        # V8: EXTENDED START DATE (June 14, 2025)
        start_date = datetime(2025, 6, 14)
        end_date = datetime(2025, 9, 30)

        trading_dates = self.get_trading_dates(start_date, end_date)
        all_vix_values = []

        for i, date in enumerate(trading_dates, 1):
            print(f"Generating {date.strftime('%Y-%m-%d')} [{i}/{len(trading_dates)}]", end='... ')

            # Generate data
            df = self.generate_day_data(date)

            # Track VIX values
            day_vix = df['vix'].iloc[0]
            all_vix_values.extend(df['vix'].unique())

            # Save to CSV
            filename = f"NIFTY_OPTIONS_5MIN_{date.strftime('%Y%m%d')}.csv"
            filepath = self.base_path / filename
            df.to_csv(filepath, index=False)

            # Update stats
            self.stats['files_created'] += 1
            self.stats['total_rows'] += len(df)
            self.stats['dates_processed'].append(date.strftime('%Y-%m-%d'))

            # Get unique strikes for this day
            unique_strikes = df['strike'].nunique()

            print(f"✓ ({len(df):,} rows, {unique_strikes} strikes, VIX: {day_vix})")

        # Calculate average VIX
        self.stats['vix_stats']['avg'] = round(np.mean(all_vix_values), 2)

        # Calculate total unique strikes
        self.stats['strike_stats']['total_unique'] = (
            self.stats['strike_stats']['max'] - self.stats['strike_stats']['min']
        ) // self.strike_interval + 1

        # Save metadata
        self.save_metadata()

        print("\n" + "="*60)
        print("V8 EXTENDED DATA GENERATION COMPLETE!")
        print(f"Files created: {self.stats['files_created']}")
        print(f"Total rows: {self.stats['total_rows']:,}")
        print(f"Strike Range: ₹{self.stats['strike_stats']['min']:,} - ₹{self.stats['strike_stats']['max']:,}")
        print(f"Total Strikes: {self.stats['strike_stats']['total_unique']}")
        print(f"VIX Range: {self.stats['vix_stats']['min']:.1f} - {self.stats['vix_stats']['max']:.1f}")
        print(f"Average VIX: {self.stats['vix_stats']['avg']:.1f}")
        print(f"Days with VIX > 30: {self.stats['vix_stats']['days_above_30']}")
        print(f"CE strikes with 0.05-0.15 delta: {self.stats['delta_coverage']['ce_low_delta_strikes']}")
        print(f"PE strikes with 0.05-0.15 delta: {self.stats['delta_coverage']['pe_low_delta_strikes']}")
        print(f"Location: {self.base_path}")

        return self.stats

    def save_metadata(self):
        """Save generation metadata"""
        metadata = {
            'version': '8.0-Extended',
            'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'period': {
                'start': '2025-06-14',
                'end': '2025-09-30'
            },
            'strike_range': {
                'min': self.stats['strike_stats']['min'],
                'max': self.stats['strike_stats']['max'],
                'interval': self.strike_interval,
                'total': self.stats['strike_stats']['total_unique']
            },
            'expiry_schedule': self.expiry_schedule,
            'vix_regimes': {
                f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}": regime
                for (start, end), regime in self.vix_regimes.items()
            },
            'stats': self.stats,
            'features': [
                'Extended period: June 14 - September 30, 2025',
                'Broader strikes: ₹22,000 - ₹30,000 (161 strikes)',
                '0.1 delta support for weekly hedges',
                'June 26 (Thursday) monthly expiry included',
                'September 30 (Tuesday) monthly expiry included',
                'VIX data with realistic regimes',
                'Periods of high volatility (VIX > 30)',
                'Smooth VIX transitions',
                'Option prices adjusted for VIX levels',
                'Volume and spreads react to VIX'
            ],
            'delta_coverage': self.stats['delta_coverage']
        }

        metadata_path = self.base_path / 'metadata'
        metadata_path.mkdir(exist_ok=True)

        with open(metadata_path / 'generation_info.json', 'w') as f:
            json.dump(metadata, f, indent=2)

def main():
    generator = V8ExtendedDataGenerator()
    stats = generator.generate_all_data()
    return stats

if __name__ == '__main__':
    main()
