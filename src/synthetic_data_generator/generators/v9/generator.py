#!/usr/bin/env python3
"""
V9 Balanced Synthetic Data Generator - 1 Hour Candles
Fixes critical issues from V8:
1. Balanced trend generation (50/50 weeks)
2. Expiry-specific option pricing (correct TTE per expiry)
3. Realistic delta-distance relationships
4. Meaningful weekly hedge premiums
5. 1-hour candles (7/day instead of 79)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import random

class V9BalancedGenerator:
    def __init__(self):
        self.base_path = Path('/Users/nitindhawan/NikAlgoBulls/zerodha_strategy/data/synthetic/hourly_v9_balanced')
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Market parameters
        self.timestamps_per_day = 7  # 1-hour candles
        self.strike_interval = 50
        self.risk_free_rate = 0.065

        # Strike range (same as V8)
        self.min_strike = 22000
        self.max_strike = 30000

        # Expiry schedule
        self.expiry_schedule = {
            'weekly': {
                6: [18, 25],             # June
                7: [3, 10, 17, 24],      # July
                8: [7, 14, 21],          # August
                9: [3, 10, 17, 24]       # September
            },
            'monthly': {
                6: 26,  # June 26 - Thursday
                7: 31,  # July 31 - Thursday
                8: 28,  # August 28 - Thursday
                9: 30   # September 30 - Tuesday
            }
        }

        # VIX regimes (same as V8)
        self.vix_regimes = {
            (datetime(2025, 6, 16), datetime(2025, 6, 30)): {'base': 16, 'range': 2, 'trend': 'stable'},
            (datetime(2025, 7, 1), datetime(2025, 7, 20)): {'base': 15, 'range': 3, 'trend': 'stable'},
            (datetime(2025, 7, 21), datetime(2025, 7, 31)): {'base': 21, 'range': 4, 'trend': 'rising'},
            (datetime(2025, 8, 1), datetime(2025, 8, 10)): {'base': 32, 'range': 3, 'trend': 'elevated'},
            (datetime(2025, 8, 11), datetime(2025, 8, 20)): {'base': 25, 'range': 3, 'trend': 'declining'},
            (datetime(2025, 8, 21), datetime(2025, 9, 5)): {'base': 17, 'range': 3, 'trend': 'stable'},
            (datetime(2025, 9, 6), datetime(2025, 9, 15)): {'base': 34, 'range': 4, 'trend': 'spike'},
            (datetime(2025, 9, 16), datetime(2025, 9, 30)): {'base': 22, 'range': 3, 'trend': 'recovering'}
        }

        # Trading holidays
        self.holidays = [datetime(2025, 8, 15)]

        # Stats tracking
        self.stats = {
            'files_created': 0,
            'total_rows': 0,
            'dates_processed': [],
            'weekly_trend': {'up': 0, 'down': 0},
            'delta_coverage': {'ce_low_delta': 0, 'pe_low_delta': 0}
        }

        # Pre-generated balanced price series
        self.price_series = None

    def generate_balanced_price_series(self, start_date, end_date):
        """
        Generate underlying price movement with enforced 50/50 weekly balance
        """
        print("\n🎲 Generating balanced price movement...")

        # Get all trading weeks
        weeks = []
        current = start_date
        while current <= end_date:
            week_start = current - timedelta(days=current.weekday())
            if week_start not in [w[0] for w in weeks]:
                weeks.append((week_start, week_start + timedelta(days=4)))
            current += timedelta(days=7)

        # Enforce 50/50 split
        total_weeks = len(weeks)
        up_weeks = total_weeks // 2
        down_weeks = total_weeks - up_weeks

        print(f"  Total weeks: {total_weeks}")
        print(f"  Up weeks: {up_weeks} ({up_weeks/total_weeks*100:.1f}%)")
        print(f"  Down weeks: {down_weeks} ({down_weeks/total_weeks*100:.1f}%)")

        # Create balanced direction list
        directions = ['UP'] * up_weeks + ['DOWN'] * down_weeks
        random.shuffle(directions)

        # Generate price series
        price_series = {}
        current_price = 25400  # Starting Nifty

        for week_num, (week_start, week_end) in enumerate(weeks):
            direction = directions[week_num]

            if direction == 'UP':
                week_change_pct = np.random.uniform(0.005, 0.015)  # +0.5% to +1.5%
                self.stats['weekly_trend']['up'] += 1
            else:
                week_change_pct = np.random.uniform(-0.015, -0.005)  # -0.5% to -1.5%
                self.stats['weekly_trend']['down'] += 1

            # Generate daily prices for this week
            week_prices = self.generate_week_prices(
                start_price=current_price,
                target_change_pct=week_change_pct,
                week_start=week_start,
                week_end=week_end
            )

            price_series.update(week_prices)
            if week_prices:
                current_price = list(week_prices.values())[-1][-1]  # Last price of week

        print(f"  ✓ Generated {len(price_series)} days of price data")
        return price_series

    def generate_week_prices(self, start_price, target_change_pct, week_start, week_end):
        """Generate daily prices for one week with target change"""
        target_end_price = start_price * (1 + target_change_pct)

        week_prices = {}
        current_date = week_start
        day_count = 0

        # Count trading days in week
        trading_days = []
        temp_date = week_start
        while temp_date <= week_end:
            if temp_date.weekday() < 5 and temp_date not in self.holidays:
                trading_days.append(temp_date)
            temp_date += timedelta(days=1)

        if not trading_days:
            return {}

        # Distribute change across days
        daily_changes = self.distribute_change(target_change_pct, len(trading_days))

        current_price = start_price
        for day, daily_change in zip(trading_days, daily_changes):
            day_open = current_price
            day_close = current_price * (1 + daily_change)

            # Generate 7 hourly prices
            hourly_prices = self.generate_hourly_prices(day_open, day_close)
            week_prices[day] = hourly_prices

            current_price = day_close

        # Adjust last day to hit exact target
        if trading_days:
            last_day = trading_days[-1]
            week_prices[last_day][-1] = target_end_price

        return week_prices

    def distribute_change(self, total_change, num_days):
        """Distribute total change across days with randomness"""
        # Generate random daily changes that sum to target
        changes = np.random.randn(num_days)
        changes = changes / changes.sum() * total_change

        # Add some randomness but constrain
        changes = changes + np.random.randn(num_days) * 0.002

        return changes

    def generate_hourly_prices(self, open_price, close_price):
        """Generate 7 hourly candles from open to close"""
        prices = [open_price]

        # Generate intermediate prices with mean reversion
        target_change = close_price - open_price

        for i in range(1, 7):
            # Gradual move toward close with noise
            progress = i / 7
            expected = open_price + (target_change * progress)
            noise = np.random.randn() * open_price * 0.001  # 0.1% noise
            price = expected + noise
            prices.append(max(open_price * 0.95, min(open_price * 1.05, price)))

        prices[-1] = close_price  # Ensure exact close

        return prices

    def get_vix_for_timestamp(self, dt, timestamp_index=0):
        """Get VIX value for specific datetime"""
        date = dt.date() if isinstance(dt, datetime) else dt
        date = datetime.combine(date, datetime.min.time())

        for (start_date, end_date), regime in self.vix_regimes.items():
            if start_date <= date <= end_date:
                base_vix = regime['base']
                vix_range = regime['range']
                trend = regime['trend']

                # Trend adjustment
                days_in_regime = (date - start_date).days
                total_days = (end_date - start_date).days + 1
                progress = days_in_regime / max(1, total_days)

                if trend == 'rising':
                    trend_adj = vix_range * progress
                elif trend == 'declining':
                    trend_adj = -vix_range * progress
                elif trend == 'spike':
                    trend_adj = vix_range * 2 * (0.5 - abs(progress - 0.5))
                else:
                    trend_adj = 0

                # Intraday variation
                intraday_factor = 1.0 + (timestamp_index / 100.0) * 0.02

                noise = np.random.normal(0, 0.5)
                vix = (base_vix + trend_adj + noise) * intraday_factor

                return round(max(10, min(50, vix)), 2)

        return 16.0

    def get_active_expiries(self, current_date):
        """Get all active expiries for a date"""
        expiries = []

        # Weekly expiries (next 4 weeks)
        for weeks_ahead in range(4):
            expiry = self.get_next_weekly_expiry(current_date + timedelta(weeks=weeks_ahead))
            if expiry and expiry >= current_date:
                expiries.append((expiry, 'weekly'))

        # Monthly expiries (current and next 2 months)
        for month_offset in range(3):
            month = ((current_date.month - 1 + month_offset) % 12) + 1
            year = current_date.year + ((current_date.month - 1 + month_offset) // 12)

            if year == 2025 and month in self.expiry_schedule['monthly']:
                day = self.expiry_schedule['monthly'][month]
                expiry = datetime(year, month, day)

                if expiry >= current_date:
                    expiries.append((expiry, 'monthly'))

        return sorted(set(expiries))[:6]  # Max 6 expiries

    def get_next_weekly_expiry(self, from_date):
        """Get next weekly expiry from date"""
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
        """Normal CDF approximation"""
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911

        sign = 1 if x >= 0 else -1
        x = abs(x) / np.sqrt(2.0)

        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t * np.exp(-x*x))

        return 0.5 * (1.0 + sign * y)

    def norm_pdf(self, x):
        """Normal PDF"""
        return np.exp(-x*x/2) / np.sqrt(2*np.pi)

    def get_iv_for_expiry(self, spot, strike, tte_days, expiry_type, vix):
        """
        Calculate IV with expiry-type-specific adjustments
        CRITICAL: Different IV for weekly vs monthly
        """
        base_iv = vix / 100.0

        # Volatility smile
        moneyness = spot / strike
        if abs(moneyness - 1) > 0.05:
            smile_adj = abs(moneyness - 1) * 0.2
            iv = base_iv * (1 + smile_adj)
        else:
            iv = base_iv

        # NEW: Expiry-type-specific term structure
        if expiry_type == 'weekly':
            if tte_days <= 3:
                iv *= 1.20  # +20% for very short TTE
            elif tte_days <= 7:
                iv *= 1.15  # +15% for weekly
        else:  # monthly
            if tte_days <= 7:
                iv *= 1.15
            elif tte_days <= 15:
                iv *= 1.08
            elif tte_days <= 30:
                iv *= 1.05

        return iv

    def calculate_option_price(self, spot, strike, tte_days, option_type, expiry_type, vix):
        """
        Black-Scholes option pricing with expiry-specific IV
        """
        if tte_days <= 0:
            intrinsic = max(0, spot - strike) if option_type == 'CE' else max(0, strike - spot)
            return max(0.05, intrinsic)

        T = tte_days / 365.0
        iv = self.get_iv_for_expiry(spot, strike, tte_days, expiry_type, vix)

        d1 = (np.log(spot/strike) + (self.risk_free_rate + 0.5*iv**2)*T) / (iv*np.sqrt(T))
        d2 = d1 - iv*np.sqrt(T)

        if option_type == 'CE':
            price = spot*self.norm_cdf(d1) - strike*np.exp(-self.risk_free_rate*T)*self.norm_cdf(d2)
        else:
            price = strike*np.exp(-self.risk_free_rate*T)*self.norm_cdf(-d2) - spot*self.norm_cdf(-d1)

        return max(0.05, round(price, 2))

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
        d2 = d1 - iv*sqrt_T

        if option_type == 'CE':
            delta = self.norm_cdf(d1)
        else:
            delta = -self.norm_cdf(-d1)

        gamma = self.norm_pdf(d1) / (spot * iv * sqrt_T) if sqrt_T > 0 else 0
        vega = spot * self.norm_pdf(d1) * sqrt_T / 100 if sqrt_T > 0 else 0

        # Theta
        term1 = -spot * self.norm_pdf(d1) * iv / (2 * sqrt_T) if sqrt_T > 0 else 0
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

    def generate_day_data(self, date):
        """
        Generate complete options data for one day
        CRITICAL: Expiry-specific pricing (not shared!)
        """
        data = []

        # Get hourly spot prices for this day
        if date not in self.price_series:
            return pd.DataFrame()

        spot_prices = self.price_series[date]

        # Generate timestamps (7 hourly candles)
        timestamps = []
        time_of_day = datetime.min.time().replace(hour=9, minute=15)
        for i in range(7):
            dt = datetime.combine(date, time_of_day)
            timestamps.append(dt)
            # Next hour
            time_of_day = (datetime.combine(date, time_of_day) + timedelta(hours=1)).time()

        # Get VIX for the day
        vix_values = [self.get_vix_for_timestamp(date, i) for i in range(7)]

        # Get active expiries
        active_expiries = self.get_active_expiries(date)

        # Generate all strikes
        strikes = list(range(self.min_strike, self.max_strike + 1, self.strike_interval))

        # Track delta coverage
        day_ce_low_delta = 0
        day_pe_low_delta = 0

        # CRITICAL: Loop through expiries FIRST (not strikes first!)
        for expiry_date, expiry_type in active_expiries:
            tte_days = (expiry_date - date).days

            # For each strike FOR THIS EXPIRY
            for strike in strikes:
                for option_type in ['CE', 'PE']:

                    # Generate prices for each timestamp for THIS expiry
                    for i, (timestamp, spot, vix) in enumerate(zip(timestamps, spot_prices, vix_values)):

                        # CRITICAL: Calculate price with correct TTE for THIS expiry
                        tte_current = tte_days - (i / self.timestamps_per_day)

                        price = self.calculate_option_price(
                            spot=spot,
                            strike=strike,
                            tte_days=tte_current,
                            option_type=option_type,
                            expiry_type=expiry_type,  # Weekly vs monthly IV
                            vix=vix
                        )

                        # Calculate Greeks
                        iv = self.get_iv_for_expiry(spot, strike, tte_current, expiry_type, vix)
                        greeks = self.calculate_greeks(spot, strike, tte_current, iv, option_type)

                        # Track 0.1 delta coverage (first timestamp only)
                        if i == 0:
                            delta_abs = abs(greeks['delta'])
                            if 0.05 <= delta_abs <= 0.15:
                                if option_type == 'CE':
                                    day_ce_low_delta += 1
                                else:
                                    day_pe_low_delta += 1

                        # Generate OHLC
                        vix_factor = vix / 15.0
                        noise = np.random.uniform(-0.015 * vix_factor, 0.015 * vix_factor, 3)
                        high = round(price * (1 + abs(noise[0])), 2)
                        low = round(price * (1 - abs(noise[1])), 2)
                        close = round(price * (1 + noise[2]), 2)

                        # Bid-Ask spread
                        moneyness = spot / strike
                        spread_pct = 0.01 * vix_factor if abs(moneyness - 1) < 0.02 else (0.02 + abs(moneyness - 1) * 0.03) * vix_factor
                        bid = round(price * (1 - spread_pct/2), 2)
                        ask = round(price * (1 + spread_pct/2), 2)

                        # Volume and OI
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
                            round(vix, 2)
                        ])

        # Update stats
        self.stats['delta_coverage']['ce_low_delta'] = max(
            self.stats['delta_coverage']['ce_low_delta'], day_ce_low_delta
        )
        self.stats['delta_coverage']['pe_low_delta'] = max(
            self.stats['delta_coverage']['pe_low_delta'], day_pe_low_delta
        )

        return pd.DataFrame(data, columns=[
            'timestamp', 'symbol', 'strike', 'option_type', 'expiry', 'expiry_type',
            'open', 'high', 'low', 'close', 'volume', 'oi', 'bid', 'ask',
            'iv', 'delta', 'gamma', 'theta', 'vega', 'underlying_price', 'vix'
        ])

    def validate_day_data(self, df, date):
        """Validate generated data for one day"""
        if df.empty:
            return True

        # Check delta-distance for weekly vs monthly
        spot = df['underlying_price'].iloc[0]

        # Weekly 0.1 delta check
        weekly = df[df['expiry_type'] == 'weekly']
        if not weekly.empty:
            weekly_ce = weekly[(weekly['option_type'] == 'CE') &
                              (weekly['delta'].abs().between(0.08, 0.12))]
            if not weekly_ce.empty:
                dist = weekly_ce.iloc[0]['strike'] - spot
                if not (800 <= dist <= 1400):
                    print(f"    ⚠️  Weekly 0.1Δ CE at {dist:.0f} pts (expected 800-1400)")

        # Monthly 0.1 delta check
        monthly = df[df['expiry_type'] == 'monthly']
        if not monthly.empty:
            monthly_ce = monthly[(monthly['option_type'] == 'CE') &
                                (monthly['delta'].abs().between(0.08, 0.12))]
            if not monthly_ce.empty:
                dist = monthly_ce.iloc[0]['strike'] - spot
                if not (1800 <= dist <= 2600):
                    print(f"    ⚠️  Monthly 0.1Δ CE at {dist:.0f} pts (expected 1800-2600)")

        return True

    def generate_all_data(self):
        """Main generation loop"""
        print("="*80)
        print("V9 BALANCED SYNTHETIC DATA GENERATOR - 1 HOUR CANDLES")
        print("="*80)
        print(f"Period: June 14 - September 30, 2025")
        print(f"Strikes: ₹{self.min_strike:,} - ₹{self.max_strike:,} (161 strikes)")
        print(f"Candles: 7 per day (1-hour intervals)")
        print(f"Output: {self.base_path}")
        print("="*80)

        # Step 1: Generate balanced price series
        start_date = datetime(2025, 6, 14)
        end_date = datetime(2025, 9, 30)

        self.price_series = self.generate_balanced_price_series(start_date, end_date)

        # Step 2: Generate options data day by day
        print("\n📊 Generating options data...")

        trading_dates = sorted(self.price_series.keys())

        for i, date in enumerate(trading_dates, 1):
            print(f"  [{i:2}/{len(trading_dates)}] {date.strftime('%Y-%m-%d')}", end=' ... ')

            # Generate data
            df = self.generate_day_data(date)

            if df.empty:
                print("⊘ No trading")
                continue

            # Validate
            self.validate_day_data(df, date)

            # Save
            filename = f"NIFTY_OPTIONS_1H_{date.strftime('%Y%m%d')}.csv"
            filepath = self.base_path / filename
            df.to_csv(filepath, index=False)

            # Update stats
            self.stats['files_created'] += 1
            self.stats['total_rows'] += len(df)
            self.stats['dates_processed'].append(date.strftime('%Y-%m-%d'))

            vix = df['vix'].iloc[0]
            strikes = df['strike'].nunique()
            print(f"✓ ({len(df):,} rows, {strikes} strikes, VIX: {vix:.1f})")

        # Save metadata
        self.save_metadata()

        # Print summary
        print("\n" + "="*80)
        print("✅ V9 DATA GENERATION COMPLETE!")
        print("="*80)
        print(f"Files created: {self.stats['files_created']}")
        print(f"Total rows: {self.stats['total_rows']:,}")
        print(f"Weekly trends: {self.stats['weekly_trend']['up']} up, {self.stats['weekly_trend']['down']} down")
        print(f"  Ratio: {self.stats['weekly_trend']['up']/(self.stats['weekly_trend']['up']+self.stats['weekly_trend']['down'])*100:.1f}% bullish")
        print(f"Delta coverage:")
        print(f"  CE 0.05-0.15Δ: {self.stats['delta_coverage']['ce_low_delta']} strikes")
        print(f"  PE 0.05-0.15Δ: {self.stats['delta_coverage']['pe_low_delta']} strikes")
        print(f"Location: {self.base_path}")
        print("="*80)

        return self.stats

    def save_metadata(self):
        """Save generation metadata"""
        metadata_path = self.base_path / 'metadata'
        metadata_path.mkdir(exist_ok=True)

        metadata = {
            'version': '9.0-Balanced-1H',
            'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'candle_interval': '1 hour',
            'timestamps_per_day': self.timestamps_per_day,
            'critical_fixes': [
                'Balanced trend generation (50/50 weekly)',
                'Expiry-specific option pricing',
                'Correct delta-distance relationships',
                'Realistic weekly premiums',
                '1-hour candles for efficiency'
            ],
            'period': {
                'start': '2025-06-14',
                'end': '2025-09-30'
            },
            'strike_range': {
                'min': self.min_strike,
                'max': self.max_strike,
                'interval': self.strike_interval,
                'total': (self.max_strike - self.min_strike) // self.strike_interval + 1
            },
            'trend_stats': self.stats['weekly_trend'],
            'stats': self.stats,
            'expiry_schedule': self.expiry_schedule
        }

        with open(metadata_path / 'generation_info.json', 'w') as f:
            json.dump(metadata, f, indent=2)

def main():
    generator = V9BalancedGenerator()
    stats = generator.generate_all_data()
    return stats

if __name__ == '__main__':
    main()
