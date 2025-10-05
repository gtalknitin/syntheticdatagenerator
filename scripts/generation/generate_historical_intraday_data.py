"""
Generate synthetic Nifty options intraday data from Jan 2023 to Sep 2025
Includes both weekly and monthly expiries with near-month contracts only
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import math
import calendar

class HistoricalIntradayDataGenerator:
    def __init__(self, base_path):
        self.base_path = base_path
        self.output_path = os.path.join(base_path, 'intraday_2023_2025')
        
        # Create directory if it doesn't exist
        os.makedirs(self.output_path, exist_ok=True)
        
        # Trading hours
        self.market_open = pd.Timedelta(hours=9, minutes=15)
        self.market_close = pd.Timedelta(hours=15, minutes=30)
        
        # Base Nifty levels for different periods
        self.base_levels = {
            2023: 18000,  # Starting level for 2023
            2024: 21500,  # Starting level for 2024
            2025: 25000   # Starting level for 2025
        }
    
    def get_holidays(self, year):
        """Get market holidays for a given year"""
        holidays = {
            2023: [
                pd.Timestamp(f'{year}-01-26'),  # Republic Day
                pd.Timestamp(f'{year}-03-07'),  # Holi
                pd.Timestamp(f'{year}-03-30'),  # Ram Navami
                pd.Timestamp(f'{year}-04-04'),  # Mahavir Jayanti
                pd.Timestamp(f'{year}-04-07'),  # Good Friday
                pd.Timestamp(f'{year}-04-14'),  # Ambedkar Jayanti
                pd.Timestamp(f'{year}-05-01'),  # Maharashtra Day
                pd.Timestamp(f'{year}-06-29'),  # Bakri Id
                pd.Timestamp(f'{year}-08-15'),  # Independence Day
                pd.Timestamp(f'{year}-09-19'),  # Ganesh Chaturthi
                pd.Timestamp(f'{year}-10-02'),  # Gandhi Jayanti
                pd.Timestamp(f'{year}-10-24'),  # Dussehra
                pd.Timestamp(f'{year}-11-14'),  # Diwali Balipratipada
                pd.Timestamp(f'{year}-11-27'),  # Guru Nanak Jayanti
            ],
            2024: [
                pd.Timestamp(f'{year}-01-26'),  # Republic Day
                pd.Timestamp(f'{year}-03-08'),  # Mahashivratri
                pd.Timestamp(f'{year}-03-25'),  # Holi
                pd.Timestamp(f'{year}-03-29'),  # Good Friday
                pd.Timestamp(f'{year}-04-11'),  # Id-Ul-Fitr
                pd.Timestamp(f'{year}-04-17'),  # Ram Navami
                pd.Timestamp(f'{year}-05-01'),  # Maharashtra Day
                pd.Timestamp(f'{year}-06-17'),  # Bakri Id
                pd.Timestamp(f'{year}-08-15'),  # Independence Day
                pd.Timestamp(f'{year}-10-02'),  # Gandhi Jayanti
                pd.Timestamp(f'{year}-11-01'),  # Diwali
                pd.Timestamp(f'{year}-11-15'),  # Guru Nanak Jayanti
            ],
            2025: [
                pd.Timestamp(f'{year}-01-26'),  # Republic Day (Sunday)
                pd.Timestamp(f'{year}-03-14'),  # Holi
                pd.Timestamp(f'{year}-03-31'),  # Id-Ul-Fitr
                pd.Timestamp(f'{year}-04-18'),  # Good Friday
                pd.Timestamp(f'{year}-05-01'),  # Maharashtra Day
                pd.Timestamp(f'{year}-06-07'),  # Bakri Id
                pd.Timestamp(f'{year}-08-15'),  # Independence Day
                pd.Timestamp(f'{year}-10-02'),  # Gandhi Jayanti
                pd.Timestamp(f'{year}-10-20'),  # Diwali
                pd.Timestamp(f'{year}-11-05'),  # Guru Nanak Jayanti
            ]
        }
        return holidays.get(year, [])
    
    def get_last_thursday(self, year, month):
        """Get the last Thursday of a given month"""
        # Get the last day of the month
        last_day = calendar.monthrange(year, month)[1]
        last_date = pd.Timestamp(year=year, month=month, day=last_day)
        
        # Find the last Thursday
        while last_date.dayofweek != 3:  # Thursday is 3
            last_date -= timedelta(days=1)
        
        return last_date
    
    def get_all_thursdays(self, year, month):
        """Get all Thursdays in a given month"""
        thursdays = []
        first_day = pd.Timestamp(year=year, month=month, day=1)
        
        # Find the first Thursday
        current = first_day
        while current.dayofweek != 3:
            current += timedelta(days=1)
        
        # Get all Thursdays in the month
        while current.month == month:
            thursdays.append(current)
            current += timedelta(days=7)
        
        return thursdays
    
    def get_near_month_expiries(self, current_date):
        """Get near-month expiries for a given date"""
        expiries = []
        
        # Current week's Thursday (weekly expiry)
        days_to_thursday = (3 - current_date.dayofweek) % 7
        if days_to_thursday == 0:  # If today is Thursday
            if current_date.hour >= 15 and current_date.minute >= 30:
                days_to_thursday = 7  # Next Thursday
        
        week_expiry = current_date + timedelta(days=days_to_thursday)
        
        # Add weekly expiry
        expiries.append((week_expiry, 'weekly'))
        
        # Get monthly expiry (last Thursday of current month)
        current_month = current_date.month
        current_year = current_date.year
        month_expiry = self.get_last_thursday(current_year, current_month)
        
        # If we've passed this month's expiry, use next month's
        if current_date > month_expiry:
            if current_month == 12:
                next_month = 1
                next_year = current_year + 1
            else:
                next_month = current_month + 1
                next_year = current_year
            
            month_expiry = self.get_last_thursday(next_year, next_month)
        
        # Add monthly expiry
        expiries.append((month_expiry, 'monthly'))
        
        # If current week expiry is very close, add next week too
        if (week_expiry - current_date).days <= 1:
            next_week_expiry = week_expiry + timedelta(days=7)
            expiries.append((next_week_expiry, 'weekly'))
        
        # Filter to keep only near-month (within 35 days)
        expiries = [(exp, typ) for exp, typ in expiries if (exp - current_date).days <= 35]
        
        return expiries
    
    def get_trading_days(self, start_date, end_date):
        """Get list of trading days (excluding weekends and holidays)"""
        trading_days = []
        current = start_date
        
        while current <= end_date:
            # Skip weekends
            if current.dayofweek < 5:
                # Skip holidays
                year_holidays = self.get_holidays(current.year)
                if current not in year_holidays:
                    trading_days.append(current)
            current += timedelta(days=1)
        
        return trading_days
    
    def generate_underlying_price(self, date):
        """Generate realistic underlying Nifty price based on historical trends"""
        year = date.year
        base_price = self.base_levels.get(year, 20000)
        
        # Add trend based on time in year
        days_in_year = (date - pd.Timestamp(year=year, month=1, day=1)).days
        yearly_trend = base_price * (1 + 0.0004 * days_in_year)  # ~15% annual growth
        
        # Add monthly seasonality
        month_factor = 1 + 0.02 * np.sin(2 * np.pi * date.month / 12)
        
        # Add random volatility
        np.random.seed(int(date.timestamp()) % 2**32)
        volatility = np.random.normal(0, base_price * 0.01)  # 1% daily volatility
        
        return round(yearly_trend * month_factor + volatility, 2)
    
    def calculate_option_price(self, spot, strike, days_to_expiry, option_type, iv=0.15):
        """Calculate option price using simplified Black-Scholes"""
        if days_to_expiry <= 0:
            if option_type == 'CE':
                return max(0, spot - strike)
            else:
                return max(0, strike - spot)
        
        time_to_expiry = days_to_expiry / 365.0
        moneyness = spot / strike
        
        if option_type == 'CE':
            intrinsic = max(0, spot - strike)
            if moneyness > 1.05:  # Deep ITM
                time_value = 10 * math.sqrt(time_to_expiry)
            elif moneyness > 0.95:  # Near ATM
                time_value = spot * 0.02 * math.sqrt(time_to_expiry)
            else:  # OTM
                time_value = spot * 0.01 * math.sqrt(time_to_expiry) * max(0, 1 - abs(1 - moneyness))
        else:  # PE
            intrinsic = max(0, strike - spot)
            if moneyness < 0.95:  # Deep ITM
                time_value = 10 * math.sqrt(time_to_expiry)
            elif moneyness < 1.05:  # Near ATM
                time_value = spot * 0.02 * math.sqrt(time_to_expiry)
            else:  # OTM
                time_value = spot * 0.01 * math.sqrt(time_to_expiry) * max(0, 1 - abs(1 - moneyness))
        
        return round(intrinsic + time_value, 2)
    
    def calculate_greeks(self, spot, strike, days_to_expiry, option_type, iv):
        """Calculate option Greeks"""
        if days_to_expiry <= 0:
            return {
                'delta': 1.0 if (option_type == 'CE' and spot > strike) else 0.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0
            }
        
        moneyness = spot / strike
        time_factor = math.sqrt(days_to_expiry / 365.0)
        
        if option_type == 'CE':
            if moneyness > 1.1:
                delta = 0.9 + 0.09 * min(1, time_factor)
            elif moneyness > 0.9:
                delta = 0.5 + 0.4 * (moneyness - 1.0)
            else:
                delta = max(0.01, 0.3 * moneyness * time_factor)
        else:  # PE
            if moneyness < 0.9:
                delta = -0.9 - 0.09 * min(1, time_factor)
            elif moneyness < 1.1:
                delta = -0.5 + 0.4 * (moneyness - 1.0)
            else:
                delta = -max(0.01, 0.3 * (2 - moneyness) * time_factor)
        
        gamma = 0.001 * math.exp(-10 * (moneyness - 1.0)**2) / max(0.1, time_factor)
        theta = -10 * math.exp(-5 * (moneyness - 1.0)**2) / max(0.1, time_factor)
        vega = spot * 0.001 * math.sqrt(time_factor) * math.exp(-2 * (moneyness - 1.0)**2)
        
        return {
            'delta': round(delta, 4),
            'gamma': round(gamma, 6),
            'theta': round(theta, 2),
            'vega': round(vega, 2)
        }
    
    def generate_intraday_data(self, date, underlying_open, underlying_close):
        """Generate 5-minute intraday data for a given date"""
        data = []
        
        # Get near-month expiries
        expiries = self.get_near_month_expiries(date)
        
        # Generate timestamps for the day
        timestamps = pd.date_range(
            start=date + self.market_open,
            end=date + self.market_close,
            freq='5min'
        )
        
        # Generate strikes (ATM +/- 800 points, 50 point intervals)
        strikes = range(int(underlying_open - 800), int(underlying_open + 850), 50)
        
        for expiry, expiry_type in expiries:
            if expiry < date:
                continue
            
            days_to_expiry = (expiry - date).days
            
            for strike in strikes:
                for option_type in ['CE', 'PE']:
                    # Skip some far OTM options to reduce file size
                    if abs(strike - underlying_open) > 500:
                        if np.random.random() > 0.4:  # Keep only 40% of far OTM
                            continue
                    
                    base_iv = 0.12 + np.random.uniform(-0.02, 0.06)
                    
                    for timestamp in timestamps:
                        # Intraday price movement
                        time_of_day = (timestamp - date).total_seconds() / 3600
                        intraday_factor = time_of_day / 6.25  # 6.25 hours trading day
                        
                        # Calculate spot price at this time
                        spot = underlying_open + (underlying_close - underlying_open) * intraday_factor
                        spot += np.random.normal(0, 5)  # Add noise
                        
                        # Calculate option price
                        option_price = self.calculate_option_price(
                            spot, strike, days_to_expiry, option_type, base_iv
                        )
                        
                        # Add price variation
                        option_price *= (1 + np.random.uniform(-0.015, 0.015))
                        option_price = max(0.05, round(option_price, 2))
                        
                        # Calculate bid-ask spread
                        spread_pct = 0.01 if option_price > 50 else 0.02
                        bid = round(option_price * (1 - spread_pct), 2)
                        ask = round(option_price * (1 + spread_pct), 2)
                        
                        # Generate volume and OI
                        base_volume = 8000 * math.exp(-abs(strike - spot) / 250)
                        volume = int(base_volume * np.random.uniform(0.5, 1.5))
                        oi = int(volume * np.random.uniform(8, 15))
                        
                        # Calculate Greeks
                        greeks = self.calculate_greeks(spot, strike, days_to_expiry, option_type, base_iv)
                        
                        data.append({
                            'timestamp': timestamp,
                            'symbol': 'NIFTY',
                            'strike': strike,
                            'option_type': option_type,
                            'expiry': expiry.strftime('%Y-%m-%d'),
                            'expiry_type': expiry_type,
                            'open': option_price,
                            'high': round(option_price * 1.015, 2),
                            'low': round(option_price * 0.985, 2),
                            'close': option_price,
                            'volume': volume,
                            'oi': oi,
                            'bid': bid,
                            'ask': ask,
                            'iv': round(base_iv, 4),
                            'delta': greeks['delta'],
                            'gamma': greeks['gamma'],
                            'theta': greeks['theta'],
                            'vega': greeks['vega'],
                            'underlying_price': round(spot, 2)
                        })
        
        return pd.DataFrame(data)
    
    def generate_month_data(self, year, month):
        """Generate all intraday data for a given month"""
        start_date = pd.Timestamp(year=year, month=month, day=1)
        
        # Get last day of month
        last_day = calendar.monthrange(year, month)[1]
        end_date = pd.Timestamp(year=year, month=month, day=last_day)
        
        # Get trading days
        trading_days = self.get_trading_days(start_date, end_date)
        
        print(f"  Generating {len(trading_days)} trading days for {calendar.month_name[month]} {year}...")
        
        # Create month directory
        month_dir = os.path.join(self.output_path, f"{year}_{month:02d}")
        os.makedirs(month_dir, exist_ok=True)
        
        for i, date in enumerate(trading_days, 1):
            # Generate underlying prices
            underlying_close = self.generate_underlying_price(date)
            underlying_open = underlying_close * np.random.uniform(0.997, 1.003)
            
            # Generate intraday data
            intraday_df = self.generate_intraday_data(date, underlying_open, underlying_close)
            
            # Save to file
            filename = f"NIFTY_OPTIONS_5MIN_{date.strftime('%Y%m%d')}.csv"
            filepath = os.path.join(month_dir, filename)
            intraday_df.to_csv(filepath, index=False)
            
            if i % 5 == 0 or i == len(trading_days):
                print(f"    Progress: {i}/{len(trading_days)} days completed")
        
        return len(trading_days)
    
    def generate_all_data(self):
        """Generate all historical intraday data from Jan 2023 to Sep 2025"""
        print("=" * 80)
        print("GENERATING HISTORICAL INTRADAY OPTIONS DATA")
        print("Period: January 2023 to September 2025")
        print("Configuration: Near-month expiries with weekly and monthly options")
        print("=" * 80)
        
        total_days = 0
        total_files = 0
        
        # Generate data for each month
        periods = []
        
        # 2023
        for month in range(1, 13):
            periods.append((2023, month))
        
        # 2024
        for month in range(1, 13):
            periods.append((2024, month))
        
        # 2025 (Jan to Sep)
        for month in range(1, 10):
            periods.append((2025, month))
        
        for year, month in periods:
            print(f"\n📅 Processing {calendar.month_name[month]} {year}...")
            
            days_generated = self.generate_month_data(year, month)
            total_days += days_generated
            total_files += days_generated
            
            print(f"  ✅ Completed: {days_generated} files generated")
        
        # Create summary file
        summary_path = os.path.join(self.output_path, 'DATA_SUMMARY.txt')
        with open(summary_path, 'w') as f:
            f.write("HISTORICAL INTRADAY OPTIONS DATA SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Period: January 2023 to September 2025\n")
            f.write(f"Total Trading Days: {total_days}\n")
            f.write(f"Total Files Generated: {total_files}\n\n")
            f.write("Directory Structure:\n")
            f.write("  /intraday_2023_2025/\n")
            f.write("    /YYYY_MM/\n")
            f.write("      NIFTY_OPTIONS_5MIN_YYYYMMDD.csv\n\n")
            f.write("Features:\n")
            f.write("  - 5-minute interval data\n")
            f.write("  - Weekly expiries (every Thursday)\n")
            f.write("  - Monthly expiries (last Thursday)\n")
            f.write("  - Near-month contracts only (max 35 days)\n")
            f.write("  - Complete Greeks calculation\n")
            f.write("  - Realistic volume and OI\n")
        
        print("\n" + "=" * 80)
        print("✅ DATA GENERATION COMPLETE!")
        print("=" * 80)
        
        print("\n📊 SUMMARY:")
        print(f"• Total Period: 33 months (Jan 2023 - Sep 2025)")
        print(f"• Total Trading Days: {total_days}")
        print(f"• Total Files Generated: {total_files}")
        print(f"• Output Location: {self.output_path}")
        print(f"• Directory Structure: YYYY_MM/NIFTY_OPTIONS_5MIN_YYYYMMDD.csv")
        
        print("\n📁 DIRECTORY STRUCTURE:")
        print("  intraday_2023_2025/")
        print("    ├── 2023_01/  (January 2023)")
        print("    ├── 2023_02/  (February 2023)")
        print("    ├── ...")
        print("    ├── 2025_08/  (August 2025)")
        print("    └── 2025_09/  (September 2025)")
        
        print("\n✨ Data includes both weekly and monthly expiries with realistic pricing!")

if __name__ == "__main__":
    base_path = "/Users/nitindhawan/NikAlgoBulls/zerodha_strategy/data/synthetic"
    generator = HistoricalIntradayDataGenerator(base_path)
    generator.generate_all_data()