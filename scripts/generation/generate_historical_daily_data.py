"""
Generate synthetic Nifty options DAILY (interday) data from Jan 2023 to Sep 2025
Includes weekly expiry change from Thursday to Wednesday
Monthly expiries remain on last Thursday
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import math
import calendar

class HistoricalDailyDataGenerator:
    def __init__(self, base_path):
        self.base_path = base_path
        self.output_path = os.path.join(base_path, 'daily_2023_2025')
        
        # Create directory if it doesn't exist
        os.makedirs(self.output_path, exist_ok=True)
        
        # Weekly expiry changed from Thursday to Wednesday
        # For this simulation, let's assume the change happened in Feb 2024
        self.weekly_expiry_change_date = pd.Timestamp('2024-02-01')
        
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
                pd.Timestamp(f'{year}-11-13'),  # Diwali
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
        last_day = calendar.monthrange(year, month)[1]
        last_date = pd.Timestamp(year=year, month=month, day=last_day)
        
        while last_date.dayofweek != 3:  # Thursday is 3
            last_date -= timedelta(days=1)
        
        return last_date
    
    def get_weekly_expiry_day(self, date):
        """Get the weekly expiry day based on date (Wednesday after Feb 2024, Thursday before)"""
        if date >= self.weekly_expiry_change_date:
            return 2  # Wednesday
        else:
            return 3  # Thursday
    
    def get_near_month_expiries(self, current_date):
        """Get near-month expiries for a given date"""
        expiries = []
        
        # Determine weekly expiry day
        weekly_expiry_day = self.get_weekly_expiry_day(current_date)
        
        # Current week's expiry
        days_to_expiry = (weekly_expiry_day - current_date.dayofweek) % 7
        if days_to_expiry == 0:  # If today is expiry day
            # Check if market has closed
            if hasattr(current_date, 'hour') and current_date.hour >= 15 and current_date.minute >= 30:
                days_to_expiry = 7
        
        week_expiry = current_date + timedelta(days=days_to_expiry)
        expiries.append((week_expiry, 'weekly'))
        
        # Monthly expiry (always last Thursday)
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
        
        expiries.append((month_expiry, 'monthly'))
        
        # Add next week's expiry if current week is expiring
        if (week_expiry - current_date).days <= 1:
            next_week_expiry = week_expiry + timedelta(days=7)
            expiries.append((next_week_expiry, 'weekly'))
        
        # Add next 2 weekly expiries for better coverage
        next_week = week_expiry + timedelta(days=7)
        if (next_week - current_date).days <= 35:
            expiries.append((next_week, 'weekly'))
        
        next_next_week = week_expiry + timedelta(days=14)
        if (next_next_week - current_date).days <= 35:
            expiries.append((next_next_week, 'weekly'))
        
        # Remove duplicates and filter to near-month only
        unique_expiries = []
        seen = set()
        for exp, typ in expiries:
            if exp not in seen and (exp - current_date).days <= 35 and exp >= current_date:
                unique_expiries.append((exp, typ))
                seen.add(exp)
        
        return unique_expiries
    
    def get_trading_days(self, start_date, end_date):
        """Get list of trading days (excluding weekends and holidays)"""
        trading_days = []
        current = start_date
        
        while current <= end_date:
            if current.dayofweek < 5:  # Weekday
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
    
    def generate_daily_data(self, start_date, end_date):
        """Generate daily options data"""
        daily_data = []
        trading_days = self.get_trading_days(start_date, end_date)
        
        for date in trading_days:
            underlying_close = self.generate_underlying_price(date)
            underlying_open = underlying_close * np.random.uniform(0.995, 1.005)
            
            # Get near-month expiries for this date
            expiries = self.get_near_month_expiries(date)
            
            # Generate strikes (ATM +/- 2000 points, 50 point intervals)
            strikes = range(int(underlying_close - 2000), int(underlying_close + 2050), 50)
            
            for expiry_date, expiry_type in expiries:
                days_to_expiry = (expiry_date - date).days
                
                # Skip if too far
                if days_to_expiry > 35:
                    continue
                
                for strike in strikes:
                    for option_type in ['CE', 'PE']:
                        # Skip some far OTM options
                        if abs(strike - underlying_close) > 1000:
                            if np.random.random() > 0.4:
                                continue
                        
                        base_iv = 0.12 + np.random.uniform(-0.02, 0.08)
                        option_price = self.calculate_option_price(
                            underlying_close, strike, days_to_expiry, option_type, base_iv
                        )
                        
                        # OHLC for the day
                        open_price = option_price * np.random.uniform(0.95, 1.05)
                        high_price = option_price * np.random.uniform(1.01, 1.10)
                        low_price = option_price * np.random.uniform(0.90, 0.99)
                        close_price = option_price
                        
                        # Volume and OI
                        base_volume = 100000 * math.exp(-abs(strike - underlying_close) / 300)
                        volume = int(base_volume * np.random.uniform(0.5, 2.0))
                        oi = int(volume * np.random.uniform(5, 20))
                        
                        # VWAP
                        vwap = (open_price + high_price + low_price + close_price) / 4
                        
                        # Bid-Ask
                        spread_pct = 0.01 if option_price > 50 else 0.02
                        bid = round(option_price * (1 - spread_pct), 2)
                        ask = round(option_price * (1 + spread_pct), 2)
                        
                        # Greeks
                        greeks = self.calculate_greeks(
                            underlying_close, strike, days_to_expiry, option_type, base_iv
                        )
                        
                        # Contracts and turnover
                        contracts_traded = volume // 25
                        turnover_cr = round((volume * option_price * 25) / 10000000, 2)
                        
                        daily_data.append({
                            'date': date.strftime('%Y-%m-%d'),
                            'symbol': 'NIFTY',
                            'strike': strike,
                            'option_type': option_type,
                            'expiry': expiry_date.strftime('%Y-%m-%d'),
                            'expiry_type': expiry_type,
                            'open': round(open_price, 2),
                            'high': round(high_price, 2),
                            'low': round(low_price, 2),
                            'close': round(close_price, 2),
                            'volume': volume,
                            'oi': oi,
                            'vwap': round(vwap, 2),
                            'bid': bid,
                            'ask': ask,
                            'iv': round(base_iv, 4),
                            'delta': greeks['delta'],
                            'gamma': greeks['gamma'],
                            'theta': greeks['theta'],
                            'vega': greeks['vega'],
                            'underlying_close': underlying_close,
                            'contracts_traded': contracts_traded,
                            'turnover': f"{turnover_cr}Cr"
                        })
        
        return pd.DataFrame(daily_data)
    
    def generate_all_data(self):
        """Generate all historical daily data from Jan 2023 to Sep 2025"""
        print("=" * 80)
        print("GENERATING HISTORICAL DAILY (INTERDAY) OPTIONS DATA")
        print("Period: January 2023 to September 2025")
        print("Configuration:")
        print("  - Weekly expiries: Thursday (until Jan 2024), Wednesday (from Feb 2024)")
        print("  - Monthly expiries: Last Thursday (throughout)")
        print("  - Near-month contracts only (max 35 days)")
        print("=" * 80)
        
        # Define periods to generate
        periods = [
            ('2023-01-01', '2023-12-31', '2023'),
            ('2024-01-01', '2024-12-31', '2024'),
            ('2025-01-01', '2025-09-30', '2025_Jan_Sep')
        ]
        
        all_data = []
        
        for start_str, end_str, label in periods:
            start_date = pd.Timestamp(start_str)
            end_date = pd.Timestamp(end_str)
            
            print(f"\n📅 Generating data for {label}...")
            
            # Generate daily data
            daily_df = self.generate_daily_data(start_date, end_date)
            all_data.append(daily_df)
            
            # Save yearly data
            filename = f'NIFTY_OPTIONS_DAILY_{label}.csv'
            filepath = os.path.join(self.output_path, filename)
            daily_df.to_csv(filepath, index=False)
            
            print(f"  ✅ Saved {len(daily_df)} records to {filename}")
            
            # Show expiry distribution
            if len(daily_df) > 0:
                unique_dates = daily_df['date'].nunique()
                unique_expiries = daily_df[['expiry', 'expiry_type']].drop_duplicates()
                
                print(f"  📊 Statistics:")
                print(f"     - Trading days: {unique_dates}")
                print(f"     - Unique expiries: {len(unique_expiries)}")
                
                # Show sample expiries
                weekly_sample = unique_expiries[unique_expiries['expiry_type'] == 'weekly']['expiry'].head(3).tolist()
                monthly_sample = unique_expiries[unique_expiries['expiry_type'] == 'monthly']['expiry'].head(3).tolist()
                
                print(f"     - Sample weekly expiries: {', '.join(weekly_sample)}")
                print(f"     - Sample monthly expiries: {', '.join(monthly_sample)}")
        
        # Combine all data
        print("\n📊 Creating combined dataset...")
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_file = os.path.join(self.output_path, 'NIFTY_OPTIONS_DAILY_2023_2025_COMPLETE.csv')
        combined_df.to_csv(combined_file, index=False)
        
        print(f"✅ Saved combined dataset: {len(combined_df)} records")
        
        # Create summary
        print("\n" + "=" * 80)
        print("✅ DATA GENERATION COMPLETE!")
        print("=" * 80)
        
        print("\n📊 FINAL SUMMARY:")
        print(f"• Total records: {len(combined_df):,}")
        print(f"• Trading days covered: {combined_df['date'].nunique()}")
        print(f"• Output location: {self.output_path}")
        
        # Verify expiry day changes
        print("\n🔄 EXPIRY DAY VERIFICATION:")
        
        # Check some dates before and after change
        jan_2024 = combined_df[combined_df['date'] == '2024-01-15']
        if len(jan_2024) > 0:
            jan_weekly = jan_2024[jan_2024['expiry_type'] == 'weekly']['expiry'].iloc[0]
            jan_day = pd.Timestamp(jan_weekly).day_name()
            print(f"  January 2024 weekly expiry: {jan_weekly} ({jan_day})")
        
        mar_2024 = combined_df[combined_df['date'] == '2024-03-15']
        if len(mar_2024) > 0:
            mar_weekly = mar_2024[mar_2024['expiry_type'] == 'weekly']['expiry'].iloc[0]
            mar_day = pd.Timestamp(mar_weekly).day_name()
            print(f"  March 2024 weekly expiry: {mar_weekly} ({mar_day})")
        
        # Show file list
        print("\n📁 FILES GENERATED:")
        for file in os.listdir(self.output_path):
            if file.endswith('.csv'):
                file_path = os.path.join(self.output_path, file)
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                print(f"  • {file} ({file_size:.1f} MB)")

if __name__ == "__main__":
    base_path = "/Users/nitindhawan/NikAlgoBulls/zerodha_strategy/data/synthetic"
    generator = HistoricalDailyDataGenerator(base_path)
    generator.generate_all_data()