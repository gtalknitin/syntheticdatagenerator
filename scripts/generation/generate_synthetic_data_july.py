"""
Generate synthetic Nifty options data starting from July 1st with both weekly and monthly expiries
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import math

class SyntheticDataGenerator:
    def __init__(self, base_path):
        self.base_path = base_path
        self.intraday_path = os.path.join(base_path, 'intraday_july')
        self.daily_path = os.path.join(base_path, 'daily_july')
        
        # Create directories if they don't exist
        os.makedirs(self.intraday_path, exist_ok=True)
        os.makedirs(self.daily_path, exist_ok=True)
        
        # Trading hours
        self.market_open = pd.Timedelta(hours=9, minutes=15)
        self.market_close = pd.Timedelta(hours=15, minutes=30)
        
    def get_expiry_dates(self, year, month):
        """Get both weekly (Thursday) and monthly (last Thursday) expiry dates"""
        expiries = []
        
        # Find all Thursdays in the month
        first_day = pd.Timestamp(year=year, month=month, day=1)
        if month == 12:
            next_month = pd.Timestamp(year=year+1, month=1, day=1)
        else:
            next_month = pd.Timestamp(year=year, month=month+1, day=1)
        
        current_day = first_day
        thursdays = []
        
        while current_day < next_month:
            if current_day.dayofweek == 3:  # Thursday
                thursdays.append(current_day)
            current_day += timedelta(days=1)
        
        # Add all Thursdays as weekly expiries
        for thursday in thursdays:
            expiries.append((thursday, 'weekly'))
        
        # Last Thursday is also the monthly expiry
        if thursdays:
            expiries.append((thursdays[-1], 'monthly'))
        
        return expiries
    
    def get_trading_days(self, start_date, end_date):
        """Get list of trading days (excluding weekends and holidays)"""
        holidays = [
            pd.Timestamp('2025-01-26'),  # Republic Day (Sunday - no impact)
            pd.Timestamp('2025-03-14'),  # Holi
            pd.Timestamp('2025-04-18'),  # Good Friday
            pd.Timestamp('2025-08-15'),  # Independence Day
            pd.Timestamp('2025-10-02'),  # Gandhi Jayanti
            pd.Timestamp('2025-10-24'),  # Diwali
        ]
        
        trading_days = []
        current = start_date
        
        while current <= end_date:
            # Skip weekends and holidays
            if current.dayofweek < 5 and current not in holidays:
                trading_days.append(current)
            current += timedelta(days=1)
        
        return trading_days
    
    def generate_underlying_price(self, date, base_price=25000):
        """Generate realistic underlying Nifty price with trend and volatility"""
        # Add trend component (slight upward bias)
        days_from_start = (date - pd.Timestamp('2025-07-01')).days
        trend = base_price * (1 + 0.0002 * days_from_start)  # 0.02% daily trend
        
        # Add random walk component
        np.random.seed(int(date.timestamp()) % 2**32)
        volatility = np.random.normal(0, base_price * 0.008)  # 0.8% daily volatility
        
        return round(trend + volatility, 2)
    
    def calculate_option_price(self, spot, strike, days_to_expiry, option_type, iv=0.15):
        """Calculate option price using simplified Black-Scholes approximation"""
        if days_to_expiry <= 0:
            # Option has expired
            if option_type == 'CE':
                return max(0, spot - strike)
            else:
                return max(0, strike - spot)
        
        time_to_expiry = days_to_expiry / 365.0
        r = 0.06  # Risk-free rate
        
        # Simplified pricing model
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
            if moneyness > 1.1:  # Deep ITM
                delta = 0.9 + 0.09 * min(1, time_factor)
            elif moneyness > 0.9:  # Near ATM
                delta = 0.5 + 0.4 * (moneyness - 1.0)
            else:  # OTM
                delta = max(0.01, 0.3 * moneyness * time_factor)
        else:  # PE
            if moneyness < 0.9:  # Deep ITM
                delta = -0.9 - 0.09 * min(1, time_factor)
            elif moneyness < 1.1:  # Near ATM
                delta = -0.5 + 0.4 * (moneyness - 1.0)
            else:  # OTM
                delta = -max(0.01, 0.3 * (2 - moneyness) * time_factor)
        
        # Gamma peaks near ATM
        gamma = 0.001 * math.exp(-10 * (moneyness - 1.0)**2) / max(0.1, time_factor)
        
        # Theta (time decay)
        theta = -10 * math.exp(-5 * (moneyness - 1.0)**2) / max(0.1, time_factor)
        
        # Vega
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
        
        # Get relevant expiries (current week and month)
        week_expiry = None
        month_expiry = None
        
        # Find current week's Thursday
        days_to_thursday = (3 - date.dayofweek) % 7
        if days_to_thursday == 0 and date.hour >= 15 and date.minute >= 30:
            days_to_thursday = 7
        week_expiry = date + timedelta(days=days_to_thursday)
        
        # Find month end expiry (last Thursday of month)
        current_month = date.month
        current_year = date.year
        
        # Get last Thursday of current month
        next_month = current_month + 1 if current_month < 12 else 1
        next_year = current_year if current_month < 12 else current_year + 1
        first_of_next = pd.Timestamp(year=next_year, month=next_month, day=1)
        
        # Find last Thursday
        current_day = first_of_next - timedelta(days=1)
        while current_day.dayofweek != 3:  # Thursday
            current_day -= timedelta(days=1)
        month_expiry = current_day
        
        # If we've passed this month's expiry, use next month's
        if date > month_expiry:
            if current_month == 12:
                next_month = 1
                next_year = current_year + 1
            else:
                next_month = current_month + 1
                next_year = current_year
                
            # Find last Thursday of next month
            if next_month == 12:
                first_of_next = pd.Timestamp(year=next_year + 1, month=1, day=1)
            else:
                first_of_next = pd.Timestamp(year=next_year, month=next_month + 1, day=1)
            
            current_day = first_of_next - timedelta(days=1)
            while current_day.dayofweek != 3:
                current_day -= timedelta(days=1)
            month_expiry = current_day
        
        # Generate timestamps for the day
        timestamps = pd.date_range(
            start=date + self.market_open,
            end=date + self.market_close,
            freq='5min'
        )
        
        # Generate strikes (ATM +/- 1000 points, 50 point intervals)
        strikes = range(int(underlying_open - 1000), int(underlying_open + 1050), 50)
        
        for expiry, expiry_type in [(week_expiry, 'weekly'), (month_expiry, 'monthly')]:
            if expiry < date:
                continue
                
            days_to_expiry = (expiry - date).days
            
            for strike in strikes:
                for option_type in ['CE', 'PE']:
                    # Skip some far OTM options for weekly to reduce data size
                    if expiry_type == 'weekly' and abs(strike - underlying_open) > 500:
                        if np.random.random() > 0.3:  # Keep only 30% of far OTM
                            continue
                    
                    base_iv = 0.12 + np.random.uniform(-0.02, 0.08)
                    
                    for timestamp in timestamps:
                        # Intraday price movement
                        time_of_day = (timestamp - date).total_seconds() / 3600
                        intraday_factor = time_of_day / 6.25  # 6.25 hours trading day
                        
                        spot = underlying_open + (underlying_close - underlying_open) * intraday_factor
                        spot += np.random.normal(0, 10)  # Add noise
                        
                        option_price = self.calculate_option_price(
                            spot, strike, days_to_expiry, option_type, base_iv
                        )
                        
                        # Add some price variation
                        option_price *= (1 + np.random.uniform(-0.02, 0.02))
                        option_price = max(0.05, round(option_price, 2))
                        
                        # Calculate bid-ask spread
                        spread_pct = 0.01 if option_price > 50 else 0.02
                        bid = round(option_price * (1 - spread_pct), 2)
                        ask = round(option_price * (1 + spread_pct), 2)
                        
                        # Generate volume and OI
                        base_volume = 10000 * math.exp(-abs(strike - spot) / 200)
                        volume = int(base_volume * np.random.uniform(0.5, 2.0))
                        oi = int(volume * np.random.uniform(5, 20))
                        
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
                            'high': round(option_price * 1.02, 2),
                            'low': round(option_price * 0.98, 2),
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
    
    def generate_daily_data(self, start_date, end_date):
        """Generate daily options data"""
        daily_data = []
        trading_days = self.get_trading_days(start_date, end_date)
        
        for date in trading_days:
            underlying_close = self.generate_underlying_price(date)
            underlying_open = underlying_close * np.random.uniform(0.995, 1.005)
            
            # Get expiries for the month
            expiries = self.get_expiry_dates(date.year, date.month)
            
            # Also get next month's expiries if we're in the last week
            if date.day > 20:
                next_month = date.month + 1 if date.month < 12 else 1
                next_year = date.year if date.month < 12 else date.year + 1
                expiries.extend(self.get_expiry_dates(next_year, next_month))
            
            # Generate strikes
            strikes = range(int(underlying_close - 2000), int(underlying_close + 2050), 50)
            
            for expiry_date, expiry_type in expiries:
                if expiry_date < date:
                    continue
                    
                days_to_expiry = (expiry_date - date).days
                
                for strike in strikes:
                    for option_type in ['CE', 'PE']:
                        # Skip some far OTM options to reduce data size
                        if abs(strike - underlying_close) > 1000:
                            if np.random.random() > 0.5:
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
                        
                        # Contracts traded and turnover
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
        """Generate all synthetic data for July 2025"""
        print("Generating synthetic Nifty options data starting from July 1, 2025...")
        print("This includes both weekly (Thursday) and monthly (last Thursday) expiries")
        
        # Generate daily data for July 2025
        start_date = pd.Timestamp('2025-07-01')
        end_date = pd.Timestamp('2025-07-31')
        
        print("\nGenerating daily data...")
        daily_df = self.generate_daily_data(start_date, end_date)
        
        # Save daily data
        daily_file = os.path.join(self.daily_path, 'NIFTY_OPTIONS_DAILY_2025_07.csv')
        daily_df.to_csv(daily_file, index=False)
        print(f"Saved daily data to: {daily_file}")
        print(f"Total daily records: {len(daily_df)}")
        
        # Show expiry distribution
        expiry_counts = daily_df.groupby(['expiry', 'expiry_type']).size().reset_index(name='count')
        print("\nExpiry distribution:")
        print(expiry_counts)
        
        # Generate intraday data for each trading day
        print("\nGenerating intraday data...")
        trading_days = self.get_trading_days(start_date, end_date)
        
        for date in trading_days:
            print(f"Generating data for {date.strftime('%Y-%m-%d')}...")
            
            underlying_close = self.generate_underlying_price(date)
            underlying_open = underlying_close * np.random.uniform(0.995, 1.005)
            
            intraday_df = self.generate_intraday_data(date, underlying_open, underlying_close)
            
            # Save intraday data
            filename = f"NIFTY_OPTIONS_5MIN_{date.strftime('%Y%m%d')}.csv"
            filepath = os.path.join(self.intraday_path, filename)
            intraday_df.to_csv(filepath, index=False)
            print(f"  Saved {len(intraday_df)} records to {filename}")
        
        print("\n✅ Data generation complete!")
        print(f"Daily data: {self.daily_path}")
        print(f"Intraday data: {self.intraday_path}")
        
        # Summary statistics
        print("\n📊 Summary Statistics:")
        print(f"Date range: July 1-31, 2025")
        print(f"Trading days: {len(trading_days)}")
        print(f"Weekly expiries: Every Thursday")
        print(f"Monthly expiry: Last Thursday of the month")
        
        # Show sample of generated data
        print("\n📋 Sample of daily data:")
        print(daily_df[['date', 'strike', 'option_type', 'expiry', 'expiry_type', 'close', 'volume']].head(10))

if __name__ == "__main__":
    base_path = "/Users/nitindhawan/NikAlgoBulls/zerodha_strategy/data/synthetic"
    generator = SyntheticDataGenerator(base_path)
    generator.generate_all_data()