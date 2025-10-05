#!/usr/bin/env python3
"""
Synthetic Options Data Generator for Zerodha Strategy Backtesting
Generates realistic options data in CSV format for backtesting
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from scipy.stats import norm
import os
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

class SyntheticOptionsDataGenerator:
    """Generate synthetic options data with realistic market characteristics"""
    
    def __init__(self):
        # Market parameters
        self.base_iv = 0.12  # 12% base implied volatility for Nifty
        self.risk_free_rate = 0.065  # 6.5% risk-free rate
        self.dividend_yield = 0.015  # 1.5% dividend yield
        
        # Trading hours
        self.market_open = pd.Timestamp("09:15:00").time()
        self.market_close = pd.Timestamp("15:30:00").time()
        
        # Strike parameters
        self.strike_interval = 50
        self.weekly_strike_range = 1000  # ATM ± 1000 for weekly
        self.monthly_strike_range = 2000  # ATM ± 2000 for monthly
        
    def black_scholes(self, S: float, K: float, T: float, r: float, 
                     sigma: float, option_type: str = 'CE') -> Dict:
        """
        Calculate option price and Greeks using Black-Scholes model
        
        Returns dict with: price, delta, gamma, theta, vega
        """
        if T <= 0:
            T = 0.001  # Minimum time to avoid division by zero
            
        # Calculate d1 and d2
        d1 = (np.log(S/K) + (r - self.dividend_yield + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        # Calculate option price
        if option_type == 'CE':
            price = S*np.exp(-self.dividend_yield*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
            delta = np.exp(-self.dividend_yield*T) * norm.cdf(d1)
        else:  # PE
            price = K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-self.dividend_yield*T)*norm.cdf(-d1)
            delta = -np.exp(-self.dividend_yield*T) * norm.cdf(-d1)
        
        # Calculate other Greeks
        gamma = np.exp(-self.dividend_yield*T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta_daily = self.calculate_theta(S, K, T, r, sigma, option_type) / 365
        vega = S * np.exp(-self.dividend_yield*T) * norm.pdf(d1) * np.sqrt(T) / 100
        
        return {
            'price': max(price, 0.05),  # Minimum price 5 paise
            'delta': delta,
            'gamma': gamma,
            'theta': theta_daily,
            'vega': vega
        }
    
    def calculate_theta(self, S: float, K: float, T: float, r: float, 
                       sigma: float, option_type: str) -> float:
        """Calculate theta (time decay) in rupees per day"""
        d1 = (np.log(S/K) + (r - self.dividend_yield + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        term1 = -S * np.exp(-self.dividend_yield*T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
        
        if option_type == 'CE':
            term2 = r * K * np.exp(-r*T) * norm.cdf(d2)
            term3 = self.dividend_yield * S * np.exp(-self.dividend_yield*T) * norm.cdf(d1)
            theta = term1 - term2 + term3
        else:  # PE
            term2 = r * K * np.exp(-r*T) * norm.cdf(-d2)
            term3 = self.dividend_yield * S * np.exp(-self.dividend_yield*T) * norm.cdf(-d1)
            theta = term1 + term2 - term3
            
        return theta
    
    def generate_iv_smile(self, moneyness: float, time_to_expiry: int) -> float:
        """Generate realistic implied volatility smile"""
        atm_iv = self.base_iv
        
        # Volatility smile based on moneyness
        if abs(moneyness) < 0.02:  # Near ATM
            iv = atm_iv
        elif moneyness > 0:  # ITM
            iv = atm_iv * (1 - 0.1 * min(moneyness, 0.2))  # Lower IV for ITM
        else:  # OTM
            iv = atm_iv * (1 + 0.2 * min(abs(moneyness), 0.3))  # Higher IV for OTM
        
        # Term structure adjustment
        if time_to_expiry <= 3:
            iv *= 1.3  # Higher IV very near expiry
        elif time_to_expiry <= 7:
            iv *= 1.15  # Elevated IV near expiry
        elif time_to_expiry > 30:
            iv *= 0.95  # Lower IV for longer dated
        
        # Add random market noise
        iv += np.random.normal(0, 0.003)
        
        # Ensure reasonable bounds
        return np.clip(iv, 0.08, 0.50)
    
    def generate_volume_oi(self, moneyness: float, is_monthly: bool, 
                          time_to_expiry: int) -> Tuple[int, int]:
        """Generate realistic volume and open interest"""
        # Base values
        if is_monthly:
            base_volume = 50000
            base_oi = 500000
        else:
            base_volume = 20000
            base_oi = 200000
        
        # Adjust for moneyness (peak at ATM)
        moneyness_factor = np.exp(-10 * moneyness**2)  # Gaussian decay
        
        # Adjust for time to expiry
        if time_to_expiry <= 1:
            expiry_factor = 3.0  # High volume on expiry
        elif time_to_expiry <= 3:
            expiry_factor = 2.0
        elif time_to_expiry <= 7:
            expiry_factor = 1.5
        else:
            expiry_factor = 1.0
        
        volume = int(base_volume * moneyness_factor * expiry_factor * np.random.uniform(0.5, 1.5))
        oi = int(base_oi * moneyness_factor * np.random.uniform(0.8, 1.2))
        
        return max(volume, 100), max(oi, 1000)
    
    def generate_bid_ask_spread(self, price: float, moneyness: float, 
                               volume: int) -> Tuple[float, float]:
        """Calculate realistic bid-ask spread"""
        # Base spread as percentage of price
        if price < 5:
            base_spread_pct = 0.10  # 10% for very low premium
        elif price < 10:
            base_spread_pct = 0.05  # 5% for low premium
        elif price < 50:
            base_spread_pct = 0.02  # 2% for medium
        elif price < 200:
            base_spread_pct = 0.01  # 1% for high
        else:
            base_spread_pct = 0.005  # 0.5% for very high
        
        # Adjust for moneyness (wider spreads for far OTM/ITM)
        if abs(moneyness) > 0.15:
            base_spread_pct *= 2
        elif abs(moneyness) > 0.10:
            base_spread_pct *= 1.5
        
        # Adjust for liquidity (tighter spreads for high volume)
        if volume > 20000:
            base_spread_pct *= 0.7
        elif volume > 10000:
            base_spread_pct *= 0.85
        
        spread = max(price * base_spread_pct, 0.05)  # Minimum 5 paise spread
        
        bid = round(price - spread/2, 2)
        ask = round(price + spread/2, 2)
        
        return max(bid, 0.05), ask
    
    def generate_intraday_pattern(self, base_price: float, time_str: str,
                                 volatility: float) -> Tuple[float, float, float, float]:
        """Generate realistic OHLC for 5-minute candle"""
        hour = int(time_str.split(':')[0])
        minute = int(time_str.split(':')[1])
        
        # Market phase volatility multipliers
        if hour == 9 and minute < 30:  # Opening
            vol_mult = 2.0
        elif hour == 9 and minute < 45:  # Early morning
            vol_mult = 1.5
        elif hour >= 14 and minute >= 30:  # Closing
            vol_mult = 1.3
        elif hour >= 12 and hour < 14:  # Lunch time
            vol_mult = 0.7
        else:  # Normal trading
            vol_mult = 1.0
        
        # Generate price changes
        returns = np.random.normal(0, volatility * vol_mult / np.sqrt(75))  # 75 candles per day
        
        open_price = base_price * (1 + np.random.normal(0, 0.001))
        close_price = open_price * (1 + returns)
        
        # High and low with realistic wicks
        high_wick = abs(np.random.normal(0, volatility * vol_mult / 200))
        low_wick = abs(np.random.normal(0, volatility * vol_mult / 200))
        
        high_price = max(open_price, close_price) * (1 + high_wick)
        low_price = min(open_price, close_price) * (1 - low_wick)
        
        # Round to 2 decimal places (5 paise tick)
        return (round(open_price, 2), 
                round(high_price, 2), 
                round(low_price, 2), 
                round(close_price, 2))
    
    def generate_5min_data(self, spot_data: pd.DataFrame, date: datetime) -> pd.DataFrame:
        """Generate 5-minute options data for a single day"""
        rows = []
        
        # Convert date to proper format for indexing
        if isinstance(date, pd.Timestamp):
            date_key = date
        else:
            date_key = pd.Timestamp(date)
            
        # Get spot price for the day
        try:
            if date_key in spot_data.index:
                day_spot = spot_data.loc[date_key]
            else:
                # Use nearest available date
                nearest_idx = spot_data.index.get_indexer([date_key], method='nearest')[0]
                day_spot = spot_data.iloc[nearest_idx]
        except Exception as e:
            print(f"Error getting spot data for {date}: {e}")
            return pd.DataFrame()
        
        spot_open = float(day_spot['Open'])
        spot_close = float(day_spot['Close'])
        spot_high = float(day_spot['High'])
        spot_low = float(day_spot['Low'])
        
        # Determine ATM strike
        atm_strike = round(spot_open / self.strike_interval) * self.strike_interval
        
        # Get next Thursday (weekly expiry)
        current_date = pd.Timestamp(date) if not isinstance(date, pd.Timestamp) else date
        days_ahead = 3 - current_date.weekday()  # Thursday is 3
        if days_ahead <= 0:
            days_ahead += 7
        weekly_expiry = current_date + timedelta(days=days_ahead)
        
        # Get last Thursday of month (monthly expiry)
        next_month = current_date.replace(day=28) + timedelta(days=4)
        monthly_expiry = next_month - timedelta(days=next_month.weekday() - 3)
        if monthly_expiry.day <= current_date.day:
            # Move to next month
            next_month = monthly_expiry + timedelta(days=7)
            monthly_expiry = next_month.replace(day=28) + timedelta(days=4)
            monthly_expiry = monthly_expiry - timedelta(days=monthly_expiry.weekday() - 3)
        
        # Generate strikes
        weekly_strikes = list(range(
            atm_strike - self.weekly_strike_range,
            atm_strike + self.weekly_strike_range + self.strike_interval,
            self.strike_interval
        ))
        
        # Sample strikes to reduce data size (every 3rd strike for far OTM)
        selected_strikes = []
        for strike in weekly_strikes:
            distance = abs(strike - atm_strike)
            if distance <= 300:  # Keep all near ATM
                selected_strikes.append(strike)
            elif distance <= 600 and strike % 100 == 0:  # Every other for medium
                selected_strikes.append(strike)
            elif strike % 150 == 0:  # Every third for far OTM
                selected_strikes.append(strike)
        
        # Generate time series for the day
        base_date = current_date.date() if hasattr(current_date, 'date') else current_date
        current_time = datetime.combine(base_date, self.market_open)
        end_time = datetime.combine(base_date, self.market_close)
        
        while current_time <= end_time:
            time_str = current_time.strftime("%H:%M")
            
            # Interpolate spot price for this time
            time_progress = (current_time.hour * 60 + current_time.minute - 9*60 - 15) / (6*60 + 15)
            current_spot = spot_open + (spot_close - spot_open) * time_progress
            
            # Add some intraday volatility
            current_spot *= (1 + np.random.normal(0, 0.001))
            
            for strike in selected_strikes:
                for option_type in ['CE', 'PE']:
                    # Skip very far ITM options (low liquidity)
                    if option_type == 'CE' and strike < current_spot - 500:
                        continue
                    if option_type == 'PE' and strike > current_spot + 500:
                        continue
                    
                    # Calculate moneyness
                    if option_type == 'CE':
                        moneyness = (current_spot - strike) / current_spot
                    else:
                        moneyness = (strike - current_spot) / current_spot
                    
                    # Time to expiry in years
                    days_to_expiry = (weekly_expiry - current_date).days
                    T = max(days_to_expiry / 365, 0.001)
                    
                    # Generate IV with smile
                    iv = self.generate_iv_smile(moneyness, days_to_expiry)
                    
                    # Calculate option price and Greeks
                    option_data = self.black_scholes(current_spot, strike, T, 
                                                    self.risk_free_rate, iv, option_type)
                    
                    # Generate volume and OI
                    volume, oi = self.generate_volume_oi(moneyness, False, days_to_expiry)
                    
                    # Adjust volume for time of day (U-shaped pattern)
                    if time_str < "10:00" or time_str > "15:00":
                        volume = int(volume * 1.5)
                    elif "11:00" < time_str < "14:00":
                        volume = int(volume * 0.7)
                    
                    # Generate OHLC
                    base_price = option_data['price']
                    ohlc = self.generate_intraday_pattern(base_price, time_str, iv)
                    
                    # Generate bid-ask
                    bid, ask = self.generate_bid_ask_spread(ohlc[3], moneyness, volume)  # Use close
                    
                    # Create row
                    row = {
                        'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S"),
                        'symbol': 'NIFTY',
                        'strike': strike,
                        'option_type': option_type,
                        'expiry': weekly_expiry.strftime("%Y-%m-%d"),
                        'open': ohlc[0],
                        'high': ohlc[1],
                        'low': ohlc[2],
                        'close': ohlc[3],
                        'volume': volume,
                        'oi': oi,
                        'bid': bid,
                        'ask': ask,
                        'iv': round(iv, 4),
                        'delta': round(option_data['delta'], 4),
                        'gamma': round(option_data['gamma'], 6),
                        'theta': round(option_data['theta'], 2),
                        'vega': round(option_data['vega'], 2),
                        'underlying_price': round(current_spot, 2)
                    }
                    rows.append(row)
            
            # Move to next 5-minute interval
            current_time += timedelta(minutes=5)
        
        return pd.DataFrame(rows)
    
    def generate_daily_data(self, spot_data: pd.DataFrame, year: int) -> pd.DataFrame:
        """Generate daily options data for a full year"""
        rows = []
        
        # Filter spot data for the year
        year_data = spot_data[spot_data.index.year == year]
        
        for date in year_data.index:
            spot_close = float(year_data.loc[date]['Close'])
            spot_open = float(year_data.loc[date]['Open'])
            spot_high = float(year_data.loc[date]['High'])
            spot_low = float(year_data.loc[date]['Low'])
            
            # Determine ATM strike
            atm_strike = round(spot_close / self.strike_interval) * self.strike_interval
            
            # Calculate expiries
            current_date = pd.Timestamp(date)
            
            # Weekly expiry (next Thursday)
            days_ahead = 3 - current_date.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            weekly_expiry = current_date + timedelta(days=days_ahead)
            
            # Monthly expiry (last Thursday)
            next_month = current_date.replace(day=28) + timedelta(days=4)
            monthly_expiry = next_month - timedelta(days=next_month.weekday() - 3)
            
            # Generate strikes for both weekly and monthly
            expiries = [
                (weekly_expiry, False, self.weekly_strike_range),
                (monthly_expiry, True, self.monthly_strike_range)
            ]
            
            for expiry_date, is_monthly, strike_range in expiries:
                days_to_expiry = (expiry_date - current_date).days
                
                if days_to_expiry <= 0:
                    continue
                
                # Generate strikes
                strikes = []
                for strike in range(atm_strike - strike_range, 
                                  atm_strike + strike_range + self.strike_interval,
                                  self.strike_interval):
                    # Sample strikes for daily data
                    distance = abs(strike - atm_strike)
                    if distance <= 500:  # Keep all near ATM
                        strikes.append(strike)
                    elif distance <= 1000 and strike % 100 == 0:
                        strikes.append(strike)
                    elif is_monthly and strike % 200 == 0:  # Sparse for far monthly
                        strikes.append(strike)
                
                for strike in strikes:
                    for option_type in ['CE', 'PE']:
                        # Calculate moneyness
                        if option_type == 'CE':
                            moneyness = (spot_close - strike) / spot_close
                        else:
                            moneyness = (strike - spot_close) / spot_close
                        
                        # Skip very far ITM
                        if abs(moneyness) > 0.15 and not is_monthly:
                            continue
                        if abs(moneyness) > 0.25 and is_monthly:
                            continue
                        
                        # Time to expiry
                        T = days_to_expiry / 365
                        
                        # Generate IV
                        iv = self.generate_iv_smile(moneyness, days_to_expiry)
                        
                        # Calculate prices at different spot levels for OHLC
                        prices = []
                        for spot in [spot_open, spot_high, spot_low, spot_close]:
                            opt = self.black_scholes(spot, strike, T, 
                                                   self.risk_free_rate, iv, option_type)
                            prices.append(opt['price'])
                        
                        # Final Greeks at close
                        option_data = self.black_scholes(spot_close, strike, T,
                                                        self.risk_free_rate, iv, option_type)
                        
                        # Volume and OI
                        volume, oi = self.generate_volume_oi(moneyness, is_monthly, days_to_expiry)
                        daily_volume = volume * 75  # Approximate daily from 5-min
                        
                        # Bid-ask
                        bid, ask = self.generate_bid_ask_spread(prices[3], moneyness, daily_volume)
                        
                        # VWAP
                        vwap = np.average(prices, weights=[0.2, 0.1, 0.1, 0.6])
                        
                        # Contracts and turnover
                        lot_size = 25  # Nifty lot size
                        contracts_traded = daily_volume // lot_size
                        turnover_cr = (vwap * daily_volume * lot_size) / 10000000
                        
                        row = {
                            'date': date.strftime("%Y-%m-%d"),
                            'symbol': 'NIFTY',
                            'strike': strike,
                            'option_type': option_type,
                            'expiry': expiry_date.strftime("%Y-%m-%d"),
                            'open': round(prices[0], 2),
                            'high': round(max(prices), 2),
                            'low': round(min(prices), 2),
                            'close': round(prices[3], 2),
                            'volume': daily_volume,
                            'oi': oi,
                            'vwap': round(vwap, 2),
                            'bid': bid,
                            'ask': ask,
                            'iv': round(iv, 4),
                            'delta': round(option_data['delta'], 4),
                            'gamma': round(option_data['gamma'], 6),
                            'theta': round(option_data['theta'], 2),
                            'vega': round(option_data['vega'], 2),
                            'underlying_close': round(spot_close, 2),
                            'contracts_traded': contracts_traded,
                            'turnover': f"{turnover_cr:.2f}Cr"
                        }
                        rows.append(row)
        
        return pd.DataFrame(rows)


def main():
    """Main function to generate all synthetic data"""
    print("Starting Synthetic Options Data Generation...")
    print("=" * 60)
    
    # Initialize generator
    generator = SyntheticOptionsDataGenerator()
    
    # Create output directories
    base_path = "/Users/nitindhawan/NikAlgoBulls/zerodha_strategy/data/synthetic"
    os.makedirs(f"{base_path}/intraday", exist_ok=True)
    os.makedirs(f"{base_path}/daily", exist_ok=True)
    
    # Fetch Nifty spot data
    print("\n1. Fetching Nifty spot data from yfinance...")
    try:
        # For 5-min data (last 60 days)
        end_date = datetime.now()
        start_date_5min = end_date - timedelta(days=60)
        
        spot_5min = yf.download("^NSEI", start=start_date_5min, end=end_date, 
                               interval="1d", progress=False)
        
        # For daily data (2023-2025)
        spot_daily = yf.download("^NSEI", start="2023-01-01", end=end_date,
                                interval="1d", progress=False)
        
        print(f"   ✓ Fetched {len(spot_5min)} days of recent data")
        print(f"   ✓ Fetched {len(spot_daily)} days of historical data")
        
    except Exception as e:
        print(f"   ✗ Error fetching data: {e}")
        return
    
    # Generate 5-minute data for last 60 days
    print("\n2. Generating 5-minute options data...")
    print("   This will take approximately 10-15 minutes...")
    
    generated_files = []
    dates_to_generate = spot_5min.index[-60:]  # Last 60 trading days
    
    for i, date in enumerate(dates_to_generate, 1):
        if i % 10 == 0:
            print(f"   Processing day {i}/{len(dates_to_generate)}...")
        
        try:
            # Generate data for the day
            df_5min = generator.generate_5min_data(spot_5min, pd.Timestamp(date))
            
            # Save to CSV
            filename = f"NIFTY_OPTIONS_5MIN_{date.strftime('%Y%m%d')}.csv"
            filepath = f"{base_path}/intraday/{filename}"
            df_5min.to_csv(filepath, index=False)
            generated_files.append(filepath)
            
        except Exception as e:
            print(f"   ⚠ Error generating data for {date}: {e}")
    
    print(f"   ✓ Generated {len(generated_files)} intraday data files")
    
    # Generate daily data for 2023-2025
    print("\n3. Generating daily options data...")
    
    for year in [2023, 2024, 2025]:
        print(f"   Generating year {year}...")
        
        try:
            df_daily = generator.generate_daily_data(spot_daily, year)
            
            # Save to CSV
            filename = f"NIFTY_OPTIONS_DAILY_{year}.csv"
            filepath = f"{base_path}/daily/{filename}"
            df_daily.to_csv(filepath, index=False)
            generated_files.append(filepath)
            
            print(f"   ✓ Generated {len(df_daily)} rows for {year}")
            
        except Exception as e:
            print(f"   ✗ Error generating daily data for {year}: {e}")
    
    # Create summary file
    print("\n4. Creating summary...")
    
    summary = {
        'generation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'total_files': len(generated_files),
        'intraday_files': len([f for f in generated_files if 'intraday' in f]),
        'daily_files': len([f for f in generated_files if 'daily' in f]),
        'data_path': base_path,
        'files': generated_files
    }
    
    summary_df = pd.DataFrame([summary])
    summary_path = f"{base_path}/generation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE!")
    print("=" * 60)
    print(f"\nGenerated {len(generated_files)} files")
    print(f"Data location: {base_path}")
    print(f"Summary saved: {summary_path}")
    
    # Display sample of generated files
    print("\nSample files generated:")
    for file in generated_files[:5]:
        print(f"  - {os.path.basename(file)}")
    if len(generated_files) > 5:
        print(f"  ... and {len(generated_files) - 5} more files")
    
    return generated_files


if __name__ == "__main__":
    files = main()