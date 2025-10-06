#!/usr/bin/env python3
"""
Fetch India VIX Data from Multiple Sources

India VIX is the volatility index for NIFTY 50, calculated by NSE.
This script attempts to fetch from multiple sources.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import requests
import yfinance as yf
import logging

logger = logging.getLogger(__name__)


class IndiaVIXFetcher:
    """Fetch India VIX data from various sources"""

    def __init__(self, output_dir: str = "data/seed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # Method 1: Yahoo Finance (Symbol: ^INDIAVIX)
    def fetch_from_yahoo(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch India VIX from Yahoo Finance

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with VIX data
        """
        print("\n📊 Method 1: Fetching from Yahoo Finance...")

        try:
            # India VIX symbol on Yahoo Finance
            ticker = yf.Ticker("^INDIAVIX")

            # Fetch historical data
            df = ticker.history(start=start_date, end=end_date)

            if df.empty:
                print("  ❌ No data from Yahoo Finance")
                return pd.DataFrame()

            # Rename columns to match our schema
            df = df.reset_index()
            df = df.rename(columns={
                'Date': 'date',
                'Open': 'vix_open',
                'High': 'vix_high',
                'Low': 'vix_low',
                'Close': 'vix_close',
                'Volume': 'vix_volume'
            })

            df = df[['date', 'vix_open', 'vix_high', 'vix_low', 'vix_close']]

            print(f"  ✅ Fetched {len(df)} days from Yahoo Finance")
            print(f"     Range: {df['date'].min()} to {df['date'].max()}")
            print(f"     VIX Range: {df['vix_close'].min():.2f} - {df['vix_close'].max():.2f}")

            return df

        except Exception as e:
            print(f"  ❌ Error fetching from Yahoo Finance: {e}")
            return pd.DataFrame()

    # Method 2: NSE India Website (Official Source)
    def fetch_from_nse(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch India VIX from NSE website (official source)

        Note: NSE has anti-scraping measures, may require headers/session
        """
        print("\n📊 Method 2: Fetching from NSE India...")

        try:
            # NSE VIX historical data URL
            url = "https://www.nseindia.com/api/historical/vixhistorical"

            # Headers to mimic browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                'Accept': 'application/json',
                'Accept-Language': 'en-US,en;q=0.9',
                'Referer': 'https://www.nseindia.com/'
            }

            # Create session
            session = requests.Session()

            # First, get the main page to set cookies
            session.get("https://www.nseindia.com/", headers=headers, timeout=10)

            # Parse dates
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')

            # NSE API requires specific date format
            params = {
                'from': start_dt.strftime('%d-%m-%Y'),
                'to': end_dt.strftime('%d-%m-%Y')
            }

            # Fetch data
            response = session.get(url, headers=headers, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()

                # Parse response (structure may vary)
                if 'data' in data:
                    df = pd.DataFrame(data['data'])

                    # Convert to our schema
                    df['date'] = pd.to_datetime(df['DATE'], format='%d-%b-%Y')
                    df = df.rename(columns={
                        'OPEN': 'vix_open',
                        'HIGH': 'vix_high',
                        'LOW': 'vix_low',
                        'CLOSE': 'vix_close'
                    })

                    df = df[['date', 'vix_open', 'vix_high', 'vix_low', 'vix_close']]

                    print(f"  ✅ Fetched {len(df)} days from NSE")
                    return df
                else:
                    print(f"  ⚠️ Unexpected NSE response format: {data.keys()}")
                    return pd.DataFrame()
            else:
                print(f"  ❌ NSE API returned status {response.status_code}")
                return pd.DataFrame()

        except Exception as e:
            print(f"  ❌ Error fetching from NSE: {e}")
            return pd.DataFrame()

    # Method 3: Calculate from NIFTY Historical Volatility (Fallback)
    def calculate_from_nifty_volatility(self, nifty_data: pd.DataFrame,
                                       window: int = 20) -> pd.DataFrame:
        """
        Calculate implied VIX from NIFTY historical volatility

        This is a fallback method when actual VIX data is unavailable.

        Args:
            nifty_data: DataFrame with NIFTY OHLC data
            window: Rolling window for volatility calculation (days)

        Returns:
            DataFrame with calculated VIX
        """
        print(f"\n📊 Method 3: Calculating from NIFTY volatility (window={window})...")

        try:
            df = nifty_data.copy()

            # Ensure we have a date column
            if 'date' not in df.columns:
                df = df.reset_index()
                df = df.rename(columns={'index': 'date'})

            # Calculate returns
            df['returns'] = df['close'].pct_change()

            # Calculate rolling volatility (annualized)
            df['vol'] = df['returns'].rolling(window).std() * np.sqrt(252)

            # Convert to VIX scale (percentage)
            df['vix_close'] = df['vol'] * 100

            # Add some realistic variation for open/high/low
            df['vix_open'] = df['vix_close'] * (1 + np.random.uniform(-0.02, 0.02, len(df)))
            df['vix_high'] = df[['vix_open', 'vix_close']].max(axis=1) * (1 + abs(np.random.uniform(0, 0.03, len(df))))
            df['vix_low'] = df[['vix_open', 'vix_close']].min(axis=1) * (1 - abs(np.random.uniform(0, 0.03, len(df))))

            # Clean up
            df = df[['date', 'vix_open', 'vix_high', 'vix_low', 'vix_close']].dropna()

            # Apply realistic bounds (India VIX typically 10-50)
            for col in ['vix_open', 'vix_high', 'vix_low', 'vix_close']:
                df[col] = df[col].clip(10, 50)

            print(f"  ✅ Calculated VIX for {len(df)} days")
            print(f"     VIX Range: {df['vix_close'].min():.2f} - {df['vix_close'].max():.2f}")

            return df

        except Exception as e:
            print(f"  ❌ Error calculating VIX: {e}")
            return pd.DataFrame()

    # Method 4: CSV Upload (Manual)
    def load_from_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load VIX data from manually downloaded CSV

        Expected CSV format:
        date,vix_open,vix_high,vix_low,vix_close
        2024-01-01,15.2,15.8,14.9,15.5
        ...
        """
        print(f"\n📊 Method 4: Loading from CSV: {filepath}...")

        try:
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date'])

            # Validate columns
            required_cols = ['date', 'vix_open', 'vix_high', 'vix_low', 'vix_close']
            if not all(col in df.columns for col in required_cols):
                print(f"  ❌ CSV missing required columns: {required_cols}")
                return pd.DataFrame()

            print(f"  ✅ Loaded {len(df)} days from CSV")
            print(f"     Range: {df['date'].min()} to {df['date'].max()}")

            return df

        except Exception as e:
            print(f"  ❌ Error loading CSV: {e}")
            return pd.DataFrame()

    def fetch_all_methods(self, start_date: str, end_date: str,
                         nifty_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Try all methods in order until one succeeds

        Priority:
        1. Yahoo Finance (easiest, most reliable)
        2. NSE India (official but may have restrictions)
        3. Calculate from NIFTY volatility (fallback)

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            nifty_data: NIFTY data for fallback calculation

        Returns:
            DataFrame with VIX data
        """
        print("="*80)
        print("INDIA VIX DATA FETCHER")
        print("="*80)
        print(f"Period: {start_date} to {end_date}")

        # Try Method 1: Yahoo Finance
        df = self.fetch_from_yahoo(start_date, end_date)
        if not df.empty:
            return df

        # Try Method 2: NSE
        df = self.fetch_from_nse(start_date, end_date)
        if not df.empty:
            return df

        # Fallback Method 3: Calculate from NIFTY
        if nifty_data is not None:
            print("\n⚠️ Could not fetch VIX data, falling back to calculated volatility")
            df = self.calculate_from_nifty_volatility(nifty_data)
            if not df.empty:
                return df

        print("\n❌ All methods failed to fetch VIX data")
        return pd.DataFrame()

    def save_vix_data(self, df: pd.DataFrame, filename: str = "india_vix.csv"):
        """Save VIX data to CSV"""
        if df.empty:
            print("No data to save")
            return

        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False)
        print(f"\n💾 Saved VIX data to: {filepath}")
        print(f"   Rows: {len(df)}")
        print(f"   Size: {filepath.stat().st_size / 1024:.1f} KB")

        return filepath


def main():
    """Example usage"""
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))

    from src.synthetic_data_generator.io.seed_data_loader import NiftySeedDataLoader

    # Load NIFTY seed data
    print("Loading NIFTY seed data...")
    nifty_loader = NiftySeedDataLoader()
    nifty_data = nifty_loader.load()
    nifty_daily = nifty_loader.get_daily_data()

    # Get date range from seed data
    start_date = nifty_daily['date'].min().strftime('%Y-%m-%d')
    end_date = nifty_daily['date'].max().strftime('%Y-%m-%d')

    print(f"\nNIFTY data range: {start_date} to {end_date}")

    # Initialize fetcher
    fetcher = IndiaVIXFetcher()

    # Try to fetch VIX data
    vix_df = fetcher.fetch_all_methods(
        start_date=start_date,
        end_date=end_date,
        nifty_data=nifty_daily
    )

    if not vix_df.empty:
        # Save to CSV
        output_path = fetcher.save_vix_data(vix_df)

        # Show summary
        print("\n" + "="*80)
        print("VIX DATA SUMMARY")
        print("="*80)
        print(f"\nPeriod: {vix_df['date'].min()} to {vix_df['date'].max()}")
        print(f"Days: {len(vix_df)}")
        print(f"\nVIX Statistics:")
        print(f"  Mean: {vix_df['vix_close'].mean():.2f}")
        print(f"  Std: {vix_df['vix_close'].std():.2f}")
        print(f"  Min: {vix_df['vix_close'].min():.2f}")
        print(f"  Max: {vix_df['vix_close'].max():.2f}")

        # Show sample
        print(f"\nFirst 5 rows:")
        print(vix_df.head().to_string(index=False))

        print("\n" + "="*80)
        print(f"✅ SUCCESS! VIX data saved to: {output_path}")
        print("="*80)
    else:
        print("\n❌ Failed to fetch VIX data from any source")
        sys.exit(1)


if __name__ == '__main__':
    main()
