#!/usr/bin/env python3
"""
Synthetic Data Adapter for V9 Balanced Data
Provides OptionsDataFetcher-compatible interface for CSV-based synthetic data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class SyntheticDataAdapter:
    """
    Adapter to load V9 synthetic data and provide OptionsDataFetcher-compatible interface
    """

    def __init__(self, data_path: str):
        """
        Initialize adapter with path to synthetic data

        Args:
            data_path: Path to synthetic data directory
        """
        self.data_path = Path(data_path)

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path not found: {data_path}")

        # Cache for loaded data
        self.data_cache = {}

        # Load metadata
        self.metadata = self.load_metadata()

        logger.info(f"Synthetic data adapter initialized: {data_path}")

    def load_metadata(self):
        """Load metadata from generation_info.json"""
        metadata_file = self.data_path / 'metadata' / 'generation_info.json'

        if metadata_file.exists():
            import json
            with open(metadata_file) as f:
                return json.load(f)

        return {}

    def get_data_for_date(self, date: datetime.date) -> pd.DataFrame:
        """
        Load data for a specific date

        Args:
            date: Date to load data for

        Returns:
            DataFrame with options data for that date
        """
        date_str = date.strftime('%Y%m%d')

        # Check cache
        if date_str in self.data_cache:
            return self.data_cache[date_str]

        # Load from CSV
        filename = f"NIFTY_OPTIONS_1H_{date_str}.csv"
        filepath = self.data_path / filename

        if not filepath.exists():
            logger.warning(f"No data file for {date}: {filename}")
            return pd.DataFrame()

        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['expiry'] = pd.to_datetime(df['expiry']).dt.date

        # Cache it
        self.data_cache[date_str] = df

        return df

    def get_monthly_expiries(self, symbol="NIFTY", months=3, from_date: Optional[datetime.date] = None) -> List[datetime.date]:
        """
        Get monthly expiry dates from synthetic data

        Args:
            symbol: Symbol (always NIFTY for synthetic data)
            months: Number of months to return
            from_date: Reference date (defaults to first available date)

        Returns:
            List of monthly expiry dates
        """
        if from_date is None:
            # Get first available date from metadata or files
            if 'stats' in self.metadata and 'dates_processed' in self.metadata['stats']:
                dates = self.metadata['stats']['dates_processed']
                if dates:
                    from_date = datetime.strptime(dates[0], '%Y-%m-%d').date()

            if from_date is None:
                # Fallback: find first file
                csv_files = sorted(self.data_path.glob('NIFTY_OPTIONS_1H_*.csv'))
                if csv_files:
                    date_str = csv_files[0].stem.split('_')[-1]
                    from_date = datetime.strptime(date_str, '%Y%m%d').date()
                else:
                    logger.error("No data files found")
                    return []

        # Load data for the reference date
        df = self.get_data_for_date(from_date)

        if df.empty:
            logger.error(f"No data for {from_date}")
            return []

        # Extract monthly expiries from data
        monthly_expiries = df[df['expiry_type'] == 'monthly']['expiry'].unique()
        monthly_expiries = sorted(monthly_expiries)

        # Filter for expiries >= from_date
        monthly_expiries = [exp for exp in monthly_expiries if exp >= from_date]

        # Return requested number
        return monthly_expiries[:months]

    def get_weekly_expiries(self, symbol="NIFTY", weeks=4, from_date: Optional[datetime.date] = None) -> List[datetime.date]:
        """
        Get weekly expiry dates from synthetic data

        Args:
            symbol: Symbol (always NIFTY for synthetic data)
            weeks: Number of weeks to return
            from_date: Reference date (defaults to first available date)

        Returns:
            List of weekly expiry dates
        """
        if from_date is None:
            # Get first available date
            if 'stats' in self.metadata and 'dates_processed' in self.metadata['stats']:
                dates = self.metadata['stats']['dates_processed']
                if dates:
                    from_date = datetime.strptime(dates[0], '%Y-%m-%d').date()

            if from_date is None:
                csv_files = sorted(self.data_path.glob('NIFTY_OPTIONS_1H_*.csv'))
                if csv_files:
                    date_str = csv_files[0].stem.split('_')[-1]
                    from_date = datetime.strptime(date_str, '%Y%m%d').date()
                else:
                    logger.error("No data files found")
                    return []

        # Load data for the reference date
        df = self.get_data_for_date(from_date)

        if df.empty:
            logger.error(f"No data for {from_date}")
            return []

        # Extract weekly expiries from data
        weekly_expiries = df[df['expiry_type'] == 'weekly']['expiry'].unique()
        weekly_expiries = sorted(weekly_expiries)

        # Filter for expiries >= from_date
        weekly_expiries = [exp for exp in weekly_expiries if exp >= from_date]

        # Return requested number
        return weekly_expiries[:weeks]

    def get_option_chain(self, symbol="NIFTY", date: Optional[datetime.date] = None) -> pd.DataFrame:
        """
        Get complete option chain for a date

        Args:
            symbol: Symbol (always NIFTY for synthetic data)
            date: Date to get chain for

        Returns:
            DataFrame with option chain
        """
        if date is None:
            date = datetime.now().date()

        return self.get_data_for_date(date)

    def get_spot_price(self, symbol="NIFTY", date: Optional[datetime.date] = None,
                       timestamp: Optional[datetime] = None) -> Optional[float]:
        """
        Get spot price for a symbol at a specific date/time

        Args:
            symbol: Symbol (always NIFTY)
            date: Date
            timestamp: Specific timestamp (if None, uses latest for that date)

        Returns:
            Spot price
        """
        if date is None:
            date = datetime.now().date()

        df = self.get_data_for_date(date)

        if df.empty:
            return None

        if timestamp is not None:
            # Get price at specific timestamp
            df_time = df[df['timestamp'] == timestamp]
            if not df_time.empty:
                return df_time.iloc[0]['underlying_price']

        # Return latest price for the date
        return df.iloc[-1]['underlying_price']

    def get_option_data(self, symbol="NIFTY", strike: int = None, option_type: str = None,
                       expiry: datetime.date = None, date: datetime.date = None) -> pd.DataFrame:
        """
        Get specific option data

        Args:
            symbol: Symbol
            strike: Strike price
            option_type: 'CE' or 'PE'
            expiry: Expiry date
            date: Trading date

        Returns:
            DataFrame with option data
        """
        if date is None:
            date = datetime.now().date()

        df = self.get_data_for_date(date)

        if df.empty:
            return pd.DataFrame()

        # Filter
        if strike is not None:
            df = df[df['strike'] == strike]

        if option_type is not None:
            df = df[df['option_type'] == option_type]

        if expiry is not None:
            df = df[df['expiry'] == expiry]

        return df

    def get_atm_strike(self, spot_price: float, symbol="NIFTY") -> int:
        """
        Get ATM strike for a spot price

        Args:
            spot_price: Current spot price
            symbol: Symbol

        Returns:
            ATM strike (rounded to nearest 50)
        """
        return round(spot_price / 50) * 50

    def get_available_dates(self) -> List[datetime.date]:
        """
        Get list of all available trading dates

        Returns:
            List of dates with data
        """
        csv_files = sorted(self.data_path.glob('NIFTY_OPTIONS_1H_*.csv'))

        dates = []
        for file in csv_files:
            date_str = file.stem.split('_')[-1]
            date = datetime.strptime(date_str, '%Y%m%d').date()
            dates.append(date)

        return dates


# Example usage and testing
if __name__ == '__main__':
    import sys

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Initialize adapter
    data_path = '/Users/nitindhawan/NikAlgoBulls/zerodha_strategy/data/synthetic/hourly_v9_balanced'

    try:
        adapter = SyntheticDataAdapter(data_path)

        print("="*80)
        print("SYNTHETIC DATA ADAPTER TEST")
        print("="*80)

        # Test 1: Get available dates
        dates = adapter.get_available_dates()
        print(f"\n1. Available Dates: {len(dates)} days")
        print(f"   First: {dates[0]}")
        print(f"   Last: {dates[-1]}")

        # Test 2: Get monthly expiries
        test_date = dates[10]  # Use a date in the middle
        monthly_exp = adapter.get_monthly_expiries("NIFTY", months=2, from_date=test_date)
        print(f"\n2. Monthly Expiries (from {test_date}):")
        for i, exp in enumerate(monthly_exp, 1):
            print(f"   {i}. {exp}")

        # Test 3: Get weekly expiries
        weekly_exp = adapter.get_weekly_expiries("NIFTY", weeks=4, from_date=test_date)
        print(f"\n3. Weekly Expiries (from {test_date}):")
        for i, exp in enumerate(weekly_exp, 1):
            print(f"   {i}. {exp}")

        # Test 4: Get spot price
        spot = adapter.get_spot_price("NIFTY", date=test_date)
        print(f"\n4. Spot Price on {test_date}: ₹{spot:.2f}")

        # Test 5: Get ATM strike
        atm = adapter.get_atm_strike(spot)
        print(f"\n5. ATM Strike: {atm}")

        # Test 6: Get option data
        if monthly_exp:
            option_data = adapter.get_option_data(
                symbol="NIFTY",
                strike=atm,
                option_type="CE",
                expiry=monthly_exp[0],
                date=test_date
            )
            print(f"\n6. Option Data (ATM CE, Monthly):")
            print(f"   Rows: {len(option_data)}")
            if not option_data.empty:
                print(f"   Sample: Strike={option_data.iloc[0]['strike']}, "
                      f"Premium=₹{option_data.iloc[0]['close']:.2f}, "
                      f"Delta={option_data.iloc[0]['delta']:.4f}")

        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED")
        print("="*80)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
