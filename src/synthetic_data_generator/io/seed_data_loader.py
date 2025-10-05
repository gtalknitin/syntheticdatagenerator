#!/usr/bin/env python3
"""
Seed Data Loader for NIFTY 1-minute historical data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class NiftySeedDataLoader:
    """Load and process NIFTY 1-minute historical seed data"""

    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize seed data loader

        Args:
            data_path: Path to seed data CSV file. If None, uses default location.
        """
        if data_path is None:
            # Default location
            project_root = Path(__file__).parent.parent.parent.parent
            data_path = project_root / "data" / "seed" / "nifty_data_min.csv"

        self.data_path = Path(data_path)

        if not self.data_path.exists():
            raise FileNotFoundError(f"Seed data not found: {self.data_path}")

        self._data = None
        logger.info(f"Seed data loader initialized: {self.data_path}")

    def load(self) -> pd.DataFrame:
        """
        Load seed data from CSV

        Returns:
            DataFrame with NIFTY 1-minute OHLCV data
        """
        if self._data is not None:
            return self._data

        logger.info("Loading seed data...")

        df = pd.read_csv(self.data_path)

        # Parse datetime with timezone
        df['date'] = pd.to_datetime(df['date'])

        # Extract date components
        df['trading_date'] = df['date'].dt.date
        df['time'] = df['date'].dt.time

        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)

        self._data = df
        logger.info(f"Loaded {len(df):,} rows from {df['date'].min()} to {df['date'].max()}")

        return df

    def get_summary(self) -> dict:
        """
        Get summary statistics of the seed data

        Returns:
            Dictionary with summary statistics
        """
        if self._data is None:
            self.load()

        df = self._data

        summary = {
            'total_rows': len(df),
            'total_days': df['trading_date'].nunique(),
            'date_range': {
                'start': str(df['date'].min()),
                'end': str(df['date'].max()),
                'days': (df['date'].max() - df['date'].min()).days
            },
            'price_range': {
                'min': float(df['low'].min()),
                'max': float(df['high'].max()),
                'mean': float(df['close'].mean()),
                'std': float(df['close'].std())
            },
            'data_quality': {
                'missing_values': df.isnull().sum().to_dict(),
                'zero_volume_pct': (df['volume'] == 0).sum() / len(df) * 100
            },
            'trading_hours': {
                'first_timestamp': str(df.groupby('trading_date')['date'].min().iloc[0].time()),
                'last_timestamp': str(df.groupby('trading_date')['date'].max().iloc[0].time()),
                'avg_candles_per_day': len(df) / df['trading_date'].nunique()
            }
        }

        return summary

    def get_daily_data(self) -> pd.DataFrame:
        """
        Aggregate 1-minute data to daily OHLCV

        Returns:
            DataFrame with daily OHLCV data
        """
        if self._data is None:
            self.load()

        df = self._data

        daily = df.groupby('trading_date').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).reset_index()

        daily.rename(columns={'trading_date': 'date'}, inplace=True)

        return daily

    def get_hourly_data(self) -> pd.DataFrame:
        """
        Aggregate 1-minute data to hourly OHLCV

        Returns:
            DataFrame with hourly OHLCV data
        """
        if self._data is None:
            self.load()

        df = self._data.copy()

        # Create hour bins
        df['hour'] = df['date'].dt.floor('H')

        hourly = df.groupby('hour').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).reset_index()

        hourly.rename(columns={'hour': 'date'}, inplace=True)

        return hourly

    def get_date_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get data for a specific date range

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame filtered to date range
        """
        if self._data is None:
            self.load()

        df = self._data

        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        return df[(df['date'] >= start) & (df['date'] <= end)]

    def validate_data(self) -> Tuple[bool, list]:
        """
        Validate seed data quality

        Returns:
            Tuple of (is_valid, list of issues)
        """
        if self._data is None:
            self.load()

        df = self._data
        issues = []

        # Check for missing values
        if df.isnull().any().any():
            issues.append(f"Missing values found: {df.isnull().sum().to_dict()}")

        # Check OHLC consistency
        invalid_ohlc = df[(df['high'] < df['low']) |
                         (df['high'] < df['open']) |
                         (df['high'] < df['close']) |
                         (df['low'] > df['open']) |
                         (df['low'] > df['close'])]

        if len(invalid_ohlc) > 0:
            issues.append(f"Invalid OHLC relationships in {len(invalid_ohlc)} rows")

        # Check for date gaps (weekdays only)
        dates = sorted(df['trading_date'].unique())
        date_gaps = []
        for i in range(1, len(dates)):
            prev_date = pd.Timestamp(dates[i-1])
            curr_date = pd.Timestamp(dates[i])

            # Count business days between
            business_days = pd.bdate_range(prev_date, curr_date, freq='B')

            if len(business_days) > 2:  # More than 1 day gap (accounting for weekends)
                date_gaps.append((dates[i-1], dates[i], len(business_days) - 1))

        if date_gaps:
            issues.append(f"Found {len(date_gaps)} date gaps (possibly holidays)")

        # Check price continuity (no huge jumps)
        df_sorted = df.sort_values('date')
        price_changes = df_sorted['close'].pct_change()
        large_jumps = price_changes[abs(price_changes) > 0.05]  # >5% in 1 minute

        if len(large_jumps) > 0:
            issues.append(f"Found {len(large_jumps)} large price jumps (>5% in 1 min)")

        is_valid = len(issues) == 0
        return is_valid, issues


def main():
    """Example usage"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Initialize loader
    loader = NiftySeedDataLoader()

    print("="*80)
    print("NIFTY SEED DATA ANALYSIS")
    print("="*80)

    # Load data
    data = loader.load()
    print(f"\n✓ Loaded {len(data):,} rows")

    # Get summary
    summary = loader.get_summary()

    print(f"\n📊 Summary:")
    print(f"  Total Days: {summary['total_days']}")
    print(f"  Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}")
    print(f"  Price Range: ₹{summary['price_range']['min']:.2f} - ₹{summary['price_range']['max']:.2f}")
    print(f"  Mean Price: ₹{summary['price_range']['mean']:.2f}")
    print(f"  Avg Candles/Day: {summary['trading_hours']['avg_candles_per_day']:.0f}")
    print(f"  Zero Volume: {summary['data_quality']['zero_volume_pct']:.1f}%")

    # Validate
    is_valid, issues = loader.validate_data()

    print(f"\n🔍 Validation:")
    if is_valid:
        print("  ✓ All quality checks passed")
    else:
        print(f"  ⚠ Found {len(issues)} issues:")
        for issue in issues:
            print(f"    - {issue}")

    # Show daily aggregation sample
    daily = loader.get_daily_data()
    print(f"\n📅 Daily Data Sample (first 5 days):")
    print(daily.head().to_string(index=False))

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
