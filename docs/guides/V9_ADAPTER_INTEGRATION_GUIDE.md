# V9 Synthetic Data Adapter - Integration Guide

## Problem Solved
**Issue**: Strategy can't find 'monthly_expiries' field in V9 synthetic data
**Root Cause**: Strategy expects OptionsDataFetcher API interface, but V9 data is CSV-based
**Solution**: SyntheticDataAdapter provides OptionsDataFetcher-compatible interface

## Integration Steps

### 1. Import the Adapter
```python
from synthetic_data_adapter import SyntheticDataAdapter
```

### 2. Initialize with Data Path
```python
# Point to V9 balanced data directory
data_path = '/Users/nitindhawan/NikAlgoBulls/zerodha_strategy/data/synthetic/hourly_v9_balanced'
fetcher = SyntheticDataAdapter(data_path)
```

### 3. Use Same Interface as Before
```python
# Get monthly expiries (returns List[datetime.date])
monthly_expiries = fetcher.get_monthly_expiries("NIFTY", months=3, from_date=date)

# Get weekly expiries (returns List[datetime.date])
weekly_expiries = fetcher.get_weekly_expiries("NIFTY", weeks=4, from_date=date)

# Get spot price
spot = fetcher.get_spot_price("NIFTY", date=date)

# Get ATM strike
atm_strike = fetcher.get_atm_strike(spot)

# Get option data
option_data = fetcher.get_option_data(
    symbol="NIFTY",
    strike=atm_strike,
    option_type="CE",
    expiry=monthly_expiries[0],
    date=date
)
```

## Available Methods

### Core Data Methods
- `get_monthly_expiries(symbol, months, from_date)` → List[datetime.date]
- `get_weekly_expiries(symbol, weeks, from_date)` → List[datetime.date]
- `get_spot_price(symbol, date, timestamp)` → float
- `get_option_chain(symbol, date)` → pd.DataFrame
- `get_option_data(symbol, strike, option_type, expiry, date)` → pd.DataFrame
- `get_atm_strike(spot_price, symbol)` → int
- `get_available_dates()` → List[datetime.date]

### Data Caching
The adapter automatically caches loaded CSV data for better performance.

## Test Results ✅

Tested on 2025-06-23:
- **Available Dates**: 79 days (2025-06-09 to 2025-09-26)
- **Monthly Expiries**: [2025-06-26, 2025-07-31]
- **Weekly Expiries**: [2025-06-25, 2025-07-03, 2025-07-10, 2025-07-17]
- **Spot Price**: ₹25,308.44
- **ATM Strike**: 25,300
- **Option Data**: 7 rows (1-hour candles), Premium ₹208.36, Delta 0.5797

## Migration Checklist

- [ ] Copy `synthetic_data_adapter.py` to your strategy's source directory
- [ ] Update imports to use `SyntheticDataAdapter` instead of `OptionsDataFetcher`
- [ ] Update initialization with V9 data path
- [ ] Test with a single trading day
- [ ] Run full backtest
- [ ] Verify both bullish and bearish positions are generated
- [ ] Confirm weekly hedges show meaningful P&L (not zero)

## V9 Data Improvements

Using V9 balanced data now provides:
1. ✅ **Balanced Trends**: 50/50 weekly bullish/bearish split (was 81% bullish in V8)
2. ✅ **Expiry-Specific Pricing**: Weekly 0.1Δ ~1000 pts, Monthly 0.1Δ ~2000 pts (realistic)
3. ✅ **Realistic Premiums**: Monthly/Weekly ratio >1.3x (correct TTE application)
4. ✅ **Efficient Data**: 1-hour candles = 91% smaller than 5-min candles
5. ✅ **Complete Greeks**: All Greeks calculated per expiry-type

## File Location

**Adapter**: `/Users/nitindhawan/NikAlgoBulls/zerodha_strategy/src/synthetic_data_adapter.py`
**V9 Data**: `/Users/nitindhawan/NikAlgoBulls/zerodha_strategy/data/synthetic/hourly_v9_balanced/`
**Metadata**: `hourly_v9_balanced/metadata/generation_info.json`
