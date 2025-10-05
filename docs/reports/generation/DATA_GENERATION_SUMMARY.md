# Synthetic Options Data Generation Summary

**Generation Date**: September 9, 2025  
**Generation Time**: ~10 minutes  
**Total Files Generated**: 44 CSV files  
**Data Location**: `/Users/nitindhawan/NikAlgoBulls/zerodha_strategy/data/synthetic/`

---

## 📊 Generated Data Overview

### 1. Five-Minute Options Data (Intraday)
- **Files**: 41 CSV files
- **Period**: Last 41 trading days (July 11, 2025 - September 9, 2025)
- **Format**: NIFTY_OPTIONS_5MIN_YYYYMMDD.csv
- **Rows per file**: ~2,500-3,000 rows
- **Time intervals**: 75 candles per day (9:15 AM to 3:30 PM)
- **Strikes covered**: ATM ± 1000 points (sampled)

#### Sample Files:
- `intraday/NIFTY_OPTIONS_5MIN_20250909.csv` - Most recent
- `intraday/NIFTY_OPTIONS_5MIN_20250908.csv`
- `intraday/NIFTY_OPTIONS_5MIN_20250905.csv`
- `intraday/NIFTY_OPTIONS_5MIN_20250904.csv`
- `intraday/NIFTY_OPTIONS_5MIN_20250903.csv`

### 2. Daily Options Data
- **Files**: 3 CSV files
- **Period**: January 1, 2023 to September 9, 2025
- **Format**: NIFTY_OPTIONS_DAILY_YYYY.csv
- **Total rows**: 94,099 across all files
  - 2023: 34,952 rows
  - 2024: 34,932 rows
  - 2025: 24,214 rows

#### Files:
- `daily/NIFTY_OPTIONS_DAILY_2023.csv`
- `daily/NIFTY_OPTIONS_DAILY_2024.csv`
- `daily/NIFTY_OPTIONS_DAILY_2025.csv`

---

## 📁 File Structure

```
zerodha_strategy/data/synthetic/
├── intraday/                    # 5-minute candle data
│   ├── NIFTY_OPTIONS_5MIN_20250711.csv
│   ├── NIFTY_OPTIONS_5MIN_20250714.csv
│   ├── ...
│   └── NIFTY_OPTIONS_5MIN_20250909.csv
├── daily/                       # Daily OHLC data
│   ├── NIFTY_OPTIONS_DAILY_2023.csv
│   ├── NIFTY_OPTIONS_DAILY_2024.csv
│   └── NIFTY_OPTIONS_DAILY_2025.csv
├── generation_summary.csv       # File list and metadata
└── DATA_GENERATION_SUMMARY.md   # This file
```

---

## 📋 Data Fields

### Intraday (5-minute) Format:
```csv
timestamp,symbol,strike,option_type,expiry,open,high,low,close,volume,oi,bid,ask,iv,delta,gamma,theta,vega,underlying_price
```

### Daily Format:
```csv
date,symbol,strike,option_type,expiry,open,high,low,close,volume,oi,vwap,bid,ask,iv,delta,gamma,theta,vega,underlying_close,contracts_traded,turnover
```

---

## 🔍 Data Characteristics

### Realistic Features Included:
1. **Black-Scholes pricing** with market adjustments
2. **Volatility smile** - Higher IV for OTM options
3. **Term structure** - Elevated IV near expiry
4. **Intraday patterns** - U-shaped volume, opening/closing volatility
5. **Bid-ask spreads** - Based on moneyness and liquidity
6. **Volume distribution** - Peak at ATM, Gaussian decay
7. **Option Greeks** - Delta, Gamma, Theta, Vega calculated

### Market Parameters Used:
- Base IV: 12% (adjusted with smile)
- Risk-free rate: 6.5%
- Dividend yield: 1.5%
- Strike interval: ₹50
- Lot size: 25 contracts

---

## 🚀 Usage Instructions

### Loading Data in Python:
```python
import pandas as pd

# Load 5-minute data
df_5min = pd.read_csv('/Users/nitindhawan/NikAlgoBulls/zerodha_strategy/data/synthetic/intraday/NIFTY_OPTIONS_5MIN_20250909.csv')

# Load daily data
df_daily = pd.read_csv('/Users/nitindhawan/NikAlgoBulls/zerodha_strategy/data/synthetic/daily/NIFTY_OPTIONS_DAILY_2025.csv')

# Filter for specific strike and type
atm_calls = df_5min[(df_5min['strike'] == 24900) & (df_5min['option_type'] == 'CE')]
```

### For Backtesting:
```python
# Example: Get option chain for specific timestamp
timestamp = '2025-09-09 10:00:00'
option_chain = df_5min[df_5min['timestamp'] == timestamp]

# Get ATM strike
underlying_price = option_chain['underlying_price'].iloc[0]
atm_strike = round(underlying_price / 50) * 50

# Filter near-ATM options
near_atm = option_chain[abs(option_chain['strike'] - atm_strike) <= 300]
```

---

## ⚠️ Important Notes

1. **This is SYNTHETIC data** - Generated using mathematical models, not real market data
2. **For backtesting only** - Results won't match actual trading performance
3. **Conservative estimates recommended** - Add 15-20% slippage for realistic results
4. **No execution modeling** - Doesn't account for market impact or partial fills
5. **Simplified microstructure** - Real markets have more complex dynamics

---

## 📍 File Access Links

### Quick Access Paths:
```bash
# Intraday data directory
cd /Users/nitindhawan/NikAlgoBulls/zerodha_strategy/data/synthetic/intraday/

# Daily data directory
cd /Users/nitindhawan/NikAlgoBulls/zerodha_strategy/data/synthetic/daily/

# View sample data
head /Users/nitindhawan/NikAlgoBulls/zerodha_strategy/data/synthetic/intraday/NIFTY_OPTIONS_5MIN_20250909.csv
```

### File Sizes:
- Intraday files: ~450-500 KB each
- Daily files: ~4-5 MB each
- Total dataset: ~40 MB uncompressed

---

## 🔧 Generator Code

The synthetic data was generated using:
- **Script**: `/Users/nitindhawan/NikAlgoBulls/zerodha_strategy/src/data_generator/synthetic_generator.py`
- **Class**: `SyntheticOptionsDataGenerator`
- **Dependencies**: pandas, numpy, yfinance, scipy

To regenerate data:
```bash
cd /Users/nitindhawan/NikAlgoBulls/zerodha_strategy
source venv/bin/activate
python src/data_generator/synthetic_generator.py
```

---

## 📈 Data Validation

The generated data has been validated for:
- ✅ Price consistency (call-put parity approximation)
- ✅ Greeks boundaries (Delta: 0-1 for calls, -1-0 for puts)
- ✅ Bid < Ask spread consistency
- ✅ Realistic volume patterns
- ✅ IV within reasonable bounds (8%-50%)
- ✅ Proper CSV formatting and headers

---

*Generated by Synthetic Options Data Generator v1.0*  
*Based on PRD: `/documentation/prd/enhancements/identified/003_synthetic_options_data_generation_prd_v1.0.md`*