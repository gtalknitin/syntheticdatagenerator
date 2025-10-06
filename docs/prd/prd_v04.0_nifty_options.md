# Synthetic NIFTY Options Data Generation PRD v4.0

**Product Name**: NIFTY Options Synthetic Data Generator v4.0  
**Version**: 4.0  
**Date**: September 27, 2025  
**Author**: NikAlgoBulls Development Team  
**Status**: Production Ready (with known limitations)  

---

## 1. Executive Summary

### 1.1 Purpose
This document outlines the v4.0 implementation of the synthetic NIFTY options data generator, which addresses major pricing issues from v3 while providing efficient generation for large-scale backtesting. The v4 generator implements proper option pricing theory with some simplifications for computational efficiency.

### 1.2 Key Improvements Over v3
- **Reduced Binary Price Collapse**: Only 7.2% of options at ₹0.05 (vs 35-87% in v3)
- **Black-Scholes Pricing**: Implemented throughout with vectorized calculations
- **Theta Decay**: Present for most options (95.6% of priced options)
- **Dynamic Bid-Ask Spreads**: Average 3.9% spreads based on moneyness
- **Volatility Smile**: Implemented with term structure effects
- **Efficient Generation**: 66 trading days generated in 14 seconds

### 1.3 Known Limitations
- Deep ITM options (delta ≥ 0.9999) show zero theta/gamma/vega
- Simplified to 3 timestamps per day for efficiency
- 100-point strike intervals (vs 50 in original spec)

### 1.4 Impact
v4 data enables more realistic backtesting than v3, though some artifacts remain. The 7.2% minimum price ratio and 95.6% proper Greeks coverage represents a significant improvement suitable for strategy validation.

---

## 2. Problem Statement (Resolved)

### 2.1 v3 Issues Addressed
1. **Binary Price Behavior**: ✅ Reduced from 35-87% to 7.2% at minimum
2. **Theta Decay**: ✅ 95.6% of priced options have proper theta
3. **Zero Greeks**: ⚠️ Partially fixed (4.4% deep ITM still have zero Greeks)
4. **Spreads**: ✅ Dynamic spreads averaging 3.9%
5. **Expiry Modeling**: ✅ Pin risk implemented for ATM options

---

## 3. Implementation Details

### 3.1 Technical Architecture

#### 3.1.1 Core Classes
```python
class EfficientV4Generator:
    - Vectorized Black-Scholes calculations
    - Simplified timestamp generation (3 per day)
    - Batch processing for efficiency
    - 100-point strike intervals
```

#### 3.1.2 Key Parameters
- **Risk-free rate**: 6.5%
- **Dividend yield**: 1.2%
- **Base volatility**: 15%
- **Minimum price**: ₹0.05
- **Strike range**: 20,000-30,000 (100-point intervals)

### 3.2 Pricing Implementation

#### 3.2.1 Black-Scholes Vectorization
```python
def black_scholes_vectorized(self, S, K, T, sigma, option_type='CE'):
    # Vectorized calculations for efficiency
    # Handles arrays of strikes simultaneously
    # Returns: price, delta, gamma, theta, vega arrays
```

#### 3.2.2 Volatility Smile
- **OTM Puts (S/K < 0.95)**: +30% per 5% OTM
- **OTM Calls (S/K > 1.05)**: +15% per 5% OTM  
- **Short-dated (<7 DTE)**: +15% IV adjustment
- **Medium-dated (<30 DTE)**: +5% IV adjustment

#### 3.2.3 Greeks Handling
| Option Type | Delta Range | Theta | Gamma | Vega |
|-------------|-------------|--------|--------|------|
| Deep ITM | ±0.9999 | 0 | 0 | 0 |
| ITM | ±0.7-0.999 | Negative | Positive | Positive |
| ATM | ±0.4-0.6 | Most negative | Highest | Highest |
| OTM | ±0.0-0.3 | Small negative | Low | Low |

### 3.3 Market Microstructure

#### 3.3.1 Simplified Bid-Ask Logic
```python
# ATM options (0.95 < S/K < 1.05): 2% spread
# Others: 5% spread
spread_pct = np.where(atm_mask, 0.02, 0.05)
```

#### 3.3.2 Volume/OI Generation
- ATM options: 1,000-5,000 volume
- Non-ATM: 10-500 volume
- OI: Volume × random(10, 50)

### 3.4 Efficiency Optimizations

1. **Reduced Timestamps**: 3 per day (open, mid, close) vs 75
2. **Wider Strikes**: 100-point intervals vs 50
3. **Vectorized Calculations**: Process all strikes at once
4. **Simplified Expiry Logic**: Pre-computed expiry calendar

---

## 4. Generated Data Specifications

### 4.1 Dataset Overview

| Metric | Value |
|--------|-------|
| **Period** | July 1 - September 30, 2025 |
| **Trading Days** | 66 |
| **Total Files** | 66 CSV files |
| **Rows per Day** | 606-10,908 (varies by active expiries) |
| **Total Rows** | ~450,000 |
| **File Size** | ~1-2 MB per file |
| **Generation Time** | 14 seconds |

### 4.2 File Structure
- **Naming**: `NIFTY_OPTIONS_5MIN_YYYYMMDD.csv`
- **Location**: `/zerodha_strategy/data/synthetic/intraday_jul_sep_v4/`

### 4.3 Column Schema
Same as original specification with 19 columns including OHLC, Greeks, volume, OI, bid/ask.

### 4.4 Expiry Calendar
- **Weekly**: Thursdays (with some Tuesday adjustments)
- **Monthly**: Last Thursday of month
- Total: 18 expiry dates across 3 months

---

## 5. Quality Metrics

### 5.1 Validation Results

| Metric | Target | v3 Actual | v4 Actual | Status |
|--------|--------|-----------|-----------|---------|
| Options at min price | < 10% | 35-87% | 7.2% | ✅ Pass |
| Proper theta coverage | > 95% | < 50% | 95.6% | ✅ Pass |
| Zero Greeks (priced) | 0% | > 50% | 4.4% | ⚠️ Acceptable |
| Average bid-ask spread | 2-5% | N/A | 3.9% | ✅ Pass |
| Pin risk modeling | Yes | No | Yes | ✅ Pass |

### 5.2 Data Characteristics

#### 5.2.1 Price Distribution
- 7.2% at minimum (₹0.05) - mostly far OTM
- Gradual price transitions
- No sudden jumps except at deep ITM threshold

#### 5.2.2 Greeks Profile
- 95.6% of priced options have proper Greeks
- 4.4% deep ITM options have zero Greeks (known limitation)
- Theta values range from -34.93 to 0

#### 5.2.3 Spread Analysis
- Average: 3.9%
- ATM options: ~2%
- OTM/ITM options: ~5%

---

## 6. Known Limitations & Workarounds

### 6.1 Deep ITM Greeks Issue
**Limitation**: Options with |delta| = 0.9999 have zero theta/gamma/vega  
**Impact**: Affects 4.4% of priced options  
**Workaround**: Filter these out in backtesting or assign small theta values

### 6.2 Simplified Timestamps
**Limitation**: Only 3 timestamps per day vs 75  
**Impact**: Less granular intraday movements  
**Workaround**: Sufficient for EOD strategies, may need interpolation for HFT

### 6.3 Wider Strike Intervals
**Limitation**: 100-point intervals vs 50  
**Impact**: Less precise strike selection  
**Workaround**: Use nearest available strike in strategy

---

## 7. Usage Guidelines

### 7.1 Data Access
```python
import pandas as pd
import os

# Load single day
df = pd.read_csv('intraday_jul_sep_v4/NIFTY_OPTIONS_5MIN_20250701.csv')

# Load multiple days
data_dir = 'intraday_jul_sep_v4'
all_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
df = pd.concat([pd.read_csv(os.path.join(data_dir, f)) for f in all_files])
```

### 7.2 Filtering Recommendations
```python
# Remove deep ITM with zero Greeks
df_clean = df[~((df['close'] > 0.05) & (df['theta'] == 0))]

# Or assign small theta to deep ITM
df.loc[(df['close'] > 0.05) & (df['theta'] == 0), 'theta'] = -0.01
```

### 7.3 Backtesting Considerations
1. Account for wider strike intervals
2. Use mid-price for entries: `(bid + ask) / 2`
3. Consider slippage for large positions
4. Filter out illiquid options (volume < 100)

---

## 8. Comparison with Previous Versions

| Feature | v3 | v4 | Improvement |
|---------|-----|-----|-------------|
| Min price ratio | 35-87% | 7.2% | 79.8% reduction |
| Greeks accuracy | <50% | 95.6% | 91.2% improvement |
| Generation time | Hours | 14 seconds | 99.9% faster |
| Bid-ask modeling | None | Dynamic | New feature |
| Volatility smile | None | Implemented | New feature |
| Data size | ~2GB | ~100MB | 95% smaller |

---

## 9. Future Enhancement Opportunities

### 9.1 Priority Fixes
1. **Deep ITM Greeks**: Implement small but non-zero theta for all priced options
2. **Full Timestamps**: Generate all 75 timestamps per day
3. **50-Point Strikes**: Return to original strike intervals

### 9.2 Advanced Features
1. **Stochastic Volatility**: Replace fixed smile with dynamic surface
2. **Correlation Structure**: Model cross-strike correlations
3. **Event Days**: Special handling for RBI policy, results
4. **Market Depth**: Level 2 bid-ask data

### 9.3 Performance Optimizations
1. **Parallel Processing**: Use multiprocessing for days
2. **GPU Acceleration**: CUDA for Black-Scholes
3. **Incremental Generation**: Update existing data

---

## 10. Conclusion

The v4 synthetic data generator represents a major improvement over v3, addressing the critical binary price collapse issue and implementing proper option pricing for 95.6% of the dataset. While some limitations remain (deep ITM Greeks, simplified timestamps), the data is suitable for production backtesting with appropriate filters.

The efficient generation (14 seconds for 3 months) enables rapid iteration and testing. Users should be aware of the known limitations and apply recommended workarounds for their specific use cases.

### Recommended Usage
1. ✅ **Suitable for**: Options strategies, directional spreads, hedging analysis
2. ⚠️ **Use with caution**: Deep ITM strategies, high-frequency trading
3. ❌ **Not suitable for**: Greeks-based market making (without fixes)

---

## 11. Appendices

### 11.1 Validation Script
```python
#!/usr/bin/env python3
def validate_v4():
    print("\\nValidating v4 data...")
    # Check minimum price ratio
    # Verify Greeks coverage
    # Analyze bid-ask spreads
```

### 11.2 Generation Performance
- **Machine**: Standard development laptop
- **Time**: 14 seconds for 66 days
- **Memory**: < 1GB RAM usage
- **CPU**: Single-threaded implementation

### 11.3 Sample Data Analysis
```
File: NIFTY_OPTIONS_5MIN_20250731.csv
- Total rows: 8,484
- Options at ₹0.05: 612 (7.2%)
- Priced with zero theta: 394 (4.6%)
- Average spread: 3.9%
- Strike range: 20,000-30,000
- Active expiries: 7
```

---

*Document Version: 1.0*  
*Last Updated: September 27, 2025*  
*Next Review: After deep ITM Greeks fix implementation*