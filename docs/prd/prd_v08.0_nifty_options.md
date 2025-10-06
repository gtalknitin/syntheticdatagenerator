# Synthetic NIFTY Options Data Generation PRD v8.0
## Extended Range with Broader Strike Coverage for Low Delta Options

**Product Name**: NIFTY Options Synthetic Data Generator v8.0
**Version**: 8.0
**Date**: October 3, 2025
**Author**: NikAlgoBulls Development Team
**Status**: Extended Range & Broader Strikes for 0.1 Delta Testing
**Previous Version**: v7.0-VIX (July 1 - September 30, 2025, Strikes ₹24,000-28,000)

---

## 1. Executive Summary

### 1.1 Purpose
This document specifies the v8.0 implementation of the synthetic NIFTY options data generator, extending the date range backward to **June 14, 2025** and expanding strike coverage to **₹22,000 - ₹30,000** to support testing of **0.1 delta weekly hedge positions** required by Nikhil's strategy.

### 1.2 Critical Improvements (v8.0)

**New in V8.0**:
- **Extended Date Range**: June 14, 2025 to September 30, 2025 (3.5+ months)
- **Broader Strike Coverage**: ₹22,000 to ₹30,000 (161 strikes, 50-point intervals)
- **0.1 Delta Support**: Far OTM options for weekly hedge testing
- **June Monthly Expiry**: June 26, 2025 (last Thursday)
- **Additional VIX Regimes**: Two more weeks of testing scenarios

**Retained from V7**:
- VIX Data Integration with realistic regimes
- September 30 Tuesday expiry fix
- High volatility testing periods (VIX > 30)
- Data continuity and quality validation
- Complete Greeks calculations

### 1.3 Key Design Principles
1. **0.1 Delta Coverage**: Enable testing of far OTM weekly hedges
2. **Extended Testing Period**: Longer backtest window for strategy validation
3. **Market Realism**: Accurate expiry schedules and volatility regimes
4. **Data Integrity**: No gaps, duplicates, or impossible values
5. **Production Readiness**: Full strategy testing capability

---

## 2. Critical Requirements Matrix

### 2.1 Mandatory Requirements (MUST HAVE)

| Requirement | Specification | Validation Criteria | Priority |
|-------------|--------------|-------------------|----------|
| **Date Range** | June 14, 2025 to September 30, 2025 | 79 trading days minimum | P0 |
| **Strike Range** | ₹22,000 to ₹30,000 (161 strikes) | All strikes available from day 1 | P0 |
| **0.1 Delta Options** | Consistently available across range | Min 10 strikes with delta 0.05-0.15 | P0 |
| **June 26 Expiry** | Last Thursday monthly expiry | Full data from June 14 onwards | P0 |
| **September 30 Expiry** | Last Tuesday monthly expiry | Full data from creation | P0 |
| **VIX Data Column** | India VIX values for all timestamps | Range 10-50, realistic patterns | P0 |
| **VIX > 30 Periods** | At least 12 days for testing exits | Min 2 separate high VIX events | P0 |
| **No Duplicates** | One row per timestamp/strike/type | Zero duplicate entries | P0 |
| **Price Continuity** | Smooth evolution, max 5% per 5min | No impossible jumps | P0 |
| **Complete Greeks** | Delta, Gamma, Theta, Vega calculated | Consistent with prices | P0 |

### 2.2 Enhanced Requirements (v8.0 Additions)

| Feature | Implementation | Benefit |
|---------|---------------|---------|
| **Far OTM Strikes** | ±2000 points from ATM at all times | 0.1 delta weekly hedges testable |
| **Extended VIX Regimes** | June volatility baseline period | More testing scenarios |
| **June Expiry** | June 26 monthly expiry support | Full cycle testing |
| **Strike Persistence** | All 161 strikes always available | Consistent option chain |
| **Multi-Month Coverage** | 3.5+ months of data | Longer strategy validation |

---

## 3. Date Range Specification

### 3.1 Trading Period

```python
DATE_RANGE_V8 = {
    "start_date": "2025-06-14",      # Saturday - first Monday is June 16
    "end_date": "2025-09-30",        # Tuesday - September monthly expiry
    "total_days": 79,                # Estimated trading days
    "months_covered": 3.5,           # Mid-June to end-September
    "purpose": "Extended testing period with broader strikes"
}
```

### 3.2 Expiry Schedule

```python
EXPIRY_SCHEDULE_V8 = {
    "monthly": {
        "2025-06-26": "Thursday",    # Last Thursday of June
        "2025-07-31": "Thursday",    # Last Thursday of July
        "2025-08-28": "Thursday",    # Last Thursday of August
        "2025-09-30": "Tuesday"      # Last Tuesday of September (CRITICAL)
    },
    "weekly": {
        "June": [
            "2025-06-18",  # Wed (transition period)
            "2025-06-25"   # Wed (one day before monthly)
        ],
        "July": [
            "2025-07-03",  # Thu
            "2025-07-10",  # Thu
            "2025-07-17",  # Thu
            "2025-07-24"   # Thu
        ],
        "August": [
            "2025-08-07",  # Thu
            "2025-08-14",  # Thu
            "2025-08-21"   # Thu
        ],
        "September": [
            "2025-09-03",  # Wed
            "2025-09-10",  # Wed
            "2025-09-17",  # Wed
            "2025-09-24"   # Tue (transition)
        ]
    }
}
```

### 3.3 Expiry Coverage Requirements

**June 26 Monthly Expiry**:
- Must exist from: June 14, 2025 (first Saturday, trading starts June 16)
- Strike coverage: All 161 strikes (₹22,000 to ₹30,000)
- Data availability: All timestamps from June 16 to June 26
- Price continuity: Smooth theta decay to expiry

**September 30 Monthly Expiry**:
- Must exist from: August 28, 2025 (day after August monthly expiry)
- Strike coverage: All 161 strikes
- Data availability: All timestamps from creation to September 30
- Price continuity: Smooth theta decay to expiry

---

## 4. Strike Range Specification

### 4.1 Expanded Strike Coverage

```python
STRIKE_RANGE_V8 = {
    "min_strike": 22000,
    "max_strike": 30000,
    "interval": 50,
    "total_strikes": 161,           # (30000-22000)/50 + 1
    "purpose": "Support 0.05-0.15 delta options"
}

# Example strike distribution for underlying at ₹26,000:
STRIKE_EXAMPLES = {
    "deep_ITM": [22000, 22050, ..., 24000],      # ~40 strikes
    "near_ATM": [24050, ..., 27950],             # ~80 strikes
    "deep_OTM": [28000, 28050, ..., 30000]       # ~40 strikes
}
```

### 4.2 Delta Coverage Validation

```python
DELTA_REQUIREMENTS = {
    # For CE options when underlying = ₹26,000
    "call_options": {
        "0.90_to_1.00_delta": "Strikes 22000-24000 (deep ITM)",
        "0.40_to_0.60_delta": "Strikes 25500-26500 (near ATM)",
        "0.10_to_0.20_delta": "Strikes 27500-28500 (hedge range)",
        "0.05_to_0.10_delta": "Strikes 28500-29500 (far OTM)"
    },

    # For PE options when underlying = ₹26,000
    "put_options": {
        "0.90_to_1.00_delta": "Strikes 28000-30000 (deep ITM)",
        "0.40_to_0.60_delta": "Strikes 25500-26500 (near ATM)",
        "0.10_to_0.20_delta": "Strikes 23500-24500 (hedge range)",
        "0.05_to_0.10_delta": "Strikes 22500-23500 (far OTM)"
    }
}
```

### 4.3 Weekly Hedge Strike Requirements

**For Nikhil's Strategy**:
- Weekly hedges use ~0.1 delta short strikes
- Protective long at 50-66% of short premium
- Must have at least 10 strikes with delta 0.05-0.15 for both CE and PE
- Strikes must be available continuously throughout trading day

**Example Weekly Hedge Scenario**:
```python
# Underlying: ₹26,000, VIX: 18
weekly_hedge_example = {
    "short_ce_strike": 28000,    # ~0.10 delta CE
    "long_ce_strike": 29000,     # ~0.03 delta CE (protection)
    "short_pe_strike": 24000,    # ~0.10 delta PE
    "long_pe_strike": 23000      # ~0.03 delta PE (protection)
}
# All these strikes must be in ₹22,000-30,000 range ✓
```

---

## 5. VIX Data Specification

### 5.1 Extended VIX Regime Design

```python
VIX_REGIMES_V8 = {
    # Period 0: June Baseline (VIX 14-18)
    "2025-06-16 to 2025-06-30": {
        "base_vix": 16,
        "range": 2,
        "description": "Initial baseline period"
    },

    # Period 1: Normal Market (VIX 12-18)
    "2025-07-01 to 2025-07-20": {
        "base_vix": 15,
        "range": 3,
        "description": "Baseline normal volatility"
    },

    # Period 2: Rising Anxiety (VIX 18-25)
    "2025-07-21 to 2025-07-31": {
        "base_vix": 21,
        "range": 4,
        "description": "Pre-event volatility rise"
    },

    # Period 3: High Volatility Event (VIX 28-35)
    "2025-08-01 to 2025-08-10": {
        "base_vix": 32,
        "range": 3,
        "description": "First VIX > 30 test period"
    },

    # Period 4: Recovery (VIX 22-28)
    "2025-08-11 to 2025-08-20": {
        "base_vix": 25,
        "range": 3,
        "description": "Post-event cooling"
    },

    # Period 5: Normal (VIX 14-20)
    "2025-08-21 to 2025-09-05": {
        "base_vix": 17,
        "range": 3,
        "description": "Return to normal"
    },

    # Period 6: Volatility Spike (VIX 30-38)
    "2025-09-06 to 2025-09-15": {
        "base_vix": 34,
        "range": 4,
        "description": "Second VIX > 30 test period"
    },

    # Period 7: Stabilization (VIX 20-25)
    "2025-09-16 to 2025-09-30": {
        "base_vix": 22,
        "range": 3,
        "description": "End period normalization"
    }
}
```

### 5.2 VIX Testing Scenarios

| Scenario | Dates | VIX Range | Testing Purpose |
|----------|-------|-----------|-----------------|
| **June Baseline** | Jun 16-30 | 14-18 | Initial period normal trading |
| **Normal Trading** | Jul 1-20 | 12-18 | Baseline strategy performance |
| **VIX > 30 Exit #1** | Aug 1-10 | 28-35 | Test exit all trades logic |
| **Resume Trading** | Aug 11 | ~25 | Test resumption when VIX < 30 |
| **VIX > 30 Exit #2** | Sep 6-15 | 30-38 | Test second exit event |
| **Edge Cases** | Various | 29.9, 30.0, 30.1 | Test threshold precision |

---

## 6. Technical Implementation

### 6.1 Data Generation Architecture

```python
class SyntheticOptionsGeneratorV8:
    """
    Extended range generator with broader strike coverage
    """

    def __init__(self):
        self.config = {
            # Date range
            'start_date': '2025-06-14',
            'end_date': '2025-09-30',
            'timestamps_per_day': 79,      # 09:15 to 15:30

            # Strike range (EXPANDED)
            'min_strike': 22000,
            'max_strike': 30000,
            'strike_interval': 50,
            'total_strikes': 161,

            # Pricing
            'min_price': 0.05,
            'risk_free_rate': 0.065,

            # Features
            'vix_column_required': True,
            'june_26_expiry': True,
            'sept_30_expiry': True,

            # Validation
            'min_delta_01_strikes': 10     # Min strikes with 0.05-0.15 delta
        }

        self.validators = [
            DuplicateValidator(),
            StrikeCoverageValidator(),      # NEW: Validate 161 strikes
            LowDeltaValidator(),            # NEW: Validate 0.1 delta availability
            ExpiryCoverageValidator(),
            VixDataValidator(),
            VixRangeValidator(),
            PriceContinuityValidator(),
            GreeksConsistencyValidator()
        ]
```

### 6.2 Strike Generation

```python
def generate_strike_chain(self):
    """
    Generate all 161 strikes from ₹22,000 to ₹30,000
    """
    strikes = []
    current_strike = 22000

    while current_strike <= 30000:
        strikes.append(current_strike)
        current_strike += 50

    assert len(strikes) == 161, f"Expected 161 strikes, got {len(strikes)}"
    return strikes
```

### 6.3 Delta Calculation & Validation

```python
def validate_delta_coverage(df, underlying_price):
    """
    Ensure sufficient 0.05-0.15 delta options available
    """
    # CE options: far OTM strikes
    ce_options = df[(df['option_type'] == 'CE') &
                    (df['delta'] >= 0.05) &
                    (df['delta'] <= 0.15)]

    # PE options: far OTM strikes
    pe_options = df[(df['option_type'] == 'PE') &
                    (df['delta'] >= 0.05) &
                    (df['delta'] <= 0.15)]

    ce_strikes = len(ce_options['strike'].unique())
    pe_strikes = len(pe_options['strike'].unique())

    assert ce_strikes >= 10, f"Only {ce_strikes} CE strikes with 0.05-0.15 delta"
    assert pe_strikes >= 10, f"Only {pe_strikes} PE strikes with 0.05-0.15 delta"

    return True
```

### 6.4 VIX-Influenced Pricing (Same as V7)

```python
def calculate_option_price_with_vix(spot, strike, tte, option_type, vix):
    """
    Price options using VIX as base volatility
    """
    # Convert VIX to volatility
    base_iv = vix / 100.0

    # Adjust for moneyness (volatility smile)
    moneyness = spot / strike
    if abs(moneyness - 1) > 0.05:
        iv = base_iv * (1 + abs(moneyness - 1) * 0.2)
    else:
        iv = base_iv

    # Calculate Black-Scholes price
    price = black_scholes(spot, strike, tte, iv, r, option_type)

    # Ensure minimum tick
    return max(0.05, round(price, 2))
```

---

## 7. Data Validation Framework

### 7.1 Critical Validations

```python
VALIDATION_SUITE_V8 = {
    "date_validation": {
        "start_date_correct": lambda df: df['timestamp'].min().date() == date(2025, 6, 16),
        "end_date_correct": lambda df: df['timestamp'].max().date() == date(2025, 9, 30),
        "min_trading_days": lambda df: df['timestamp'].dt.date.nunique() >= 79
    },

    "strike_validation": {
        "min_strike_22k": lambda df: df['strike'].min() == 22000,
        "max_strike_30k": lambda df: df['strike'].max() == 30000,
        "total_strikes_161": lambda df: df['strike'].nunique() == 161,
        "strike_interval_50": lambda df: verify_strike_intervals(df, 50)
    },

    "delta_validation": {
        "ce_low_delta_available": lambda df: count_delta_range(df, 'CE', 0.05, 0.15) >= 10,
        "pe_low_delta_available": lambda df: count_delta_range(df, 'PE', 0.05, 0.15) >= 10,
        "delta_consistency": lambda df: verify_delta_moneyness(df)
    },

    "expiry_validation": {
        "june_26_exists": lambda df: '2025-06-26' in df['expiry'].unique(),
        "june_26_strikes": lambda df: len(df[df['expiry']=='2025-06-26']['strike'].unique()) == 161,
        "sept_30_exists": lambda df: '2025-09-30' in df['expiry'].unique(),
        "sept_30_strikes": lambda df: len(df[df['expiry']=='2025-09-30']['strike'].unique()) == 161
    },

    "vix_validation": {
        "vix_column_exists": lambda df: 'vix' in df.columns,
        "vix_range_valid": lambda df: (df['vix'] >= 10).all() and (df['vix'] <= 50).all(),
        "vix_above_30_days": lambda df: (df.groupby('date')['vix'].first() > 30).sum() >= 12,
        "vix_smooth_transitions": lambda df: verify_vix_transitions(df)
    },

    "data_quality": {
        "no_duplicates": lambda df: not df.duplicated(['timestamp','strike','option_type','expiry']).any(),
        "price_continuity": lambda df: verify_price_jumps(df, max_jump=0.05),
        "greeks_validity": lambda df: verify_greeks_consistency(df)
    }
}
```

### 7.2 Success Criteria

The generator is successful when:

1. ✅ Date range: June 16, 2025 to September 30, 2025 (79+ days)
2. ✅ Strike range: ₹22,000 to ₹30,000 (161 strikes)
3. ✅ 0.1 delta coverage: 10+ CE and 10+ PE strikes with delta 0.05-0.15
4. ✅ June 26 (Thursday) expiry exists with all 161 strikes
5. ✅ September 30 (Tuesday) expiry exists with all 161 strikes
6. ✅ VIX column present in all files
7. ✅ At least 12 days with VIX > 30
8. ✅ Zero duplicate timestamp/strike combinations
9. ✅ Price movements < 5% per 5 minutes
10. ✅ All Greeks properly calculated
11. ✅ Bid < Price < Ask for all options
12. ✅ Volume increases during high VIX periods

---

## 8. File Structure & Output

### 8.1 Directory Organization

```
zerodha_strategy/data/synthetic/
├── intraday_v8_extended/         # NEW: V8 production version
│   ├── NIFTY_OPTIONS_5MIN_20250616.csv  # First Monday
│   ├── NIFTY_OPTIONS_5MIN_20250626.csv  # June monthly expiry
│   ├── NIFTY_OPTIONS_5MIN_20250701.csv
│   ├── NIFTY_OPTIONS_5MIN_20250801.csv  # VIX > 30
│   ├── NIFTY_OPTIONS_5MIN_20250827.csv  # Sept 30 expiry appears
│   ├── NIFTY_OPTIONS_5MIN_20250910.csv  # VIX peak ~38
│   ├── NIFTY_OPTIONS_5MIN_20250930.csv  # Final day
│   └── metadata/
│       └── generation_info.json
├── intraday_v7_vix/              # Previous version
└── archive/                       # Old versions
```

### 8.2 CSV Schema (21 columns - Same as V7)

```python
SCHEMA_V8 = {
    # Core identifiers
    'timestamp': 'datetime64[ns]',
    'symbol': 'str',              # NIFTY
    'strike': 'int32',            # 22000-30000 in 50 point intervals
    'option_type': 'category',    # CE/PE
    'expiry': 'datetime64[ns]',   # Includes 2025-06-26, 2025-09-30
    'expiry_type': 'category',    # weekly/monthly

    # Price data
    'open': 'float32',
    'high': 'float32',
    'low': 'float32',
    'close': 'float32',

    # Market data
    'volume': 'int32',
    'oi': 'int32',
    'bid': 'float32',
    'ask': 'float32',

    # Greeks
    'iv': 'float32',
    'delta': 'float32',           # Critical: Must support 0.05-0.15 range
    'gamma': 'float32',
    'theta': 'float32',
    'vega': 'float32',

    # Reference
    'underlying_price': 'float32',
    'vix': 'float32'              # India VIX
}
```

---

## 9. Testing Capabilities Enabled

### 9.1 Strategy Testing

- **Entry Logic**: Test in normal VIX conditions (June, early July)
- **Exit Logic**: Validate VIX > 30 exit all trades (August, September)
- **Resume Logic**: Verify trading resumes when VIX < 30
- **Weekly Hedges**: Test 0.1 delta short strikes with protective longs
- **June Expiry**: Test monthly positions on June 26
- **September Expiry**: Test monthly positions on September 30
- **Extended Backtest**: 3.5+ months of continuous data
- **Risk Management**: Validate all risk thresholds

### 9.2 0.1 Delta Weekly Hedge Testing

**Test Scenarios**:
```python
# Scenario 1: Bullish monthly, bearish weekly hedge
monthly_position = {
    "type": "bull_call_spread",
    "long": 26000,   # ATM CE
    "short": 26500   # OTM CE
}

weekly_hedge = {
    "type": "bear_put_spread",
    "short": 24000,  # ~0.10 delta PE (needs ₹22k-30k range)
    "long": 23500    # ~0.05 delta PE (protection)
}

# Scenario 2: Bearish monthly, bullish weekly hedge
monthly_position = {
    "type": "bear_put_spread",
    "long": 26000,   # ATM PE
    "short": 25500   # OTM PE
}

weekly_hedge = {
    "type": "bull_call_spread",
    "short": 28000,  # ~0.10 delta CE (needs ₹22k-30k range)
    "long": 28500    # ~0.05 delta CE (protection)
}
```

### 9.3 Market Conditions

- **June Baseline**: VIX 14-18 (Jun 16-30)
- **Low Volatility**: VIX 12-18 (July 1-20)
- **Rising Volatility**: VIX 18-25 (July 21-31)
- **High Volatility**: VIX 30+ (Aug 1-10, Sep 6-15)
- **Recovery**: VIX declining (Aug 11-20, Sep 16-30)

### 9.4 Edge Cases

- VIX exactly at 30.0 threshold
- Rapid intraday VIX changes
- 0.1 delta options during high VIX (spreads widen)
- Options behavior near expiry with high VIX
- June 26 and September 30 expiry handling
- Far OTM strikes with very low premiums (₹0.05 minimum)

---

## 10. Quality Metrics

### 10.1 Data Statistics Targets

| Metric | Target | V7 Achieved | V8 Target |
|--------|--------|-------------|-----------|
| Total rows | 3.5M+ | 3.78M | 6.5M+ |
| Trading days | 65 | 65 | 79+ |
| Strikes per day | 80+ | 81 | 161 |
| VIX > 30 days | 10+ | 12 | 12+ |
| June 26 strikes | N/A | N/A | 161 |
| Sept 30 strikes | 80+ | 81 | 161 |
| Duplicate rows | 0 | 0 | 0 |
| Max price jump | <5% | <5% | <5% |
| 0.1 delta CE strikes | N/A | ~5 | 10+ |
| 0.1 delta PE strikes | N/A | ~5 | 10+ |

### 10.2 VIX Statistics (Extended)

- **Range**: 13-41
- **Average**: ~22
- **Days > 30**: 12+ (15%+ of trading days)
- **Peak**: ~39 (September)
- **Smooth transitions**: Yes
- **June baseline**: 14-18

---

## 11. Migration from Previous Versions

### 11.1 Version Comparison

| Version | Date Range | Strikes | Strike Range | 0.1Δ Support | VIX | Status |
|---------|-----------|---------|--------------|--------------|-----|--------|
| V5 | Jul 1-Sep 25 | 81 | ₹24k-28k | ❌ Limited | ❌ No | Deprecated |
| V6 | Jul 1-Sep 25 | 81 | ₹24k-28k | ❌ Limited | ❌ No | Superseded |
| V7 | Jul 1-Sep 30 | 81 | ₹24k-28k | ⚠️ Partial | ✅ Yes | Superseded |
| **V8** | **Jun 14-Sep 30** | **161** | **₹22k-30k** | **✅ Full** | **✅ Yes** | **Production** |

### 11.2 Migration Steps

1. Generate V8 data to `intraday_v8_extended/`
2. Validate all 161 strikes present
3. Verify 0.1 delta options available
4. Test weekly hedge strike selection
5. Validate June 26 and September 30 expiries
6. Update strategy code to use V8 data
7. Archive V7 to `archive/` directory

---

## 12. Implementation Checklist

### 12.1 Pre-Generation

- [ ] Define June 16-30 VIX baseline (14-18)
- [ ] Create 161-strike generation logic
- [ ] Update Black-Scholes calculator for broader range
- [ ] Implement June 26 expiry logic
- [ ] Extend expiry schedule through June

### 12.2 Generation

- [ ] Generate June 16-30 files (11 trading days)
- [ ] Generate July 1-31 files (23 trading days)
- [ ] Generate August 1-31 files (21 trading days)
- [ ] Generate September 1-30 files (22 trading days)
- [ ] Create metadata/generation_info.json

### 12.3 Validation

- [ ] Validate 79+ trading days
- [ ] Validate 161 strikes in all files
- [ ] Validate June 26 expiry coverage
- [ ] Validate September 30 expiry coverage
- [ ] Validate 0.1 delta CE options (10+ strikes)
- [ ] Validate 0.1 delta PE options (10+ strikes)
- [ ] Validate VIX > 30 periods (12+ days)
- [ ] Validate no duplicates
- [ ] Validate price continuity
- [ ] Validate Greeks consistency

### 12.4 Documentation

- [ ] Create V8_GENERATION_SUMMARY.md
- [ ] Document 0.1 delta strike locations
- [ ] Document weekly hedge test scenarios
- [ ] Create migration guide from V7
- [ ] Update strategy PRD references

---

## 13. Success Metrics

The V8.0 generator successfully provides:

1. **Extended Period**: 3.5+ months (June 14 - September 30, 2025)
2. **Broader Strikes**: 161 strikes covering ₹22,000 to ₹30,000
3. **0.1 Delta Support**: Full coverage for weekly hedge testing
4. **Complete Testing**: All strategy features including low delta hedges
5. **Market Realism**: Accurate expiry dates and volatility patterns
6. **Data Quality**: No duplicates, smooth prices, valid Greeks
7. **Production Ready**: Suitable for comprehensive strategy validation

---

## 14. Risk Mitigation

### 14.1 Data Size Management

**Challenge**: 161 strikes × 79 days × 79 timestamps = 6.5M+ rows

**Solutions**:
- Efficient CSV writing with compression
- Chunked generation by month
- Memory-efficient data structures
- Daily file size limit monitoring

### 14.2 Far OTM Option Pricing

**Challenge**: Very low premiums near ₹0.05 minimum

**Solutions**:
- Strict ₹0.05 minimum price enforcement
- Realistic bid-ask spreads (avoid ₹0.00 bids)
- Volume tapers for far OTM (low but non-zero)
- Greeks calculated correctly even for low premiums

### 14.3 Validation Complexity

**Challenge**: More strikes = more validation points

**Solutions**:
- Automated validation suite
- Statistical sampling for large datasets
- Critical path validation (0.1 delta strikes)
- Daily spot checks during generation

---

## 15. Document History

- v1.0: Initial requirements (July 2025)
- v2.0: Added strike persistence (August 2025)
- v3.0: Binary price issues identified (September 2025)
- v4.0: Partial fixes, efficiency focus (September 27, 2025)
- v5.0: Enhanced realism, validation framework (September 30, 2025)
- v6.0: VIX integration, Sept 30 expiry fix (October 1, 2025)
- v7.0: Production ready with VIX data (October 1, 2025)
- **v8.0: Extended date range, broader strikes for 0.1 delta support (October 3, 2025)**

---

**Status**: 📝 SPECIFICATION READY - AWAITING GENERATION
**Target Location**: `/zerodha_strategy/data/synthetic/intraday_v8_extended/`
**Generator**: `generate_v8_extended.py` (to be created)
**Validation**: All criteria defined and ready

---

## 16. Appendix: Strike-Delta Mapping Reference

### 16.1 Expected Delta Ranges (Underlying: ₹26,000, VIX: 18)

**Call Options (CE)**:
```
Strike Range    Expected Delta    Use Case
₹22,000-24,000  0.90-1.00        Deep ITM (not typically traded)
₹24,000-25,500  0.60-0.90        ITM monthly spreads
₹25,500-26,500  0.40-0.60        ATM monthly spreads (primary)
₹26,500-27,500  0.20-0.40        OTM monthly spreads
₹27,500-28,500  0.10-0.20        Weekly hedge candidates
₹28,500-29,500  0.05-0.10        Far OTM hedges (V8 enabled)
₹29,500-30,000  <0.05            Very far OTM (edge testing)
```

**Put Options (PE)**:
```
Strike Range    Expected Delta    Use Case
₹28,000-30,000  0.90-1.00        Deep ITM (not typically traded)
₹26,500-28,000  0.60-0.90        ITM monthly spreads
₹25,500-26,500  0.40-0.60        ATM monthly spreads (primary)
₹24,500-25,500  0.20-0.40        OTM monthly spreads
₹23,500-24,500  0.10-0.20        Weekly hedge candidates
₹22,500-23,500  0.05-0.10        Far OTM hedges (V8 enabled)
₹22,000-22,500  <0.05            Very far OTM (edge testing)
```

**Note**: Delta values will vary with VIX levels and time to expiry. Values shown are approximate for 30-day options with VIX at 18.
