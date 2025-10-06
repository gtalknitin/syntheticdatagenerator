# Synthetic Data Generator PRD v2.0
## Time-Series Based NIFTY Options Data Generation with Greeks Evolution

**Version**: 2.0
**Date**: 2025-09-30
**Status**: Production-Ready Specification
**Previous Version Issues**: V1.0/V5 fundamental flaws corrected

---

## Executive Summary

This PRD defines the requirements for a **corrected** synthetic NIFTY options data generator that creates realistic, time-series based option price movements using proper Greeks calculations. This version addresses all critical flaws identified in V5, particularly the duplicate timestamp catastrophe and random price generation issues.

**Key Principle**: Every option price must evolve from its previous price using Greeks-based mathematics, never randomly generated.

---

## 1. CRITICAL REQUIREMENTS - MANDATORY

### 1.1 Data Structure Integrity

```python
MANDATORY_RULES = {
    "unique_timestamps": "One and only one row per timestamp/strike/option_type",
    "time_progression": "Timestamps must increment by exactly 5 minutes",
    "price_continuity": "Each price derived from previous price + Greeks effects",
    "no_duplicates": "Zero tolerance for duplicate entries",
    "complete_chain": "All strikes present for every timestamp"
}
```

### 1.2 Time Series Evolution Model

```python
# CORE PRICING FORMULA - Must be implemented exactly
def calculate_new_price(previous_price, greeks, market_changes):
    """
    Black-Scholes price evolution formula
    Every price MUST be calculated using this method
    """
    price_change = (
        greeks['delta'] * market_changes['spot_change'] +
        0.5 * greeks['gamma'] * (market_changes['spot_change'] ** 2) +
        greeks['theta'] * market_changes['time_elapsed'] +
        greeks['vega'] * market_changes['iv_change'] +
        greeks['rho'] * market_changes['rate_change']
    )

    new_price = previous_price + price_change
    return max(new_price, MIN_OPTION_PRICE)  # 0.05
```

### 1.3 Greeks Calculation Requirements

```python
GREEKS_CONSTRAINTS = {
    "delta": {
        "calls": [0.0001, 0.9999],  # Must be positive
        "puts": [-0.9999, -0.0001],  # Must be negative
        "smooth_transition": True,    # No jumps
    },
    "gamma": {
        "range": [0.000001, 1.0],    # Always positive
        "peak": "at_the_money",      # Highest at ATM
        "never_zero": True            # Even deep ITM has small gamma
    },
    "theta": {
        "range": [-1000, -0.0001],   # Always negative
        "acceleration": "near_expiry", # Increases near expiry
        "never_positive": True         # Time decay always present
    },
    "vega": {
        "range": [0.0001, 1000],     # Always positive
        "peak": "at_the_money",      # Highest at ATM
        "term_structure": True        # Varies with time to expiry
    }
}
```

---

## 2. GENERATOR ARCHITECTURE

### 2.1 State Management System

```python
class OptionDataGenerator:
    """Core generator with mandatory state management"""

    def __init__(self):
        # MANDATORY: Maintain price history
        self.price_history = {}  # {(strike, type, expiry): [prices]}
        self.greeks_cache = {}   # Cache for efficiency
        self.last_timestamp = None
        self.last_underlying = None

    def generate_day_data(self, date):
        """Generate data maintaining temporal continuity"""

        # Step 1: Initialize if first timestamp
        if self.last_timestamp is None:
            self._initialize_opening_prices(date)

        # Step 2: Generate each timestamp sequentially
        all_data = []
        timestamps = self._get_trading_timestamps(date)

        for timestamp in timestamps:
            # CRITICAL: Each timestamp processed once
            timestamp_data = self._evolve_to_timestamp(timestamp)
            all_data.extend(timestamp_data)
            self.last_timestamp = timestamp

        return pd.DataFrame(all_data)

    def _evolve_to_timestamp(self, timestamp):
        """Evolve all option prices to new timestamp"""

        # Calculate market changes
        new_underlying = self._get_underlying_price(timestamp)
        spot_change = new_underlying - self.last_underlying
        time_elapsed = 5 / (390 * 365.25)  # 5 min as year fraction

        timestamp_data = []

        # Process each option ONCE
        for option_key in self.price_history.keys():
            strike, opt_type, expiry = option_key

            # Get previous price
            previous_price = self.price_history[option_key][-1]

            # Get Greeks (cached for efficiency)
            greeks = self._get_greeks(strike, opt_type, expiry, timestamp)

            # Calculate new price using evolution formula
            new_price = self._evolve_price(
                previous_price, greeks, spot_change, time_elapsed
            )

            # Update history
            self.price_history[option_key].append(new_price)

            # Add to output
            timestamp_data.append({
                'timestamp': timestamp,
                'strike': strike,
                'option_type': opt_type,
                'expiry': expiry,
                'close': new_price,
                **greeks,  # Include Greeks in output
                'underlying_price': new_underlying
            })

        self.last_underlying = new_underlying
        return timestamp_data
```

### 2.2 Price Evolution Implementation

```python
def _evolve_price(self, previous_price, greeks, spot_change, time_elapsed):
    """
    MANDATORY: Use Greeks-based evolution, not random generation
    """

    # Calculate each Greek contribution
    delta_effect = greeks['delta'] * spot_change
    gamma_effect = 0.5 * greeks['gamma'] * (spot_change ** 2)
    theta_effect = greeks['theta'] * time_elapsed

    # Simplified IV change model (can be enhanced)
    iv_change = self._calculate_iv_change(spot_change)
    vega_effect = greeks['vega'] * iv_change

    # Total change
    total_change = delta_effect + gamma_effect + theta_effect + vega_effect

    # New price with bounds
    new_price = previous_price + total_change

    # CRITICAL: Sanity checks
    # 1. Minimum price
    new_price = max(new_price, 0.05)

    # 2. Maximum movement cap (prevent data artifacts)
    max_move_pct = min(abs(spot_change / self.last_underlying) * 10, 0.10)
    actual_move_pct = abs(new_price - previous_price) / previous_price

    if actual_move_pct > max_move_pct:
        # Cap the movement
        direction = 1 if new_price > previous_price else -1
        new_price = previous_price * (1 + direction * max_move_pct)

    return round(new_price, 2)
```

---

## 3. DATA SPECIFICATIONS

### 3.1 Temporal Structure

```python
TEMPORAL_SPECS = {
    "trading_day": {
        "start": "09:15:00",
        "end": "15:30:00",
        "intervals": 5,  # minutes
        "timestamps_per_day": 79
    },
    "continuity_rules": {
        "price_t": "derived_from_price_t_minus_1",
        "no_gaps": True,
        "no_duplicates": True,
        "sequential_only": True
    }
}
```

### 3.2 Output Schema

```python
DATA_SCHEMA = {
    # Unique identifiers (combination must be unique)
    'timestamp': 'datetime64[ns]',  # 2025-07-01 09:15:00
    'strike': 'int32',              # 25000
    'option_type': 'category',      # CE/PE
    'expiry': 'date',               # 2025-07-10

    # Price data (evolved, not random)
    'open': 'float32',   # First price of 5-min window
    'high': 'float32',   # Max during 5-min window
    'low': 'float32',    # Min during 5-min window
    'close': 'float32',  # Last price (PRIMARY)

    # Market microstructure
    'bid': 'float32',    # close - spread/2
    'ask': 'float32',    # close + spread/2
    'volume': 'int32',   # Based on moneyness
    'oi': 'int32',       # Open interest

    # Greeks (used for price evolution)
    'iv': 'float32',     # Implied volatility
    'delta': 'float32',  # Price sensitivity to spot
    'gamma': 'float32',  # Delta sensitivity to spot
    'theta': 'float32',  # Time decay (per day)
    'vega': 'float32',   # IV sensitivity
    'rho': 'float32',    # Interest rate sensitivity

    # Reference data
    'underlying_price': 'float32',  # Current spot price
    'tte_days': 'float32',         # Time to expiry in days
    'moneyness': 'float32'         # Spot/Strike ratio
}
```

---

## 4. VALIDATION REQUIREMENTS

### 4.1 Mandatory Pre-Release Validations

```python
class DataValidator:
    """Every generated dataset MUST pass these validations"""

    def validate_structure(self, df):
        """Check data structure integrity"""

        # No duplicate timestamps
        duplicates = df.groupby(['timestamp', 'strike', 'option_type']).size()
        assert duplicates.max() == 1, "Duplicate timestamps found!"

        # Proper time series
        for (strike, opt_type), group in df.groupby(['strike', 'option_type']):
            timestamps = group['timestamp'].sort_values()
            diffs = timestamps.diff()[1:]
            assert (diffs == pd.Timedelta(minutes=5)).all(), "Time gaps found!"

        return True

    def validate_price_movements(self, df):
        """Check price movements are realistic"""

        for (strike, opt_type), group in df.groupby(['strike', 'option_type']):
            group = group.sort_values('timestamp')
            price_changes = group['close'].pct_change().abs()

            # Max 10% movement per 5 minutes
            assert price_changes.max() < 0.10, f"Extreme move: {price_changes.max()}"

        return True

    def validate_greeks(self, df):
        """Check Greeks follow theoretical constraints"""

        # Delta range
        assert df[df['option_type']=='CE']['delta'].between(0, 1).all()
        assert df[df['option_type']=='PE']['delta'].between(-1, 0).all()

        # Gamma positive
        assert (df['gamma'] >= 0).all(), "Negative gamma found!"

        # Theta negative
        assert (df['theta'] <= 0).all(), "Positive theta found!"

        # Vega positive
        assert (df['vega'] >= 0).all(), "Negative vega found!"

        return True

    def validate_time_series_continuity(self, df):
        """Check price evolution is continuous"""

        for (strike, opt_type), group in df.groupby(['strike', 'option_type']):
            prices = group.sort_values('timestamp')['close']

            # Autocorrelation should be high (prices related to previous)
            if len(prices) > 10:
                autocorr = prices.autocorr(lag=1)
                assert autocorr > 0.9, f"Low autocorrelation: {autocorr}"

        return True
```

### 4.2 Quality Metrics

```python
QUALITY_METRICS = {
    "structural": {
        "zero_duplicates": {"target": 0, "mandatory": True},
        "complete_timestamps": {"target": "100%", "mandatory": True},
        "price_continuity": {"target": "R² > 0.95", "mandatory": True}
    },
    "behavioral": {
        "max_5min_move": {"target": "<10%", "mandatory": True},
        "greeks_validity": {"target": "100%", "mandatory": True},
        "theta_decay_visible": {"target": "Yes", "mandatory": True}
    },
    "statistical": {
        "price_distribution": {"target": "Log-normal", "validate": True},
        "volatility_clustering": {"target": "Present", "validate": True},
        "mean_reversion": {"target": "Observable", "validate": True}
    }
}
```

---

## 5. IMPLEMENTATION GUIDELINES

### 5.1 Critical Success Factors

```python
SUCCESS_FACTORS = {
    1: "NEVER create multiple rows for same timestamp/strike/type",
    2: "ALWAYS derive price from previous price using Greeks",
    3: "NEVER use random price generation",
    4: "ALWAYS maintain state between timestamps",
    5: "NEVER process same timestamp twice",
    6: "ALWAYS validate before release",
    7: "NEVER allow duplicate timestamps in output"
}
```

### 5.2 Common Pitfalls to Avoid

```python
AVOID_THESE = {
    "nested_loops": "Don't create multiple entries per timestamp",
    "random_prices": "Don't generate prices randomly",
    "stateless_generation": "Don't forget previous prices",
    "missing_validation": "Don't skip validation steps",
    "greeks_shortcuts": "Don't hardcode Greeks values",
    "time_gaps": "Don't skip timestamps",
    "duplicate_processing": "Don't process timestamp multiple times"
}
```

---

## 6. TESTING REQUIREMENTS

### 6.1 Unit Tests

```python
def test_no_duplicate_timestamps():
    """Critical: Ensure no timestamp duplication"""
    df = generator.generate_day_data('2025-07-01')

    duplicates = df.groupby(['timestamp', 'strike', 'option_type']).size()
    assert duplicates.max() == 1
```

### 6.2 Integration Tests

```python
def test_price_evolution():
    """Ensure prices evolve, not jump randomly"""
    df = generator.generate_day_data('2025-07-01')

    # Get one option's price series
    option = df[(df['strike']==25000) & (df['option_type']=='CE')]
    prices = option.sort_values('timestamp')['close']

    # Check autocorrelation (should be high)
    assert prices.autocorr() > 0.95

    # Check no extreme jumps
    assert prices.pct_change().abs().max() < 0.10
```

### 6.3 Backtest Validation

```python
def test_realistic_pnl():
    """Ensure backtests produce realistic results"""

    # Run simple buy-and-hold strategy
    results = run_backtest(strategy='buy_atm_hold_1day')

    # Check PnL is realistic
    assert results['max_single_day_return'] < 0.50  # <50% max
    assert results['avg_daily_return'] < 0.10        # <10% average
    assert results['theta_decay_visible'] == True    # Time decay present
```

---

## 7. PERFORMANCE SPECIFICATIONS

### 7.1 Generation Performance

| Metric | Target | Maximum |
|--------|--------|---------|
| 1 day generation | <2 sec | 5 sec |
| 1 month generation | <30 sec | 60 sec |
| 3 months generation | <90 sec | 180 sec |
| Memory usage | <2 GB | 4 GB |
| CPU cores | 1 | 4 |

### 7.2 Data Quality Performance

| Metric | Target | Minimum |
|--------|--------|---------|
| Structural validity | 100% | 100% |
| Greeks accuracy | 99.9% | 99% |
| Price continuity R² | >0.95 | >0.90 |
| Max 5-min move | <10% | <15% |
| Validation pass rate | 100% | 95% |

---

## 8. MIGRATION FROM V1/V5

### 8.1 Breaking Changes

1. **Data Structure**: No more duplicate timestamps
2. **Price Logic**: Greeks-based evolution, not random
3. **State Management**: Generator maintains history
4. **Validation**: Mandatory before release

### 8.2 Migration Checklist

- [ ] Remove all V5 data files
- [ ] Update generator to V2 architecture
- [ ] Implement state management
- [ ] Add Greeks evolution logic
- [ ] Create validation suite
- [ ] Test with sample data
- [ ] Validate no duplicates
- [ ] Generate full dataset
- [ ] Run backtest validation
- [ ] Deploy to production

---

## 9. SUCCESS CRITERIA

The V2 generator will be considered successful when:

1. ✅ **Zero duplicate timestamps** - Groupby check returns max count of 1
2. ✅ **Price continuity R² > 0.95** - High autocorrelation
3. ✅ **Max 5-min move < 10%** - No extreme jumps
4. ✅ **100% Greeks validity** - All constraints met
5. ✅ **Realistic backtest results** - No impossible profits
6. ✅ **All validations pass** - 100% pass rate
7. ✅ **Theta decay visible** - Time decay in prices
8. ✅ **State maintained** - Prices evolve from previous

---

## 10. APPENDIX: Key Formulas

### Black-Scholes Greeks

```python
# Delta
delta_call = exp(-q*T) * N(d1)
delta_put = -exp(-q*T) * N(-d1)

# Gamma (same for calls and puts)
gamma = exp(-q*T) * n(d1) / (S * sigma * sqrt(T))

# Theta
theta_call = -(S*n(d1)*sigma*exp(-q*T))/(2*sqrt(T)) - r*K*exp(-r*T)*N(d2) + q*S*exp(-q*T)*N(d1)
theta_put = -(S*n(d1)*sigma*exp(-q*T))/(2*sqrt(T)) + r*K*exp(-r*T)*N(-d2) - q*S*exp(-q*T)*N(-d1)

# Vega (same for calls and puts)
vega = S * exp(-q*T) * n(d1) * sqrt(T)

# Where:
# d1 = (ln(S/K) + (r - q + 0.5*sigma²)*T) / (sigma*sqrt(T))
# d2 = d1 - sigma*sqrt(T)
# N() = Cumulative normal distribution
# n() = Normal probability density
```

---

**Document Version**: 2.0
**Status**: Ready for Implementation
**Priority**: CRITICAL - Fixes fundamental V1/V5 flaws

---

*End of PRD V2.0*