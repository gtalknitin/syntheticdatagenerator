# Synthetic NIFTY Options Data Generation PRD v9.0
## Balanced Trends with Expiry-Specific Option Pricing - 1 Hour Candles

**Product Name**: NIFTY Options Synthetic Data Generator v9.0
**Version**: 9.0
**Date**: October 4, 2025
**Author**: NikAlgoBulls Development Team
**Status**: CRITICAL FIX - Addresses Fundamental Pricing & Trend Issues
**Candle Interval**: 1 Hour (reduced from 5 minutes for efficiency)
**Previous Version**: v8.0-Extended (June 14 - September 30, 2025, 5-min candles)

---

## 1. Executive Summary

### 1.1 Purpose
This document specifies the v9.0 implementation addressing **critical flaws** discovered in v8.0 synthetic data that made it unsuitable for strategy testing:

**Critical Issues Fixed**:
1. ❌ **Bullish trend bias** (81% bullish weeks caused unidirectional strategy signals)
2. ❌ **Shared pricing model** (same option data used for all expiries regardless of TTE)
3. ❌ **Delta-distance mismatch** (0.1 delta at wrong strikes for weekly vs monthly)
4. ❌ **Unrealistic weekly premiums** (too low due to incorrect TTE application)
5. ❌ **Zero weekly hedge profit** (wrong strikes + wrong premiums)
6. ✅ **Reduced to 1-hour candles** (7 timestamps/day vs 79, faster generation, smaller files)

### 1.2 Root Cause Analysis

**Problem**: v8.0 generated option prices once per strike, then applied to ALL expiries:
```python
# V8.0 FLAW:
base_price = calculate_price(spot, strike, tte_days, vix)  # Single TTE
for expiry in all_expiries:
    # Same base_price used! Wrong TTE for most expiries!
    data.append([..., base_price, ...])
```

**Impact**:
- Weekly options (1-7 days TTE) priced as if 30 days TTE
- Monthly options (30+ days TTE) priced incorrectly
- Delta values wrong for each expiry type
- 0.1 delta for weekly at ~1300 points vs monthly at ~2300 points (should be ~1000 vs ~2000)

### 1.3 Critical Improvements (v9.0)

**NEW ARCHITECTURE**:
1. ✅ **Expiry-Specific Pricing**: Each expiry calculates independent option prices based on actual TTE
2. ✅ **Balanced Trend Generation**: 50/50 bullish/bearish weeks (not 81% bullish)
3. ✅ **Realistic Delta-Distance**: 0.1 delta weekly ~1000 pts, monthly ~2000 pts from ATM
4. ✅ **Proper Weekly Premiums**: Based on short TTE (1-7 days), not monthly TTE
5. ✅ **Meaningful Weekly Hedges**: Correct strikes + correct premiums = real P&L contribution
6. ✅ **Bidirectional Testing**: Strategy tested in both bullish and bearish market conditions

**RETAINED FROM V8**:
- Extended date range (June 14 - September 30, 2025)
- Broad strike coverage (₹22,000 - ₹30,000, 161 strikes)
- VIX data integration
- Complete Greeks calculations
- Validation framework

---

## 2. Critical Fixes - Technical Specification

### 2.1 Fix #1: Balanced Trend Generation

**V8.0 Issue**:
```python
# Random walk caused 81% bullish weeks
trend = np.cumsum(np.random.randn(num_candles) * 5)  # Cumulative = bias!
```

**V9.0 Solution**:
```python
def generate_balanced_price_movement(start_date, end_date):
    """
    Generate underlying price with enforced balanced weekly trends
    """
    weeks = []

    # Pre-define weekly directions (50/50 split)
    total_weeks = 16  # Jun 16 - Sep 30
    bullish_weeks = 8
    bearish_weeks = 8

    # Randomize order but maintain balance
    directions = ['UP'] * bullish_weeks + ['DOWN'] * bearish_weeks
    np.random.shuffle(directions)

    # Generate price movement for each week
    current_price = 25400  # Starting NIFTY
    for week_num, direction in enumerate(directions):
        if direction == 'UP':
            # Bullish week: +0.5% to +1.5% for the week
            week_change_pct = np.random.uniform(0.005, 0.015)
        else:
            # Bearish week: -0.5% to -1.5% for the week
            week_change_pct = np.random.uniform(-0.015, -0.005)

        # Generate daily movements within weekly constraint
        daily_prices = generate_week_with_constraint(
            current_price,
            week_change_pct,
            num_days=5
        )

        weeks.append(daily_prices)
        current_price = daily_prices[-1]

    return concatenate_weeks(weeks)

def generate_week_with_constraint(start_price, target_change_pct, num_days):
    """
    Generate daily prices that achieve target weekly change
    while having daily volatility
    """
    target_end_price = start_price * (1 + target_change_pct)

    # Generate random daily changes that sum to target
    daily_changes = generate_constrained_walk(num_days, target_change_pct)

    prices = [start_price]
    for change in daily_changes:
        prices.append(prices[-1] * (1 + change))

    # Adjust final day to hit exact target
    prices[-1] = target_end_price

    return prices
```

**Validation**:
```python
TREND_VALIDATION_V9 = {
    "weekly_balance": {
        "up_weeks_min": 6,      # 37.5% minimum
        "up_weeks_max": 10,     # 62.5% maximum
        "down_weeks_min": 6,
        "down_weeks_max": 10,
        "target_ratio": 0.5     # 50/50 ideal
    },
    "daily_balance": {
        "up_days_min": 25,      # 35% minimum
        "up_days_max": 45,      # 55% maximum
        "total_days": 76
    },
    "bias_score": {
        "max_acceptable": 1,    # Was +2 in v8.0
        "checks": [
            "overall_change_pct < 1.5%",
            "weekly_ratio in [0.4, 0.6]",
            "no_streaks > 3_weeks"
        ]
    }
}
```

### 2.2 Fix #2: Expiry-Specific Option Pricing

**V8.0 Issue**:
```python
# Single pricing for all expiries
base_price = calculate_base_price(spot, strike, tte_days, vix)
prices = generate_intraday_prices(base_price, timestamps, vix)

for expiry_date, expiry_type in active_expiries:
    # WRONG: Same prices used for 1-day weekly and 30-day monthly!
    for price in prices:
        data.append([..., price, ...])
```

**V9.0 Solution**:
```python
def generate_day_data(self, date):
    """
    Generate options data with expiry-specific pricing
    """
    data = []

    # Get active expiries
    active_expiries = self.get_active_expiries(date)

    # Generate data FOR EACH EXPIRY INDEPENDENTLY
    for expiry_date, expiry_type in active_expiries:

        # Calculate actual TTE for THIS expiry
        tte_days = (expiry_date - date).days

        # Generate prices SPECIFIC to this expiry's TTE
        for strike in strikes:
            for option_type in ['CE', 'PE']:

                # CRITICAL: Price calculated with THIS expiry's TTE
                prices_for_this_expiry = self.generate_option_prices(
                    spot_prices=spot_prices,
                    strike=strike,
                    tte_days=tte_days,        # Correct TTE!
                    option_type=option_type,
                    expiry_type=expiry_type,  # Used for IV adjustment
                    vix_values=vix_values
                )

                # Each expiry gets its own pricing
                for i, (timestamp, spot, price, vix) in enumerate(
                    zip(timestamps, spot_prices, prices_for_this_expiry, vix_values)
                ):
                    # Greeks calculated with correct TTE
                    greeks = self.calculate_greeks(
                        spot=spot,
                        strike=strike,
                        tte_days=tte_days - (i / self.timestamps_per_day),
                        iv=self.get_iv(spot, strike, tte_days, expiry_type, vix),
                        option_type=option_type
                    )

                    data.append([
                        timestamp, strike, option_type,
                        expiry_date, expiry_type,
                        price, greeks['delta'], ...
                    ])

    return pd.DataFrame(data)
```

### 2.3 Fix #3: Expiry-Type-Aware Volatility

**V9.0 Enhancement**:
```python
def get_iv_for_expiry_type(self, spot, strike, tte_days, expiry_type, vix):
    """
    Calculate IV with expiry-type-specific adjustments
    """
    # Base IV from VIX
    base_iv = vix / 100.0

    # Volatility smile (same as before)
    moneyness = spot / strike
    if abs(moneyness - 1) > 0.05:
        smile_adjustment = abs(moneyness - 1) * 0.2
        iv = base_iv * (1 + smile_adjustment)
    else:
        iv = base_iv

    # NEW: Term structure adjustment based on expiry type
    if expiry_type == 'weekly':
        # Weekly options: Higher IV for very short TTE
        if tte_days <= 3:
            iv *= 1.20  # +20% for short-dated options
        elif tte_days <= 7:
            iv *= 1.10  # +10% for weekly
    else:  # monthly
        # Monthly options: Standard term structure
        if tte_days <= 7:
            iv *= 1.15
        elif tte_days <= 15:
            iv *= 1.08
        elif tte_days <= 30:
            iv *= 1.05
        # else: base IV for longer dated

    return iv
```

### 2.4 Fix #4: Delta-Distance Validation

**V9.0 Validation**:
```python
def validate_delta_distance_by_expiry(df, date):
    """
    Validate delta-strike relationships are correct for each expiry type
    """
    spot = df['underlying_price'].iloc[0]

    # Weekly options (1-7 days TTE)
    weekly = df[df['expiry_type'] == 'weekly']
    weekly_ce_01delta = weekly[
        (weekly['option_type'] == 'CE') &
        (weekly['delta'].abs() >= 0.08) &
        (weekly['delta'].abs() <= 0.12)
    ]

    if len(weekly_ce_01delta) > 0:
        weekly_distance = weekly_ce_01delta.iloc[0]['strike'] - spot

        # Weekly 0.1 delta should be ~800-1200 points from ATM
        assert 800 <= weekly_distance <= 1400, \
            f"Weekly 0.1Δ CE at {weekly_distance:.0f} pts (expected 800-1400)"

        # Premium should be meaningful (₹40-100 for Nifty)
        weekly_premium = weekly_ce_01delta.iloc[0]['close']
        assert 30 <= weekly_premium <= 150, \
            f"Weekly 0.1Δ premium ₹{weekly_premium:.2f} unrealistic"

    # Monthly options (20-35 days TTE)
    monthly = df[df['expiry_type'] == 'monthly']
    monthly_ce_01delta = monthly[
        (monthly['option_type'] == 'CE') &
        (monthly['delta'].abs() >= 0.08) &
        (monthly['delta'].abs() <= 0.12)
    ]

    if len(monthly_ce_01delta) > 0:
        monthly_distance = monthly_ce_01delta.iloc[0]['strike'] - spot

        # Monthly 0.1 delta should be ~1800-2400 points from ATM
        assert 1800 <= monthly_distance <= 2600, \
            f"Monthly 0.1Δ CE at {monthly_distance:.0f} pts (expected 1800-2600)"

        # Premium should be higher than weekly (more time value)
        monthly_premium = monthly_ce_01delta.iloc[0]['close']
        assert 50 <= monthly_premium <= 200, \
            f"Monthly 0.1Δ premium ₹{monthly_premium:.2f} unrealistic"

        # Monthly premium should be > weekly premium
        if len(weekly_ce_01delta) > 0:
            assert monthly_premium > weekly_premium * 1.3, \
                f"Monthly premium (₹{monthly_premium:.2f}) should be >1.3x weekly (₹{weekly_premium:.2f})"

    return True
```

---

## 3. Data Architecture Changes

### 3.1 V8.0 vs V9.0 Data Generation Flow

**V8.0 (FLAWED)**:
```
For each day:
  ├─ Generate spot prices (79 timestamps)
  ├─ Get active expiries [weekly1, weekly2, monthly]
  ├─ For each strike:
  │   ├─ Calculate ONE base price (using arbitrary TTE)
  │   ├─ Generate intraday variations
  │   └─ For each expiry:
  │       └─ Use SAME prices (WRONG!)
  └─ Save to CSV
```

**V9.0 (CORRECT)**:
```
For each day:
  ├─ Generate spot prices (79 timestamps)
  ├─ Get active expiries [weekly1, weekly2, monthly]
  └─ For each expiry:  ← MOVED UP!
      ├─ Calculate TTE for THIS expiry
      ├─ For each strike:
      │   ├─ Calculate option price with CORRECT TTE
      │   ├─ Calculate Greeks with CORRECT TTE
      │   ├─ Calculate IV based on expiry type
      │   └─ Generate intraday variations
      └─ Append to data
  └─ Save to CSV
```

### 3.2 Performance Considerations

**Challenge**: More calculations (each expiry priced independently)

**V9.0 Optimizations**:
```python
class OptimizedV9Generator:
    def __init__(self):
        # Pre-calculate VIX adjustments
        self.vix_cache = {}

        # Vectorize Black-Scholes
        self.use_numpy_arrays = True

        # Batch similar TTEs
        self.batch_size = 1000

    def generate_batch_prices(self, strikes, tte_days, expiry_type, vix):
        """
        Vectorized pricing for entire strike chain at once
        """
        strikes_array = np.array(strikes)

        # Vectorized Black-Scholes
        ce_prices = self.vectorized_bs_call(
            spot=self.current_spot,
            strikes=strikes_array,
            tte=tte_days,
            iv=self.get_iv_vector(strikes_array, tte_days, expiry_type, vix)
        )

        pe_prices = self.vectorized_bs_put(
            spot=self.current_spot,
            strikes=strikes_array,
            tte=tte_days,
            iv=self.get_iv_vector(strikes_array, tte_days, expiry_type, vix)
        )

        return ce_prices, pe_prices
```

---

## 4. Enhanced Validation Framework

### 4.1 Critical Validations (NEW)

```python
VALIDATION_SUITE_V9 = {
    # V8 validations (retained)
    "date_validation": {...},
    "strike_validation": {...},
    "vix_validation": {...},

    # NEW: Trend balance validations
    "trend_validation": {
        "weekly_balance": lambda df: validate_weekly_balance(df),
        "up_weeks_range": lambda df: 6 <= count_up_weeks(df) <= 10,
        "down_weeks_range": lambda df: 6 <= count_down_weeks(df) <= 10,
        "no_long_streaks": lambda df: max_streak_length(df) <= 3,
        "bias_score": lambda df: calculate_bias_score(df) <= 1
    },

    # NEW: Expiry-specific validations
    "expiry_pricing_validation": {
        "weekly_delta_distance": lambda df: validate_weekly_delta_distance(df),
        "monthly_delta_distance": lambda df: validate_monthly_delta_distance(df),
        "weekly_premiums_realistic": lambda df: validate_weekly_premiums(df),
        "monthly_premiums_higher": lambda df: monthly_premium_ratio(df) > 1.3,
        "tte_correctness": lambda df: verify_tte_in_prices(df)
    },

    # NEW: Strategy testing readiness
    "strategy_validation": {
        "bullish_weeks_exist": lambda df: count_up_weeks(df) >= 6,
        "bearish_weeks_exist": lambda df: count_down_weeks(df) >= 6,
        "weekly_hedge_strikes": lambda df: validate_hedge_strikes(df),
        "hedge_premiums_meaningful": lambda df: validate_hedge_pnl(df)
    }
}
```

### 4.2 Delta-Distance Validation Matrix

```python
DELTA_DISTANCE_MATRIX_V9 = {
    "weekly_options": {
        # 1-7 days TTE
        "0.10_delta_CE": {
            "min_distance": 800,    # Points from ATM
            "max_distance": 1400,
            "min_premium": 30,      # Rupees
            "max_premium": 150
        },
        "0.10_delta_PE": {
            "min_distance": 800,
            "max_distance": 1400,
            "min_premium": 30,
            "max_premium": 150
        }
    },

    "monthly_options": {
        # 20-35 days TTE
        "0.10_delta_CE": {
            "min_distance": 1800,   # Points from ATM
            "max_distance": 2600,
            "min_premium": 50,      # Rupees
            "max_premium": 200
        },
        "0.10_delta_PE": {
            "min_distance": 1800,
            "max_distance": 2600,
            "min_premium": 50,
            "max_premium": 200
        }
    },

    "premium_ratios": {
        "monthly_to_weekly": {
            "min_ratio": 1.3,       # Monthly should be 1.3x-2.5x weekly
            "max_ratio": 2.5,
            "typical": 1.8
        }
    }
}
```

---

## 5. Testing Scenarios Enabled

### 5.1 Balanced Strategy Testing

**V8.0 Problem**: 81% bullish weeks → Strategy only tested Bull Call Spreads

**V9.0 Solution**: 50/50 weeks → Strategy tests BOTH directions

```python
TESTING_COVERAGE_V9 = {
    "market_conditions": {
        "bullish_weeks": 8,              # 50% - Test Bull Call Spreads
        "bearish_weeks": 8,              # 50% - Test Bear Put Spreads
        "bullish_entries": "~8-10",      # Wednesdays in up weeks
        "bearish_entries": "~8-10"       # Wednesdays in down weeks
    },

    "monthly_positions": {
        "bull_call_spreads": "~8 entries",
        "bear_put_spreads": "~8 entries",
        "profit_targets": "50-70% tested in both",
        "stop_losses": "5% tested in both"
    },

    "weekly_hedges": {
        "bear_call_spreads": "~8 hedges (when monthly bullish)",
        "bull_put_spreads": "~8 hedges (when monthly bearish)",
        "delta_01_strikes": "Correct for weekly TTE",
        "premiums": "Realistic (₹40-100 range)",
        "pnl_contribution": "Meaningful (not zero)"
    }
}
```

### 5.2 Weekly Hedge Realism Test

**Example: Bullish Monthly + Bearish Weekly Hedge**

```python
# Date: June 18, 2025 (Wednesday)
# Spot: 25,400
# Weekly expiry: June 18 (0 days TTE at EOD)
# Monthly expiry: June 26 (8 days TTE)

monthly_position = {
    "type": "bull_call_spread",
    "long_strike": 25400,   # ATM CE (monthly)
    "short_strike": 25650,  # 250 pts OTM
    "long_premium": 180,    # 8 days TTE → realistic
    "short_premium": 90,    # 8 days TTE → realistic
    "net_debit": 90
}

weekly_hedge = {
    "type": "bear_call_spread",
    "short_strike": 26400,  # ~1000 pts OTM = 0.1 delta (weekly!)
    "long_strike": 26650,   # Protection
    "short_premium": 50,    # 0 days TTE → CORRECT (not ₹5!)
    "long_premium": 25,     # 0 days TTE → CORRECT
    "net_credit": 25
}

# V8.0: weekly_hedge.short_premium = ₹5 (WRONG - used monthly TTE)
# V9.0: weekly_hedge.short_premium = ₹50 (CORRECT - uses weekly TTE)
```

---

## 6. Implementation Specification

### 6.1 Core Generator Class

```python
class V9BalancedGenerator:
    """
    V9.0 Generator with balanced trends and expiry-specific pricing
    """

    def __init__(self):
        self.config = {
            # Dates (same as V8)
            'start_date': '2025-06-14',
            'end_date': '2025-09-30',

            # Strikes (same as V8)
            'min_strike': 22000,
            'max_strike': 30000,
            'strike_interval': 50,

            # NEW: Trend balance enforcement
            'enforce_balanced_trends': True,
            'target_up_weeks': 8,
            'target_down_weeks': 8,
            'max_streak': 3,

            # NEW: Expiry-specific pricing
            'independent_expiry_pricing': True,
            'weekly_iv_boost': 1.15,
            'monthly_iv_base': 1.0
        }

    def generate_all_data(self):
        """Main generation loop"""

        # Step 1: Generate balanced underlying price movement
        price_series = self.generate_balanced_price_series(
            start_date=self.start_date,
            end_date=self.end_date,
            target_up_weeks=8,
            target_down_weeks=8
        )

        # Step 2: Generate options data day by day
        for date in trading_dates:
            spot_prices = price_series[date]

            df_day = self.generate_day_data_v9(
                date=date,
                spot_prices=spot_prices,
                method='expiry_specific'  # NEW!
            )

            # Validate day
            self.validate_day_v9(df_day, date)

            # Save
            self.save_day_data(df_day, date)

    def generate_balanced_price_series(self, start_date, end_date,
                                       target_up_weeks, target_down_weeks):
        """
        Generate price movement with enforced weekly balance
        """
        # Pre-define weekly directions
        directions = ['UP'] * target_up_weeks + ['DOWN'] * target_down_weeks
        random.shuffle(directions)

        # Generate constrained weekly movements
        price_series = {}
        current_price = 25400

        for week_num, direction in enumerate(directions):
            week_data = self.generate_balanced_week(
                start_price=current_price,
                direction=direction,
                week_num=week_num
            )

            price_series.update(week_data)
            current_price = week_data[list(week_data.keys())[-1]][-1]

        return price_series

    def generate_day_data_v9(self, date, spot_prices, method='expiry_specific'):
        """
        Generate options data with expiry-specific pricing
        """
        data = []
        active_expiries = self.get_active_expiries(date)

        # CRITICAL: Loop through expiries FIRST
        for expiry_date, expiry_type in active_expiries:
            tte_days = (expiry_date - date).days

            # Generate for all strikes for THIS expiry
            for strike in self.strikes:
                for option_type in ['CE', 'PE']:

                    # Price with CORRECT TTE for this expiry
                    option_data = self.generate_option_timeseries(
                        date=date,
                        strike=strike,
                        option_type=option_type,
                        expiry_date=expiry_date,
                        expiry_type=expiry_type,
                        tte_days=tte_days,
                        spot_prices=spot_prices
                    )

                    data.extend(option_data)

        return pd.DataFrame(data, columns=self.schema)
```

### 6.2 Balanced Week Generation

```python
def generate_balanced_week(self, start_price, direction, week_num):
    """
    Generate one week of trading with target direction
    """
    if direction == 'UP':
        # Bullish week: +0.5% to +1.5% total
        target_change = random.uniform(0.005, 0.015)
    else:
        # Bearish week: -0.5% to -1.5% total
        target_change = random.uniform(-0.015, -0.005)

    target_end = start_price * (1 + target_change)

    # Generate 5 days with intraday volatility
    daily_changes = self.distribute_change_across_days(
        total_change=target_change,
        num_days=5
    )

    week_data = {}
    current = start_price

    for day_offset, daily_change in enumerate(daily_changes):
        day_date = self.start_date + timedelta(weeks=week_num, days=day_offset)

        if day_date.weekday() < 5:  # Weekday
            intraday_prices = self.generate_intraday_prices(
                open_price=current,
                close_price=current * (1 + daily_change),
                num_candles=79
            )

            week_data[day_date] = intraday_prices
            current = intraday_prices[-1]

    # Ensure exact target hit
    last_date = list(week_data.keys())[-1]
    week_data[last_date][-1] = target_end

    return week_data
```

---

## 7. Success Criteria

### 7.1 Mandatory Requirements (V9.0)

| Requirement | Specification | Validation | Priority |
|-------------|--------------|------------|----------|
| **Balanced Trends** | 6-10 up weeks, 6-10 down weeks | Weekly ratio check | P0 |
| **Expiry-Specific Pricing** | Each expiry priced with correct TTE | TTE validation | P0 |
| **Weekly 0.1Δ Distance** | 800-1400 pts from ATM | Delta-distance check | P0 |
| **Monthly 0.1Δ Distance** | 1800-2600 pts from ATM | Delta-distance check | P0 |
| **Weekly Premiums** | ₹30-150 for 0.1Δ | Premium range check | P0 |
| **Monthly Premiums** | ₹50-200 for 0.1Δ | Premium range check | P0 |
| **Premium Ratio** | Monthly > 1.3x Weekly | Ratio validation | P0 |
| **Bias Score** | ≤ 1 (was +2 in V8) | Multi-factor check | P0 |
| **Hedge Profitability** | Non-zero P&L contribution | Backtest validation | P0 |

### 7.2 Data Quality Metrics

| Metric | V8.0 Actual | V9.0 Target | Status |
|--------|-------------|-------------|--------|
| Up weeks | 81% (13/16) | 50% (8/16) | ❌→✅ |
| Down weeks | 19% (3/16) | 50% (8/16) | ❌→✅ |
| Weekly 0.1Δ distance | ~1305 pts | ~1000 pts | ❌→✅ |
| Monthly 0.1Δ distance | ~2305 pts | ~2000 pts | ⚠️→✅ |
| Weekly 0.1Δ premium | ~₹51 | ~₹60 | ⚠️→✅ |
| Monthly 0.1Δ premium | ~₹69 | ~₹110 | ❌→✅ |
| Premium ratio | 1.35x | 1.8x | ❌→✅ |
| Bias score | +2 | ≤1 | ❌→✅ |

---

## 8. Migration from V8.0

### 8.1 Breaking Changes

```python
BREAKING_CHANGES_V9 = {
    "pricing_model": {
        "v8": "Shared pricing across expiries",
        "v9": "Expiry-specific pricing",
        "impact": "All option prices change",
        "action": "Regenerate all data"
    },

    "trend_generation": {
        "v8": "Random walk (biased)",
        "v9": "Balanced constrained walk",
        "impact": "Underlying prices change",
        "action": "Regenerate all data"
    },

    "delta_distances": {
        "v8": "Inconsistent (wrong TTE)",
        "v9": "Expiry-type consistent",
        "impact": "Strike selection changes",
        "action": "Update strategy strike logic"
    }
}
```

### 8.2 Migration Checklist

- [ ] Backup V8.0 data to `archive/intraday_v8_extended/`
- [ ] Update generator to V9.0 architecture
- [ ] Implement balanced trend generation
- [ ] Implement expiry-specific pricing loop
- [ ] Add V9.0-specific validators
- [ ] Generate new data to `intraday_v9_balanced/`
- [ ] Validate all 161 strikes present
- [ ] Validate balanced weekly trends (6-10 up, 6-10 down)
- [ ] Validate weekly delta-distances (800-1400 pts)
- [ ] Validate monthly delta-distances (1800-2600 pts)
- [ ] Validate premium ratios (monthly > 1.3x weekly)
- [ ] Run sample strategy backtest
- [ ] Verify weekly hedges show P&L contribution
- [ ] Update strategy code to use V9 data
- [ ] Document lessons learned

---

## 9. Validation Test Cases

### 9.1 Trend Balance Tests

```python
def test_trend_balance():
    """Test balanced trend generation"""
    df = load_all_v9_data()

    # Test 1: Weekly balance
    weekly = df.resample('W', on='timestamp')['underlying_price'].agg(['first', 'last'])
    weekly['change'] = weekly['last'] - weekly['first']
    up_weeks = (weekly['change'] > 0).sum()
    down_weeks = (weekly['change'] < 0).sum()

    assert 6 <= up_weeks <= 10, f"Up weeks {up_weeks} not in [6,10]"
    assert 6 <= down_weeks <= 10, f"Down weeks {down_weeks} not in [6,10]"

    # Test 2: No long streaks
    max_up_streak = calculate_max_streak(weekly['change'] > 0)
    max_down_streak = calculate_max_streak(weekly['change'] < 0)

    assert max_up_streak <= 3, f"Bullish streak {max_up_streak} too long"
    assert max_down_streak <= 3, f"Bearish streak {max_down_streak} too long"

    # Test 3: Bias score
    bias_score = calculate_bias_score(df)
    assert bias_score <= 1, f"Bias score {bias_score} too high"
```

### 9.2 Expiry-Specific Pricing Tests

```python
def test_expiry_specific_pricing():
    """Test that weekly and monthly options priced differently"""
    df = load_day_v9('2025-06-18')

    spot = df['underlying_price'].iloc[0]

    # Get 0.1 delta strikes for weekly
    weekly = df[df['expiry_type'] == 'weekly']
    weekly_01 = weekly[
        (weekly['option_type'] == 'CE') &
        (weekly['delta'].between(0.08, 0.12))
    ].iloc[0]

    # Get 0.1 delta strikes for monthly
    monthly = df[df['expiry_type'] == 'monthly']
    monthly_01 = monthly[
        (monthly['option_type'] == 'CE') &
        (monthly['delta'].between(0.08, 0.12))
    ].iloc[0]

    # Test 1: Different strikes (weekly closer)
    weekly_dist = weekly_01['strike'] - spot
    monthly_dist = monthly_01['strike'] - spot

    assert 800 <= weekly_dist <= 1400, f"Weekly 0.1Δ at {weekly_dist} pts"
    assert 1800 <= monthly_dist <= 2600, f"Monthly 0.1Δ at {monthly_dist} pts"
    assert monthly_dist > weekly_dist * 1.5, "Monthly not far enough"

    # Test 2: Different premiums (monthly higher)
    weekly_prem = weekly_01['close']
    monthly_prem = monthly_01['close']

    assert 30 <= weekly_prem <= 150, f"Weekly premium ₹{weekly_prem}"
    assert 50 <= monthly_prem <= 200, f"Monthly premium ₹{monthly_prem}"
    assert monthly_prem > weekly_prem * 1.3, \
        f"Monthly (₹{monthly_prem}) not > 1.3x weekly (₹{weekly_prem})"
```

### 9.3 Strategy Readiness Tests

```python
def test_strategy_readiness():
    """Test data supports bidirectional strategy testing"""
    df = load_all_v9_data()

    # Test 1: Both directions available
    up_weeks = count_bullish_weeks(df)
    down_weeks = count_bearish_weeks(df)

    assert up_weeks >= 6, f"Only {up_weeks} bullish weeks"
    assert down_weeks >= 6, f"Only {down_weeks} bearish weeks"

    # Test 2: Weekly hedges have meaningful premiums
    for date in get_wednesdays(df):
        day_df = df[df['timestamp'].dt.date == date]

        # Check weekly 0.1 delta options exist with good premiums
        weekly_hedges = find_weekly_hedge_strikes(day_df)

        for hedge in weekly_hedges:
            assert hedge['premium'] >= 30, \
                f"Weak hedge premium ₹{hedge['premium']} on {date}"

    # Test 3: Simulated backtest shows hedge P&L
    results = run_simple_strategy_test(df)

    monthly_pnl = results['monthly_total_pnl']
    weekly_pnl = results['weekly_total_pnl']

    assert abs(weekly_pnl) > 0, "Weekly hedges show zero P&L!"
    assert abs(weekly_pnl / monthly_pnl) > 0.05, \
        "Weekly hedges contribute < 5% (too low)"
```

---

## 10. File Structure

### 10.1 Directory Organization

```
zerodha_strategy/data/synthetic/
├── hourly_v9_balanced/                # NEW: V9 production (1-hour candles)
│   ├── NIFTY_OPTIONS_1H_20250616.csv
│   ├── NIFTY_OPTIONS_1H_20250617.csv
│   ├── ...
│   ├── NIFTY_OPTIONS_1H_20250930.csv
│   └── metadata/
│       ├── generation_info.json
│       ├── validation_report.json    # NEW: Detailed validation
│       ├── trend_analysis.json       # NEW: Trend stats
│       └── delta_distance_report.json # NEW: Delta validation
│
├── intraday_v8_extended/              # Previous version
│   └── [archived - bullish bias]
│
└── archive/
    ├── intraday_v7_vix/
    └── [older versions]
```

### 10.2 Enhanced Metadata

```json
{
  "version": "9.0-Balanced",
  "generation_date": "2025-10-04 XX:XX:XX",
  "critical_fixes": [
    "Balanced trend generation (50/50 weekly)",
    "Expiry-specific option pricing",
    "Correct delta-distance relationships",
    "Realistic weekly option premiums",
    "Meaningful weekly hedge P&L"
  ],
  "trend_stats": {
    "up_weeks": 8,
    "down_weeks": 8,
    "up_week_pct": 50.0,
    "max_bullish_streak": 3,
    "max_bearish_streak": 3,
    "bias_score": 0,
    "overall_change_pct": 0.85
  },
  "delta_validation": {
    "weekly_01_delta_CE": {
      "avg_distance": 1050,
      "distance_range": [880, 1280],
      "avg_premium": 62,
      "premium_range": [38, 95]
    },
    "monthly_01_delta_CE": {
      "avg_distance": 2100,
      "distance_range": [1920, 2380],
      "avg_premium": 115,
      "premium_range": [78, 168]
    },
    "premium_ratio": 1.85
  },
  "expiry_coverage": {
    "weekly_expiries": 15,
    "monthly_expiries": 4,
    "expiries_with_correct_pricing": "all",
    "independent_tte_calculations": true
  }
}
```

---

## 11. Performance Impact Analysis

### 11.1 Computational Cost

**V8.0**:
- Option pricing: 1 calculation per strike × 161 strikes = 161 calcs
- Total per day: 161 × 2 (CE/PE) × 79 timestamps = ~25,400 calcs

**V9.0**:
- Option pricing: 1 calculation per strike × per expiry
- Expiries per day: ~6 (3 weekly + 3 monthly)
- Total per day: 161 × 2 × 6 × 7 = ~13,500 calcs (faster than V8!)

**Optimization Strategy**:
```python
# Vectorize Black-Scholes for entire strike chain
def optimized_v9_pricing(strikes, expiry_date, expiry_type):
    """
    Vectorized pricing - calculate all strikes at once
    """
    tte = (expiry_date - current_date).days
    strikes_array = np.array(strikes)

    # Vectorized calculations
    ce_prices, ce_greeks = vectorized_bs_call(
        spot=spot_array,
        strikes=strikes_array,
        tte=tte,
        iv=iv_array
    )

    pe_prices, pe_greeks = vectorized_bs_put(...)

    return ce_prices, pe_prices, ce_greeks, pe_greeks

# Expected performance: Actually faster than V8 due to 1-hour candles!
# V8: ~5 min/day (79 timestamps), V9: ~2 min/day (7 timestamps)
# Total: ~2.5 hours for 76 days
```

---

## 12. Risk Mitigation

### 12.1 Data Quality Risks

| Risk | V8.0 Impact | V9.0 Mitigation | Status |
|------|-------------|-----------------|--------|
| Bullish bias | Strategy untested bearish | Enforced 50/50 balance | ✅ Fixed |
| Wrong TTE pricing | Unrealistic premiums | Expiry-specific loops | ✅ Fixed |
| Low weekly premiums | Zero hedge P&L | Correct TTE calculation | ✅ Fixed |
| Delta mismatches | Wrong strike selection | Expiry-aware validation | ✅ Fixed |

### 12.2 Implementation Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|---------|------------|
| Performance issues | Medium | High | Vectorized calculations |
| Validation failures | Low | High | Comprehensive test suite |
| Data inconsistencies | Low | Medium | Expiry-by-expiry validation |
| Storage requirements | Low | Low | Same as V8 (~1.2GB) |

---

## 13. Document History

- v1.0 - v7.0: See V8.0 PRD
- **v8.0**: Extended range, broader strikes (October 3, 2025)
- **v9.0**: Balanced trends, expiry-specific pricing (October 4, 2025)

**Key Learnings**:
1. Random walks create unintended bias → Use constrained generation
2. Shared pricing across expiries → Calculate independently
3. Delta validation must be expiry-specific
4. Weekly options need short-TTE pricing
5. Strategy testing requires bidirectional market data

---

**Status**: 📝 SPECIFICATION COMPLETE - READY FOR IMPLEMENTATION
**Target Location**: `/zerodha_strategy/data/synthetic/hourly_v9_balanced/`
**Generator**: `generate_v9_balanced.py` (to be created)
**Estimated Generation Time**: ~2.5 hours (76 days × ~2 min/day with 1-hour candles)
**Priority**: **P0 CRITICAL** - V8.0 data unsuitable for strategy testing

---

## 14. Appendix: V8 vs V9 Comparison

### 14.1 Side-by-Side Example

**Date: June 18, 2025, 09:15 AM**
**Spot: 25,400**
**Weekly expiry: June 18 (0 days)**
**Monthly expiry: June 26 (8 days)**

| Metric | V8.0 (WRONG) | V9.0 (CORRECT) |
|--------|--------------|----------------|
| **Weekly 0.1Δ CE Strike** | 26,700 | 26,400 |
| **Weekly 0.1Δ CE Distance** | +1,300 pts | +1,000 pts |
| **Weekly 0.1Δ CE Premium** | ₹51 | ₹65 |
| **Monthly 0.1Δ CE Strike** | 27,700 | 27,400 |
| **Monthly 0.1Δ CE Distance** | +2,300 pts | +2,000 pts |
| **Monthly 0.1Δ CE Premium** | ₹69 | ₹118 |
| **Premium Ratio (M/W)** | 1.35x | 1.82x |
| **TTE Used (Weekly)** | Mixed/Wrong | 0 days ✅ |
| **TTE Used (Monthly)** | Mixed/Wrong | 8 days ✅ |

### 14.2 Backtest Impact Estimate

| Metric | V8.0 Result | V9.0 Expected |
|--------|-------------|---------------|
| Bull Call Spreads | ~13 entries | ~8 entries |
| Bear Put Spreads | ~3 entries | ~8 entries |
| Weekly hedge P&L | ~₹0 (broken) | ~₹15-25K total |
| Total strategy P&L | Bullish only | Bidirectional |
| Strategy validation | Incomplete | Complete ✅ |

---

**END OF DOCUMENT**
