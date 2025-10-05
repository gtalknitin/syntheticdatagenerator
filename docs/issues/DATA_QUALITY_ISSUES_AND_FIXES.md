# Data Quality Issues & Fix Plan

**Date**: October 5, 2025
**Reporter**: User's friend
**Status**: Critical - Requires immediate fix

## Identified Issues

### 1. Underlying Price Not Reflecting True Price

**Problem**: Synthetic generators (V7/V8/V9) simulate underlying price movements instead of using actual NIFTY data.

**Impact**:
- All option prices are based on simulated spot prices
- Backtests using this data won't reflect real market conditions
- Strategy performance will be unrealistic

**Current Implementation** (V9):
```python
# generator.py line 105-128
current_price = 25400  # Starting Nifty
# ... generates random walk price movement
```

**Fix**:
- Use real NIFTY 1-minute data from `data/seed/nifty_data_min.csv` (Jan 2024 - Oct 2025)
- 438 trading days, 373 candles/day
- Aggregate to hourly for V9 generator

---

### 2. Greeks Calculations Are Incorrect

**Problem**: Deep ITM options showing unrealistic delta values.

**Example**:
- 23000 Call, 48 hours to expiry
- If spot is ~25000, this is 2000 points ITM
- Current delta: **0.67** ❌
- Expected delta: **~0.95-0.99** ✓

**Root Cause Analysis**:

Looking at `core/base_generator.py` Black-Scholes implementation:

```python
# Line 51-52
delta = np.exp(-self.dividend_yield*T) * norm.cdf(d1)  # For CE
```

This is correct, so the issue likely stems from:

1. **Incorrect IV calculation** (line 89-113):
   - For deep ITM, moneyness > 0, code applies: `iv = atm_iv * (1 - 0.1 * min(moneyness, 0.2))`
   - This REDUCES IV for ITM, which is backwards
   - Should increase IV for OTM, not decrease for ITM

2. **Time to expiry calculation**:
   ```python
   # V9 generator.py line 441
   tte_current = tte_days - (i / self.timestamps_per_day)
   ```
   - If tte_days = 2 (48 hours) and this goes negative, delta gets distorted

3. **Dividend yield parameter**:
   ```python
   # base_generator.py line 24
   self.dividend_yield = 0.015  # 1.5%
   ```
   - This exponential term `np.exp(-0.015 * T)` reduces delta
   - For T = 2/365 = 0.0055, exp(-0.015 * 0.0055) ≈ 0.99992
   - Not the issue

**The Real Problem**: IV smile logic in `generate_iv_smile()` (line 89-113)

```python
if moneyness > 0:  # ITM for CE
    iv = atm_iv * (1 - 0.1 * min(moneyness, 0.2))  # WRONG: Reduces IV
```

For a 23000 CE with spot at 25000:
- moneyness = (25000 - 23000) / 25000 = 0.08
- iv = 0.12 * (1 - 0.1 * 0.08) = 0.12 * 0.992 = 0.119

This is close to ATM IV, so not the main issue.

**Actual Issue**: Let me check the delta calculation more carefully.

For deep ITM Call (K=23000, S=25000, T=48hrs):
- d1 = (ln(25000/23000) + (0.065 + 0.5*0.12²)*(2/365)) / (0.12*sqrt(2/365))
- d1 = (ln(1.087) + 0.065054 * 0.00548) / (0.12 * 0.074)
- d1 = (0.0833 + 0.000356) / 0.00888
- d1 = 9.4

For d1 = 9.4:
- norm.cdf(9.4) ≈ 1.0
- delta = exp(-0.015 * 0.00548) * 1.0 ≈ 0.9999

**This should give delta ≈ 1.0, not 0.67!**

The issue must be in the **IV calculation per expiry type** (V9 generator.py line 302-331):

```python
def get_iv_for_expiry(self, spot, strike, tte_days, expiry_type, vix):
    # ...
    moneyness = spot / strike  # THIS IS WRONG!
```

**FOUND IT!** Line 310:
```python
moneyness = spot / strike  # For S=25000, K=23000: 1.087
if abs(moneyness - 1) > 0.05:  # abs(0.087) = 0.087 > 0.05, TRUE
    smile_adj = abs(moneyness - 1) * 0.2  # 0.087 * 0.2 = 0.0174
    iv = base_iv * (1 + smile_adj)  # Increases IV by 1.74%
```

This increases IV slightly, which would DECREASE delta for deep ITM. But not enough to make it 0.67.

**Need to check actual V9 generated data to see what's happening.**

---

### 3. Volume, OI, Bid Unrealistic for 23000 PE

**Problem**: Options far from ATM showing unrealistic liquidity parameters.

**Example**: 23000 PE (if spot ~25000)
- This is 2000 points OTM
- Should have: LOW volume, WIDE spreads, LOW OI
- Current: Shows high volume/OI?

**Root Cause** (`base_generator.py` line 115-142):

```python
def generate_volume_oi(self, moneyness: float, is_monthly: bool,
                      time_to_expiry: int) -> Tuple[int, int]:
    # ...
    moneyness_factor = np.exp(-10 * moneyness**2)  # Gaussian decay
```

For 23000 PE with S=25000:
- moneyness = (23000 - 25000) / 25000 = -0.08
- moneyness_factor = exp(-10 * 0.0064) = exp(-0.064) = 0.938

This gives 94% of base volume! Should be much lower for 2000 points OTM.

**Fix**: Use absolute moneyness and steeper decay:
```python
abs_moneyness = abs((strike - spot) / spot)  # Distance from ATM
moneyness_factor = np.exp(-20 * abs_moneyness**2)  # Steeper Gaussian
```

For 2000 points OTM:
- abs_moneyness = 0.08
- factor = exp(-20 * 0.0064) = exp(-0.128) = 0.88 (still too high)

Need even steeper:
```python
# Use points-based distance instead
distance_pct = abs(strike - spot) / spot
if distance_pct > 0.05:  # >5% OTM
    moneyness_factor = np.exp(-50 * distance_pct)
```

For 8% OTM: exp(-50 * 0.08) = exp(-4) = 0.018 = 1.8% of base (more realistic)

---

## Fix Implementation Plan

### Phase 1: Use Real Underlying Prices (HIGH PRIORITY)

1. **Modify V9 Generator** to load real NIFTY data:
   ```python
   from synthetic_data_generator.io.seed_data_loader import NiftySeedDataLoader

   # In __init__:
   self.seed_loader = NiftySeedDataLoader()
   self.real_prices = self.seed_loader.get_hourly_data()
   ```

2. **Replace simulated price generation**:
   - Remove `generate_balanced_price_series()`
   - Use `self.real_prices` directly
   - Match dates from seed data (Jan 2024 - Oct 2025)

3. **Benefits**:
   - Underlying prices match reality
   - Volatility patterns match real market
   - Strategy backtests will be realistic

### Phase 2: Fix Greeks Calculations (CRITICAL)

1. **Fix IV Smile** in `base_generator.py`:
   ```python
   def generate_iv_smile(self, moneyness: float, time_to_expiry: int) -> float:
       atm_iv = self.base_iv

       # Use absolute distance from ATM
       abs_moneyness = abs(moneyness)

       # Realistic IV smile: increases for OTM on both sides
       if abs_moneyness < 0.02:  # Near ATM
           iv = atm_iv
       else:  # OTM/ITM wings
           # Quadratic smile
           iv = atm_iv * (1 + 0.5 * abs_moneyness**2 * 100)

       # Term structure
       # ... (keep existing)

       return np.clip(iv, 0.08, 0.50)
   ```

2. **Validate delta calculations**:
   - Add unit tests for known cases:
     - Deep ITM (2000 pts): delta ≈ 0.95-0.99
     - ATM: delta ≈ 0.50
     - Deep OTM (2000 pts): delta ≈ 0.01-0.05

3. **Add Greeks validation** in generator:
   ```python
   def validate_greeks(self, delta, spot, strike, option_type):
       if option_type == 'CE':
           if strike < spot - 2000:  # Deep ITM
               assert delta > 0.90, f"Deep ITM CE delta too low: {delta}"
           elif strike > spot + 2000:  # Deep OTM
               assert delta < 0.10, f"Deep OTM CE delta too high: {delta}"
   ```

### Phase 3: Fix Volume/OI/Bid-Ask (IMPORTANT)

1. **Steeper liquidity decay**:
   ```python
   def generate_volume_oi(self, spot: float, strike: int, ...):
       # Distance in absolute points
       distance = abs(strike - spot)

       # Realistic decay based on distance
       if distance < 100:  # Very near ATM
           decay_factor = 1.0
       elif distance < 500:
           decay_factor = np.exp(-0.003 * distance)  # Gradual
       elif distance < 1000:
           decay_factor = np.exp(-0.008 * distance)  # Faster
       else:  # Far OTM/ITM
           decay_factor = np.exp(-0.015 * distance)  # Very fast

       volume = int(base_volume * decay_factor * ...)
       return max(volume, 10)  # Minimum 10 contracts
   ```

2. **Wider spreads for illiquid strikes**:
   ```python
   def generate_bid_ask_spread(self, price, spot, strike, volume):
       distance_pct = abs(strike - spot) / spot

       if distance_pct > 0.10:  # >10% from ATM
           base_spread_pct = 0.20  # 20% spread
       elif distance_pct > 0.05:
           base_spread_pct = 0.10  # 10% spread
       else:
           base_spread_pct = 0.02  # 2% spread (ATM)

       # Adjust for volume
       if volume < 100:
           base_spread_pct *= 2  # Double spread for low volume
   ```

---

## Testing Strategy

### 1. Unit Tests for Greeks

Create `tests/unit/test_greeks_accuracy.py`:
```python
def test_deep_itm_call_delta():
    """Deep ITM call should have delta near 1.0"""
    gen = SyntheticOptionsDataGenerator()
    result = gen.black_scholes(S=25000, K=23000, T=2/365, r=0.065, sigma=0.12, option_type='CE')
    assert result['delta'] > 0.90, f"Deep ITM delta {result['delta']} should be > 0.90"

def test_deep_otm_call_delta():
    """Deep OTM call should have delta near 0"""
    gen = SyntheticOptionsDataGenerator()
    result = gen.black_scholes(S=25000, K=27000, T=2/365, r=0.065, sigma=0.12, option_type='CE')
    assert result['delta'] < 0.10, f"Deep OTM delta {result['delta']} should be < 0.10"

def test_atm_call_delta():
    """ATM call should have delta near 0.5"""
    gen = SyntheticOptionsDataGenerator()
    result = gen.black_scholes(S=25000, K=25000, T=30/365, r=0.065, sigma=0.12, option_type='CE')
    assert 0.45 < result['delta'] < 0.55, f"ATM delta {result['delta']} should be ~0.50"
```

### 2. Integration Tests with Real Data

Create `tests/integration/test_real_data_generation.py`:
```python
def test_v9_with_real_prices():
    """Test V9 generator using real NIFTY seed data"""
    gen = V9BalancedGenerator(use_real_prices=True)

    # Generate for one day
    date = datetime(2024, 6, 14)
    df = gen.generate_day_data(date)

    # Validate underlying matches seed
    seed = gen.seed_loader.get_date_range("2024-06-14", "2024-06-14")
    assert df['underlying_price'].iloc[0] == pytest.approx(seed['open'].iloc[0], rel=0.01)
```

### 3. Data Quality Report

Generate comparison report:
```bash
python scripts/validation/compare_synthetic_vs_real.py \
  --synthetic data/generated/v9_balanced/hourly/ \
  --real data/seed/nifty_data_min.csv \
  --output docs/reports/validation/v9_vs_real_comparison.md
```

---

## Timeline

- **Week 1**: Phase 1 (Real prices integration) + Phase 2 (Greeks fixes)
- **Week 2**: Phase 3 (Volume/OI fixes) + Testing
- **Week 3**: Validation, documentation, V10 release

---

## Success Metrics

✓ Deep ITM options (>2000 pts): Delta > 0.90
✓ Deep OTM options (>2000 pts): Delta < 0.10
✓ Volume for 10% OTM strikes: < 5% of ATM volume
✓ Bid-ask spread for illiquid strikes: > 10%
✓ Underlying prices match real NIFTY data within 0.1%

---

## Next Steps

1. Review this analysis with team
2. Create feature branch: `fix/data-quality-improvements`
3. Implement Phase 1 (real prices)
4. Add unit tests for Greeks
5. Regenerate V10 data with fixes
6. Compare V9 vs V10 quality metrics
