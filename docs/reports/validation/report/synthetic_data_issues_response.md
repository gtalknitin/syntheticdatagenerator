# Response to Unrealistic Price Movements Report
## Analysis of NikAlgoBulls Synthetic Data Issues

**Document Date**: September 27, 2025  
**Analyzed Dataset**: zerodha_strategy/data/synthetic/intraday_jul_sep_v3/  
**Response To**: unrealistic_price_movements_report.md  
**Author**: NikAlgoBulls Development Team  

---

## Executive Summary

After thorough analysis of our synthetic options data against the reported issues, we confirm that our dataset exhibits similar unrealistic price behavior patterns. While the specific manifestation differs (prices collapse to ₹0.05 instead of ₹0.01), the fundamental issues with option pricing models, theta decay, and expiry behavior are present in our data as well.

## Key Findings in Our Data

### 1. Binary Price Collapse to ₹0.05

Our analysis reveals that instead of the reported ₹0.01/₹0.00 prices, our synthetic data uses ₹0.05 as the minimum price. However, the issue remains fundamentally the same:

| Expiry Date | Total Options at ₹0.05 | Percentage of Dataset |
|-------------|------------------------|----------------------|
| July 3, 2025 | 8,002 | ~35% |
| July 31, 2025 | 8,593 | ~38% |
| August 29, 2025 | 20,008 | ~87% |
| September 26, 2025 | 18,334 | ~80% |

### 2. Lack of Gradual Theta Decay

Our data exhibits the same binary behavior criticized in the report:
- Options either have "normal" value or immediately drop to ₹0.05
- No gradual decay pattern (e.g., 5.00 → 3.00 → 1.00 → 0.50 → 0.05)
- Far OTM options show ₹0.05 pricing even with significant time to expiry

### 3. Specific Examples from Our Data

#### Example 1: July 3 Data Issues
```
Underlying Price: 25034.55
Expiry: July 10 (7 days away)
Issue: ALL PUT options from 20000-24950 strikes show ₹0.05
Expected: These should have time value ranging from ₹0.10 to ₹50+
```

#### Example 2: September 26 Near-Expiry Behavior
```
24500 PE (September 30 expiry - 4 days):
- Shows limited price movement (7.88 to 11.32)
- But 23000 PE and below: All at ₹0.05
- No gradual decay visible
```

### 4. Greeks Inconsistencies

Our data shows additional issues not mentioned in the report:
- Options priced at ₹0.05 have theta = -0.0000
- These same options show delta = -0.0000
- This is impossible - options with time to expiry must have non-zero Greeks

## Comparison with Reported Issues

| Issue Category | Reported Problem | Our Data Problem | Severity |
|----------------|-----------------|------------------|----------|
| Minimum Price | ₹0.01 or ₹0.00 | ₹0.05 | Same issue, different threshold |
| Decay Pattern | 98-100% in 0-1 days | Binary drop to ₹0.05 | Equally severe |
| Theta Modeling | Missing gradual decay | No decay curve | Confirmed |
| Greeks Accuracy | Not mentioned | Zero Greeks for non-zero options | Additional issue |
| Expiry Behavior | All options → 0 | Far OTM → 0.05 immediately | Confirmed |

## Impact on Nikhil's Strategy Backtesting

The data issues have specific implications for our strategy:

### 1. **Monthly Spread Positions**
- **Issue**: Far OTM protective legs show ₹0.05 immediately
- **Impact**: Overestimates net credit received, underestimates risk
- **Example**: Selling 25000/24700 put spread when spot at 25200 would show unrealistic profits

### 2. **Weekly Hedge Positions**
- **Issue**: 0.1 delta hedges collapse to ₹0.05 too quickly
- **Impact**: Hedges appear worthless, inflating strategy returns
- **Real Impact**: Would lose 50-80% of hedge value, not 99%

### 3. **Risk Management**
- **Stop Loss**: 5% position stop loss might never trigger due to binary pricing
- **Position Resets**: 50-70% profit targets hit artificially fast
- **Capital Allocation**: Risk calculations based on false premises

## Root Cause Analysis

### 1. **Simplified Pricing Model**
```python
# Current approach (suspected):
if moneyness < threshold:
    price = 0.05
else:
    price = black_scholes_price()

# Should be:
price = black_scholes_price()
price = max(price, minimum_tick_size)
```

### 2. **Missing Market Microstructure**
- No modeling of market maker behavior
- Absence of pin risk near ATM strikes
- No volatility smile implementation
- Missing bid-ask spread dynamics

### 3. **Expiry Day Modeling**
- Options treated as binary (ITM/OTM) instead of continuous
- No consideration for gamma risk
- Settlement price vs spot price not differentiated

## Recommendations for Immediate Fixes

### 1. **Implement Proper Black-Scholes Throughout**
```python
def calculate_option_price(spot, strike, time_to_expiry, volatility, rate, option_type):
    # Always calculate theoretical price first
    theoretical_price = black_scholes(spot, strike, time_to_expiry, volatility, rate, option_type)
    
    # Apply market realism
    if theoretical_price < 0.05:
        # Don't just set to 0.05, add some randomness
        price = max(0.05, theoretical_price + random.uniform(0, 0.05))
    else:
        price = theoretical_price
    
    return price
```

### 2. **Fix Theta Decay Curves**
- 30+ DTE: Daily decay of 1-3% of premium
- 15-30 DTE: Daily decay of 3-5% of premium  
- 7-15 DTE: Daily decay of 5-10% of premium
- 0-7 DTE: Accelerating decay, but maintain minimum value for time

### 3. **Realistic Expiry Behavior**
- Maintain time value until last 30 minutes
- Model gamma effects near ATM
- Add volatility surge on expiry day
- Implement proper settlement mechanics

### 4. **Market Microstructure**
```python
# Add bid-ask spreads based on moneyness
def get_bid_ask_spread(moneyness, days_to_expiry, volume):
    base_spread = 0.05  # ₹0.05 minimum
    
    # Wider spreads for far OTM
    moneyness_factor = abs(moneyness - 1.0) * 2
    
    # Wider spreads near expiry
    expiry_factor = 1 / (days_to_expiry + 1)
    
    # Liquidity adjustment
    volume_factor = 1 / (volume / 1000 + 1)
    
    spread = base_spread * (1 + moneyness_factor + expiry_factor + volume_factor)
    return min(spread, theoretical_price * 0.10)  # Cap at 10% of price
```

## Validation Framework

To ensure fixes are effective:

### 1. **Price Continuity Tests**
- Track individual option prices over time
- Verify smooth decay curves
- Check no sudden jumps to minimum

### 2. **Greeks Validation**
- Ensure theta matches actual price decay
- Verify delta/gamma relationships
- Cross-check with theoretical models

### 3. **Statistical Tests**
- Distribution of returns should be realistic
- Win rates should align with probability theory
- Average profits should not exceed theoretical maximums

## Timeline for Fixes

| Phase | Action | Timeline |
|-------|--------|----------|
| 1 | Implement proper Black-Scholes | 1 week |
| 2 | Add theta decay curves | 1 week |
| 3 | Fix expiry day behavior | 3 days |
| 4 | Add market microstructure | 1 week |
| 5 | Validation and testing | 3 days |
| **Total** | **Complete overhaul** | **3-4 weeks** |

## Conclusion

We acknowledge that our synthetic data exhibits the same fundamental flaws identified in the external report. The issues are severe enough to render backtest results unreliable and potentially misleading. The binary price behavior, lack of proper theta decay, and unrealistic expiry dynamics must be addressed before any meaningful strategy validation can occur.

### Immediate Actions:
1. **Suspend current backtesting** until data is fixed
2. **Implement proposed fixes** with priority on pricing model
3. **Re-generate entire dataset** with new models
4. **Validate against real market data** where available
5. **Document all assumptions** in generator code

### Long-term Improvements:
1. Consider purchasing historical options data for validation
2. Implement more sophisticated models (Heston, SABR)
3. Add regime-switching capabilities
4. Model market events and volatility clustering

The credibility of our backtesting results depends entirely on the quality of synthetic data. We must prioritize these fixes to ensure our strategy development is based on realistic market conditions.

---

*Acknowledgment: We thank the external analysis team for their detailed report, which has helped identify critical flaws in our synthetic data generation process.*