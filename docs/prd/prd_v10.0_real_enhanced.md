# Synthetic Data Generator PRD v10.0

**Version**: 10.0 - Real Data Enhanced
**Date**: October 5, 2025
**Status**: Specification - Ready for Implementation
**Previous Versions**: V9 (Balanced Hourly), V8 (Extended), V7 (VIX-based)

---

## Executive Summary

V10 represents a **fundamental shift** from simulated to **real-data-enhanced** synthetic options generation. Instead of simulating underlying price movements, V10 uses actual NIFTY 1-minute historical data (Jan 2024 - Oct 2025) as the foundation, generating realistic options data on top of real market movements.

### Key Improvements Over V9

| Issue | V9 Approach | V10 Approach | Impact |
|-------|-------------|--------------|---------|
| **Underlying Prices** | Simulated random walk | Real NIFTY 1-min data aggregated to hourly | 100% realistic price movements |
| **Greeks Accuracy** | Incorrect for deep ITM/OTM | Fixed Black-Scholes with validated deltas | Greeks match real options |
| **VIX Data** | Regime-based simulation | Real India VIX data (sourced separately) | Accurate volatility environment |
| **Volume/OI** | Gaussian decay (too optimistic) | Exponential decay based on distance | Realistic illiquidity for far strikes |
| **Bid-Ask Spreads** | Simple percentage | Distance + volume-based | Realistic transaction costs |

---

## 1. Data Sources

### 1.1 Primary: NIFTY Spot Prices (Seed Data)

**Source**: `data/seed/nifty_data_min.csv`

**Specifications**:
- **Format**: CSV with columns: date, open, high, low, close, volume
- **Frequency**: 1-minute candles
- **Period**: January 1, 2024 to October 3, 2025 (438 trading days)
- **Timezone**: IST (+05:30)
- **Rows**: 163,389 candles (~373 per day)
- **Price Range**: ₹21,137 - ₹26,277

**Aggregation to Hourly**:
```python
# Hour bins: 9:15-10:15, 10:15-11:15, ..., 14:15-15:15
# 7 hourly candles per day (matching V9 structure)

hourly_candles = {
    'H1': '09:15-10:15',
    'H2': '10:15-11:15',
    'H3': '11:15-12:15',
    'H4': '12:15-13:15',
    'H5': '13:15-14:15',
    'H6': '14:15-15:15',
    'H7': '15:15-15:30'  # Last 15 mins
}
```

**OHLCV Aggregation**:
- Open: First 1-min candle's open in the hour
- High: Max of all 1-min highs in the hour
- Low: Min of all 1-min lows in the hour
- Close: Last 1-min candle's close in the hour
- Volume: Sum of all 1-min volumes (though currently 0 in seed data)

### 1.2 Secondary: India VIX Data

**Source**: To be obtained from NSE or financial data provider

**Specifications**:
- **Format**: CSV with columns: date, vix_open, vix_high, vix_low, vix_close
- **Frequency**: Daily (will interpolate for hourly)
- **Period**: Match NIFTY seed data (Jan 2024 - Oct 2025)

**If VIX data unavailable**, fallback to calculated ATM IV:
```python
# Calculate ATM implied volatility from historical price volatility
rolling_window = 20  # days
historical_vol = nifty_returns.rolling(rolling_window).std() * sqrt(252)
implied_vix = historical_vol * 100  # Convert to VIX scale

# Apply realistic bounds
vix = np.clip(implied_vix, 10, 50)
```

**Hourly VIX Interpolation**:
```python
# Linear interpolation within day
# Add intraday variation: +/- 2% random walk
hourly_vix[h] = daily_vix * (1 + np.random.uniform(-0.02, 0.02))
```

---

## 2. Options Data Structure

### 2.1 Output Schema (CSV Format)

```
timestamp,symbol,strike,option_type,expiry,expiry_type,
open,high,low,close,volume,oi,bid,ask,
iv,delta,gamma,theta,vega,
underlying_price,vix
```

**Column Specifications**:

| Column | Type | Example | Description |
|--------|------|---------|-------------|
| timestamp | datetime | 2024-06-14 09:15:00 | Start of hourly candle (IST) |
| symbol | string | NIFTY | Always "NIFTY" |
| strike | int | 25000 | Strike price (₹) |
| option_type | string | CE / PE | Call or Put |
| expiry | date | 2024-06-20 | Expiry date |
| expiry_type | string | weekly / monthly | Contract type |
| open | float | 245.50 | Option premium at candle open (₹) |
| high | float | 248.25 | Highest premium in candle (₹) |
| low | float | 243.10 | Lowest premium in candle (₹) |
| close | float | 246.80 | Option premium at candle close (₹) |
| volume | int | 15420 | Contracts traded (lots) |
| oi | int | 234560 | Open interest (lots) |
| bid | float | 246.00 | Best bid price (₹) |
| ask | float | 247.60 | Best ask price (₹) |
| iv | float | 0.1245 | Implied volatility (decimal) |
| delta | float | 0.5234 | Option delta [-1, 1] |
| gamma | float | 0.000234 | Option gamma |
| theta | float | -12.45 | Option theta (₹/day) |
| vega | float | 45.67 | Option vega (₹ per IV point) |
| underlying_price | float | 25043.25 | NIFTY spot at timestamp (₹) |
| vix | float | 15.67 | India VIX at timestamp |

### 2.2 File Naming Convention

```
NIFTY_OPTIONS_1H_YYYYMMDD.csv
```

Examples:
- `NIFTY_OPTIONS_1H_20240614.csv`
- `NIFTY_OPTIONS_1H_20250930.csv`

### 2.3 Directory Structure

```
data/generated/v10_real_enhanced/
├── hourly/
│   ├── NIFTY_OPTIONS_1H_20240101.csv
│   ├── NIFTY_OPTIONS_1H_20240102.csv
│   └── ... (438 files)
├── metadata/
│   ├── generation_info.json
│   ├── quality_report.json
│   └── greeks_validation.json
└── analysis/
    ├── delta_distribution.png
    ├── volume_profile.png
    └── iv_smile.png
```

---

## 3. Options Specifications

### 3.1 Strike Range

**Range**: ₹18,000 - ₹30,000 (based on seed data price range ₹21,137 - ₹26,277)

**Interval**: ₹50 (standard Nifty options)

**Total Strikes**: (30000 - 18000) / 50 + 1 = **241 strikes**

**Strike Filtering** (to reduce data size):
- Keep ALL strikes within ±500 points of spot (ATM region)
- Keep every 2nd strike (₹100 interval) for ±500 to ±1500 points
- Keep every 4th strike (₹200 interval) beyond ±1500 points

**Example** for spot = ₹25,000:
```
ATM region (24500-25500): All ₹50 strikes = 21 strikes
Near region (23000-24500, 25500-27000): ₹100 strikes = 30 strikes
Far region (18000-23000, 27000-30000): ₹200 strikes = 50 strikes
Total per expiry: ~101 strikes
```

### 3.2 Expiry Schedule

Based on NSE NIFTY options calendar:

**Weekly Expiries**: Every Thursday
**Monthly Expiries**: Last Thursday of each month

**Active Expiries** (at any given date):
- **Current week expiry** (0-7 days)
- **Next 3 weekly expiries** (7-28 days)
- **Current month expiry** (0-30 days)
- **Next 2 monthly expiries** (30-90 days)

**Maximum**: 6 unique expiries per timestamp (4 weekly + 2-3 monthly, with overlap)

**Expiry Calculation**:
```python
def get_weekly_expiries(date: datetime, count: int = 4) -> List[datetime]:
    """Get next N Thursday expiries"""
    expiries = []
    current = date
    while len(expiries) < count:
        days_ahead = (3 - current.weekday()) % 7  # Thursday = 3
        if days_ahead == 0 and current.date() > date.date():
            days_ahead = 7
        expiry = current + timedelta(days=days_ahead)
        expiries.append(expiry)
        current = expiry + timedelta(days=1)
    return expiries

def get_monthly_expiry(year: int, month: int) -> datetime:
    """Get last Thursday of month"""
    # Get last day of month
    if month == 12:
        last_day = datetime(year, month, 31)
    else:
        last_day = datetime(year, month + 1, 1) - timedelta(days=1)

    # Find last Thursday
    while last_day.weekday() != 3:  # Thursday
        last_day -= timedelta(days=1)

    return last_day
```

---

## 4. Pricing Model (Black-Scholes)

### 4.1 Fixed Parameters

```python
RISK_FREE_RATE = 0.065      # 6.5% (RBI repo rate)
DIVIDEND_YIELD = 0.0        # Nifty index (no dividend in BS model)
LOT_SIZE = 25               # Nifty 50 lot size (as of 2024)
```

### 4.2 Black-Scholes Implementation

**Formula**:
```
Call Price:  C = S * N(d1) - K * e^(-rT) * N(d2)
Put Price:   P = K * e^(-rT) * N(-d2) - S * N(-d1)

where:
d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)
d2 = d1 - σ√T

S = Spot price (from seed data)
K = Strike price
r = Risk-free rate (0.065)
σ = Implied volatility (see section 4.3)
T = Time to expiry (years)
N(x) = Cumulative normal distribution
```

**Critical Fix from V9**:
```python
def black_scholes(self, S: float, K: float, T: float, sigma: float,
                  option_type: str) -> dict:
    """
    Calculate option price and Greeks

    VALIDATED IMPLEMENTATION:
    - Deep ITM (K < S-2000 for CE): delta > 0.90 ✓
    - ATM (K ≈ S): delta ≈ 0.50 ✓
    - Deep OTM (K > S+2000 for CE): delta < 0.10 ✓
    """

    if T <= 0:
        # At expiry: intrinsic value only
        if option_type == 'CE':
            price = max(S - K, 0)
            delta = 1.0 if S > K else 0.0
        else:
            price = max(K - S, 0)
            delta = -1.0 if K > S else 0.0

        return {
            'price': round(max(price, 0.05), 2),  # Min ₹0.05
            'delta': delta,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0
        }

    # Calculate d1, d2
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S/K) + (self.risk_free_rate + 0.5*sigma**2)*T) / (sigma*sqrt_T)
    d2 = d1 - sigma*sqrt_T

    # Price
    if option_type == 'CE':
        price = S * norm.cdf(d1) - K * np.exp(-self.risk_free_rate*T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    else:  # PE
        price = K * np.exp(-self.risk_free_rate*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = -norm.cdf(-d1)

    # Greeks
    gamma = norm.pdf(d1) / (S * sigma * sqrt_T)
    vega = S * norm.pdf(d1) * sqrt_T / 100  # Per 1% IV change

    # Theta (per day)
    theta_annual = -(S * norm.pdf(d1) * sigma) / (2 * sqrt_T)
    if option_type == 'CE':
        theta_annual -= self.risk_free_rate * K * np.exp(-self.risk_free_rate*T) * norm.cdf(d2)
    else:
        theta_annual += self.risk_free_rate * K * np.exp(-self.risk_free_rate*T) * norm.cdf(-d2)
    theta = theta_annual / 365

    return {
        'price': round(max(price, 0.05), 2),
        'delta': round(delta, 4),
        'gamma': round(max(gamma, 0), 6),
        'theta': round(theta, 2),
        'vega': round(vega, 2)
    }
```

### 4.3 Implied Volatility (IV) Calculation

**Base IV Source**: VIX / 100 (if VIX data available), else calculated from historical vol

**IV Smile Structure** (realistic):

```python
def calculate_iv(self, spot: float, strike: int, tte_days: int,
                 expiry_type: str, vix: float) -> float:
    """
    Calculate implied volatility with realistic smile

    CRITICAL FIXES:
    1. Symmetric smile (OTM on both sides has higher IV)
    2. Moneyness calculated correctly
    3. Term structure accounts for time decay
    """

    # Base IV from VIX
    atm_iv = vix / 100.0

    # Moneyness (log-moneyness for smile)
    log_moneyness = np.log(spot / strike)

    # Volatility smile (symmetric U-shape)
    # ATM has lowest IV, wings have higher IV
    smile_adjustment = 0.15 * log_moneyness**2  # Quadratic

    # Strike distance adjustment (for very far strikes)
    distance_pct = abs(strike - spot) / spot
    if distance_pct > 0.15:  # >15% from ATM
        smile_adjustment += 0.3 * (distance_pct - 0.15)

    iv = atm_iv * (1 + smile_adjustment)

    # Term structure
    if tte_days <= 2:
        iv *= 1.25  # +25% for very short TTE (weekend gamma)
    elif tte_days <= 7:
        iv *= 1.15  # +15% for weekly options
    elif tte_days <= 15:
        iv *= 1.08  # +8% for mid-term
    elif tte_days <= 30:
        iv *= 1.03  # +3% for monthly
    # else: use base IV for longer dated

    # Expiry-type adjustment
    if expiry_type == 'weekly' and tte_days <= 3:
        iv *= 1.10  # Additional 10% for expiry week

    # Random noise (±2%)
    iv *= (1 + np.random.uniform(-0.02, 0.02))

    # Bounds
    return np.clip(iv, 0.08, 0.60)  # 8% to 60% IV


# Example IV values for VIX=15, Spot=25000:
#
# Strike    Distance    TTE=2d    TTE=7d    TTE=30d
# 23000     -8% (ITM)   0.192     0.180     0.165
# 24000     -4% (ITM)   0.168     0.161     0.155
# 25000      0% (ATM)   0.150     0.150     0.150
# 26000     +4% (OTM)   0.168     0.161     0.155
# 27000     +8% (OTM)   0.192     0.180     0.165
```

**Validation**:
- ATM IV should be lowest
- Symmetric smile (CE and PE at same distance have same IV)
- Short-dated options have higher IV (term structure)

---

## 5. Volume, Open Interest, and Bid-Ask

### 5.1 Volume Generation

**Realistic Volume Profile**:

```python
def generate_volume(self, spot: float, strike: int, tte_days: int,
                    expiry_type: str, hour_index: int) -> int:
    """
    Generate realistic volume based on multiple factors

    FIXES:
    1. Steep decay for far strikes (not Gaussian)
    2. Expiry week volume surge
    3. Intraday U-shaped pattern
    """

    # Base volume by expiry type
    if expiry_type == 'monthly':
        base_volume = 8000  # contracts per hour
    else:  # weekly
        base_volume = 3000

    # Distance-based decay (CRITICAL FIX)
    distance = abs(strike - spot)

    if distance < 100:  # Within ₹100 of spot
        distance_factor = 1.0
    elif distance < 500:  # ₹100-500 away
        distance_factor = np.exp(-0.004 * distance)  # Gradual decay
    elif distance < 1500:  # ₹500-1500 away
        distance_factor = np.exp(-0.008 * distance)  # Faster decay
    else:  # >₹1500 away
        distance_factor = np.exp(-0.012 * distance)  # Very fast decay

    # Example decay factors:
    # Distance ₹0:    1.000 (100% of base)
    # Distance ₹250:  0.368 (37% of base)
    # Distance ₹500:  0.135 (13% of base)
    # Distance ₹1000: 0.018 (2% of base)
    # Distance ₹2000: 0.0003 (0.03% of base)

    # Time to expiry factor
    if tte_days <= 1:
        tte_factor = 4.0  # 4x volume on expiry day
    elif tte_days <= 3:
        tte_factor = 2.5  # 2.5x in last 3 days
    elif tte_days <= 7:
        tte_factor = 1.5  # 1.5x in expiry week
    else:
        tte_factor = 1.0

    # Intraday pattern (U-shaped: high at open/close)
    intraday_factors = [
        1.8,  # H1: 09:15-10:15 (high opening vol)
        1.2,  # H2: 10:15-11:15
        0.9,  # H3: 11:15-12:15
        0.7,  # H4: 12:15-13:15 (lunch lull)
        0.9,  # H5: 13:15-14:15
        1.3,  # H6: 14:15-15:15
        1.5   # H7: 15:15-15:30 (closing surge)
    ]
    intraday_factor = intraday_factors[hour_index]

    # VIX boost (higher volatility = higher volume)
    vix_factor = 1.0 + (max(vix - 15, 0) / 50)  # +2% per VIX point above 15

    # Random variation (±30%)
    random_factor = np.random.lognormal(0, 0.3)

    # Calculate final volume
    volume = int(
        base_volume *
        distance_factor *
        tte_factor *
        intraday_factor *
        vix_factor *
        random_factor
    )

    return max(volume, 5)  # Minimum 5 contracts
```

### 5.2 Open Interest (OI) Generation

```python
def generate_oi(self, spot: float, strike: int, tte_days: int,
                expiry_type: str) -> int:
    """
    Generate realistic open interest

    OI builds up over time and concentrates at ATM
    """

    # Base OI (higher than volume as it accumulates)
    if expiry_type == 'monthly':
        base_oi = 150000
    else:
        base_oi = 60000

    # Distance decay (same as volume but less steep)
    distance = abs(strike - spot)
    if distance < 100:
        distance_factor = 1.0
    elif distance < 500:
        distance_factor = np.exp(-0.003 * distance)
    elif distance < 1500:
        distance_factor = np.exp(-0.006 * distance)
    else:
        distance_factor = np.exp(-0.009 * distance)

    # OI decays as expiry approaches (positions closed)
    if tte_days <= 1:
        tte_factor = 0.3  # 70% positions closed on expiry day
    elif tte_days <= 3:
        tte_factor = 0.6  # 40% closed in last 3 days
    elif tte_days <= 7:
        tte_factor = 0.9
    else:
        tte_factor = 1.0 + (tte_days / 30) * 0.2  # Builds up over time

    # Random variation (±25%)
    random_factor = np.random.lognormal(0, 0.25)

    oi = int(base_oi * distance_factor * tte_factor * random_factor)

    return max(oi, 10)
```

### 5.3 Bid-Ask Spread

```python
def generate_bid_ask(self, price: float, spot: float, strike: int,
                     volume: int, oi: int) -> Tuple[float, float]:
    """
    Generate realistic bid-ask spread

    FIXES:
    1. Wider spreads for far strikes
    2. Wider spreads for low liquidity
    3. Minimum ₹0.05 tick size
    """

    # Base spread as percentage of price
    if price < 5:
        base_spread_pct = 0.15  # 15% for very cheap options
    elif price < 20:
        base_spread_pct = 0.08  # 8%
    elif price < 50:
        base_spread_pct = 0.04  # 4%
    elif price < 200:
        base_spread_pct = 0.02  # 2%
    else:
        base_spread_pct = 0.01  # 1% for expensive options

    # Distance adjustment (wider for far strikes)
    distance_pct = abs(strike - spot) / spot
    if distance_pct > 0.10:  # >10% from ATM
        base_spread_pct *= 2.5
    elif distance_pct > 0.05:  # >5% from ATM
        base_spread_pct *= 1.5

    # Liquidity adjustment (wider for low volume)
    if volume < 50:
        base_spread_pct *= 3.0  # Very illiquid
    elif volume < 200:
        base_spread_pct *= 2.0
    elif volume < 1000:
        base_spread_pct *= 1.3

    # Calculate spread
    spread = max(price * base_spread_pct, 0.05)  # Min ₹0.05

    # Bid/ask (ensure bid < price < ask)
    mid = price
    bid = round(mid - spread/2, 1)  # Round to ₹0.10
    ask = round(mid + spread/2, 1)

    # Ensure minimum tick
    bid = max(bid, 0.05)
    ask = max(ask, bid + 0.05)

    return (bid, ask)
```

---

## 6. OHLC Generation

For each hourly candle, generate realistic OHLC based on underlying movement:

```python
def generate_ohlc(self, base_price: float, underlying_ohlc: dict,
                  strike: int, option_type: str, T: float, iv: float) -> dict:
    """
    Generate option OHLC from underlying OHLC

    Uses Black-Scholes to price at each point (O/H/L/C) of underlying
    """

    # Calculate option price at each underlying level
    prices = []
    for spot in [underlying_ohlc['open'], underlying_ohlc['high'],
                 underlying_ohlc['low'], underlying_ohlc['close']]:

        option_data = self.black_scholes(
            S=spot,
            K=strike,
            T=T,
            sigma=iv,
            option_type=option_type
        )
        prices.append(option_data['price'])

    # For calls: high underlying -> high option (direct relationship)
    # For puts: high underlying -> low option (inverse relationship)
    if option_type == 'CE':
        return {
            'open': prices[0],   # At underlying open
            'high': max(prices), # At underlying high
            'low': min(prices),  # At underlying low
            'close': prices[3]   # At underlying close
        }
    else:  # PE
        return {
            'open': prices[0],   # At underlying open
            'high': max(prices), # At underlying low (inverse)
            'low': min(prices),  # At underlying high (inverse)
            'close': prices[3]   # At underlying close
        }

    # Note: This ensures realistic OHLC relationships:
    # - Low <= Open, Close <= High (always true)
    # - OHLC moves with underlying appropriately
```

---

## 7. Data Generation Workflow

### 7.1 Initialization

```python
class V10RealEnhancedGenerator:
    def __init__(self):
        # Load real NIFTY data
        self.seed_loader = NiftySeedDataLoader()
        self.nifty_1min = self.seed_loader.load()
        self.nifty_hourly = self.seed_loader.get_hourly_data()

        # Load VIX data (if available)
        self.vix_data = self.load_vix_data()
        if self.vix_data is None:
            # Calculate from historical volatility
            self.vix_data = self.calculate_vix_from_returns()

        # Output path
        self.output_path = Path('data/generated/v10_real_enhanced/hourly')
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Stats tracking
        self.stats = {
            'files_generated': 0,
            'total_rows': 0,
            'greeks_validation': {
                'deep_itm_delta_passed': 0,
                'deep_otm_delta_passed': 0,
                'atm_delta_passed': 0,
                'total_checks': 0
            },
            'volume_validation': {
                'far_strikes_low_volume': 0,
                'atm_high_volume': 0,
                'total_checks': 0
            }
        }
```

### 7.2 Generation Loop

```python
def generate_all_data(self):
    """
    Main generation loop

    For each trading day in seed data:
      1. Get real NIFTY hourly OHLC
      2. Get VIX for the day
      3. Determine active expiries
      4. For each hour:
          For each expiry:
              For each strike:
                  For each option_type (CE/PE):
                      - Calculate IV
                      - Calculate Greeks
                      - Generate OHLC
                      - Generate volume/OI
                      - Generate bid/ask
                      - Validate
      5. Save daily CSV
      6. Update stats
    """

    trading_dates = self.nifty_hourly['date'].dt.date.unique()

    print(f"Generating V10 data for {len(trading_dates)} trading days...")

    for date_idx, date in enumerate(trading_dates, 1):
        print(f"[{date_idx}/{len(trading_dates)}] {date}", end=' ')

        # Get data for this date
        day_data = self.nifty_hourly[
            self.nifty_hourly['date'].dt.date == date
        ]

        if len(day_data) == 0:
            print("⊘ No data")
            continue

        # Get VIX
        vix_value = self.get_vix_for_date(date)

        # Get active expiries
        expiries = self.get_active_expiries(date)

        # Generate options data
        options_df = self.generate_day_options(date, day_data, vix_value, expiries)

        # Validate
        validation = self.validate_day_data(options_df, date)

        # Save
        filename = f"NIFTY_OPTIONS_1H_{date.strftime('%Y%m%d')}.csv"
        filepath = self.output_path / filename
        options_df.to_csv(filepath, index=False)

        # Update stats
        self.stats['files_generated'] += 1
        self.stats['total_rows'] += len(options_df)

        print(f"✓ ({len(options_df):,} rows, VIX: {vix_value:.1f}, "
              f"{validation['pass_rate']*100:.1f}% validated)")

    # Save metadata
    self.save_metadata()

    # Print summary
    self.print_summary()
```

### 7.3 Per-Day Generation

```python
def generate_day_options(self, date, hourly_data, vix, expiries):
    """
    Generate all options for one trading day
    """
    rows = []

    # For each hourly candle (7 per day)
    for hour_idx, (idx, hour_row) in enumerate(hourly_data.iterrows()):
        timestamp = hour_row['date']
        spot_ohlc = {
            'open': hour_row['open'],
            'high': hour_row['high'],
            'low': hour_row['low'],
            'close': hour_row['close']
        }
        spot_close = spot_ohlc['close']

        # Determine strikes for this spot level
        strikes = self.get_strikes_for_spot(spot_close)

        # For each expiry
        for expiry_date, expiry_type in expiries:
            tte_days = (expiry_date - date).days
            tte_years = tte_days / 365.0

            # For each strike
            for strike in strikes:
                # For each option type
                for option_type in ['CE', 'PE']:

                    # Calculate IV
                    iv = self.calculate_iv(
                        spot=spot_close,
                        strike=strike,
                        tte_days=tte_days,
                        expiry_type=expiry_type,
                        vix=vix
                    )

                    # Calculate Greeks at close
                    greeks = self.black_scholes(
                        S=spot_close,
                        K=strike,
                        T=tte_years,
                        sigma=iv,
                        option_type=option_type
                    )

                    # Generate OHLC
                    ohlc = self.generate_ohlc(
                        base_price=greeks['price'],
                        underlying_ohlc=spot_ohlc,
                        strike=strike,
                        option_type=option_type,
                        T=tte_years,
                        iv=iv
                    )

                    # Generate volume
                    volume = self.generate_volume(
                        spot=spot_close,
                        strike=strike,
                        tte_days=tte_days,
                        expiry_type=expiry_type,
                        hour_index=hour_idx
                    )

                    # Generate OI
                    oi = self.generate_oi(
                        spot=spot_close,
                        strike=strike,
                        tte_days=tte_days,
                        expiry_type=expiry_type
                    )

                    # Generate bid/ask
                    bid, ask = self.generate_bid_ask(
                        price=ohlc['close'],
                        spot=spot_close,
                        strike=strike,
                        volume=volume,
                        oi=oi
                    )

                    # Validate Greeks
                    self.validate_greeks(greeks, spot_close, strike, option_type)

                    # Create row
                    row = {
                        'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                        'symbol': 'NIFTY',
                        'strike': strike,
                        'option_type': option_type,
                        'expiry': expiry_date.strftime('%Y-%m-%d'),
                        'expiry_type': expiry_type,
                        'open': ohlc['open'],
                        'high': ohlc['high'],
                        'low': ohlc['low'],
                        'close': ohlc['close'],
                        'volume': volume,
                        'oi': oi,
                        'bid': bid,
                        'ask': ask,
                        'iv': round(iv, 4),
                        'delta': greeks['delta'],
                        'gamma': greeks['gamma'],
                        'theta': greeks['theta'],
                        'vega': greeks['vega'],
                        'underlying_price': round(spot_close, 2),
                        'vix': round(vix, 2)
                    }

                    rows.append(row)

    return pd.DataFrame(rows)
```

---

## 8. Validation & Quality Assurance

### 8.1 Greeks Validation

```python
def validate_greeks(self, greeks: dict, spot: float, strike: int,
                    option_type: str):
    """
    Validate Greeks are realistic

    CRITICAL CHECKS (fixing V9 issues):
    """
    delta = greeks['delta']
    distance = abs(strike - spot)

    # Check 1: Deep ITM should have high delta
    if option_type == 'CE' and strike < spot - 2000:  # Deep ITM call
        if delta > 0.90:
            self.stats['greeks_validation']['deep_itm_delta_passed'] += 1
        else:
            logger.warning(
                f"Deep ITM CE delta too low: K={strike}, S={spot:.0f}, "
                f"delta={delta:.2f} (expected >0.90)"
            )

    # Check 2: Deep OTM should have low delta
    elif option_type == 'CE' and strike > spot + 2000:  # Deep OTM call
        if delta < 0.10:
            self.stats['greeks_validation']['deep_otm_delta_passed'] += 1
        else:
            logger.warning(
                f"Deep OTM CE delta too high: K={strike}, S={spot:.0f}, "
                f"delta={delta:.2f} (expected <0.10)"
            )

    # Check 3: ATM should have delta ≈ 0.5
    elif abs(strike - spot) < 100:  # ATM
        if 0.40 < abs(delta) < 0.60:
            self.stats['greeks_validation']['atm_delta_passed'] += 1
        else:
            logger.warning(
                f"ATM delta off: K={strike}, S={spot:.0f}, "
                f"delta={delta:.2f} (expected ~0.50)"
            )

    self.stats['greeks_validation']['total_checks'] += 1

    # Bounds checks
    assert -1.0 <= delta <= 1.0, f"Delta out of bounds: {delta}"
    assert greeks['gamma'] >= 0, f"Gamma negative: {greeks['gamma']}"
    assert greeks['theta'] <= 0, f"Theta positive: {greeks['theta']}"
    assert greeks['vega'] >= 0, f"Vega negative: {greeks['vega']}"
```

### 8.2 Volume Profile Validation

```python
def validate_volume_profile(self, df: pd.DataFrame, spot: float):
    """
    Validate volume concentrates at ATM
    """
    # Get ATM strike
    atm_strike = round(spot / 50) * 50

    # Get volume by distance
    df['distance'] = abs(df['strike'] - spot)

    # ATM region (±200 points)
    atm_volume = df[df['distance'] < 200]['volume'].sum()

    # Far region (>1500 points)
    far_volume = df[df['distance'] > 1500]['volume'].sum()

    # ATM should have >>more volume
    if atm_volume > far_volume * 10:
        self.stats['volume_validation']['atm_high_volume'] += 1

    # Far strikes should have low volume
    far_avg = df[df['distance'] > 1500]['volume'].mean()
    if far_avg < 100:  # <100 contracts/hour
        self.stats['volume_validation']['far_strikes_low_volume'] += 1

    self.stats['volume_validation']['total_checks'] += 1
```

### 8.3 Data Completeness

```python
def validate_completeness(self, df: pd.DataFrame, date):
    """
    Check data completeness
    """
    issues = []

    # Check for missing values
    if df.isnull().any().any():
        issues.append(f"Missing values: {df.isnull().sum().to_dict()}")

    # Check hourly coverage (should be 7 hours)
    hours = df['timestamp'].nunique()
    if hours != 7:
        issues.append(f"Expected 7 hourly candles, got {hours}")

    # Check OHLC validity
    invalid = df[
        (df['high'] < df['low']) |
        (df['high'] < df['open']) |
        (df['high'] < df['close']) |
        (df['low'] > df['open']) |
        (df['low'] > df['close'])
    ]
    if len(invalid) > 0:
        issues.append(f"Invalid OHLC in {len(invalid)} rows")

    # Check bid/ask validity
    invalid_spread = df[(df['bid'] >= df['ask']) | (df['bid'] >= df['close'])]
    if len(invalid_spread) > 0:
        issues.append(f"Invalid bid/ask in {len(invalid_spread)} rows")

    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'pass_rate': 1.0 - len(invalid) / len(df)
    }
```

---

## 9. Output Specifications

### 9.1 Dataset Size Estimate

**Per Day**:
- Hours: 7
- Expiries: ~6 (4 weekly + 2 monthly)
- Strikes per expiry: ~101 (after filtering)
- Option types: 2 (CE, PE)
- **Rows per day**: 7 × 6 × 101 × 2 ≈ **8,484 rows**

**Full Dataset** (438 days):
- **Total rows**: 438 × 8,484 ≈ **3.7 million rows**
- **Estimated size**: ~450 MB (uncompressed CSV)

### 9.2 Metadata

`metadata/generation_info.json`:
```json
{
  "version": "10.0-RealEnhanced",
  "generation_date": "2025-10-05T17:00:00",
  "data_source": {
    "underlying": "Real NIFTY 1-minute data (Jan 2024 - Oct 2025)",
    "vix": "Calculated from historical volatility",
    "seed_file": "data/seed/nifty_data_min.csv"
  },
  "period": {
    "start": "2024-01-01",
    "end": "2025-10-03",
    "trading_days": 438
  },
  "specifications": {
    "frequency": "1 hour",
    "candles_per_day": 7,
    "strike_range": [18000, 30000],
    "strike_interval": 50,
    "total_strikes": 241,
    "filtered_strikes_per_expiry": 101,
    "expiries_per_day": 6
  },
  "improvements_over_v9": [
    "Real underlying prices from seed data",
    "Fixed Greeks calculations (deep ITM/OTM deltas)",
    "Realistic volume decay for far strikes",
    "Improved bid-ask spreads based on liquidity",
    "Validated IV smile (symmetric, quadratic)"
  ],
  "stats": {
    "files_generated": 438,
    "total_rows": 3715992,
    "greeks_validation": {
      "deep_itm_pass_rate": 0.98,
      "deep_otm_pass_rate": 0.97,
      "atm_pass_rate": 0.95
    },
    "volume_validation": {
      "atm_concentration": 0.92,
      "far_strike_low_volume": 0.89
    }
  }
}
```

---

## 10. Success Criteria

### 10.1 Functional Requirements

✅ All 438 trading days from seed data covered
✅ 7 hourly candles per day (matching V9 structure)
✅ Real NIFTY prices used as underlying
✅ 6 active expiries per timestamp
✅ Greeks calculated with Black-Scholes
✅ Volume/OI/bid-ask generated realistically

### 10.2 Quality Requirements

✅ **Greeks Accuracy**:
- Deep ITM (>2000 pts): Delta > 0.90 (>95% of cases)
- Deep OTM (>2000 pts): Delta < 0.10 (>95% of cases)
- ATM (±100 pts): Delta 0.40-0.60 (>90% of cases)

✅ **Volume Realism**:
- Far strikes (>1500 pts): <5% of ATM volume
- ATM region: >10x volume of far strikes
- Expiry day surge: 3-4x normal volume

✅ **Bid-Ask Spreads**:
- Liquid strikes (<500 pts): <5% spread
- Illiquid strikes (>1500 pts): >15% spread
- Low volume options: Wide spreads (>20%)

✅ **Data Integrity**:
- No missing values
- Valid OHLC relationships (Low ≤ Open,Close ≤ High)
- Bid < Close < Ask
- Positive volume and OI

### 10.3 Performance Requirements

⏱️ Generation time: <2 hours for full dataset (438 days)
💾 Output size: ~450 MB (acceptable for hourly data)
🔍 Validation time: <10 minutes

---

## 11. Implementation Timeline

### Phase 1: Core Implementation (Week 1)
- [ ] Create V10RealEnhancedGenerator class
- [ ] Integrate NiftySeedDataLoader
- [ ] Implement fixed Black-Scholes pricing
- [ ] Implement realistic IV calculation
- [ ] Add unit tests for Greeks validation

### Phase 2: Volume/OI/Spreads (Week 1-2)
- [ ] Implement realistic volume generation
- [ ] Implement OI generation
- [ ] Implement bid-ask spread logic
- [ ] Add validation tests

### Phase 3: Generation & Validation (Week 2)
- [ ] Implement main generation loop
- [ ] Generate full dataset (438 days)
- [ ] Run validation suite
- [ ] Generate quality report

### Phase 4: Documentation & Release (Week 2-3)
- [ ] Update CLAUDE.md with V10 info
- [ ] Create V10 user guide
- [ ] Create comparison report (V9 vs V10)
- [ ] Release V10.0

---

## 12. Next Steps

1. **Review this specification** with stakeholders
2. **Obtain VIX data** (if available) for Jan 2024 - Oct 2025
3. **Create feature branch**: `feature/v10-real-enhanced`
4. **Implement core components** following this spec
5. **Run validation tests** continuously during development
6. **Generate sample data** (1 week) for quick validation
7. **Generate full dataset** once validation passes
8. **Create quality comparison** with V9 data

---

## Appendix A: Example Data Rows

Sample output for **2024-06-14 09:15:00**, Spot = ₹25,043, VIX = 15.2:

```csv
timestamp,symbol,strike,option_type,expiry,expiry_type,open,high,low,close,volume,oi,bid,ask,iv,delta,gamma,theta,vega,underlying_price,vix
2024-06-14 09:15:00,NIFTY,25000,CE,2024-06-20,weekly,142.50,145.20,140.80,143.10,8420,142300,142.50,143.70,0.1523,0.5234,0.000421,-8.45,52.34,25043.25,15.2
2024-06-14 09:15:00,NIFTY,25000,PE,2024-06-20,weekly,138.20,141.00,136.50,139.40,7890,138200,138.80,140.00,0.1534,0.4823,0.000418,-8.21,51.87,25043.25,15.2
2024-06-14 09:15:00,NIFTY,26000,CE,2024-06-20,weekly,28.40,29.50,27.80,28.90,2340,45600,28.20,29.60,0.1867,0.1245,0.000234,-3.21,28.45,25043.25,15.2
2024-06-14 09:15:00,NIFTY,24000,PE,2024-06-20,weekly,34.50,35.80,33.90,34.70,1980,38900,34.10,35.30,0.1823,0.1389,0.000241,-3.45,29.12,25043.25,15.2
```

Note:
- 26000 CE is ~1000 pts OTM: low delta (0.12), low volume (2340)
- 25000 CE/PE are ATM: high delta (~0.5), high volume (>7000)
- Bid-ask spreads tight for ATM (~₹1.20), wider for OTM (~₹1.40)

---

**END OF PRD v10.0**
