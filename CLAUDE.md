# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Synthetic Data Generator is a Python package for generating high-quality synthetic options market data for algorithmic trading backtests. The project generates realistic Nifty options data with proper market dynamics, volatility patterns, and Greeks calculations.

**Current Version**: 9.0 (Balanced Hourly)

## Development Commands

### Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/synthetic_data_generator

# Run specific test suite
pytest tests/unit/
pytest tests/integration/
```

### Code Quality

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/ tests/
```

## Architecture

### Generator Evolution (Multi-Version System)

The codebase maintains **three generator versions** (V7, V8, V9), each in `src/synthetic_data_generator/generators/`. They share a common architecture but have different trade-offs:

- **V7** (VIX-based): 551 MB, 5-minute candles, basic VIX integration
- **V8** (Extended): 1.3 GB, 5-minute candles, enhanced features
- **V9** (Current): 130 MB, 1-hour candles, balanced trends with critical fixes

**Why multiple versions exist**: Each version represents progressive improvements in data quality, with V9 being production-ready. Earlier versions are kept for comparison and legacy backtest compatibility.

### Core Components

1. **Base Generator** (`src/synthetic_data_generator/core/base_generator.py`)
   - Black-Scholes pricing model implementation
   - Greeks calculations (delta, gamma, theta, vega)
   - IV smile generation with moneyness and term structure
   - Volume/OI modeling with liquidity patterns
   - OHLC generation with intraday patterns

2. **V9 Generator** (`src/synthetic_data_generator/generators/v9/generator.py`)
   - **Critical Feature**: Balanced price series generation (50/50 weekly trend split)
   - **Critical Feature**: Expiry-specific pricing (weekly vs monthly have different IVs)
   - **Critical Feature**: Correct delta-distance relationships (0.1 delta CE weekly at 800-1400 pts from spot)
   - Generates 7 hourly candles per day (9:15 AM to 3:15 PM)
   - VIX regime modeling with trend adjustments
   - Expiry schedule: Weekly (Thursdays) + Monthly (last Thursday)
   - Strike range: ₹22,000 - ₹30,000 (50-point intervals)

3. **Adapters** (`src/synthetic_data_generator/adapters/`)
   - Bridge between CSV data files and backtest frameworks
   - **V9Adapter** (`v9_adapter.py`): Provides OptionsDataFetcher-compatible interface
   - Methods: `get_monthly_expiries()`, `get_weekly_expiries()`, `get_option_chain()`, `get_spot_price()`
   - Implements date-based caching for performance

4. **Analytics** (`src/synthetic_data_generator/analytics/`)
   - `delta_analyzer.py`: Delta-distance validation
   - `price_analyzer.py`: Price continuity checks
   - `trend_analyzer.py`: Weekly trend balance verification
   - `validators.py`: Data quality metrics

5. **Visualization** (`src/synthetic_data_generator/visualization/`)
   - OHLC chart generation with plotly
   - TradingView-style interactive charts
   - Uses Playwright for chart rendering

### Critical Data Quality Constraints

When modifying generators, these relationships MUST be preserved:

1. **Delta-Distance Relationships**:
   - Weekly 0.1Δ CE: 800-1400 points from spot
   - Monthly 0.1Δ CE: 1800-2600 points from spot
   - These emerge from correct expiry-specific IV modeling

2. **Trend Balance**:
   - Weekly price movements must be ~50% up, ~50% down
   - Achieved via pre-generation of balanced price series in V9

3. **Expiry-Specific Pricing**:
   - Weekly and monthly options at the same strike/date have DIFFERENT prices
   - They use different time-to-expiry (TTE) values
   - Weekly options have higher IV for TTE <= 7 days

4. **Price Continuity**:
   - OHLC relationships: Low <= Open,Close <= High
   - Bid < Last < Ask
   - Greeks must be mathematically consistent with prices

## Data Structure

### Generated Data Location

```
data/generated/
├── v7_vix/intraday/         # 551 MB, 67 CSV files
├── v8_extended/intraday/    # 1.3 GB, 78 CSV files
└── v9_balanced/hourly/      # 130 MB, 80 CSV files (current)
```

### CSV Schema (V9)

Columns: `timestamp`, `symbol`, `strike`, `option_type`, `expiry`, `expiry_type`, `open`, `high`, `low`, `close`, `volume`, `oi`, `bid`, `ask`, `iv`, `delta`, `gamma`, `theta`, `vega`, `underlying_price`, `vix`

**Key Fields**:
- `expiry_type`: "weekly" or "monthly" - critical for correct pricing
- `timestamp`: Hourly intervals (7 per day)
- `underlying_price`: Nifty spot price at that timestamp
- `vix`: India VIX value (regime-based)

### File Naming

- V9: `NIFTY_OPTIONS_1H_YYYYMMDD.csv`
- V8: `NIFTY_OPTIONS_5MIN_YYYYMMDD.csv`

## Key Workflows

### Generating New Data

```python
from synthetic_data_generator.generators.v9 import V9BalancedGenerator

generator = V9BalancedGenerator()
stats = generator.generate_all_data()  # Generates full dataset
```

The generator:
1. Creates balanced price series (50/50 weekly trends)
2. For each trading day, generates 7 hourly timestamps
3. For each timestamp, prices ALL active expiries (weekly + monthly)
4. Calculates expiry-specific prices and Greeks
5. Validates delta-distance relationships
6. Saves to CSV with metadata

### Using Data in Backtests

```python
from synthetic_data_generator.adapters import V9Adapter

# Initialize with data path
adapter = V9Adapter()
adapter.load_data(start_date="2023-01-01", end_date="2023-12-31")

# Get expiries
weekly_expiries = adapter.get_weekly_expiries(weeks=4)
monthly_expiries = adapter.get_monthly_expiries(months=2)

# Get option chain
chain = adapter.get_option_chain(date=backtest_date)

# Get specific option
option_data = adapter.get_option_data(
    strike=25000,
    option_type="CE",
    expiry=weekly_expiries[0],
    date=backtest_date
)
```

### Validation

```python
from synthetic_data_generator.analytics import QualityMetrics

metrics = QualityMetrics()
report = metrics.analyze("data/generated/v9_balanced/hourly/")
print(report.summary())
```

## Common Pitfalls

1. **Don't share prices across expiries**: Each (strike, option_type, expiry) combination needs independent pricing with its own TTE calculation

2. **Don't ignore expiry_type**: Weekly and monthly options have different IV curves even at same strike

3. **Trend balance**: Simply using random walks creates biased trends; V9 enforces 50/50 via pre-generation

4. **Greeks validation**: Delta must be in [-1, 1], gamma/vega >= 0, theta typically <= 0

5. **Volume/OI realism**: ATM options have highest liquidity (Gaussian decay from ATM)

## Package Structure

```
src/synthetic_data_generator/
├── core/           # Base generator with Black-Scholes model
├── generators/     # V7, V8, V9 generator implementations
├── adapters/       # Data access layer for backtests
├── analytics/      # Quality metrics and validators
├── visualization/  # Chart generation
├── io/            # File I/O utilities
└── utils/         # Helper functions
```

## Migration Context

This project was extracted from the NikAlgoBulls trading strategy repository on 2025-10-05. The original data generation was embedded in the backtest code; this standalone package provides:

- Clean separation of data generation from strategy logic
- Multiple generator versions for comparison
- Comprehensive testing and validation
- Adapter layer for integration with any backtest framework

See `MIGRATION_SUMMARY.md` for full extraction details.
