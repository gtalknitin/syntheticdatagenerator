# Synthetic Data Generator

**Version**: 9.0 (Current)
**Python**: 3.8+

A comprehensive Python package for generating high-quality synthetic options market data for algorithmic trading backtests. This project evolved from the NikAlgoBulls trading strategy development to become a standalone, production-ready data generation system.

## 🎯 Project Overview

The Synthetic Data Generator creates realistic Nifty options market data with support for multiple timeframes (5-minute intraday, hourly), incorporating market microstructure patterns, volatility dynamics, and Greeks modeling. Currently on version 9.0 (balanced hourly data), the system has evolved through multiple iterations to achieve production-quality synthetic data.

### Key Features

- **Multiple Generator Versions**: V7 (VIX-based), V8 (Extended), V9 (Balanced - Current)
- **Realistic Market Dynamics**: Trend modeling, volatility clustering, Greeks calculations
- **Comprehensive Analytics**: Data quality metrics, delta analysis, trend analysis
- **Flexible Adapters**: Easy integration with backtesting frameworks
- **Rich Visualization**: OHLC charts, TradingView-style plots
- **Extensive Documentation**: API docs, user guides, PRDs

## 🚀 Quick Start

### Installation

```bash
# Clone repository
cd /Users/nitindhawan/SyntheticDataGenerator

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Basic Usage

```python
from synthetic_data_generator.generators.v9 import V9BalancedGenerator
from synthetic_data_generator.adapters import V9Adapter

# Generate data
generator = V9BalancedGenerator()
data = generator.generate(
    start_date="2023-01-01",
    end_date="2023-12-31",
    frequency="1H"
)

# Use with backtest
adapter = V9Adapter()
backtest_data = adapter.load_data("2023-01-01", "2023-12-31")
```

## 📊 Data Versions

### V9 - Balanced Hourly (Current)
- **Size**: 130 MB (80 CSV files)
- **Frequency**: Hourly
- **Period**: 2023-2024
- **Status**: Production-ready
- **Location**: `data/generated/v9_balanced/hourly/`

### V8 - Extended Intraday
- **Size**: 1.3 GB (78 CSV files)
- **Frequency**: 5-minute
- **Period**: 2023-2024
- **Status**: Stable
- **Location**: `data/generated/v8_extended/intraday/`

### V7 - VIX-Based Intraday
- **Size**: 551 MB (67 CSV files)
- **Frequency**: 5-minute
- **Period**: 2023-2024
- **Status**: Legacy
- **Location**: `data/generated/v7_vix/intraday/`

## 📁 Project Structure

```
SyntheticDataGenerator/
├── src/synthetic_data_generator/    # Main package
│   ├── core/                        # Core engine
│   ├── generators/                  # V7, V8, V9 generators
│   ├── adapters/                    # Data adapters
│   ├── analytics/                   # Quality metrics & analysis
│   ├── visualization/               # Charts & plots
│   └── utils/                       # Utilities
├── scripts/                         # Executable scripts
│   ├── generation/                  # Generation scripts
│   ├── validation/                  # Validation scripts
│   └── maintenance/                 # Maintenance scripts
├── data/                            # Generated datasets
├── docs/                            # Documentation
├── tests/                           # Test suite
└── examples/                        # Usage examples
```

## 📚 Documentation

- **[Getting Started Guide](docs/guides/getting_started.md)** - Installation and first run
- **[Quick Start Tutorial](docs/guides/quick_start.md)** - Step-by-step tutorial
- **[API Reference](docs/api/)** - Complete API documentation
- **[Configuration Guide](docs/guides/configuration.md)** - Configuration options
- **[PRD Documentation](docs/prd/)** - Product requirements (all versions)

## 🔧 Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/synthetic_data_generator

# Run specific test suite
pytest tests/unit/test_generators/
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

## 📈 Usage Examples

### Generate Custom Dataset

```python
from synthetic_data_generator.generators.v9 import V9BalancedGenerator

generator = V9BalancedGenerator()
generator.generate(
    start_date="2024-01-01",
    end_date="2024-03-31",
    output_dir="data/custom/"
)
```

### Validate Data Quality

```python
from synthetic_data_generator.analytics import QualityMetrics

metrics = QualityMetrics()
report = metrics.analyze("data/generated/v9_balanced/hourly/")
print(report.summary())
```

### Create Visualizations

```python
from synthetic_data_generator.visualization import OHLCCharts

plotter = OHLCCharts()
plotter.plot_chart(
    data_file="data/generated/v9_balanced/hourly/NIFTY_2023-01-01.csv",
    output_file="charts/nifty_jan_2023.html"
)
```

## 🛠️ Configuration

Configuration is managed through `config/` directory:

- `base_config.py` - Base configuration
- `development.py` - Development settings
- `production.py` - Production settings
- `logging.yaml` - Logging configuration

## 📊 Data Quality

All generated data undergoes comprehensive quality checks:

- ✅ Price continuity validation
- ✅ Delta distance accuracy
- ✅ Trend consistency
- ✅ Greeks calculations verification
- ✅ Volatility clustering patterns

Quality reports available in `docs/reports/validation/`

## 🤝 Contributing

This is a private project extracted from NikAlgoBulls trading strategy development. For internal use.

## 📝 License

MIT License - See LICENSE file for details

## 📞 Contact & Support

Part of the NikAlgoBulls algorithmic trading project.

**Related Projects**:
- NikAlgoBulls - Main trading strategy repository

## 🔄 Version History

- **v9.0** (Current) - Balanced hourly data with improved quality
- **v8.0** - Extended intraday with enhanced features
- **v7.0** - VIX-based intraday generation
- **v1.0-v6.0** - Historical development versions (archived)

See [CHANGELOG.md](CHANGELOG.md) for detailed version history.

## 🎯 Roadmap

- [ ] Real-time data generation API
- [ ] Additional market conditions (high volatility events)
- [ ] Multi-asset support (Bank Nifty, Fin Nifty)
- [ ] Machine learning-based pattern generation
- [ ] Cloud deployment support

---

**Generated from**: Zerodha Strategy Development Project
**Migration Date**: October 5, 2025
**Migration Plan**: v2.0
