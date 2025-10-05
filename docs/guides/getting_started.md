# Getting Started with Synthetic Data Generator

Welcome to the Synthetic Data Generator! This guide will help you get up and running quickly.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- 5 GB free disk space (for data and processing)

## Installation

### 1. Clone or Navigate to Repository

```bash
cd /Users/nitindhawan/SyntheticDataGenerator
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

### 3. Install Dependencies

```bash
# Install production dependencies
pip install -r requirements.txt

# For development (includes testing, linting, etc.)
pip install -r requirements-dev.txt
```

### 4. Install Package in Development Mode

```bash
pip install -e .
```

### 5. Verify Installation

```bash
python -c "import synthetic_data_generator; print(synthetic_data_generator.__version__)"
```

You should see: `9.0.0`

## Quick Start

### Load Pre-Generated Data

The repository includes pre-generated datasets ready to use:

```python
from synthetic_data_generator.adapters import V9Adapter

# Initialize adapter
adapter = V9Adapter()

# Load data for a specific period
data = adapter.load_data(
    start_date="2023-01-01",
    end_date="2023-01-31"
)

print(f"Loaded {len(data)} rows of data")
print(data.head())
```

### Generate New Data

To generate custom datasets:

```python
from synthetic_data_generator.generators.v9 import V9BalancedGenerator

# Initialize generator
generator = V9BalancedGenerator()

# Generate data
generator.generate(
    start_date="2024-01-01",
    end_date="2024-03-31",
    output_dir="data/custom/"
)
```

### Analyze Data Quality

```python
from synthetic_data_generator.analytics import QualityMetrics

# Initialize quality metrics
metrics = QualityMetrics()

# Analyze dataset
report = metrics.analyze("data/generated/v9_balanced/hourly/")

# Print summary
print(report.summary())
```

### Create Visualizations

```python
from synthetic_data_generator.visualization import OHLCCharts

# Initialize plotter
plotter = OHLCCharts()

# Generate chart
plotter.plot_chart(
    data_file="data/generated/v9_balanced/hourly/NIFTY_OPTIONS_1H_20230101.csv",
    output_file="charts/nifty_jan_2023.html"
)
```

## Available Data Versions

### V9 - Balanced Hourly (Current - Recommended)
- **Location**: `data/generated/v9_balanced/hourly/`
- **Frequency**: 1 Hour
- **Size**: 130 MB
- **Status**: ✅ Production-ready

### V8 - Extended Intraday
- **Location**: `data/generated/v8_extended/intraday/`
- **Frequency**: 5 Minutes
- **Size**: 1.3 GB
- **Status**: 🟢 Stable

### V7 - VIX-Based Intraday
- **Location**: `data/generated/v7_vix/intraday/`
- **Frequency**: 5 Minutes
- **Size**: 551 MB
- **Status**: 🟡 Legacy

## Configuration

### Environment Variables

Copy the environment template:

```bash
cp .env.template .env
```

Edit `.env` to customize settings:

```bash
# Generator Settings
GENERATOR_VERSION=v9
DATA_OUTPUT_DIR=data/generated
LOG_LEVEL=INFO

# Data Generation Parameters
START_DATE=2023-01-01
END_DATE=2024-12-31
FREQUENCY=1H
```

### Configuration Files

Configuration is managed in `config/` directory:

- `base_config.py` - Base configuration
- `development.py` - Development settings
- `production.py` - Production settings

## Running Scripts

### Generate Data

```bash
# V9 (current)
python scripts/generation/generate_synthetic_data_july.py

# Historical data
python scripts/generation/generate_historical_intraday_data.py
```

### Validate Data

```bash
python scripts/validation/verify_v3_data.py
```

## Next Steps

- **[Quick Start Tutorial](quick_start.md)** - Step-by-step tutorial
- **[Configuration Guide](configuration.md)** - Detailed configuration options
- **[API Reference](../api/)** - Complete API documentation
- **[Examples](../../examples/)** - Code examples and notebooks

## Troubleshooting

### Import Errors

If you encounter import errors:

```bash
# Reinstall in development mode
pip install -e .
```

### Data Loading Errors

Check that data files exist:

```bash
ls -la data/generated/v9_balanced/hourly/
```

### Memory Issues

For large datasets, process in chunks:

```python
# Use iterator
for chunk in adapter.load_data_chunked(start_date, end_date, chunksize=10000):
    process(chunk)
```

## Support

For issues or questions:
- Check the [Troubleshooting Guide](troubleshooting.md)
- Review [API Documentation](../api/)
- See [Examples](../../examples/)

---

**Last Updated**: October 5, 2025
**Version**: 9.0.0
