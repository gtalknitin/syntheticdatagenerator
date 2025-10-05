# Quick Start Tutorial

This tutorial will walk you through using the Synthetic Data Generator in 5 minutes.

## Step 1: Verify Installation

```bash
cd /Users/nitindhawan/SyntheticDataGenerator
source venv/bin/activate
python -c "import synthetic_data_generator; print('Version:', synthetic_data_generator.__version__)"
```

**Expected Output**: `Version: 9.0.0`

## Step 2: Load Pre-Generated Data

Create a new Python file `example_load_data.py`:

```python
from synthetic_data_generator.adapters import V9Adapter
import pandas as pd

# Initialize adapter
adapter = V9Adapter()

# Load data for January 2023
data = adapter.load_data(
    start_date="2023-01-01",
    end_date="2023-01-31"
)

# Display basic information
print(f"✓ Loaded {len(data):,} rows")
print(f"✓ Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
print(f"✓ Columns: {', '.join(data.columns)}")
print("\nFirst 5 rows:")
print(data.head())
```

Run it:

```bash
python example_load_data.py
```

## Step 3: Analyze Data Quality

Create `example_analysis.py`:

```python
from synthetic_data_generator.analytics import DeltaAnalyzer
import pandas as pd

# Load a single day's data
df = pd.read_csv("data/generated/v9_balanced/hourly/NIFTY_OPTIONS_1H_20230103.csv")

# Analyze delta distance
analyzer = DeltaAnalyzer()
delta_stats = analyzer.analyze(df)

print("Delta Distance Analysis:")
print(f"✓ Mean distance: {delta_stats['mean']:.4f}")
print(f"✓ Std deviation: {delta_stats['std']:.4f}")
print(f"✓ Within target range: {delta_stats['in_range_pct']:.1f}%")
```

Run it:

```bash
python example_analysis.py
```

## Step 4: Create a Simple Visualization

Create `example_chart.py`:

```python
from synthetic_data_generator.visualization import OHLCCharts
import os

# Create output directory
os.makedirs("output_charts", exist_ok=True)

# Initialize chart generator
plotter = OHLCCharts()

# Generate interactive chart for one day
plotter.plot_chart(
    data_file="data/generated/v9_balanced/hourly/NIFTY_OPTIONS_1H_20230103.csv",
    output_file="output_charts/nifty_jan03_2023.html"
)

print("✓ Chart generated: output_charts/nifty_jan03_2023.html")
print("  Open the HTML file in your browser to view")
```

Run it:

```bash
python example_chart.py
open output_charts/nifty_jan03_2023.html  # Opens in default browser
```

## Step 5: Generate Custom Data

Create `example_generate.py`:

```python
from synthetic_data_generator.generators.v9 import V9BalancedGenerator
import os

# Create custom output directory
output_dir = "data/custom_generation"
os.makedirs(output_dir, exist_ok=True)

# Initialize generator
generator = V9BalancedGenerator()

# Generate one week of data
print("Generating custom dataset...")
generator.generate(
    start_date="2024-01-01",
    end_date="2024-01-07",
    output_dir=output_dir
)

print(f"✓ Data generated in {output_dir}/")
```

Run it:

```bash
python example_generate.py
```

**Note**: Data generation can take several minutes depending on the date range.

## Common Use Cases

### Use Case 1: Backtest Integration

```python
from synthetic_data_generator.adapters import V9Adapter

# Load data for backtesting
adapter = V9Adapter()
backtest_data = adapter.load_data(
    start_date="2023-01-01",
    end_date="2023-12-31"
)

# Use with your backtesting framework
# ... your backtest code here ...
```

### Use Case 2: Data Quality Validation

```python
from synthetic_data_generator.analytics import QualityMetrics

metrics = QualityMetrics()
report = metrics.analyze("data/generated/v9_balanced/hourly/")

# Check if data meets quality standards
if report.is_valid():
    print("✓ Data quality is acceptable")
else:
    print("✗ Data quality issues found")
    print(report.get_issues())
```

### Use Case 3: Compare Generator Versions

```python
import pandas as pd

# Load same period from different versions
v7_data = pd.read_csv("data/generated/v7_vix/intraday/NIFTY_OPTIONS_5M_20230103.csv")
v9_data = pd.read_csv("data/generated/v9_balanced/hourly/NIFTY_OPTIONS_1H_20230103.csv")

# Compare characteristics
print(f"V7 rows: {len(v7_data)}, V9 rows: {len(v9_data)}")
print(f"V7 mean IV: {v7_data['iv'].mean():.4f}")
print(f"V9 mean IV: {v9_data['iv'].mean():.4f}")
```

## Understanding the Data Format

Each CSV file contains:

| Column | Description | Example |
|--------|-------------|---------|
| `timestamp` | Date and time | 2023-01-03 09:15:00 |
| `open` | Opening price | 152.50 |
| `high` | Highest price | 155.75 |
| `low` | Lowest price | 151.25 |
| `close` | Closing price | 154.00 |
| `volume` | Trading volume | 125000 |
| `strike` | Option strike | 19000 |
| `option_type` | CE or PE | CE |
| `delta` | Option delta | 0.4523 |
| `gamma` | Option gamma | 0.0123 |
| `theta` | Option theta | -0.4521 |
| `vega` | Option vega | 15.23 |
| `iv` | Implied volatility | 0.1856 |

## Next Steps

Now that you've completed the quick start:

1. **Explore Examples**: Check out `examples/` directory for more code samples
2. **Read the Guides**: See `docs/guides/` for detailed documentation
3. **Review API Docs**: Study `docs/api/` for complete API reference
4. **Try Jupyter Notebooks**: Open `examples/notebooks/` for interactive examples

## Tips & Best Practices

### Memory Management
```python
# For large datasets, use chunking
for chunk in adapter.load_data_chunked("2023-01-01", "2023-12-31", chunksize=10000):
    process(chunk)
```

### Caching Results
```python
# Cache loaded data to avoid repeated reads
cache_file = "data/cache/jan_2023.pkl"
if os.path.exists(cache_file):
    data = pd.read_pickle(cache_file)
else:
    data = adapter.load_data("2023-01-01", "2023-01-31")
    data.to_pickle(cache_file)
```

### Quality Checks
```python
# Always validate data before use
assert not data.isnull().any().any(), "Data contains null values"
assert len(data) > 0, "Data is empty"
assert data['timestamp'].is_monotonic_increasing, "Timestamps not sorted"
```

## Troubleshooting

### Issue: "Module not found"
**Solution**: Reinstall in development mode
```bash
pip install -e .
```

### Issue: "Data file not found"
**Solution**: Verify data exists
```bash
ls -la data/generated/v9_balanced/hourly/
```

### Issue: "Memory error with large datasets"
**Solution**: Use chunked loading
```python
adapter.load_data_chunked(start, end, chunksize=5000)
```

---

**Congratulations!** You've completed the quick start tutorial. You're now ready to use the Synthetic Data Generator in your projects.

For more advanced usage, see the [Advanced Usage Guide](advanced_usage.md).
