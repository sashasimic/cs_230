# Dummy Data Generation

Scripts for generating synthetic time series data for testing and development.

## Overview

Use these scripts when you don't have access to real BigQuery data or need to quickly test the pipeline.

## Scripts

### `generate_dummy_data.py` - Generate Synthetic Datasets

Create dummy datasets with realistic financial time series patterns.

```bash
# Generate default dataset (10,000 samples, 10 features)
python scripts/dummy/generate_dummy_data.py

# Custom configuration
python scripts/dummy/generate_dummy_data.py \
  --n-samples 50000 \
  --n-features 20 \
  --output-dir data/dummy

# Specify tickers and date range
python scripts/dummy/generate_dummy_data.py \
  --tickers SPY QQQ IWM \
  --start-date 2020-01-01 \
  --end-date 2024-12-31
```

**Parameters:**
- `--n-samples`: Number of time steps (default: 10,000)
- `--n-features`: Number of features (default: 10)
- `--output-dir`: Output directory (default: data/dummy)
- `--tickers`: Ticker symbols to generate (default: SPY)
- `--start-date`: Start date for synthetic data
- `--end-date`: End date for synthetic data

**Output:**
- `data/dummy/tickers_dummy.parquet` - Synthetic OHLCV data
- Matches BigQuery schema for compatibility

### `dummy_generator.py` - Synthetic Data Generator Module

Utility module used by `generate_dummy_data.py` to create realistic synthetic financial data.

**Features:**
- Generates random walk price data
- Creates realistic volume patterns
- Adds market noise and volatility
- Exports to parquet format matching BigQuery structure

**Used by:** `generate_dummy_data.py` (not called directly)

## Use Cases

### 1. Testing Pipeline Without BigQuery

```bash
# Generate dummy data
python scripts/dummy/generate_dummy_data.py --n-samples 5000

# Use dummy data in feature engineering
python scripts/02_features/build_features.py --input data/dummy/tickers_dummy.parquet

# Train model on dummy data
python scripts/03_training/train_model.py
```

### 2. Rapid Prototyping

```bash
# Quick test with small dataset
python scripts/dummy/generate_dummy_data.py --n-samples 1000 --n-features 5
```

### 3. CI/CD Testing

```bash
# Generate minimal test data for automated tests
python scripts/dummy/generate_dummy_data.py \
  --n-samples 100 \
  --output-dir tests/fixtures
```

### 4. Development Without GCP Access

```bash
# Generate realistic multi-ticker dataset
python scripts/dummy/generate_dummy_data.py \
  --tickers SPY QQQ IWM RSP \
  --n-samples 10000 \
  --start-date 2020-01-01
```

## Data Format

Generated data matches the BigQuery schema:

```python
{
    'ticker': str,           # Ticker symbol
    'timestamp': datetime,   # Timestamp
    'date': date,           # Date
    'open': float,          # Opening price
    'high': float,          # High price
    'low': float,           # Low price
    'close': float,         # Closing price
    'volume': int,          # Trading volume
    'frequency': str        # Data frequency (e.g., 'daily')
}
```

## Synthetic Data Characteristics

- **Price movement**: Random walk with drift
- **Volatility**: Realistic intraday ranges (high/low)
- **Volume**: Log-normal distribution with trends
- **Noise**: Gaussian noise added for realism
- **No weekends**: Only trading days generated
- **Correlations**: Can specify cross-ticker correlations

## Tips

- **Start small**: Use 1,000-5,000 samples for quick tests
- **Match real data**: Use similar feature counts as production
- **Test edge cases**: Generate data with missing values, outliers
- **Version control**: Don't commit large dummy datasets (add to `.gitignore`)

## Limitations

⚠️ **Dummy data is for testing only:**
- No real market patterns or relationships
- Cannot be used for actual predictions
- May not expose all data quality issues
- Training on dummy data won't produce useful models

## Integration with Pipeline

Dummy data integrates seamlessly with the rest of the pipeline:

```
1. Generate dummy data
   python scripts/dummy/generate_dummy_data.py
   ↓
2. Build features
   python scripts/02_features/tft/tft_data_loader.py
   ↓
3. Train model
   python scripts/03_training/tft/tft_train_local.py
```

All downstream scripts work identically with dummy or real data!