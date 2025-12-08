# Feature Engineering Scripts

Scripts for preparing time-series features for model training.

## Structure

```
02_features/
├── tft/
│   └── tft_data_loader.py    # TFT-specific data loader
└── diagnose_nan.py            # Debug NaN values in data
```

## TFT Data Loader

### `tft/tft_data_loader.py` - Multi-Ticker Data Loader

Prepares time-series data for Temporal Fusion Transformer (TFT) training.

#### Features

- **Multi-ticker support**: Loads data for multiple tickers simultaneously
- **BigQuery integration**: Fetches OHLCV + synthetic indicators
- **Time-series sequences**: Creates lookback windows for sequential modeling
- **Normalization**: Standardizes features (fitted on train set only to prevent leakage)
- **Train/val/test splits**: Chronological splitting with configurable ratios
- **Caching**: Saves to NumPy arrays for fast subsequent loads
- **Export options**: Raw CSV/Parquet for validation, normalized arrays for training

#### Usage

```bash
# Load data (uses cache if available)
python scripts/02_features/tft/tft_data_loader.py

# Force reload from BigQuery (ignore cache)
python scripts/02_features/tft/tft_data_loader.py --reload

# Export debug data to temp/ directory
python scripts/02_features/tft/tft_data_loader.py --export-temp
```

#### Configuration

Uses `configs/model_tft_config.yaml`:

```yaml
data:
  tickers:
    symbols:
      - SPY
      - QQQ
      - IWM
      - RSP
    frequency: daily
    raw_features:
      - open
      - high
      - low
      - close
      - volume
    synthetic_features:
      - sma_50
      - sma_200
  
  start_date: "2022-01-01"
  end_date: "2025-05-05"
  lookback_window: 192  # Input sequence length
  prediction_horizons: [4, 8, 16]  # Multi-horizon targets
  
  train_ratio: 0.70
  val_ratio: 0.15
  # test_ratio: 0.15 (implicit)
  
  normalize: true
  normalization_method: standard  # or 'minmax'
```

#### Data Pipeline

```
1. Fetch from BigQuery
   ↓
2. Join raw OHLCV + synthetic indicators
   ↓
3. Add time features (hour, day_of_week, cyclical encodings)
   ↓
4. Create sequences (lookback windows)
   ↓
5. Split chronologically (train/val/test)
   ↓
6. Normalize (fit on train only)
   ↓
7. Save to NumPy arrays
```

#### Output Structure

##### `data/raw/` - Validation Data (Raw Prices)
```
data/raw/
├── tft_features.csv       # Human-readable CSV
└── tft_features.parquet   # Compressed Parquet
```

**Columns:**
- `timestamp`, `ticker`, `date`
- OHLCV: `open`, `high`, `low`, `close`, `volume`
- Indicators: `sma_50`, `sma_200`, etc.
- Targets: `target_4periods_ahead`, `target_8periods_ahead`, `target_16periods_ahead`

##### `data/processed/` - Training Data (Normalized Arrays)
```
data/processed/
├── X_train.npy          # Training sequences [samples, lookback, features]
├── y_train.npy          # Training targets [samples, horizons]
├── ts_train.npy         # Training timestamps [samples]
├── X_val.npy            # Validation sequences
├── y_val.npy            # Validation targets
├── ts_val.npy           # Validation timestamps
├── X_test.npy           # Test sequences
├── y_test.npy           # Test targets
├── ts_test.npy          # Test timestamps
├── scalers.pkl          # Fitted scalers for inverse transform
├── metadata.yaml        # Dataset metadata and statistics
└── feature_names.txt    # Feature list and usage notes
```

**Array Shapes:**
- `X`: `[num_samples, lookback_window, num_features]` - Input sequences
- `y`: `[num_samples, num_horizons]` - Multi-horizon targets
- `ts`: `[num_samples]` - Timestamps for each sequence

#### Example: Loading Processed Data

```python
import numpy as np
import pickle

# Load training data
X_train = np.load('data/processed/X_train.npy')
y_train = np.load('data/processed/y_train.npy')
ts_train = np.load('data/processed/ts_train.npy')

# Load scalers for inverse transform
with open('data/processed/scalers.pkl', 'rb') as f:
    scalers = pickle.load(f)

print(f"Training sequences: {X_train.shape}")  # e.g., (2198, 192, 7)
print(f"Training targets: {y_train.shape}")    # e.g., (2198, 3)
```

#### Normalization

**Features normalized:**
- OHLCV prices and volume
- Technical indicators (SMA, RSI, MACD, etc.)

**Features NOT normalized:**
- Time features (already in good ranges):
  - `hour_sin`, `hour_cos`
  - `day_sin`, `day_cos`
  - `month_sin`, `month_cos`
  - `is_weekend`

**Method:**
- StandardScaler (default): `(x - mean) / std`
- MinMaxScaler (optional): `(x - min) / (max - min)`

**Important:** Scalers are fitted ONLY on training data to prevent data leakage!

#### Multi-Horizon Targets

Targets are computed as **true k-period forward returns**:

```python
# For each horizon h in [4, 8, 16]
target_h = (close[t+h] - close[t]) / close[t]
```

Example:
- `target_4periods_ahead`: 4-day forward return
- `target_8periods_ahead`: 8-day forward return
- `target_16periods_ahead`: 16-day forward return

#### Caching Behavior

**First run:**
1. Fetches from BigQuery
2. Processes and normalizes
3. Saves to `data/processed/*.npy`
4. Takes ~30-60 seconds

**Subsequent runs:**
1. Loads from `data/processed/*.npy`
2. Skips BigQuery and processing
3. Takes ~1-2 seconds

**Force refresh:**
```bash
python scripts/02_features/tft/tft_data_loader.py --reload
```

## Utilities

### `diagnose_nan.py` - NaN Diagnostics

Debug tool for identifying NaN values in data.

```bash
python scripts/02_features/diagnose_nan.py
```

**Checks:**
- Missing values by column
- NaN patterns across time
- Potential causes (warmup periods, data gaps, etc.)

## Common Workflows

### Initial Data Preparation
```bash
# 1. Ensure data is loaded in BigQuery
python scripts/01_extract/tickers_load.py

# 2. Generate features and sequences
python scripts/02_features/tft/tft_data_loader.py

# 3. Verify output
ls -lh data/processed/
```

### Update Features After Config Change
```bash
# Force regenerate with new config
python scripts/02_features/tft/tft_data_loader.py --reload
```

### Debug NaN Issues
```bash
# Check for NaN values
python scripts/02_features/diagnose_nan.py

# Export raw data for inspection
python scripts/02_features/tft/tft_data_loader.py --export-temp
```

## Integration with Training

The data loader is used by training scripts:

```python
from scripts.02_features.tft.tft_data_loader import create_data_loaders

# Load data (uses cache if available)
dataloaders, scalers = create_data_loaders(
    config_path='configs/model_tft_config.yaml',
    force_refresh=False
)

# Train model
for batch_X, batch_y in dataloaders['train']:
    # batch_X: [batch_size, lookback, features]
    # batch_y: [batch_size, horizons]
    ...
```

## Performance Tips

- **Use caching**: Don't use `--reload` unless config/data changed
- **Batch size**: Adjust in config based on available memory
- **Lookback window**: Longer windows = more memory, better context
- **NumPy arrays**: Much faster than loading from BigQuery every time
- **Parquet format**: Compressed and fast for large datasets

## Troubleshooting

**Issue: NaN values in features**
- Check warmup periods for indicators (e.g., SMA_200 needs 200 bars)
- Verify data completeness in BigQuery
- Run `diagnose_nan.py` for detailed analysis

**Issue: Memory errors**
- Reduce `lookback_window` in config
- Reduce number of tickers
- Use smaller `batch_size`

**Issue: Slow loading**
- Use cached NumPy arrays (don't use `--reload`)
- Check BigQuery query performance
- Consider using Parquet instead of CSV for raw data