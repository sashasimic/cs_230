# Scripts Directory

Organized by workflow stage for the CS230 inflation predictor project.

## Directory Structure

```
scripts/
├── 01_extract/              # Phase 1: Data Extraction from BigQuery
├── 02_features/             # Phase 2: Feature Engineering  
├── 03_training/             # Phase 3: Model Training
└── dummy/                   # Dummy/Synthetic Data Generation
```

---

## Phase 1: Data Extraction

### `01_extract/extract_from_bigquery.py`

**Extract raw OHLCV data from BigQuery**

Extract stock/index data from BigQuery and save as parquet for local processing.

**Usage:**
```bash
# Check data availability for configured tickers
python scripts/01_extract/extract_from_bigquery.py --check-availability

# Extract all tickers to parquet
python scripts/01_extract/extract_from_bigquery.py

# Force re-download
python scripts/01_extract/extract_from_bigquery.py --force

# Add/remove tickers
python scripts/01_extract/extract_from_bigquery.py --add-ticker AAPL MSFT
python scripts/01_extract/extract_from_bigquery.py --remove-ticker AAPL
python scripts/01_extract/extract_from_bigquery.py --replace-tickers SPY QQQ
```

**Configuration:**
- Edit `configs/tickers.yaml` to configure tickers, date ranges, BigQuery details
- Outputs to `data/raw/stocks_raw.parquet` by default

**Requirements:**
- GCP authentication (see main README)
- BigQuery access (Data Viewer + Job User roles)

---

### `01_extract/verify_extraction.py`

**Verify extracted raw stock data quality**

Run data quality checks on extracted raw OHLCV data from Phase 1.

**Usage:**
```bash
# Full verification
python scripts/01_extract/verify_extraction.py

# Check specific ticker with detailed stats
python scripts/01_extract/verify_extraction.py --ticker SPY

# Verify custom file
python scripts/01_extract/verify_extraction.py --file data/raw/custom.parquet
```

**Checks:**
- ✅ No weekends (trading days only)
- ✅ Complete OHLCV columns for all tickers
- ✅ No duplicate dates
- ✅ No unusual date gaps (>4 days)
- ✅ Missing values summary
- ✅ Data quality issues (negative/zero prices)

---

## Phase 2: Feature Engineering

### `02_features/build_features.py`

**Build features from raw OHLCV data**

Transform raw stock data into ML-ready features with technical indicators and multi-horizon targets.

**Usage:**
```bash
# Use default config
python scripts/02_features/build_features.py

# Use custom config
python scripts/02_features/build_features.py --config configs/features_custom.yaml

# Override output directory
python scripts/02_features/build_features.py --output-dir data/processed_v2

# Skip sequences (for MLP models)
python scripts/02_features/build_features.py --no-sequences
```

**What it does:**
- Loads raw OHLCV data from Phase 1
- Computes technical indicators (returns, volatility, MA, RSI, momentum)
- Creates weighted multi-horizon targets (1M, 3M, 6M)
- Removes highly correlated features
- Scales features per-ticker
- Splits into train/val/test (chronological)
- Saves processed parquet files

**Outputs:**
- `data/processed/train.parquet` - Training set
- `data/processed/val.parquet` - Validation set
- `data/processed/test.parquet` - Test set
- `data/processed/scaler.pkl` - Fitted scalers
- `data/processed/feature_names.txt` - Feature list

**Configuration:**
Edit `configs/features.yaml` to customize:
- Input tickers and columns
- Technical indicators
- Target horizons and weights
- Feature engineering options
- Train/val/test split ratios

---

### `02_features/verify_features.py`

**Verify processed feature data quality**

Run comprehensive checks on processed train/val/test data.

**Usage:**
```bash
# Verify default processed data
python scripts/02_features/verify_features.py

# Verify custom directory
python scripts/02_features/verify_features.py --data-dir data/processed_v2
```

**Checks performed:**
- ✅ Column consistency across splits
- ✅ Missing values in features/targets
- ✅ Feature scaling (mean~0, std~1)
- ✅ Data leakage (temporal overlap)
- ✅ Target distribution statistics
- ✅ Outlier detection (>3 std)
- ✅ Scaler and feature names files

**Exit codes:**
- `0` - All checks passed
- `1` - Some checks failed

---

## Phase 3: Model Training

### `03_training/test_architectures.py`

**Test model architectures**

Quick script to test MLP, LSTM, and Transformer architectures before full training.

**Usage:**
```bash
python scripts/03_training/test_architectures.py
```

**Tests:**
- Model instantiation
- Forward pass with correct shapes
- Gradient flow check
- Quick overfitting test on small batch

---

## Dummy Data Generation

### `dummy/generate_dummy_data.py`

**Generate synthetic time series data**

Create dummy datasets when you don't have access to real BigQuery data.

**Usage:**
```bash
# Generate default dataset (10,000 samples, 10 features)
python scripts/dummy/generate_dummy_data.py

# Custom configuration
python scripts/dummy/generate_dummy_data.py \
  --n-samples 50000 \
  --n-features 20 \
  --output-dir data/dummy
```

**Use cases:**
- Testing pipeline without BigQuery access
- Rapid prototyping
- CI/CD testing

---

### `dummy/dummy_generator.py`

**Synthetic financial data generator module**

Utility module used by `generate_dummy_data.py` to create realistic synthetic OHLCV data.

**Features:**
- Generates random walk price data
- Creates realistic volume patterns
- Adds market noise and volatility
- Exports to parquet format matching BigQuery structure

**Used by:** `generate_dummy_data.py`