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

### `01_extract/extract_tickers.py`

**Extract raw OHLCV data from BigQuery**

Extract stock/index data from BigQuery and save as parquet for local processing.

**Configuration:** Edit `configs/tickers.yaml` to set default tickers, date range, BigQuery project/dataset/table, and output path.

**Usage:**
```bash
# Extract using config file (configs/tickers.yaml)
python scripts/01_extract/extract_tickers.py

# Check data availability for configured tickers
python scripts/01_extract/extract_tickers.py --check-availability

# Force re-download even if file exists
python scripts/01_extract/extract_tickers.py --force

# Override tickers from command line (temporary, doesn't modify config)
python scripts/01_extract/extract_tickers.py --tickers SPY QQQ IWM

# Override date range (overrides config)
python scripts/01_extract/extract_tickers.py --start-date 2015-10-28 --end-date 2025-10-24

# Modify ticker list in config file (permanently updates tickers.yaml)
python scripts/01_extract/extract_tickers.py --add-tickers AAPL MSFT --save-config
python scripts/01_extract/extract_tickers.py --remove-tickers VIX --save-config
python scripts/01_extract/extract_tickers.py --replace-tickers SPY QQQ --save-config

# Override output path
python scripts/01_extract/extract_tickers.py --output data/raw/custom.parquet
```

**Configuration File (`configs/tickers.yaml`):**
- **Tickers**: List of stock/ETF symbols to extract
- **Date range**: Start/end dates (default: 2014-01-01 to today)
- **BigQuery**: Project ID, dataset ID, table name
- **Output**: File path and compression settings
- **Columns**: Which columns to extract (OHLCV, VWAP, etc.)

**Command-line arguments override config file settings.** Use `--save-config` to permanently update the config file.

**Requirements:**
- GCP authentication (see main README)
- BigQuery access (Data Viewer + Job User roles)

---

### `01_extract/extract_google_trends.py`

**Extract Google Trends data for inflation indicators**

Fetch daily search interest data for inflation-related keywords from Google Trends (US region).

**Configuration:** Edit `configs/google_trends.yaml` to set default keywords, date range, region, and API settings.

**Usage:**
```bash
# Extract using config file (configs/google_trends.yaml)
python scripts/01_extract/extract_google_trends.py

# Use custom config file
python scripts/01_extract/extract_google_trends.py --config configs/my_trends.yaml

# Test mode (last 30 days only)
python scripts/01_extract/extract_google_trends.py --test

# Override keywords from command line (overrides config)
python scripts/01_extract/extract_google_trends.py --keywords "inflation" "gas prices" "rent prices"

# Override date range (overrides config)
python scripts/01_extract/extract_google_trends.py --start-date 2015-10-28 --end-date 2025-10-24

# Full extraction with aggressive retry settings (if hitting rate limits)
python scripts/01_extract/extract_google_trends.py --start-date 2015-10-28 --end-date 2025-10-24 --throttle 10 --max-retries 8 --force

# Override region (e.g., UK)
python scripts/01_extract/extract_google_trends.py --geo UK

# Override throttling and retry settings
python scripts/01_extract/extract_google_trends.py --throttle 10.0 --max-retries 8

# Force re-download even if file exists
python scripts/01_extract/extract_google_trends.py --force
```

**Features:**
- ✅ Daily aggregate data
- ✅ US region by default (configurable)
- ✅ Automatic chunking for date ranges > 270 days
- ✅ **Progress tracking** - Shows chunk progress, percentage, and elapsed time
- ✅ **Exponential backoff retry** - Automatically retries on 429 rate limits (5 retries default)
- ✅ **Smart throttling** - 5s default with random jitter to avoid rate limiting
- ✅ **Countdown timers** - Shows remaining wait time during long retries
- ✅ Handles API errors gracefully
- ✅ Skip existing files unless `--force` is used

**Configuration File (`configs/google_trends.yaml`):**
- **Keywords**: List of search terms to track (default: inflation, food prices, gas prices, rent prices, etc.)
- **Date range**: Start/end dates (default: 2015-10-28 to today)
- **Region**: Geographic area (default: US)
- **API settings**: Throttle and retry parameters

**Command-line arguments override config file settings.**

**Output:**
- `data/raw/google_trends.parquet`
- Columns: `date` + one column per keyword (0-100 scale)

**Requirements:**
- Internet connection
- No authentication needed (public API)

**Progress Output Example:**
```
[Chunk 3/15] (20.0% complete)
Fetching data for: 2020-07-01 2021-03-27
Elapsed time: 45.2s
  ✓ Success. Waiting 5.3s before next request...
```

**Notes:**
- Google Trends data is relative (0-100 scale)
- Daily data limited to 270 days per request (auto-handled)
- Progress tracking shows chunk number, percentage, and elapsed time
- If rate limited (429 error), automatically retries with exponential backoff
- For long date ranges, expect 10-30 minutes of runtime
- Increase `--throttle` if you continue to hit rate limits

---

### `01_extract/verify_ticker_extraction.py`

**Verify extracted raw stock data quality**

Run data quality checks on extracted raw OHLCV data from Phase 1.

**Usage:**
```bash
# Full verification
python scripts/01_extract/verify_ticker_extraction.py

# Check specific ticker with detailed stats (includes date range)
python scripts/01_extract/verify_ticker_extraction.py --ticker SPY

# Verify custom file
python scripts/01_extract/verify_ticker_extraction.py --file data/raw/custom.parquet
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

## Phase 3: Model Training & Testing

### `03_training/inspect_model.py`

**Quick model architecture inspection (no data required)**

View model architecture, layer details, and parameter counts without running a forward pass.

**Usage:**
```bash
# Inspect LSTM model
python scripts/03_training/inspect_model.py --model-type lstm

# Inspect MLP with custom config
python scripts/03_training/inspect_model.py --model-type mlp --config configs/mlp_multi_horizon.yaml

# Custom input shape
python scripts/03_training/inspect_model.py --model-type lstm --seq-len 60 --n-features 200
```

**Features:**
- ✅ Fast (no forward pass)
- ✅ Shows model.summary()
- ✅ Counts trainable/non-trainable parameters
- ✅ Works with any config file

**When to use:**
- Quick architecture overview
- Comparing model sizes
- Documentation/reporting

---

### `03_training/test_architectures.py`

**Integration test with dummy data forward pass**

Test model can compile, run predictions, and produce correct output shapes using random dummy data.

**Usage:**
```bash
# Test LSTM with real feature count
python scripts/03_training/test_architectures.py --model-type lstm --n-features 163

# Test multi-horizon MLP
python scripts/03_training/test_architectures.py --model-type mlp --config configs/mlp_multi_horizon.yaml

# Custom test parameters
python scripts/03_training/test_architectures.py --model-type gru --seq-len 60 --batch-size 10
```

**Tests:**
- ✅ Model instantiation
- ✅ Forward pass with random data
- ✅ Output shape validation
- ✅ Multi-horizon output verification

**Important:** Uses **dummy random data**, not real parquet data. For testing architecture only.

**Does NOT:**
- ❌ Does NOT use real data from parquet files
- ❌ Does NOT train or evaluate

**When to use:**
- Before starting expensive training
- After modifying model architecture
- Debugging shape mismatches

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