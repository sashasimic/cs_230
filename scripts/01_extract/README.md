# Data Extraction Scripts

Scripts for fetching and loading financial data from various sources into BigQuery.

## Overview

This folder contains scripts for:
- **Loading OHLCV data** from Polygon.io API
- **Computing synthetic indicators** (SMA, RSI, MACD, etc.)
- **Verifying data quality** and alignment
- **Setting up BigQuery tables** and schemas

## Core Scripts

### Data Loading

#### `tickers_load.py` - Main Orchestrator
Main pipeline for loading multi-ticker data.

```bash
# Load all tickers from config
python scripts/01_extract/tickers_load.py

# Load specific ticker
python scripts/01_extract/tickers_load.py --ticker SPY

# Force reload (overwrite existing data)
python scripts/01_extract/tickers_load.py --reload
```

**Features:**
- Loads multiple tickers from `configs/tickers.yaml`
- Delegates to Polygon and synthetic loaders
- Runs verification after loading
- Handles date range defaults from config

#### `tickers_load_polygon.py` - Polygon.io Data Fetcher
Fetches raw OHLCV data from Polygon.io API.

**Features:**
- Handles pagination and rate limiting
- Supports multiple frequencies (15m, 1h, daily)
- Deduplicates on (ticker, timestamp, frequency)
- Merges into BigQuery staging → main table

#### `tickers_load_synthetic.py` - Technical Indicators
Computes synthetic indicators from raw OHLCV data.

**Features:**
- Calculates SMA, EMA, RSI, MACD, Bollinger Bands, ATR, etc.
- Respects warmup periods for each indicator
- Stores in separate BigQuery table
- Skips existing data unless `--reload` is used

### Data Verification

#### `tickers_verify.py` - Main Verification Orchestrator
Delegates verification to specialized scripts.

```bash
python scripts/01_extract/tickers_verify.py \
  --ticker SPY --frequency daily \
  --start 2024-01-01 --end 2025-05-05 \
  --export
```

**Output:**
- Console: Verification summary with statistics
- Files (with `--export`): `temp/tickers_verification_output.{parquet,csv}`

#### `tickers_verify_polygon.py` - Raw Data Verification
Verifies OHLCV data quality.

**Checks:**
- Missing timestamps and gaps
- Weekend/holiday exclusion for stock data
- Data range validation
- Volume and price sanity checks

#### `tickers_verify_synthetic.py` - Indicator Verification
Verifies synthetic indicators.

**Checks:**
- Indicator coverage across date range
- Warmup period validation
- Alignment with raw data
- Value range checks

### Setup

#### `setup_bigquery_tables.py` - BigQuery Schema Setup
Creates BigQuery tables with proper schemas.

```bash
python scripts/01_extract/setup_bigquery_tables.py
```

**Creates:**
- `tickers_raw` - OHLCV data table
- `tickers_raw_staging` - Staging table for merges
- `tickers_synthetic` - Technical indicators table
- `tickers_synthetic_staging` - Staging for indicators

## Configuration

### Environment Variables (`.env`)
```bash
GCP_PROJECT_ID=your-project-id
BQ_DATASET=raw_dataset
BQ_TABLE=tickers_raw
BQ_SYNTHETIC_TABLE=tickers_synthetic
POLYGON_API_KEY=your-polygon-api-key
```

### Tickers Config (`configs/tickers.yaml`)
```yaml
tickers:
  inflation:
    - SPY  # S&P 500
    - QQQ  # NASDAQ 100
  credit:
    - LQD  # Investment Grade Bonds
    - HYG  # High Yield Bonds
  commodities:
    - GLD  # Gold
    - USO  # Oil

default_frequency: daily
default_start_date: "2022-01-01"
default_end_date: "2025-05-05"

indicators:
  - sma_50
  - sma_200
  - ema_12
  - ema_26
  - rsi_14
  - macd
  - bb_upper
  - bb_lower
```

## Data Flow

```
1. Polygon.io API
   ↓
2. tickers_load_polygon.py
   ↓
3. BigQuery (tickers_raw_staging)
   ↓
4. Merge to tickers_raw (deduplicate)
   ↓
5. tickers_load_synthetic.py
   ↓
6. BigQuery (tickers_synthetic_staging)
   ↓
7. Merge to tickers_synthetic
   ↓
8. tickers_verify.py (validation)
```

## BigQuery Schema

### Raw Table (`tickers_raw`)
```sql
CREATE TABLE tickers_raw (
  ticker STRING NOT NULL,
  timestamp TIMESTAMP NOT NULL,
  date DATE NOT NULL,
  open FLOAT64,
  high FLOAT64,
  low FLOAT64,
  close FLOAT64,
  volume INT64,
  frequency STRING NOT NULL
)
PARTITION BY DATE(timestamp)
CLUSTER BY ticker, frequency;
```

### Synthetic Table (`tickers_synthetic`)
```sql
CREATE TABLE tickers_synthetic (
  ticker STRING NOT NULL,
  timestamp TIMESTAMP NOT NULL,
  frequency STRING NOT NULL,
  -- Moving Averages
  sma_50 FLOAT64,
  sma_200 FLOAT64,
  ema_12 FLOAT64,
  ema_26 FLOAT64,
  -- Momentum
  rsi_14 FLOAT64,
  macd FLOAT64,
  macd_signal FLOAT64,
  macd_hist FLOAT64,
  -- Volatility
  bb_upper FLOAT64,
  bb_middle FLOAT64,
  bb_lower FLOAT64,
  atr_14 FLOAT64
)
PARTITION BY DATE(timestamp)
CLUSTER BY ticker, frequency;
```

## Common Workflows

### Initial Setup
```bash
# 1. Create BigQuery tables
python scripts/01_extract/setup_bigquery_tables.py

# 2. Load all tickers from config
python scripts/01_extract/tickers_load.py

# 3. Verify data quality
python scripts/01_extract/tickers_verify.py \
  --ticker SPY --frequency daily \
  --start 2024-01-01 --end 2025-05-05 \
  --export
```

### Daily Updates
```bash
# Load latest data (incremental)
python scripts/01_extract/tickers_load.py
```

### Reload Specific Ticker
```bash
# Force reload SPY data
python scripts/01_extract/tickers_load.py --ticker SPY --reload
```

### Export Data for Analysis
```bash
# Export combined raw + synthetic data
python scripts/01_extract/tickers_verify.py \
  --ticker SPY --frequency daily \
  --start 2024-01-01 --end 2025-05-05 \
  --export

# Output: temp/tickers_verification_output.{parquet,csv}
```

## Error Handling

- **API rate limits**: Scripts automatically handle Polygon.io rate limits with retries
- **Missing data**: Gaps are logged and can be inspected with verification scripts
- **Duplicates**: Automatic deduplication on (ticker, timestamp, frequency) during merge
- **Warmup periods**: Synthetic indicators respect required warmup periods (e.g., SMA_200 needs 200 bars)
- **Weekends/holidays**: Verification scripts can exclude expected market closures

## Output Files

- **BigQuery tables**: `{project}.{dataset}.tickers_raw` and `tickers_synthetic`
- **Verification exports**: `temp/tickers_verification_output.{parquet,csv}`
- **Logs**: Console output with detailed progress, statistics, and error messages

## Tips

- Use `--reload` sparingly to avoid unnecessary API calls
- Run verification after loading to catch data quality issues early
- Export verification data for offline analysis and debugging
- Check `configs/tickers.yaml` for default date ranges and ticker lists
- Monitor Polygon.io API usage to stay within rate limits