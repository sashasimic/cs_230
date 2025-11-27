# Data Extraction Scripts

Scripts for fetching and loading financial data from various sources into BigQuery.

## Overview

This folder contains scripts for:
- **Loading OHLCV data** from Polygon.io API (tickers)
- **Loading GDELT sentiment data** from Google's public BigQuery dataset
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

#### `gdelt_load.py` - GDELT Sentiment Data Fetcher
Fetches news sentiment data from Google's public GDELT BigQuery dataset.

```bash
# Load using config (requires topic group selection)
python scripts/01_extract/gdelt_load.py \
  --config configs/gdelt.yaml \
  --topic-group inflation_prices

# Load monetary policy sentiment
python scripts/01_extract/gdelt_load.py \
  --config configs/gdelt.yaml \
  --topic-group monetary_policy

# Reload existing data (clears old data for this topic group + frequency)
python scripts/01_extract/gdelt_load.py \
  --config configs/gdelt.yaml \
  --topic-group inflation_prices \
  --reload

# Load with custom date range
python scripts/01_extract/gdelt_load.py \
  --topic-group inflation_prices \
  --start-date 2024-01-01 \
  --end-date 2024-12-31
```

**Features:**
- Queries GDELT GKG (Global Knowledge Graph) public dataset
- **Topic groups**: Organized into `inflation_prices` and `monetary_policy`
- **US-focused**: Filters for articles mentioning US locations
- **Required parameter**: `--topic-group` must be specified
- Aggregates sentiment to configurable frequency (15m, 1h, 4h, 1d, 1w)
- Weighted sentiment scores by article word count
- Saves directly to BigQuery with `topic_group_id` tagging
- `--reload` flag clears existing data for this topic group + frequency
- **Duplicate prevention**: Automatically checks for existing data before loading

### Data Verification

#### `tickers_verify.py` - Main Verification Orchestrator
Master script that orchestrates both raw and synthetic data verification.

**Quick Mode (Multi-Ticker Summary):**
```bash
# Verify specific tickers
python scripts/01_extract/tickers_verify.py \
  --tickers CORN CANE COW UBC UGA \
  --frequency daily \
  --quick \
  --exclude-weekends-and-holidays

# Verify ticker group
python scripts/01_extract/tickers_verify.py \
  --group inflation \
  --frequency daily \
  --quick \
  --exclude-weekends-and-holidays
```

**Detailed Mode (Single Ticker):**
```bash
python scripts/01_extract/tickers_verify.py \
  --ticker SPY --frequency daily \
  --start 2024-01-01 --end 2025-05-05 \
  --export
```

**Output:**
- **Quick mode**: Summary table showing data ranges, gaps, and status for all tickers
- **Detailed mode**: Full verification report with statistics and optional export
- Files (with `--export`): `temp/tickers_verification_output.{parquet,csv}`

#### `tickers_verify_polygon.py` - Raw Data Verification
Verifies OHLCV data quality from Polygon.io.

**Quick Mode Examples:**
```bash
# Quick verify multiple tickers
python scripts/01_extract/tickers_verify_polygon.py \
  --tickers CORN CANE COW UBC \
  --frequency daily \
  --quick \
  --exclude-weekends-and-holidays

# Quick verify ticker group
python scripts/01_extract/tickers_verify_polygon.py \
  --group inflation \
  --frequency daily \
  --quick
```

**Checks:**
- Missing timestamps and gaps
- Weekend/holiday exclusion for stock data (use `--exclude-weekends-and-holidays`)
- Data range validation
- Volume and price sanity checks
- Duplicate detection

#### `tickers_verify_synthetic.py` - Indicator Verification
Verifies synthetic indicators computed from raw data.

**Quick Mode Examples:**
```bash
# Quick verify multiple tickers
python scripts/01_extract/tickers_verify_synthetic.py \
  --tickers CORN CANE COW UBC \
  --frequency daily \
  --quick

# Quick verify ticker group
python scripts/01_extract/tickers_verify_synthetic.py \
  --group commodities \
  --frequency daily \
  --quick
```

**Checks:**
- Indicator coverage across date range
- Warmup period validation
- Alignment with raw data
- Value range checks
- Gap detection

#### `gdelt_verify.py` - GDELT Data Verification
Verifies GDELT sentiment data quality and completeness.

```bash
# Verify inflation prices sentiment (requires topic group)
python scripts/01_extract/gdelt_verify.py \
  --config configs/gdelt.yaml \
  --topic-group inflation_prices \
  --frequency 1d

# Verify monetary policy sentiment
python scripts/01_extract/gdelt_verify.py \
  --config configs/gdelt.yaml \
  --topic-group monetary_policy \
  --frequency 1d

# Full verification for specific date range
python scripts/01_extract/gdelt_verify.py \
  --topic-group inflation_prices \
  --start 2025-11-01 \
  --end 2025-11-10 \
  --frequency 1d

# Quick completeness check only
python scripts/01_extract/gdelt_verify.py \
  --topic-group inflation_prices \
  --frequency 1d \
  --completeness-only

# Export sample data
python scripts/01_extract/gdelt_verify.py \
  --topic-group inflation_prices \
  --frequency 1d \
  --export
```

**Checks:**
- Data completeness (missing intervals)
- Sentiment value ranges (tone, polarity)
- Duplicate timestamps (per topic group)
- Article count statistics
- Extreme sentiment dates

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
BQ_GDELT_TABLE=gdelt_sentiment
BQ_GDELT_STAGING_TABLE=gdelt_sentiment_staging
POLYGON_API_KEY=your-polygon-api-key
```

### Tickers Config (`configs/tickers.yaml`)
```yaml
# Ticker groups for quick verification
ticker_groups:
  inflation:
    - DBC      # Broad Commodities Index
    - USO      # Crude Oil
    - GLD      # Gold
    - TIP      # TIPS
    # ... more inflation-related tickers
  
  commodities:
    - DBC
    - USO
    - CORN
    - CANE
    # ... more commodity tickers
  
  equity:
    - SPY
    - QQQ
    - IWM
    - RSP
  
  # More groups: energy, agriculture, sectors, etc.

# Flat list (backwards compatibility)
tickers:
  - SPY  # S&P 500
  - QQQ  # NASDAQ 100
  - IWM  # Russell 2000
  - RSP  # Equal-Weight S&P 500
  # ... more tickers

date_range:
  start_date: '2015-11-12'
  end_date: '2025-11-13'

indicators:
  - returns
  - volatility_20
  - momentum_10
  - sma_50
  - sma_200
  # ... more indicators
```

### GDELT Config (`configs/gdelt.yaml`)
```yaml
# Topic Groups - organized by theme
topic_groups:
  # INFLATION & PRICES
  inflation_prices:
    description: "Inflation, consumer prices, and cost of living"
    topics:
      - INFLATION
      - CONSUMER_PRICE
      - PRICE_STABILITY
      - COST_OF_LIVING
      - FOOD_PRICE
      - HOUSING_PRICES
      - NATGASPRICE
      - FUELPRICES
      - FOOD_STAPLE
  
  # MONETARY POLICY & CENTRAL BANKS
  monetary_policy:
    description: "Federal Reserve, interest rates, and monetary policy"
    topics:
      - FED
      - FEDERAL_RESERVE
      - INTEREST_RATE
      - MONETARY_POLICY
      - CENTRAL_BANK
      - FOMC
      - JEROME_POWELL

date_range:
  start_date: '2015-11-12'
  end_date: '2025-11-13'

aggregation:
  frequency: '1d'  # Daily aggregation (15m, 1h, 4h, 1d, 1w)
  weight_by_word_count: true

bigquery:
  table_name: 'gdelt_sentiment'
  staging_table_name: 'gdelt_sentiment_staging'
  partition_by: 'date'

quality:
  require_tone: true
  min_word_count: 0
```

## Data Flow

### Tickers Pipeline
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

### GDELT Pipeline
```
1. GDELT Public BigQuery Dataset (gdelt-bq.gdeltv2.gkg)
   ↓
2. gdelt_load.py (query & filter by topics)
   ↓
3. Resample to target frequency (15m → 1d)
   ↓
4. BigQuery (gdelt_sentiment_staging)
   ↓
5. Merge to gdelt_sentiment (deduplicate)
   ↓
6. gdelt_verify.py (validation)
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

### GDELT Table (`gdelt_sentiment`)
```sql
CREATE TABLE gdelt_sentiment (
  timestamp TIMESTAMP NOT NULL,
  date DATE NOT NULL,
  frequency STRING NOT NULL,
  topic_group_id STRING NOT NULL,  -- 'inflation_prices' or 'monetary_policy'
  weighted_avg_tone FLOAT64,
  weighted_avg_positive FLOAT64,
  weighted_avg_negative FLOAT64,
  weighted_avg_polarity FLOAT64,
  num_articles INT64,
  num_sources INT64,
  total_word_count INT64,
  min_tone FLOAT64,
  max_tone FLOAT64,
  ingestion_timestamp TIMESTAMP
)
PARTITION BY date
CLUSTER BY topic_group_id, frequency;
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

# Quick verify all data
python scripts/01_extract/tickers_verify.py \
  --group all \
  --frequency daily \
  --quick \
  --exclude-weekends-and-holidays
```

### Reload Specific Ticker
```bash
# Force reload SPY data
python scripts/01_extract/tickers_load.py --ticker SPY --reload

# Verify after reload
python scripts/01_extract/tickers_verify_polygon.py \
  --ticker SPY \
  --frequency daily \
  --check-gaps \
  --exclude-weekends-and-holidays
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

### Load GDELT Sentiment Data
```bash
# Load inflation prices sentiment
python scripts/01_extract/gdelt_load.py \
  --config configs/gdelt.yaml \
  --topic-group inflation_prices

# Load monetary policy sentiment
python scripts/01_extract/gdelt_load.py \
  --config configs/gdelt.yaml \
  --topic-group monetary_policy

# Verify data quality
python scripts/01_extract/gdelt_verify.py \
  --config configs/gdelt.yaml \
  --topic-group inflation_prices \
  --frequency 1d
```

## Working Without Polygon API Access

If you don't have a Polygon.io API key, you can still work with existing ticker data.

**Automatic Skip Logic:**
- Scripts automatically check if data exists in BigQuery before calling Polygon API
- If data is found for a ticker+frequency, the Polygon API call is **automatically skipped**
- This allows users without API access to work seamlessly with existing data
- Use `--reload` flag only if you need to force re-fetching from Polygon

**Example - Running without API key:**
```bash
# This works fine if data already exists in BigQuery
python scripts/01_extract/tickers_load.py --frequency daily

# Output:
[1/5] 📊 Processing SPY...
  ⏭️  SKIPPED: SPY already has data in BigQuery (use --reload to overwrite)
[2/5] 📊 Processing QQQ...
  ⏭️  SKIPPED: QQQ already has data in BigQuery (use --reload to overwrite)
...
✅ Pipeline completed successfully
```

**If data doesn't exist:**
```bash
# Clear error message if API key is missing and data doesn't exist
[1/5] 📊 Processing NEWticker...
  ❌ ERROR: NEWticker has no data in BigQuery and POLYGON_API_KEY is not set
     Cannot fetch data from Polygon.io without API key
```

## Ticker Groups

Ticker groups allow you to verify related tickers together:

**Available Groups:**
- `inflation` - Commodities, energy, TIPS (inflation indicators)
- `commodities` - All commodity ETFs
- `energy` - Energy sector and oil/gas
- `agriculture` - Food and agricultural commodities
- `equity` - Major equity indices
- `sectors` - All sector ETFs
- `fixed_income` - Treasury bonds and TIPS
- `credit` - Credit market indicators
- `currency` - USD and Euro
- `real_estate` - Real estate tickers
- `all` - Complete list of all tickers

**Usage:**
```bash
# Verify inflation group
python scripts/01_extract/tickers_verify.py \
  --group inflation \
  --frequency daily \
  --quick
```

## Error Handling

- **No Polygon API key**: Scripts automatically skip tickers that already have data in BigQuery
- **API rate limits**: Scripts automatically handle Polygon.io rate limits with retries
- **Missing data**: Gaps are logged and can be inspected with verification scripts
- **Duplicates**: Automatic deduplication on (ticker, timestamp, frequency) during merge
- **Warmup periods**: Synthetic indicators respect required warmup periods (e.g., SMA_200 needs 200 bars)
- **Weekends/holidays**: Use `--exclude-weekends-and-holidays` flag to exclude expected market closures (~72 hour weekend gaps and ~96 hour holiday gaps)
- **Gap detection**: Quick mode shows gap counts; detailed mode shows specific gap dates and durations

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