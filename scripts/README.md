# Utility Scripts

This directory contains standalone utility scripts for the CS230 project.

## Available Scripts

### `extract_tickers_from_bigquery.py`

**Phase 1: Extract raw OHLCV data from BigQuery**

Extract stock/index data from BigQuery and save as parquet for local processing.

**Usage:**
```bash
# Check data availability for configured tickers
python scripts/extract_tickers_from_bigquery.py --check-availability

# Extract all tickers to parquet
python scripts/extract_tickers_from_bigquery.py

# Force re-download
python scripts/extract_tickers_from_bigquery.py --force

# Add/remove tickers
python scripts/extract_tickers_from_bigquery.py --add-ticker AAPL MSFT
python scripts/extract_tickers_from_bigquery.py --remove-ticker AAPL
python scripts/extract_tickers_from_bigquery.py --replace-tickers SPY QQQ
```

**Configuration:**
- Edit `configs/tickers.yaml` to configure tickers, date ranges, BigQuery details
- Outputs to `data/raw/stocks_raw.parquet` by default

**Requirements:**
- GCP authentication (see main README)
- BigQuery access (Data Viewer + Job User roles)

---

### `verify_raw_stocks_data.py`

**Verify extracted parquet data quality**

Run data quality checks on extracted raw stock data.

**Usage:**
```bash
# Full verification
python scripts/verify_raw_stocks_data.py

# Check specific ticker with detailed stats
python scripts/verify_raw_stocks_data.py --ticker SPY

# Verify custom file
python scripts/verify_raw_stocks_data.py --file data/raw/custom.parquet
```

**Checks:**
- ✅ No weekends (trading days only)
- ✅ Complete OHLCV columns for all tickers
- ✅ No duplicate dates
- ✅ No unusual date gaps (>4 days)
- ✅ Missing values summary
- ✅ Data quality issues (negative/zero prices)

---

### `generate_data.py`

**Generate synthetic time series data for testing**

Create dummy datasets when you don't have access to real data.

**Usage:**
```bash
# Generate default dataset (10,000 samples, 10 features)
python scripts/generate_data.py

# Custom configuration
python scripts/generate_data.py \
  --n-samples 50000 \
  --n-features 20 \
  --output-dir data/dummy