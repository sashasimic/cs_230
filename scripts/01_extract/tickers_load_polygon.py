"""Ingest data from Polygon.io and save to BigQuery for Inflation prediction.

Fetches OHLCV data and/or technical indicators from Polygon API with configurable 
frequency (hourly, daily) and saves to BigQuery using staging + MERGE pattern.
"""
import os
import sys
import argparse
from datetime import datetime, timedelta, timezone
import requests
from google.cloud import bigquery
import pandas as pd
import time
from pathlib import Path
import yaml

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configuration
PROJECT_ID = os.environ.get('GCP_PROJECT_ID')
DATASET_ID = os.environ.get('BQ_DATASET', 'raw_dataset')
TABLE_NAME = os.environ.get('BQ_TABLE', 'raw_ohlcv')
STAGING_TABLE_NAME = os.environ.get('BQ_STAGING_TABLE', 'raw_ohlcv_staging')
INDICATORS_TABLE = os.environ.get('BQ_INDICATORS_TABLE', 'technical_indicators')
INDICATORS_STAGING_TABLE = os.environ.get('BQ_INDICATORS_STAGING_TABLE', 'technical_indicators_staging')
POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY')

RATE_LIMIT_DELAY = int(os.environ.get('RATE_LIMIT_DELAY', '12'))  # Seconds between API calls

# Technical indicator configurations
INDICATOR_CONFIGS = {
    'SMA': [10, 20, 50, 200],  # Simple Moving Average windows
    'EMA': [10, 20, 50, 200],  # Exponential Moving Average windows
    'RSI': [14],                # Relative Strength Index periods
    'MACD': [(12, 26, 9)]       # MACD fast, slow, signal periods
}

# Frequency mapping for Polygon API
FREQUENCY_MAP = {
    'hourly': {'multiplier': 1, 'timespan': 'hour'},
    'daily': {'multiplier': 1, 'timespan': 'day'},
    '1h': {'multiplier': 1, 'timespan': 'hour'},
    '1d': {'multiplier': 1, 'timespan': 'day'},
    '5m': {'multiplier': 5, 'timespan': 'minute'},
    '15m': {'multiplier': 15, 'timespan': 'minute'},
}


def fetch_ohlcv_data(
    ticker: str, 
    start_date: str, 
    end_date: str, 
    frequency: str = 'daily'
) -> list:
    """
    Fetch OHLCV data from Polygon.io with configurable frequency and pagination.
    
    Args:
        ticker: Stock symbol (e.g., 'SPY', 'SPY')
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        frequency: Data frequency ('hourly', 'daily', '5m', '15m', etc.)
    
    Returns:
        List of OHLCV bars
    """
    freq_config = FREQUENCY_MAP.get(frequency, FREQUENCY_MAP['daily'])
    multiplier = freq_config['multiplier']
    timespan = freq_config['timespan']
    
    # Handle tickers
    if ticker == 'SPY':
        polygon_ticker = 'SPY'
    elif '-' in ticker and ticker.startswith('X:'):
        polygon_ticker = ticker
    elif '-USD' in ticker:
        # Convert other currency
        base = ticker.replace('-USD', '')
        polygon_ticker = f'X:{base}USD'
    else:
        polygon_ticker = ticker
    
    url = f"https://api.polygon.io/v2/aggs/ticker/{polygon_ticker}/range/{multiplier}/{timespan}/{start_date}/{end_date}"
    params = {
        'adjusted': 'true',
        'sort': 'asc',
        'limit': 50000,
        'apiKey': POLYGON_API_KEY
    }
    
    all_results = []
    page = 1
    
    try:
        while url:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') == 'OK' and 'results' in data:
                results = data['results']
                all_results.extend(results)
                
                # Extract date range from this page
                if results:
                    first_date = datetime.fromtimestamp(results[0]['t'] / 1000).strftime('%Y-%m-%d')
                    last_date = datetime.fromtimestamp(results[-1]['t'] / 1000).strftime('%Y-%m-%d')
                    date_range = f"{first_date} to {last_date}"
                else:
                    date_range = "no data"
                
                if page == 1:
                    print(f"    📊 Page {page}: Got {len(results)} bars for {ticker} ({frequency}) [{date_range}]")
                else:
                    print(f"    📊 Page {page}: Got {len(results)} more bars (total: {len(all_results)}) [{date_range}]")
                
                # Check for pagination
                next_url = data.get('next_url')
                if next_url:
                    # Polygon's next_url doesn't include API key, add it back
                    url = next_url
                    params = {'apiKey': POLYGON_API_KEY}  # Re-add API key for next page
                    page += 1
                    time.sleep(RATE_LIMIT_DELAY)  # Rate limit between pages
                else:
                    # No more pages
                    break
            elif data.get('status') == 'ERROR':
                print(f"    ❌ API Error: {data.get('error', 'Unknown error')}")
                break
            else:
                # No results
                break
        
        if all_results:
            print(f"    ✅ Total: {len(all_results)} bars fetched across {page} page(s)")
        
        return all_results
        
    except Exception as e:
        print(f"    ❌ API Error: {str(e)}")
        return all_results if all_results else []


def convert_ohlcv_to_rows(ticker: str, bars: list, frequency: str = 'daily') -> list:
    """
    Convert OHLCV bars to BigQuery row format.
    
    Args:
        ticker: Stock symbol
        bars: List of OHLCV bars from Polygon
        frequency: Data frequency
    
    Returns:
        List of dictionaries ready for BigQuery insertion
    """
    rows = []
    for bar in bars:
        dt = datetime.fromtimestamp(bar['t'] / 1000)
        
        row = {
            'ticker': ticker,
            'timestamp': dt.strftime('%Y-%m-%d %H:%M:%S'),
            'date': dt.date().strftime('%Y-%m-%d'),
            'frequency': frequency,
            'open': bar.get('o'),
            'high': bar.get('h'),
            'low': bar.get('l'),
            'close': bar.get('c'),
            'volume': bar.get('v'),
            'vwap': bar.get('vw'),
            'transactions': bar.get('n'),
            'ingested_at': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        }
        rows.append(row)
    
    return rows


def fetch_technical_indicator(
    ticker: str,
    indicator_type: str,
    start_date: str,
    end_date: str,
    frequency: str = 'daily',
    window: int = None,
    **kwargs
) -> list:
    """
    Fetch technical indicator data from Polygon.io.
    
    Args:
        ticker: Stock symbol
        indicator_type: Type of indicator (SMA, EMA, RSI, MACD)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        frequency: Data frequency
        window: Time window/period for indicator
        **kwargs: Additional indicator-specific parameters
    
    Returns:
        List of indicator data points
    """
    freq_config = FREQUENCY_MAP.get(frequency, FREQUENCY_MAP['daily'])
    multiplier = freq_config['multiplier']
    timespan = freq_config['timespan']
    
    # Handle tickers
    if ticker == 'SPY':
        polygon_ticker = 'SPY'
    elif '-' in ticker and ticker.startswith('X:'):
        polygon_ticker = ticker
    elif '-USD' in ticker:
        base = ticker.replace('-USD', '')
        polygon_ticker = f'X:{base}USD'
    else:
        polygon_ticker = ticker
    
    # Build indicator URL based on type
    base_url = f"https://api.polygon.io/v1/indicators"
    
    if indicator_type in ['SMA', 'EMA']:
        indicator_lower = indicator_type.lower()
        url = f"{base_url}/{indicator_lower}/{polygon_ticker}"
        params = {
            'timespan': timespan,
            'multiplier': multiplier,  # Required for proper aggregation
            'adjusted': 'true',
            'window': window,
            'series_type': 'close',
            'timestamp.gte': start_date,
            'timestamp.lte': end_date,
            'order': 'asc',
            'limit': 5000,
            'apiKey': POLYGON_API_KEY
        }
    elif indicator_type == 'RSI':
        url = f"{base_url}/rsi/{polygon_ticker}"
        params = {
            'timespan': timespan,
            'multiplier': multiplier,  # Required for proper aggregation
            'adjusted': 'true',
            'window': window,
            'series_type': 'close',
            'timestamp.gte': start_date,
            'timestamp.lte': end_date,
            'order': 'asc',
            'limit': 5000,
            'apiKey': POLYGON_API_KEY
        }
    elif indicator_type == 'MACD':
        url = f"{base_url}/macd/{polygon_ticker}"
        fast = kwargs.get('fast_period', 12)
        slow = kwargs.get('slow_period', 26)
        signal = kwargs.get('signal_period', 9)
        params = {
            'timespan': timespan,
            'multiplier': multiplier,  # Required for proper aggregation
            'adjusted': 'true',
            'short_window': fast,
            'long_window': slow,
            'signal_window': signal,
            'series_type': 'close',
            'timestamp.gte': start_date,
            'timestamp.lte': end_date,
            'order': 'asc',
            'limit': 5000,
            'apiKey': POLYGON_API_KEY
        }
    else:
        print(f"    ⚠️  Unknown indicator type: {indicator_type}")
        return []
    
    all_results = []
    page = 1
    
    try:
        while url:
            # Debug: Log API request on first page
            if page == 1:
                print(f"      DEBUG: API URL: {url}")
                print(f"      DEBUG: Params: {params}")
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') == 'OK' and 'results' in data and 'values' in data['results']:
                results = data['results']['values']
                all_results.extend(results)
                
                # Get date range for this page
                if results:
                    first_ts = datetime.fromtimestamp(results[0]['timestamp'] / 1000)
                    last_ts = datetime.fromtimestamp(results[-1]['timestamp'] / 1000)
                    date_range = f"{first_ts.strftime('%Y-%m-%d')} to {last_ts.strftime('%Y-%m-%d')}"
                else:
                    date_range = "N/A"
                
                if page == 1:
                    print(f"      Page {page}: Got {len(results)} {indicator_type} values ({date_range})")
                    # Debug: Show sample timestamps from first page
                    if len(results) >= 10:
                        print(f"      DEBUG: First 5 timestamps:")
                        for i in range(5):
                            ts = datetime.fromtimestamp(results[i]['timestamp'] / 1000)
                            print(f"        [{i}] {ts}")
                        print(f"      DEBUG: Last 5 timestamps:")
                        for i in range(-5, 0):
                            ts = datetime.fromtimestamp(results[i]['timestamp'] / 1000)
                            print(f"        [{len(results)+i}] {ts}")
                        # Check interval between first two bars
                        if len(results) >= 2:
                            ts1 = datetime.fromtimestamp(results[0]['timestamp'] / 1000)
                            ts2 = datetime.fromtimestamp(results[1]['timestamp'] / 1000)
                            interval_minutes = (ts2 - ts1).total_seconds() / 60
                            print(f"      DEBUG: Interval between first 2 bars: {interval_minutes:.1f} minutes")
                else:
                    print(f"      Page {page}: Got {len(results)} more values (total: {len(all_results)}) ({date_range})")
                
                # Check for pagination
                next_url = data.get('next_url')
                if next_url:
                    url = next_url
                    params = {'apiKey': POLYGON_API_KEY}
                    page += 1
                    time.sleep(RATE_LIMIT_DELAY)
                else:
                    break
            else:
                if data.get('status') == 'ERROR':
                    print(f"      ❌ API Error: {data.get('error', 'Unknown error')}")
                break
        
        return all_results
        
    except Exception as e:
        print(f"      ❌ API Error: {str(e)}")
        return all_results if all_results else []


def aggregate_indicators_to_wide_format(
    ticker: str,
    frequency: str,
    all_indicator_data: dict
) -> list:
    """
    Aggregate all indicators into wide format (one row per timestamp).
    
    Args:
        ticker: Stock symbol
        frequency: Data frequency
        all_indicator_data: Dict mapping (indicator_type, window) -> list of values
    
    Returns:
        List of dictionaries in wide format ready for BigQuery insertion
    """
    # Collect all unique timestamps across all indicators
    all_timestamps = set()
    for values in all_indicator_data.values():
        for val in values:
            all_timestamps.add(val['timestamp'])
    
    # Build rows for each timestamp
    rows = []
    for ts in sorted(all_timestamps):
        dt = datetime.fromtimestamp(ts / 1000)
        
        row = {
            'ticker': ticker,
            'timestamp': dt.strftime('%Y-%m-%d %H:%M:%S'),
            'date': dt.date().strftime('%Y-%m-%d'),
            'frequency': frequency,
            # Initialize all indicator columns to None
            'sma_10': None, 'sma_20': None, 'sma_50': None, 'sma_200': None,
            'ema_10': None, 'ema_20': None, 'ema_50': None, 'ema_200': None,
            'rsi_14': None,
            'macd_value': None, 'macd_signal': None, 'macd_histogram': None,
            'ingested_at': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Fill in values for this timestamp
        for (ind_type, window), values in all_indicator_data.items():
            # Find value for this timestamp
            matching_val = next((v for v in values if v['timestamp'] == ts), None)
            if matching_val:
                if ind_type == 'SMA':
                    row[f'sma_{window}'] = matching_val.get('value')
                elif ind_type == 'EMA':
                    row[f'ema_{window}'] = matching_val.get('value')
                elif ind_type == 'RSI':
                    row['rsi_14'] = matching_val.get('value')
                elif ind_type == 'MACD':
                    row['macd_value'] = matching_val.get('value')
                    row['macd_signal'] = matching_val.get('signal')
                    row['macd_histogram'] = matching_val.get('histogram')
        
        rows.append(row)
    
    return rows


def save_to_bigquery(bq_client: bigquery.Client, rows: list, table_id: str) -> int:
    """
    Save rows to BigQuery using load job.
    
    Args:
        bq_client: BigQuery client
        rows: List of row dictionaries
        table_id: Fully qualified table ID
    
    Returns:
        Number of rows inserted
    """
    if not rows:
        return 0
    
    try:
        job_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
            source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        )
        
        job = bq_client.load_table_from_json(rows, table_id, job_config=job_config)
        job.result()  # Wait for completion
        
        print(f"  ✅ Inserted {len(rows)} rows to BigQuery")
        return len(rows)
        
    except Exception as e:
        print(f"  ❌ BigQuery error: {str(e)}")
        return 0


def get_latest_timestamp(bq_client: bigquery.Client, table: str, ticker: str, frequency: str):
    """
    Get the latest timestamp for a ticker/frequency combination from BigQuery.
    
    Args:
        bq_client: BigQuery client
        table: Table ID to query
        ticker: Ticker symbol
        frequency: Data frequency
    
    Returns:
        Latest timestamp as datetime object, or None if no data exists
    """
    query = f"""
    SELECT MAX(timestamp) as latest_timestamp
    FROM `{table}`
    WHERE ticker = @ticker AND frequency = @frequency
    """
    
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("ticker", "STRING", ticker),
            bigquery.ScalarQueryParameter("frequency", "STRING", frequency),
        ]
    )
    
    try:
        result = bq_client.query(query, job_config=job_config).to_dataframe()
        if not result.empty and pd.notna(result['latest_timestamp'].iloc[0]):
            return result['latest_timestamp'].iloc[0]
        return None
    except Exception as e:
        print(f"  ⚠️  Warning: Could not query latest timestamp: {str(e)}")
        return None


def purge_ticker_data(
    bq_client: bigquery.Client,
    ohlcv_table: str,
    indicators_table: str,
    tickers: list,
    data_type: str = 'raw'
):
    """
    Delete ALL data for specified tickers (all dates, all frequencies).
    
    Args:
        bq_client: BigQuery client
        ohlcv_table: OHLCV table ID to delete from
        indicators_table: Indicators table ID to delete from
        tickers: List of tickers to purge
        data_type: Which data to purge - 'raw', 'indicators', or 'both'
    """
    print("\n" + "=" * 80)
    print("   🚨 PURGING ALL DATA FOR TICKERS 🚨")
    print("=" * 80)
    print()
    
    ticker_list = "', '".join(tickers)
    delete_query_template = "DELETE FROM `{}` WHERE ticker IN ('{}')" 
    
    print(f"⚠️  WARNING: Deleting ALL records for {len(tickers)} ticker(s):")
    print(f"   Tickers: {', '.join(tickers)}")
    print(f"   Data type: {data_type.upper()}")
    print(f"   This will remove data for ALL dates and ALL frequencies!")
    print()
    
    # Confirmation prompt
    response = input("Type 'DELETE' to confirm: ")
    if response != 'DELETE':
        print("❌ Purge cancelled")
        print()
        return
    
    print("\n🗑️  Executing deletion...")
    
    total_deleted = 0
    
    try:
        # Purge OHLCV data
        if data_type in ['raw', 'both']:
            print(f"  Purging OHLCV data...")
            delete_query = delete_query_template.format(ohlcv_table, ticker_list)
            delete_job = bq_client.query(delete_query)
            delete_job.result()
            stats = delete_job._properties.get('statistics', {}).get('query', {})
            rows_deleted = int(stats.get('numDmlAffectedRows', 0))
            print(f"  ✅ Deleted {rows_deleted:,} OHLCV rows")
            total_deleted += rows_deleted
        
        # Purge indicator data
        if data_type in ['indicators', 'both']:
            print(f"  Purging technical indicators data...")
            delete_query = delete_query_template.format(indicators_table, ticker_list)
            delete_job = bq_client.query(delete_query)
            delete_job.result()
            stats = delete_job._properties.get('statistics', {}).get('query', {})
            rows_deleted = int(stats.get('numDmlAffectedRows', 0))
            print(f"  ✅ Deleted {rows_deleted:,} indicator rows")
            total_deleted += rows_deleted
        
        print(f"\n✅ Total purged: {total_deleted:,} rows")
        print()
        
    except Exception as e:
        print(f"❌ Purge failed: {str(e)}")
        print()


def flush_existing_data(
    bq_client: bigquery.Client,
    ohlcv_table: str,
    indicators_table: str,
    tickers: list,
    start_date: str,
    end_date: str,
    frequency: str,
    data_type: str = 'raw'
):
    """
    Flush existing data for the specified tickers and date range.
    
    Args:
        bq_client: BigQuery client
        ohlcv_table: OHLCV table ID to flush from
        indicators_table: Indicators table ID to flush from
        tickers: List of tickers to flush
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        frequency: Data frequency
        data_type: Which data to flush - 'raw', 'indicators', or 'both'
    """
    print("\n" + "=" * 80)
    print("   Flushing Existing Data")
    print("=" * 80)
    print()
    
    ticker_list = "', '".join(tickers)
    delete_query_template = """
    DELETE FROM `{}`
    WHERE ticker IN ('{}')
      AND date BETWEEN '{}' AND '{}'
      AND frequency = '{}'
    """
    
    print(f"🗑️  Deleting existing data for {len(tickers)} ticker(s)...")
    print(f"   Date range: {start_date} to {end_date}")
    print(f"   Frequency: {frequency}")
    print(f"   Data type: {data_type.upper()}")
    
    total_deleted = 0
    
    try:
        # Flush OHLCV data
        if data_type in ['raw', 'both']:
            delete_query = delete_query_template.format(
                ohlcv_table, ticker_list, start_date, end_date, frequency
            )
            delete_job = bq_client.query(delete_query)
            delete_job.result()
            stats = delete_job._properties.get('statistics', {}).get('query', {})
            rows_deleted = int(stats.get('numDmlAffectedRows', 0))
            print(f"  ✅ Deleted {rows_deleted:,} OHLCV rows")
            total_deleted += rows_deleted
        
        # Flush indicator data
        if data_type in ['indicators', 'both']:
            delete_query = delete_query_template.format(
                indicators_table, ticker_list, start_date, end_date, frequency
            )
            delete_job = bq_client.query(delete_query)
            delete_job.result()
            stats = delete_job._properties.get('statistics', {}).get('query', {})
            rows_deleted = int(stats.get('numDmlAffectedRows', 0))
            print(f"  ✅ Deleted {rows_deleted:,} indicator rows")
            total_deleted += rows_deleted
        
        print(f"\nTotal deleted: {total_deleted:,} rows")
        print()
        
    except Exception as e:
        print(f"❌ Flush failed: {str(e)}")
        print()


def merge_staging_to_main(
    bq_client: bigquery.Client, 
    staging_table: str, 
    main_table: str
):
    """
    Merge data from staging to main table with deduplication.
    
    Args:
        bq_client: BigQuery client
        staging_table: Staging table ID
        main_table: Main table ID
    """
    print("\n" + "=" * 80)
    print("   Merging Staging to Main Table")
    print("=" * 80)
    print()
    
    try:
        # Check staging count
        query = f"SELECT COUNT(*) as count FROM `{staging_table}`"
        result = bq_client.query(query).result()
        staging_count = list(result)[0]['count']
        
        if staging_count == 0:
            print("⚠️  Staging table is empty, nothing to merge")
            return
        
        print(f"📊 Staging table has {staging_count:,} rows")
        print(f"🔄 Merging to main table (deduplicating by ticker + timestamp + frequency)...")
        
        # Perform MERGE
        merge_query = f"""
        MERGE `{main_table}` T
        USING `{staging_table}` S
        ON T.ticker = S.ticker 
           AND T.timestamp = S.timestamp 
           AND T.frequency = S.frequency
        WHEN NOT MATCHED THEN
          INSERT ROW
        """
        
        merge_job = bq_client.query(merge_query)
        merge_job.result()
        
        # Get stats
        stats = merge_job._properties.get('statistics', {}).get('query', {})
        rows_affected = stats.get('numDmlAffectedRows', 'unknown')
        
        print(f"✅ Merge complete! Rows inserted to main table: {rows_affected}")
        print()
        
        # Truncate staging
        print("🧹 Truncating staging table...")
        truncate_query = f"TRUNCATE TABLE `{staging_table}`"
        bq_client.query(truncate_query).result()
        print("✅ Staging table truncated")
        print()
        
    except Exception as e:
        print(f"❌ Merge failed: {str(e)}")
        print("   Staging data preserved for manual review")
        print()


def ingest_polygon_data(
    tickers: list, 
    start_date: str, 
    end_date: str,
    frequency: str = 'daily',
    auto_merge: bool = True,
    flush_existing: bool = False,
    purge_tickers: bool = False,
    data_type: str = 'raw',
    indicators_filter: set = None,
    top_up: bool = False
):
    """
    Main ingestion function for Polygon to BigQuery.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        frequency: Data frequency ('hourly', 'daily', etc.)
        auto_merge: Whether to automatically merge staging to main
        flush_existing: Whether to delete existing data for date range before ingestion
        purge_tickers: Whether to delete ALL data for tickers (ignores date range)
        data_type: Type of data to fetch - 'raw', 'indicators', or 'both'
        indicators_filter: Set of specific indicators to fetch (e.g., {'SMA-10', 'RSI-14'})
        top_up: Whether to auto-detect latest data and fill from there (ignores start_date)
    """
    print("=" * 80)
    print("   Inflation prediction - Polygon to BigQuery Ingestion")
    print("=" * 80)
    print(f"\nProject: {PROJECT_ID}")
    print(f"Dataset: {DATASET_ID}")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Frequency: {frequency.upper()}")
    print(f"Data type: {data_type.upper()}")
    print()
    
    # Block technical indicators loading (disabled feature)
    if data_type in ['indicators', 'both']:
        print("❌ " + "=" * 76)
        print("❌ POLYGON TECHNICAL INDICATORS LOADING IS CURRENTLY DISABLED")
        print("❌ " + "=" * 76)
        print()
        print("⚠️  Polygon API indicators are disabled for now.")
        print()
        print("✅ Please use SYNTHETIC INDICATORS instead:")
        print("   Synthetic indicators are computed locally from raw OHLCV data")
        print("   and offer more flexibility and faster processing.")
        print()
        print("   Run this command:")
        print("   python scripts/01_extract/tickers_load_synthetic.py \\")
        print(f"     --tickers {' '.join(tickers)} \\")
        print(f"     --start {start_date} \\")
        print(f"     --end {end_date} \\")
        print(f"     --frequency {frequency}")
        print()
        print("   See scripts/README.md for more details on synthetic indicators.")
        print()
        return
    
    # Validate configuration
    if not PROJECT_ID:
        print("❌ Error: GCP_PROJECT_ID not set")
        return
    
    # Note: POLYGON_API_KEY is checked later only if we need to fetch data
    # If data already exists and --reload is not set, we skip the API call
    # This allows users without Polygon access to work with existing data
    if not POLYGON_API_KEY and (flush_existing or purge_tickers):
        print("❌ Error: POLYGON_API_KEY not set (required for --reload/--purge)")
        return
    
    if frequency not in FREQUENCY_MAP:
        print(f"❌ Error: Invalid frequency '{frequency}'. Use: {', '.join(FREQUENCY_MAP.keys())}")
        return
    
    # Initialize clients
    print("🔧 Initializing BigQuery client...")
    bq_client = bigquery.Client(project=PROJECT_ID)
    
    # Table IDs for OHLCV
    staging_table = f"{PROJECT_ID}.{DATASET_ID}.{STAGING_TABLE_NAME}"
    main_table = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_NAME}"
    
    # Table IDs for indicators
    indicators_staging_table = f"{PROJECT_ID}.{DATASET_ID}.{INDICATORS_STAGING_TABLE}"
    indicators_main_table = f"{PROJECT_ID}.{DATASET_ID}.{INDICATORS_TABLE}"
    
    # Purge ALL ticker data if requested (takes precedence over flush)
    if purge_tickers:
        purge_ticker_data(bq_client, main_table, indicators_main_table, tickers, data_type)
        # If purge-only (no dates provided), skip ingestion
        if start_date == '2000-01-01' and end_date == '2000-01-01':
            print("\n✅ Purge complete! Skipping data ingestion.\n")
            return
        # Otherwise continue to ingest after purge
    elif flush_existing:
        flush_existing_data(
            bq_client, 
            main_table,
            indicators_main_table,
            tickers, 
            start_date, 
            end_date, 
            frequency,
            data_type
        )
    
    # Statistics
    total_ohlcv_rows = 0
    total_indicator_rows = 0
    total_api_calls = 0
    skipped_tickers = []
    
    # Process each ticker
    for ticker_idx, ticker in enumerate(tickers, 1):
        print(f"[{ticker_idx}/{len(tickers)}] 📊 Processing {ticker}...")
        
        # Check if ticker already has data (unless reload flag is set)
        # This allows users without Polygon API access to work with existing data
        if not flush_existing and not purge_tickers:
            ticker_exists = check_ticker_exists(bq_client, ticker, frequency)
            if ticker_exists:
                print(f"  ⏭️  SKIPPED: {ticker} already has data in BigQuery (use --reload to overwrite)")
                print()
                skipped_tickers.append(ticker)
                continue
        
        # If we reach here, we need to fetch from Polygon - check API key
        if not POLYGON_API_KEY:
            print(f"  ❌ ERROR: {ticker} has no data in BigQuery and POLYGON_API_KEY is not set")
            print(f"     Cannot fetch data from Polygon.io without API key")
            print()
            skipped_tickers.append(ticker)
            continue
        
        # Top-up mode: Check for latest data and adjust start_date
        ticker_start_date = start_date
        ticker_end_date = end_date
        
        if top_up:
            # Check latest timestamp for this ticker
            latest_ts = get_latest_timestamp(bq_client, main_table, ticker, frequency)
            
            if latest_ts:
                print(f"  🔍 Top-up mode details:")
                print(f"     Latest timestamp in BigQuery: {latest_ts.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Convert to date string, add 1 period based on frequency
                if frequency in ['daily', '1d']:
                    next_date = latest_ts + timedelta(days=1)
                    period_name = "1 day"
                elif frequency in ['hourly', '1h']:
                    next_date = latest_ts + timedelta(hours=1)
                    period_name = "1 hour"
                elif frequency == '15m':
                    next_date = latest_ts + timedelta(minutes=15)
                    period_name = "15 minutes"
                elif frequency == '5m':
                    next_date = latest_ts + timedelta(minutes=5)
                    period_name = "5 minutes"
                else:
                    next_date = latest_ts + timedelta(days=1)  # Default to daily
                    period_name = "1 day (default)"
                
                print(f"     Next period ({period_name}): {next_date.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"     Requested end date: {ticker_end_date}")
                
                ticker_start_date = next_date.strftime('%Y-%m-%d')
                
                # Parse end_date as END OF DAY timestamp for proper comparison
                # Make it timezone-aware (UTC) to match next_date from BigQuery
                import pytz
                end_datetime = datetime.strptime(ticker_end_date, '%Y-%m-%d').replace(hour=23, minute=59, second=59)
                end_datetime = pytz.UTC.localize(end_datetime)
                
                print(f"     End datetime (EOD): {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"     Comparison: {next_date.strftime('%Y-%m-%d %H:%M:%S')} > {end_datetime.strftime('%Y-%m-%d %H:%M:%S')} ? {next_date > end_datetime}")
                
                # Skip if next period is after end of day
                if next_date > end_datetime:
                    print(f"  ✅ Already up-to-date! Latest data is at or past end date.")
                    print(f"  ℹ️  Note: Top-up only fills forward from latest timestamp, not gaps in history.")
                    print(f"  💡 To fill historical gaps (e.g., 2015-11-12 to 2016-01-03), use --start/--end without --top-up")
                    continue
                else:
                    print(f"  🔄 Fetching from {ticker_start_date} to {ticker_end_date}")
            else:
                print(f"  📌 No existing data found. Using start date: {ticker_start_date}")
        
        # Fetch raw OHLCV data
        if data_type in ['raw', 'both']:
            print(f"  📈 Fetching OHLCV data...")
            bars = fetch_ohlcv_data(ticker, ticker_start_date, ticker_end_date, frequency)
            
            if bars:
                rows = convert_ohlcv_to_rows(ticker, bars, frequency)
                print(f"  💾 Saving {len(rows)} OHLCV rows to staging...")
                total_ohlcv_rows += save_to_bigquery(bq_client, rows, staging_table)
                total_api_calls += 1
            else:
                print(f"  ⚠️  No OHLCV data fetched for {ticker}")
        
        # Fetch technical indicators
        if data_type in ['indicators', 'both']:
            print(f"  📉 Fetching technical indicators...")
            
            # Collect all indicator data first, then aggregate to wide format
            all_indicator_data = {}
            
            # Fetch each indicator type
            for ind_type in ['SMA', 'EMA', 'RSI', 'MACD']:
                if ind_type in INDICATOR_CONFIGS:
                    for window in INDICATOR_CONFIGS[ind_type]:
                        # Check filter
                        if indicators_filter:
                            if ind_type == 'MACD':
                                if 'MACD' not in indicators_filter:
                                    continue
                            else:
                                spec = f"{ind_type}-{window}"
                                if spec not in indicators_filter:
                                    continue
                        
                        print(f"    🔹 Fetching {ind_type}-{window if ind_type != 'MACD' else 'default'}...")
                        
                        if ind_type == 'MACD':
                            values = fetch_technical_indicator(
                                ticker, ind_type, ticker_start_date, ticker_end_date, frequency, window=None
                            )
                            window_val = 'default'
                        else:
                            values = fetch_technical_indicator(
                                ticker, ind_type, ticker_start_date, ticker_end_date, frequency, window=window
                            )
                            window_val = window
                    
                        if values:
                            all_indicator_data[(ind_type, window_val)] = values
                            total_api_calls += 1
                        else:
                            print(f"      ⚠️  No {ind_type} data")
                    
                        time.sleep(RATE_LIMIT_DELAY)
                    
                    time.sleep(RATE_LIMIT_DELAY)
            
            # Aggregate all indicators into wide format and save
            if all_indicator_data:
                print(f"  📊 Aggregating indicators to wide format...")
                rows = aggregate_indicators_to_wide_format(ticker, frequency, all_indicator_data)
                print(f"  💾 Saving {len(rows)} indicator rows to staging...")
                total_indicator_rows += save_to_bigquery(bq_client, rows, indicators_staging_table)
        
        print(f"  ✅ {ticker} complete!\n")
    
    # Merge staging to main tables
    if auto_merge:
        if data_type in ['raw', 'both'] and total_ohlcv_rows > 0:
            merge_staging_to_main(bq_client, staging_table, main_table)
        
        if data_type in ['indicators', 'both'] and total_indicator_rows > 0:
            print("\n" + "=" * 80)
            print("   Merging Indicators Staging to Main Table")
            print("=" * 80)
            print()
            merge_staging_to_main(bq_client, indicators_staging_table, indicators_main_table)
    print("=" * 80)
    print("   Ingestion Complete!")
    print("=" * 80)
    print("📊 Statistics:")
    print(f"  Tickers requested: {len(tickers)}")
    print(f"  Tickers processed: {len(tickers) - len(skipped_tickers)}")
    if skipped_tickers:
        print(f"  Tickers skipped: {len(skipped_tickers)} (already exist: {', '.join(skipped_tickers)})")
    if data_type in ['raw', 'both']:
        print(f"  OHLCV rows inserted: {total_ohlcv_rows:,}")
    print(f"  Total API calls: {total_api_calls}")
    print(f"  Frequency: {frequency}")
    print(f"  Data type: {data_type}")
    print()


def load_config(config_path: str = 'configs/tickers.yaml') -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config YAML file
    
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"⚠️  Config file not found: {config_path}")
        return None


def load_tickers_from_config(config_path: str = 'configs/tickers.yaml') -> list:
    """
    Load ticker list from configuration file.
    
    Args:
        config_path: Path to tickers YAML config
    
    Returns:
        List of ticker symbols
    """
    config = load_config(config_path)
    if config:
        return config.get('tickers', [])
    return []


def check_ticker_exists(bq_client: bigquery.Client, ticker: str, frequency: str) -> bool:
    """Check if ticker data already exists in BigQuery.
    
    Args:
        bq_client: BigQuery client
        ticker: Ticker symbol
        frequency: Data frequency
    
    Returns:
        True if data exists, False otherwise
    """
    table_id = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_NAME}"
    query = f"""
    SELECT COUNT(*) as count
    FROM `{table_id}`
    WHERE ticker = '{ticker}' AND frequency = '{frequency}'
    LIMIT 1
    """
    
    try:
        result = bq_client.query(query).to_dataframe()
        return result['count'].iloc[0] > 0
    except Exception:
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Ingest data from Polygon.io to BigQuery for Inflation prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Load all tickers from config file (default behavior)
  python scripts/01_extract/tickers_load_polygon.py --start 2024-01-01 --end 2024-12-31 --frequency daily
  
  # Load specific tickers (overrides config)
  python scripts/01_extract/tickers_load_polygon.py --tickers SPY QQQ --start 2024-01-01 --end 2024-12-31 --frequency daily
  
  # Reload/overwrite existing data
  python scripts/01_extract/tickers_load_polygon.py --tickers SPY --start 2024-01-01 --end 2024-12-31 --frequency daily --reload
  
  # Use custom config file
  python scripts/01_extract/tickers_load_polygon.py --config configs/my_tickers.yaml --start 2024-01-01 --end 2024-12-31
        """
    )
    
    parser.add_argument(
        '--tickers',
        nargs='+',
        help='Ticker symbols to ingest (e.g., SPY QQQ). If not specified, loads all tickers from config file.'
    )
    parser.add_argument(
        '--config',
        default='configs/tickers.yaml',
        help='Path to tickers YAML config file (default: configs/tickers.yaml)'
    )
    parser.add_argument(
        '--start', '--start-date',
        dest='start_date',
        required=False,
        help='Start date (YYYY-MM-DD) - not required for --purge'
    )
    parser.add_argument(
        '--end', '--end-date',
        dest='end_date',
        required=False,
        help='End date (YYYY-MM-DD) - not required for --purge'
    )
    parser.add_argument(
        '--frequency',
        choices=['hourly', 'daily', '1h', '1d', '5m', '15m'],
        default='daily',
        help='Data frequency (default: daily)'
    )
    parser.add_argument(
        '--reload',
        '--flush',
        dest='flush',
        action='store_true',
        help='Reload/overwrite existing data for tickers (deletes existing data for date range before ingestion)'
    )
    parser.add_argument(
        '--purge',
        action='store_true',
        help='Delete ALL data for tickers (all dates, all frequencies) - requires confirmation'
    )
    parser.add_argument(
        '--no-merge',
        action='store_true',
        help='Do not automatically merge staging to main table'
    )
    parser.add_argument(
        '--data-type',
        choices=['raw', 'indicators', 'both'],
        default='raw',
        help='Type of data to fetch: raw (OHLCV only - indicators are DISABLED, use tickers_load_synthetic.py instead)'
    )
    parser.add_argument(
        '--indicators',
        nargs='+',
        help='Specific indicators to fetch (DISABLED - use tickers_load_synthetic.py instead)'
    )
    parser.add_argument(
        '--top-up',
        action='store_true',
        help='Auto-detect latest data per ticker and fill from there to end date. Uses config date_range if no dates specified.'
    )

# ... (rest of the code remains the same)
    
    args = parser.parse_args()
    
    # Get tickers from args or config
    tickers = args.tickers or []
    config = None
    
    # Try to load config (default or specified)
    if not args.tickers:
        # No tickers specified on command line, load from config
        config = load_config(args.config)
        if config and 'tickers' in config:
            tickers = config['tickers']
            print(f"📋 Loaded {len(tickers)} tickers from {args.config}")
            print(f"   Tickers: {', '.join(tickers[:10])}{'...' if len(tickers) > 10 else ''}")
        else:
            print(f"❌ Error: No tickers found in config file {args.config}")
            sys.exit(1)
    else:
        # Tickers specified on command line
        print(f"📋 Using {len(tickers)} ticker(s) from command line")
    
    if not tickers:
        print("❌ Error: No tickers specified. Use --tickers or ensure config file has tickers")
        sys.exit(1)
    
    # Validate dates
    if args.purge and not args.start_date and not args.end_date:
        # Purge-only mode: use dummy dates (will be ignored)
        start_date = '2000-01-01'
        end_date = '2000-01-01'
    elif args.top_up:
        # Top-up mode: start_date from config or required, end_date defaults to today
        if not args.start_date:
            if config and 'date_range' in config:
                start_date = config['date_range'].get('start_date', '2015-11-10')
                print(f"📅 Using config start_date: {start_date}")
            else:
                print("❌ Error: --start date required for --top-up (or provide --config with date_range)")
                sys.exit(1)
        else:
            start_date = args.start_date
        
        # End date defaults to today for top-up
        if not args.end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
            print(f"📅 Top-up to today: {end_date}")
        else:
            end_date = args.end_date
    else:
        # Normal mode: both dates required
        if not args.start_date or not args.end_date:
            print("❌ Error: --start and --end dates are required")
            sys.exit(1)
        start_date = args.start_date
        end_date = args.end_date
    
    # Parse indicators filter if provided
    indicators_filter = None
    if args.indicators:
        indicators_filter = set(args.indicators)
        print(f"📌 Filtering indicators: {', '.join(sorted(indicators_filter))}\n")
    
    # Run ingestion
    ingest_polygon_data(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        frequency=args.frequency,
        auto_merge=not args.no_merge,
        flush_existing=args.flush,
        purge_tickers=args.purge,
        data_type=args.data_type,
        indicators_filter=indicators_filter,
        top_up=args.top_up
    )
