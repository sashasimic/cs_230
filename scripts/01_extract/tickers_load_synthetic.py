"""Compute synthetic indicators from raw OHLCV data and save to BigQuery.

Calculates technical indicators locally from existing raw_ohlcv data and saves
to BigQuery using staging + MERGE pattern. Supports selective indicator computation.
"""
import os
import sys
import argparse
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import pytz
import yaml
from google.cloud import bigquery

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configuration
PROJECT_ID = os.environ.get('GCP_PROJECT_ID')
DATASET_ID = os.environ.get('BQ_DATASET', 'raw_dataset')
RAW_TABLE = os.environ.get('BQ_TABLE', 'raw_ohlcv')
SYNTHETIC_TABLE = os.environ.get('BQ_SYNTHETIC_TABLE', 'synthetic_indicators')
SYNTHETIC_STAGING_TABLE = os.environ.get('BQ_SYNTHETIC_STAGING_TABLE', 'synthetic_indicators_staging')

# Available synthetic indicators grouped by category
INDICATOR_GROUPS = {
    'returns': ['returns', 'returns_5', 'log_returns'],
    'volatility': ['volatility_5', 'volatility_10', 'volatility_20'],
    'momentum': ['momentum_5', 'momentum_10', 'roc_5', 'roc_10', 'macd', 'macd_signal', 'macd_hist'],
    'volume': ['volume_ma_5', 'volume_ma_10', 'volume_ratio', 'obv'],
    'range': ['atr_14', 'high_low_ratio'],
    'bollinger': ['bb_upper_20', 'bb_middle_20', 'bb_lower_20', 'bb_width'],
    'sma': ['sma_10', 'sma_20', 'sma_50', 'sma_200'],
    'ema': ['ema_12', 'ema_26']
}


def calculate_indicator_warmup_dates(
    bq_client: bigquery.Client,
    ticker: str,
    requested_start_date: str,
    frequency: str,
    periods_required: int,
    indicator_name: str = "indicator"
) -> tuple:
    """
    Calculate warmup dates for indicators that use rolling windows/averages.
    
    This ensures indicators can compute from the requested start date by:
    1. Fetching data from earlier (warmup period)
    2. Computing indicator values only from the requested start date onwards
    3. Aborts if insufficient warmup data is available
    
    Args:
        bq_client: BigQuery client
        ticker: Ticker symbol
        requested_start_date: User-requested start date (YYYY-MM-DD)
        frequency: Data frequency ('15m', '1h', 'daily', etc.)
        periods_required: Number of periods needed for warmup (e.g., 200 for SMA_200)
        indicator_name: Name of indicator for logging (e.g., 'sma_200')
    
    Returns:
        Tuple of (warmup_fetch_start, indicator_value_start, actual_warmup_periods):
        - warmup_fetch_start: Earliest date to fetch raw data from (str)
        - indicator_value_start: First date where indicator will have values (datetime, tz-aware)
        - actual_warmup_periods: Actual number of periods available for warmup (int)
    
    Raises:
        SystemExit: If insufficient warmup data is available
    """
    from datetime import datetime, timedelta
    
    # Frequency to period mapping (periods per day)
    freq_periods_per_day = {
        '5m': 288,
        '15m': 96,
        '1h': 24,
        'hourly': 24,
        '1d': 1,
        'daily': 1
    }
    
    periods_per_day = freq_periods_per_day.get(frequency, 1)
    
    # Calculate ideal warmup start (go back periods_required)
    requested_start_dt = datetime.strptime(requested_start_date, '%Y-%m-%d')
    days_to_rewind = int((periods_required / periods_per_day) * 1.2) + 5  # 20% buffer + 5 days
    ideal_warmup_start = (requested_start_dt - timedelta(days=days_to_rewind)).strftime('%Y-%m-%d')
    
    # Query raw data to find actual earliest available date
    table_id = f"{PROJECT_ID}.{DATASET_ID}.{RAW_TABLE}"
    query = f"""
    SELECT 
        MIN(timestamp) as earliest_timestamp,
        COUNT(*) as total_rows
    FROM `{table_id}`
    WHERE ticker = '{ticker}'
      AND frequency = '{frequency}'
      AND date >= '{ideal_warmup_start}'
      AND date < '{requested_start_date}'
    """
    
    result = bq_client.query(query).to_dataframe()
    
    if result.empty or pd.isna(result['earliest_timestamp'].iloc[0]):
        # No data before requested start - check if any data exists at all
        fallback_query = f"""
        SELECT MIN(timestamp) as earliest_timestamp
        FROM `{table_id}`
        WHERE ticker = '{ticker}' AND frequency = '{frequency}'
        """
        fallback_result = bq_client.query(fallback_query).to_dataframe()
        
        if fallback_result.empty or pd.isna(fallback_result['earliest_timestamp'].iloc[0]):
            # No data at all - ABORT
            print(f"\n❌ ERROR: No raw data found for {ticker} ({frequency})")
            print(f"   Indicator: {indicator_name} requires {periods_required} periods of warmup")
            print(f"   Requested start: {requested_start_date}")
            print(f"   Solution: Load raw data using tickers_load_polygon.py first")
            sys.exit(1)
        else:
            # Data exists but starts after ideal warmup - check if enough
            earliest_ts = fallback_result['earliest_timestamp'].iloc[0]
            time_diff = requested_start_dt - earliest_ts.to_pydatetime().replace(tzinfo=None)
            actual_warmup_periods = int((time_diff.total_seconds() / 86400) * periods_per_day)
            
            # Set warmup_fetch_start from earliest available
            warmup_fetch_start = earliest_ts.strftime('%Y-%m-%d')
            
            # Check if sufficient warmup available
            if actual_warmup_periods < periods_required:
                # Insufficient warmup - adjust indicator_value_start forward to allow sufficient warmup
                if frequency in ['5m', '15m', '1h', 'hourly']:
                    # Intraday: add periods as time
                    minutes_per_period = {'5m': 5, '15m': 15, '1h': 60, 'hourly': 60}.get(frequency, 60)
                    adjusted_start = earliest_ts + timedelta(minutes=minutes_per_period * periods_required)
                else:
                    # Daily: add periods as days
                    adjusted_start = earliest_ts + timedelta(days=periods_required)
                
                indicator_value_start = adjusted_start
                
                print(f"  ⚠️  Adjusted start date for {indicator_name} (insufficient warmup)")
                print(f"     Warmup fetch start: {warmup_fetch_start}")
                print(f"     Requested start: {requested_start_date}")
                print(f"     Indicator values from: {indicator_value_start.date()} (adjusted forward)")
                print(f"     Warmup periods: {actual_warmup_periods} rows ({periods_required} required)")
                print(f"     Note: First {periods_required} periods used for warmup")
            else:
                # Sufficient warmup - calculate where indicator values start (after warmup)
                if frequency in ['5m', '15m', '1h', 'hourly']:
                    minutes_per_period = {'5m': 5, '15m': 15, '1h': 60, 'hourly': 60}.get(frequency, 60)
                    indicator_value_start = earliest_ts + timedelta(minutes=minutes_per_period * periods_required)
                else:
                    indicator_value_start = earliest_ts + timedelta(days=periods_required)
                
                print(f"  ✅ Warmup data available for {indicator_name}")
                print(f"     Warmup fetch start: {warmup_fetch_start}")
                print(f"     Indicator values from: {indicator_value_start.date()} (after {periods_required} period warmup)")
                print(f"     Requested start: {requested_start_date}")
                print(f"     Warmup periods: {actual_warmup_periods} rows ({periods_required} required)")
    else:
        # Warmup data exists - calculate forward-filled periods available
        earliest_ts = result['earliest_timestamp'].iloc[0]
        warmup_fetch_start = earliest_ts.strftime('%Y-%m-%d')
        
        # Calculate forward-filled periods between earliest_ts and requested_start_date
        # This matches what fetch_raw_ohlcv does with pd.date_range
        time_diff = requested_start_dt - earliest_ts.to_pydatetime().replace(tzinfo=None)
        
        if frequency in ['5m', '15m', '1h', 'hourly']:
            # Intraday: calculate periods based on minutes
            minutes_per_period = {'5m': 5, '15m': 15, '1h': 60, 'hourly': 60}.get(frequency, 60)
            actual_warmup_periods = int(time_diff.total_seconds() / 60 / minutes_per_period)
        else:
            # Daily
            actual_warmup_periods = int(time_diff.total_seconds() / 86400)
        
        # Check if sufficient warmup available (forward-filled periods)
        if actual_warmup_periods < periods_required:
            # Insufficient warmup - adjust indicator_value_start forward
            if frequency in ['5m', '15m', '1h', 'hourly']:
                minutes_per_period = {'5m': 5, '15m': 15, '1h': 60, 'hourly': 60}.get(frequency, 60)
                adjusted_start = earliest_ts + timedelta(minutes=minutes_per_period * periods_required)
            else:
                adjusted_start = earliest_ts + timedelta(days=periods_required)
            
            indicator_value_start = adjusted_start
            
            print(f"  ⚠️  Adjusted start date for {indicator_name} (insufficient warmup)")
            print(f"     Warmup fetch start: {warmup_fetch_start}")
            print(f"     Requested start: {requested_start_date}")
            print(f"     Indicator values from: {indicator_value_start.date()} (adjusted forward)")
            print(f"     Warmup periods available: {actual_warmup_periods} / {periods_required} required")
            print(f"     Note: First {periods_required} periods used for warmup")
        else:
            # Sufficient warmup - calculate where indicator values start (after warmup)
            if frequency in ['5m', '15m', '1h', 'hourly']:
                minutes_per_period = {'5m': 5, '15m': 15, '1h': 60, 'hourly': 60}.get(frequency, 60)
                indicator_value_start = earliest_ts + timedelta(minutes=minutes_per_period * periods_required)
            else:
                indicator_value_start = earliest_ts + timedelta(days=periods_required)
            
            print(f"  ✅ Warmup data available for {indicator_name}")
            print(f"     Warmup fetch start: {warmup_fetch_start}")
            print(f"     Indicator values from: {indicator_value_start.date()} (after {periods_required} period warmup)")
            print(f"     Requested start: {requested_start_date}")
            print(f"     Warmup periods available: {actual_warmup_periods} / {periods_required} required")
    
    return warmup_fetch_start, indicator_value_start, actual_warmup_periods


def calculate_warmup_days(frequency: str, max_periods: int) -> int:
    """Calculate days needed for warmup based on frequency and periods.
    
    Args:
        frequency: Data frequency ('15m', '1h', 'daily', etc.)
        max_periods: Maximum periods needed (e.g., 200 for SMA_200)
    
    Returns:
        Number of days to rewind
    """
    # Bars per day
    bars_per_day = {
        '5m': 288,
        '15m': 96,
        '1h': 24,
        'hourly': 24,
        '1d': 1,
        'daily': 1
    }
    
    bars = bars_per_day.get(frequency, 1)
    # Add 20% buffer for weekends/gaps
    days = int((max_periods / bars) * 1.2) + 5
    return days


def fetch_raw_ohlcv(
    bq_client: bigquery.Client,
    ticker: str,
    start_date: str,
    end_date: str,
    frequency: str,
    warmup_days: int = 0
) -> pd.DataFrame:
    """Fetch raw OHLCV data with optional warmup period extension.
    
    Args:
        warmup_days: Days to extend backwards for indicator warmup
    
    Returns:
        DataFrame with forward-filled timestamps
    """
    from datetime import datetime, timedelta
    
    table_id = f"{PROJECT_ID}.{DATASET_ID}.{RAW_TABLE}"
    
    # Extend start date for warmup
    if warmup_days > 0:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        extended_start = (start_dt - timedelta(days=warmup_days)).strftime('%Y-%m-%d')
        print(f"  ⏪ Rewinding {warmup_days} days for warmup: {extended_start} → {start_date}")
    else:
        extended_start = start_date
    
    query = f"""
    SELECT 
        ticker,
        timestamp,
        date,
        frequency,
        open,
        high,
        low,
        close,
        volume,
        vwap
    FROM `{table_id}`
    WHERE ticker = '{ticker}'
      AND date BETWEEN '{extended_start}' AND '{end_date}'
      AND frequency = '{frequency}'
    ORDER BY timestamp ASC
    """
    
    print(f"  💾 Fetching raw OHLCV data from BigQuery...")
    df = bq_client.query(query).to_dataframe()
    print(f"  ✅ Fetched {len(df):,} rows")
    
    if df.empty or len(df) < 2:
        return df
    
    # No forward-filling - use only actual trading days
    # This ensures synthetic indicators are computed only for days when market was open
    # Holidays and weekends will naturally have gaps, which is correct behavior
    
    return df


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute return-based indicators."""
    df['returns'] = df['close'].pct_change()  # 1-period return
    df['returns_5'] = df['close'].pct_change(periods=5)  # 5-period return
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    return df


def compute_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """Compute volatility indicators."""
    df['volatility_5'] = df['returns'].rolling(window=5).std()
    df['volatility_10'] = df['returns'].rolling(window=10).std()
    df['volatility_20'] = df['returns'].rolling(window=20).std()
    return df


def compute_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """Compute momentum indicators."""
    df['momentum_5'] = df['close'] - df['close'].shift(5)
    df['momentum_10'] = df['close'] - df['close'].shift(10)
    df['roc_5'] = ((df['close'] - df['close'].shift(5)) / df['close'].shift(5)) * 100
    df['roc_10'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100
    
    # MACD (Moving Average Convergence Divergence)
    # Use min_periods for warmup consistency with other indicators
    ema_12 = df['close'].ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False, min_periods=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False, min_periods=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    return df


def compute_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute volume-based indicators."""
    df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
    df['volume_ma_10'] = df['volume'].rolling(window=10).mean()
    df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma_10']  # Volume relative to 10-period average
    
    # On-Balance Volume (OBV) - cumulative volume flow
    obv = np.zeros(len(df))
    obv[0] = df['volume'].iloc[0]
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv[i] = obv[i-1] + df['volume'].iloc[i]  # Price up: add volume
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv[i] = obv[i-1] - df['volume'].iloc[i]  # Price down: subtract volume
        else:
            obv[i] = obv[i-1]  # Price unchanged: keep OBV
    df['obv'] = obv
    
    return df


def compute_range_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute price range indicators."""
    # Average True Range (ATR)
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr_14'] = df['tr'].rolling(window=14).mean()
    df.drop('tr', axis=1, inplace=True)
    
    # High-Low ratio
    df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
    return df


def compute_bollinger_bands(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Bollinger Bands."""
    df['bb_middle_20'] = df['close'].rolling(window=20).mean()
    rolling_std = df['close'].rolling(window=20).std()
    df['bb_upper_20'] = df['bb_middle_20'] + (2 * rolling_std)
    df['bb_lower_20'] = df['bb_middle_20'] - (2 * rolling_std)
    df['bb_width'] = (df['bb_upper_20'] - df['bb_lower_20']) / df['bb_middle_20']
    return df


def compute_sma(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Simple Moving Averages with minimum periods requirement."""
    df['sma_10'] = df['close'].rolling(window=10, min_periods=10).mean()
    df['sma_20'] = df['close'].rolling(window=20, min_periods=20).mean()
    df['sma_50'] = df['close'].rolling(window=50, min_periods=50).mean()
    df['sma_200'] = df['close'].rolling(window=200, min_periods=200).mean()
    return df


def compute_ema(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Exponential Moving Averages."""
    # Use min_periods to ensure warmup consistency with other indicators
    df['ema_12'] = df['close'].ewm(span=12, adjust=False, min_periods=12).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False, min_periods=26).mean()
    return df


def compute_synthetic_indicators(
    df: pd.DataFrame,
    indicators_filter: set = None
) -> pd.DataFrame:
    """Compute all selected synthetic indicators."""
    print(f"  📊 Computing synthetic indicators...")
    
    # Determine which indicator groups to compute
    if indicators_filter:
        groups_to_compute = set()
        for group, indicators in INDICATOR_GROUPS.items():
            if any(ind in indicators_filter for ind in indicators):
                groups_to_compute.add(group)
    else:
        groups_to_compute = set(INDICATOR_GROUPS.keys())
    
    # Compute selected indicators
    if 'returns' in groups_to_compute:
        df = compute_returns(df)
        print(f"    ✅ Returns")
    
    if 'volatility' in groups_to_compute:
        if 'returns' not in df.columns:
            df = compute_returns(df)  # Volatility needs returns
        df = compute_volatility(df)
        print(f"    ✅ Volatility")
    
    if 'momentum' in groups_to_compute:
        df = compute_momentum(df)
        print(f"    ✅ Momentum")
    
    if 'volume' in groups_to_compute:
        df = compute_volume_indicators(df)
        print(f"    ✅ Volume indicators")
    
    if 'range' in groups_to_compute:
        df = compute_range_indicators(df)
        print(f"    ✅ Range indicators")
    
    if 'bollinger' in groups_to_compute:
        df = compute_bollinger_bands(df)
        print(f"    ✅ Bollinger Bands")
    
    if 'sma' in groups_to_compute:
        df = compute_sma(df)
        print(f"    ✅ Simple Moving Averages (SMA)")
    
    if 'ema' in groups_to_compute:
        df = compute_ema(df)
        print(f"    ✅ Exponential Moving Averages (EMA)")
    
    return df


def prepare_output_rows(
    df: pd.DataFrame,
    indicators_filter: set = None
) -> list:
    """Prepare rows for BigQuery insertion in wide format."""
    # Base columns
    base_cols = ['ticker', 'timestamp', 'date', 'frequency']
    
    # All possible indicator columns
    all_indicator_cols = []
    for indicators in INDICATOR_GROUPS.values():
        all_indicator_cols.extend(indicators)
    
    # Filter to selected indicators or all
    if indicators_filter:
        indicator_cols = [col for col in all_indicator_cols if col in indicators_filter and col in df.columns]
    else:
        indicator_cols = [col for col in all_indicator_cols if col in df.columns]
    
    # Select columns
    output_cols = base_cols + indicator_cols
    df_output = df[output_cols].copy()
    
    # Add ingested_at timestamp
    df_output['ingested_at'] = datetime.now(timezone.utc)
    
    # Convert timestamp to ISO string format for BigQuery
    # BigQuery TIMESTAMP type accepts ISO 8601 strings
    if 'timestamp' in df_output.columns:
        df_output['timestamp'] = df_output['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Convert date to string format BEFORE to_dict to prevent integer serialization
    # This fixes the "Value 16753 for partition column date" error
    if 'date' in df_output.columns:
        df_output['date'] = df_output['date'].apply(lambda x: x.strftime('%Y-%m-%d') if hasattr(x, 'strftime') else str(x))
    
    # Convert to records
    rows = df_output.to_dict('records')
    
    # Convert other datetime fields and handle NaN/None values
    for row in rows:
        # Convert ingested_at to ISO string
        if 'ingested_at' in row and isinstance(row['ingested_at'], datetime):
            row['ingested_at'] = row['ingested_at'].isoformat()
        
        # Convert NaN to None for ONLY the indicators in the output (don't fill missing ones)
        for col in indicator_cols:
            if col in row:
                if pd.isna(row[col]):
                    row[col] = None
                elif isinstance(row[col], (float, int)):
                    # Keep numeric values as-is, but convert NaN/inf to None
                    if not pd.isna(row[col]) and pd.notna(row[col]):
                        row[col] = float(row[col])
                    else:
                        row[col] = None
    
    return rows


def save_to_bigquery(bq_client: bigquery.Client, rows: list, table_id: str) -> int:
    """Save rows to BigQuery using load_table_from_json (supports historical data >10 years)."""
    if not rows:
        return 0
    
    print(f"  💾 Loading {len(rows):,} rows to BigQuery...")
    
    try:
        # Use load_table_from_json instead of insert_rows_json
        # This avoids the 10-year streaming partition limit
        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        )
        
        job = bq_client.load_table_from_json(
            rows,
            table_id,
            job_config=job_config
        )
        
        # Wait for job to complete
        job.result()
        
        print(f"  ✅ Successfully loaded {len(rows):,} rows")
        return len(rows)
        
    except Exception as e:
        print(f"  ❌ Error loading data: {str(e)}")
        if hasattr(e, 'errors') and e.errors:
            print(f"  ❌ Job errors:")
            for error in e.errors[:10]:
                print(f"     {error}")
        return 0


def merge_staging_to_main(
    bq_client: bigquery.Client,
    staging_table: str,
    main_table: str
):
    """Merge data from staging to main table with deduplication."""
    print("\n" + "=" * 80)
    print("   Merging Staging to Main Table")
    print("=" * 80)
    print()
    
    merge_query = f"""
    MERGE `{main_table}` T
    USING `{staging_table}` S
    ON T.ticker = S.ticker 
       AND T.timestamp = S.timestamp 
       AND T.frequency = S.frequency
    WHEN MATCHED THEN
      UPDATE SET
        T.returns = COALESCE(S.returns, T.returns),
        T.returns_5 = COALESCE(S.returns_5, T.returns_5),
        T.log_returns = COALESCE(S.log_returns, T.log_returns),
        T.volatility_5 = COALESCE(S.volatility_5, T.volatility_5),
        T.volatility_10 = COALESCE(S.volatility_10, T.volatility_10),
        T.volatility_20 = COALESCE(S.volatility_20, T.volatility_20),
        T.momentum_5 = COALESCE(S.momentum_5, T.momentum_5),
        T.momentum_10 = COALESCE(S.momentum_10, T.momentum_10),
        T.roc_5 = COALESCE(S.roc_5, T.roc_5),
        T.roc_10 = COALESCE(S.roc_10, T.roc_10),
        T.volume_ma_5 = COALESCE(S.volume_ma_5, T.volume_ma_5),
        T.volume_ma_10 = COALESCE(S.volume_ma_10, T.volume_ma_10),
        T.volume_ratio = COALESCE(S.volume_ratio, T.volume_ratio),
        T.obv = COALESCE(S.obv, T.obv),
        T.atr_14 = COALESCE(S.atr_14, T.atr_14),
        T.high_low_ratio = COALESCE(S.high_low_ratio, T.high_low_ratio),
        T.bb_upper_20 = COALESCE(S.bb_upper_20, T.bb_upper_20),
        T.bb_middle_20 = COALESCE(S.bb_middle_20, T.bb_middle_20),
        T.bb_lower_20 = COALESCE(S.bb_lower_20, T.bb_lower_20),
        T.bb_width = COALESCE(S.bb_width, T.bb_width),
        T.sma_10 = COALESCE(S.sma_10, T.sma_10),
        T.sma_20 = COALESCE(S.sma_20, T.sma_20),
        T.sma_50 = COALESCE(S.sma_50, T.sma_50),
        T.sma_200 = COALESCE(S.sma_200, T.sma_200),
        T.ema_12 = COALESCE(S.ema_12, T.ema_12),
        T.ema_26 = COALESCE(S.ema_26, T.ema_26),
        T.macd = COALESCE(S.macd, T.macd),
        T.macd_signal = COALESCE(S.macd_signal, T.macd_signal),
        T.macd_hist = COALESCE(S.macd_hist, T.macd_hist),
        T.ingested_at = S.ingested_at
    WHEN NOT MATCHED THEN
      INSERT (ticker, timestamp, date, frequency, returns, returns_5, log_returns,
              volatility_5, volatility_10, volatility_20,
              momentum_5, momentum_10, roc_5, roc_10,
              volume_ma_5, volume_ma_10, volume_ratio, obv,
              atr_14, high_low_ratio,
              bb_upper_20, bb_middle_20, bb_lower_20, bb_width,
              sma_10, sma_20, sma_50, sma_200,
              ema_12, ema_26, macd, macd_signal, macd_hist,
              ingested_at)
      VALUES (S.ticker, S.timestamp, S.date, S.frequency, S.returns, S.returns_5, S.log_returns,
              S.volatility_5, S.volatility_10, S.volatility_20,
              S.momentum_5, S.momentum_10, S.roc_5, S.roc_10,
              S.volume_ma_5, S.volume_ma_10, S.volume_ratio, S.obv,
              S.atr_14, S.high_low_ratio,
              S.bb_upper_20, S.bb_middle_20, S.bb_lower_20, S.bb_width,
              S.sma_10, S.sma_20, S.sma_50, S.sma_200,
              S.ema_12, S.ema_26, S.macd, S.macd_signal, S.macd_hist,
              S.ingested_at)
    """
    
    print("🔄 Running MERGE query...")
    merge_job = bq_client.query(merge_query)
    merge_job.result()
    
    stats = merge_job._properties.get('statistics', {}).get('query', {})
    rows_affected = stats.get('numDmlAffectedRows', 'unknown')
    
    print(f"✅ Merged {rows_affected} rows")
    
    # Truncate staging table
    print("\n🗑️  Truncating staging table...")
    truncate_query = f"TRUNCATE TABLE `{staging_table}`"
    truncate_job = bq_client.query(truncate_query)
    truncate_job.result()
    print("✅ Staging table cleared")
    print()


def purge_ticker_data(
    bq_client: bigquery.Client,
    table: str,
    tickers: list
):
    """Delete ALL synthetic indicator data for specified tickers."""
    print("\n" + "=" * 80)
    print("   🚨 PURGING ALL SYNTHETIC INDICATORS FOR TICKERS 🚨")
    print("=" * 80)
    print()
    
    ticker_list = "', '".join(tickers)
    
    print(f"⚠️  WARNING: Deleting ALL synthetic indicators for {len(tickers)} ticker(s):")
    print(f"   Tickers: {', '.join(tickers)}")
    print(f"   This will remove data for ALL dates and ALL frequencies!")
    print()
    
    response = input("Type 'DELETE' to confirm: ")
    if response != 'DELETE':
        print("❌ Purge cancelled")
        print()
        return
    
    try:
        delete_query = f"DELETE FROM `{table}` WHERE ticker IN ('{ticker_list}')"
        print("\n🗑️  Executing deletion...")
        delete_job = bq_client.query(delete_query)
        delete_job.result()
        
        stats = delete_job._properties.get('statistics', {}).get('query', {})
        rows_deleted = int(stats.get('numDmlAffectedRows', 0))
        
        print(f"✅ Purged {rows_deleted:,} rows")
        print()
    except Exception as e:
        print(f"❌ Purge failed: {str(e)}")
        print()


def get_latest_timestamp(bq_client: bigquery.Client, table: str, ticker: str, frequency: str):
    """Get the latest timestamp for a ticker/frequency from BigQuery.
    
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
    except Exception as e:
        print(f"   Error checking latest timestamp: {str(e)}")
    
    return None


def flush_existing_data(
    bq_client: bigquery.Client,
    table: str,
    tickers: list,
    start_date: str,
    end_date: str,
    frequency: str,
    indicators_filter: set = None
):
    """Flush existing synthetic indicator data for specified range.
    
    Args:
        indicators_filter: If provided, only NULL out these specific indicators.
                          If None, delete entire rows.
    """
    print("\n" + "=" * 80)
    print("   Flushing Existing Synthetic Indicators")
    print("=" * 80)
    print()
    
    ticker_list = "', '".join(tickers)
    
    try:
        if indicators_filter:
            # Partial update: Set specific indicator columns to NULL
            set_clauses = [f"{ind} = NULL" for ind in indicators_filter]
            set_clause = ", ".join(set_clauses)
            
            update_query = f"""
            UPDATE `{table}`
            SET {set_clause}
            WHERE ticker IN ('{ticker_list}')
              AND date BETWEEN '{start_date}' AND '{end_date}'
              AND frequency = '{frequency}'
            """
            
            print(f"  Updating specific indicators for {len(tickers)} ticker(s)...")
            print(f"   Indicators: {', '.join(sorted(indicators_filter))}")
            print(f"   Date range: {start_date} to {end_date}")
            print(f"   Frequency: {frequency}")
            print(f"   Action: Setting columns to NULL (preserving other indicators)")
            
            update_job = bq_client.query(update_query)
            update_job.result()
            
            stats = update_job._properties.get('statistics', {}).get('query', {})
            rows_updated = int(stats.get('numDmlAffectedRows', 0))
            
            print(f"  Updated {rows_updated:,} rows (set {len(indicators_filter)} indicators to NULL)")
            print()
        else:
            # Full delete: Remove entire rows
            delete_query = f"""
            DELETE FROM `{table}`
            WHERE ticker IN ('{ticker_list}')
              AND date BETWEEN '{start_date}' AND '{end_date}'
              AND frequency = '{frequency}'
            """
            
            print(f"  Deleting ALL indicators for {len(tickers)} ticker(s)...")
            print(f"   Date range: {start_date} to {end_date}")
            print(f"   Frequency: {frequency}")
            
            delete_job = bq_client.query(delete_query)
            delete_job.result()
            
            stats = delete_job._properties.get('statistics', {}).get('query', {})
            rows_deleted = int(stats.get('numDmlAffectedRows', 0))
            
            print(f"  Deleted {rows_deleted:,} entire rows")
            print()
    except Exception as e:
        print(f"  Flush failed: {str(e)}")
        print()


def compute_and_save_indicators(
    tickers: list,
    start_date: str,
    end_date: str,
    frequency: str = 'daily',
    auto_merge: bool = True,
    flush_existing: bool = False,
    purge_tickers: bool = False,
    indicators_filter: set = None,
    top_up: bool = False,
    flush_only: bool = False,
    dry_run: bool = False
):
    """Main function to compute and save synthetic indicators."""
    print("=" * 80)
    print("   Inflation prediction - Synthetic Indicators Computation")
    print("=" * 80)
    print(f"\nProject: {PROJECT_ID}")
    print(f"Dataset: {DATASET_ID}")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Frequency: {frequency.upper()}")
    if indicators_filter:
        print(f"Indicators: {', '.join(sorted(indicators_filter))}")
    if dry_run:
        print(f"Mode: DRY RUN (no data will be loaded)")
    print()
    
    # Validate configuration
    if not PROJECT_ID:
        print("  Error: GCP_PROJECT_ID not set")
        return
    
    # Initialize BigQuery client
    print("  Initializing BigQuery client...")
    bq_client = bigquery.Client(project=PROJECT_ID)
    
    # Table IDs
    staging_table = f"{PROJECT_ID}.{DATASET_ID}.{SYNTHETIC_STAGING_TABLE}"
    main_table = f"{PROJECT_ID}.{DATASET_ID}.{SYNTHETIC_TABLE}"
    
    # Handle purge/flush
    if purge_tickers:
        purge_ticker_data(bq_client, main_table, tickers)
        if start_date == '2000-01-01' and end_date == '2000-01-01':
            print("\n  Purge complete! Skipping computation.\n")
            return
    elif flush_existing:
        flush_existing_data(bq_client, main_table, tickers, start_date, end_date, frequency, indicators_filter)
        if flush_only:
            print("\n✅ Flush complete! Exiting without computation.\n")
            return
    
    # Statistics
    total_rows_inserted = 0
    total_tickers_processed = 0
    skipped_tickers = []
    
    # Determine max warmup needed
    max_periods = 0
    if indicators_filter:
        period_map = {
            # Returns (pct_change and shift need 1 period)
            'returns': 1, 'returns_5': 5, 'log_returns': 1,
            # SMAs
            'sma_10': 10, 'sma_20': 20, 'sma_50': 50, 'sma_200': 200,
            # EMAs
            'ema_12': 12, 'ema_26': 26,
            # Volatility (needs +1 for returns calculation: window + 1)
            'volatility_5': 6, 'volatility_10': 11, 'volatility_20': 21,
            # Momentum (MACD needs EMA26=26, signal needs +9=35)
            'momentum_5': 5, 'momentum_10': 10, 'roc_5': 5, 'roc_10': 10,
            'macd': 26, 'macd_signal': 35, 'macd_hist': 35,
            # Volume (volume_ratio inherits from volume_ma_10, obv needs 1 for close comparison)
            'volume_ma_5': 5, 'volume_ma_10': 10, 'volume_ma_20': 20, 'volume_ratio': 10, 'obv': 1,
            # Range and Bollinger (ATR needs +1 for close.shift(1), high_low_ratio no warmup)
            'atr_14': 15, 'high_low_ratio': 0,
            'bb_upper_20': 20, 'bb_middle_20': 20, 'bb_lower_20': 20, 'bb_width': 20
        }
        for ind, periods in period_map.items():
            if ind in indicators_filter:
                max_periods = max(max_periods, periods)
    else:
        # All indicators - SMA_200 is the largest
        max_periods = 200
    
    warmup_days = calculate_warmup_days(frequency, max_periods) if max_periods > 0 else 0
    if warmup_days > 0:
        print(f"\n  Warmup: {max_periods} periods = {warmup_days} days for {frequency}\n")
    
    # Clear staging table before loading new data
    print("\n  🗑️  Truncating staging table...")
    truncate_query = f"TRUNCATE TABLE `{staging_table}`"
    bq_client.query(truncate_query).result()
    print("  ✅ Staging table cleared\n")
    
    # Process each ticker
    for ticker_idx, ticker in enumerate(tickers, 1):
        print(f"[{ticker_idx}/{len(tickers)}]  Processing {ticker}...")
        
        # Skip if ticker already has data (unless flush_existing or purge_tickers)
        if not flush_existing and not purge_tickers:
            if check_ticker_exists(bq_client, ticker, frequency):
                print(f"  ⏭️  SKIPPED: {ticker} already has synthetic indicators (use --reload to overwrite)\n")
                skipped_tickers.append(ticker)
                continue
        
        # Top-up mode: Check for latest data and adjust start_date
        ticker_start_date = start_date
        ticker_end_date = end_date
        
        if top_up:
            # Check latest timestamp for this ticker
            latest_ts = get_latest_timestamp(bq_client, main_table, ticker, frequency)
            
            if latest_ts:
                print(f"   Top-up mode details:")
                print(f"     Latest timestamp in BigQuery: {latest_ts.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Add 1 period based on frequency
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
                    next_date = latest_ts + timedelta(days=1)
                    period_name = "1 day (default)"
                
                print(f"     Next period ({period_name}): {next_date.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"     Requested end date: {ticker_end_date}")
                
                ticker_start_date = next_date.strftime('%Y-%m-%d')
                
                # Parse end_date as END OF DAY timestamp for proper comparison
                # Make it timezone-aware (UTC) to match next_date from BigQuery
                end_datetime = datetime.strptime(ticker_end_date, '%Y-%m-%d').replace(hour=23, minute=59, second=59)
                end_datetime = pytz.UTC.localize(end_datetime)
                
                print(f"     End datetime (EOD): {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"     Comparison: {next_date.strftime('%Y-%m-%d %H:%M:%S')} > {end_datetime.strftime('%Y-%m-%d %H:%M:%S')} ? {next_date > end_datetime}")
                
                # Skip if next period is after end of day
                if next_date > end_datetime:
                    print(f"   Already up-to-date! Latest data is at or past end date.")
                    print(f"   Note: Top-up only fills forward from latest timestamp, not gaps in history.")
                    continue
                else:
                    print(f"   Computing from {ticker_start_date} to {ticker_end_date}")
            else:
                print(f"   No existing data found. Using start date: {ticker_start_date}")
        
        # Determine indicator name for logging
        indicator_name = f"max warmup ({max_periods} periods)"
        if indicators_filter:
            # Find the indicator with max periods
            period_map = {
                'sma_10': 10, 'sma_20': 20, 'sma_50': 50, 'sma_200': 200,
                'volatility_5': 5, 'volatility_10': 10, 'volatility_20': 20,
                'bb_upper_20': 20, 'bb_middle_20': 20, 'bb_lower_20': 20, 'bb_width': 20,
                'atr_14': 14, 'volume_ma_5': 5, 'volume_ma_10': 10
            }
            for ind, periods in period_map.items():
                if ind in indicators_filter and periods == max_periods:
                    indicator_name = ind
                    break
        
        # Calculate proper warmup dates for average-based indicators
        warmup_fetch_start, indicator_value_start, actual_warmup = calculate_indicator_warmup_dates(
            bq_client, ticker, ticker_start_date, frequency, max_periods, indicator_name
        )
        
        # If dry-run, calculate what would be loaded
        if dry_run:
            # Query to get actual date range from raw data
            table_id = f"{PROJECT_ID}.{DATASET_ID}.{RAW_TABLE}"
            range_query = f"""
            SELECT 
                MIN(timestamp) as first_timestamp,
                MAX(timestamp) as last_timestamp
            FROM `{table_id}`
            WHERE ticker = '{ticker}'
              AND frequency = '{frequency}'
              AND timestamp >= '{warmup_fetch_start}'
              AND timestamp <= '{ticker_end_date}'
            """
            
            range_result = bq_client.query(range_query).to_dataframe()
            
            if not range_result.empty and pd.notna(range_result['last_timestamp'].iloc[0]):
                first_ts = range_result['first_timestamp'].iloc[0]
                last_ts = range_result['last_timestamp'].iloc[0]
                
                # Calculate forward-filled continuous time series
                # This matches what fetch_raw_ohlcv does with pd.date_range
                freq_map = {
                    '5m': '5min',
                    '15m': '15min',
                    '1h': '1H',
                    'hourly': '1H',
                    '1d': '1D',
                    'daily': '1D'
                }
                pandas_freq = freq_map.get(frequency, '1D')
                
                # Total rows with forward-fill (from warmup_fetch_start to end)
                complete_range = pd.date_range(
                    start=first_ts,
                    end=last_ts,
                    freq=pandas_freq
                )
                total_rows_with_ff = len(complete_range)
                
                # Indicator rows (from indicator_value_start onwards)
                indicator_range = pd.date_range(
                    start=indicator_value_start,
                    end=last_ts,
                    freq=pandas_freq
                )
                indicator_rows = len(indicator_range)
                
                print(f"\n  📊 DRY RUN: Data availability check")
                print(f"     Indicator values would span: {indicator_value_start.date()} to {last_ts.date()}")
                print(f"     Total rows to fetch (with forward-fill): {total_rows_with_ff:,}")
                print(f"     Indicator entries to load: {indicator_rows:,} (warmup excluded)")
                print(f"     Indicators to compute: {len(indicators_filter) if indicators_filter else 'all'}")
                print(f"\n  🎯 Skipping actual data load...\n")
            else:
                print(f"\n  ⚠️  DRY RUN: No raw data found in range")
                print(f"     Would attempt to fetch from {warmup_fetch_start} to {ticker_end_date}")
                print(f"\n  🎯 Skipping actual data load...\n")
            
            continue
        
        # Fetch raw OHLCV from warmup start
        print(f"\n  📊 Fetching and computing indicators...")
        print(f"     Fetching from: {warmup_fetch_start}")
        print(f"     Computing to: {ticker_end_date}")
        
        df = fetch_raw_ohlcv(bq_client, ticker, warmup_fetch_start, ticker_end_date, frequency, warmup_days=0)
        
        if df.empty:
            print(f"   ❌ No raw OHLCV data found for {ticker}")
            print(f"   ℹ️  Run tickers_load_polygon.py first to fetch raw data")
            continue
        
        print(f"     Raw data fetched: {len(df):,} rows ({df['timestamp'].min().date()} to {df['timestamp'].max().date()})")
        
        # Compute synthetic indicators on full dataset (including warmup period)
        df = compute_synthetic_indicators(df, indicators_filter)
        
        # Filter: only keep rows where ANY requested indicator has a non-NULL value
        # This allows each indicator to start from its own warmup period
        if indicators_filter:
            indicator_cols = [col for col in indicators_filter if col in df.columns]
            if indicator_cols:
                df_before = len(df)
                
                # Log per-indicator first valid dates BEFORE filtering
                print(f"\n  📊 Indicator warmup completion dates:")
                total_rows = len(df)
                for ind in sorted(indicator_cols):
                    first_valid_idx = df[df[ind].notna()].index.min()
                    ind_row_count = df[df[ind].notna()].shape[0]
                    null_count = df[ind].isna().sum()
                    
                    if pd.notna(first_valid_idx):
                        first_valid_date = df.loc[first_valid_idx, 'timestamp'].date()
                        pct = (ind_row_count / total_rows) * 100
                        print(f"     {ind:20s}: {ind_row_count:6,} valid, {null_count:5,} nulls ({pct:5.1f}%) - from {first_valid_date}")
                    else:
                        print(f"     {ind:20s}: {ind_row_count:6,} valid, {null_count:5,} nulls (  0.0%) - No valid data")
                
                # Keep rows where at least one indicator is not null
                has_data = df[indicator_cols].notna().any(axis=1)
                df = df[has_data]
                df_after = len(df)
                warmup_excluded = df_before - df_after
                first_valid = df['timestamp'].min().date()
                print(f"\n  🎯 Saving {df_after:,} total rows from {first_valid} onwards")
                print(f"     Warmup periods excluded: {warmup_excluded} rows")
            else:
                print(f"  ⚠️  No indicator columns found in data")
        else:
            # No filter, keep all computed rows
            df_before = len(df)
            df = df[df['timestamp'] >= indicator_value_start]
            df_after = len(df)
            warmup_excluded = df_before - df_after
            print(f"  🎯 Saving {df_after:,} rows from {indicator_value_start.date()} onwards")
            print(f"     Warmup periods excluded: {warmup_excluded} rows ({max_periods} periods for {indicator_name})")
        
        if df.empty:
            print(f"   No valid data after warmup period")
            continue
        
        # Deduplicate by (ticker, timestamp, frequency) before saving
        # Keep last occurrence to ensure we have the most recent data
        df = df.drop_duplicates(subset=['ticker', 'timestamp', 'frequency'], keep='last')
        
        # Store dataframe for per-indicator summary before converting to rows
        df_for_summary = df.copy()
        
        # Prepare rows for BigQuery
        rows = prepare_output_rows(df, indicators_filter)
        
        # Save to staging
        print(f"\n  💾 Saving to BigQuery staging...")
        total_rows_inserted += save_to_bigquery(bq_client, rows, staging_table)
        print(f"     Saved {len(rows):,} rows to staging table")
        
        # Show actual per-indicator statistics from saved data
        if indicators_filter:
            print(f"\n  📊 Saved Per-Indicator Data:")
            indicator_cols = [col for col in indicators_filter if col in df_for_summary.columns]
            
            for ind in sorted(indicator_cols):
                # Get actual row count and first date with non-null values
                ind_data = df_for_summary[df_for_summary[ind].notna()]
                if len(ind_data) > 0:
                    ind_count = len(ind_data)
                    ind_first = ind_data['timestamp'].min().date()
                    ind_last = ind_data['timestamp'].max().date()
                    print(f"     {ind:20s}: {ind_count:,} rows ({ind_first} to {ind_last})")
                else:
                    print(f"     {ind:20s}: 0 rows (no valid data)")
        
        print(f"\n  ✅ {ticker} complete!\n")
        total_tickers_processed += 1
    
    # Merge staging to main
    if auto_merge and total_rows_inserted > 0 and not dry_run:
        merge_staging_to_main(bq_client, staging_table, main_table)
    
    # Final summary
    print("=" * 80)
    if dry_run:
        print("   DRY RUN Complete!")
    else:
        print("   Computation Complete!")
    print("=" * 80)
    if dry_run:
        print("  Summary:")
        print(f"  Tickers analyzed: {len(tickers)}")
        print("=" * 80)
    print("   Computation Complete!")
    print("=" * 80)
    print("📊 Statistics:")
    print(f"  Tickers requested: {len(tickers)}")
    print(f"  Tickers processed: {total_tickers_processed}")
    if skipped_tickers:
        print(f"  Tickers skipped: {len(skipped_tickers)} (already exist: {', '.join(skipped_tickers)})")
    print(f"  Total rows inserted: {total_rows_inserted:,}")
    print(f"  Frequency: {frequency}")
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
    """Check if ticker synthetic indicators already exist in BigQuery.
    
    Args:
        bq_client: BigQuery client
        ticker: Ticker symbol
        frequency: Data frequency
    
    Returns:
        True if data exists, False otherwise
    """
    table_id = f"{PROJECT_ID}.{DATASET_ID}.{SYNTHETIC_TABLE}"
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


def main():
    parser = argparse.ArgumentParser(
        description='Compute synthetic indicators from raw OHLCV data and save to BigQuery',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Compute indicators for all tickers from config file (default behavior)
  python scripts/01_extract/tickers_load_synthetic.py --start 2024-01-01 --end 2024-12-31 --frequency daily
  
  # Compute for specific tickers (overrides config)
  python scripts/01_extract/tickers_load_synthetic.py --tickers SPY QQQ --start 2024-01-01 --end 2024-12-31 --frequency daily
  
  # Reload/overwrite existing data
  python scripts/01_extract/tickers_load_synthetic.py --tickers SPY --start 2024-01-01 --end 2024-12-31 --frequency daily --reload
  
  # Compute specific indicators only
  python scripts/01_extract/tickers_load_synthetic.py --tickers SPY --start 2024-01-01 --end 2024-12-31 --indicators returns volatility_5 sma_20
  
  # Use custom config file
  python scripts/01_extract/tickers_load_synthetic.py --config configs/my_tickers.yaml --start 2024-01-01 --end 2024-12-31
        """
    )
    
    parser.add_argument(
        '--tickers',
        nargs='+',
        help='Ticker symbols to process (e.g., SPY QQQ). If not specified, loads all tickers from config file.'
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
        '--indicators',
        nargs='+',
        help='Specific indicators to compute (e.g., returns volatility_5 momentum_10). If not specified, computes all.'
    )
    parser.add_argument(
        '--reload',
        '--flush',
        dest='flush',
        action='store_true',
        help='Reload/overwrite existing data for tickers (deletes existing data for date range before computation)'
    )
    parser.add_argument(
        '--flush-only',
        action='store_true',
        help='Only flush/clear indicators without recomputing. Requires --indicators. If dates not specified, flushes ALL entries.'
    )
    parser.add_argument(
        '--purge',
        action='store_true',
        help='Delete ALL synthetic indicators for tickers - requires confirmation'
    )
    parser.add_argument(
        '--no-merge',
        action='store_true',
        help='Do not automatically merge staging to main table'
    )
    parser.add_argument(
        '--top-up',
        action='store_true',
        help='Incremental mode: auto-detect latest data and fill gaps to end date (default: today)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Test mode: calculate warmup dates and check data availability without loading'
    )
    
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
    
    # Validate flush-only
    if args.flush_only:
        # Automatically enable flush when using flush-only
        args.flush = True
        
        if not args.indicators:
            print("❌ Error: --flush-only requires --indicators (specify which indicators to clear)")
            sys.exit(1)
        
        # If dates not specified, flush ALL entries for the indicators
        if not args.start_date and not args.end_date:
            start_date = '2000-01-01'
            end_date = '2099-12-31'
            print(f"🗑️  Flush-only mode: No dates specified, flushing ALL entries")
        elif not args.start_date:
            print("❌ Error: --flush-only with --end requires --start date")
            sys.exit(1)
        else:
            start_date = args.start_date
            end_date = args.end_date or datetime.now().strftime('%Y-%m-%d')
    # Validate dates
    elif args.purge and not args.start_date and not args.end_date:
        start_date = '2000-01-01'
        end_date = '2000-01-01'
    elif args.top_up:
        # Top-up mode: start_date required (for warmup calculation), end_date defaults to today
        if not args.start_date:
            print("  Error: --start date required for --top-up (needed for warmup calculation)")
            sys.exit(1)
        start_date = args.start_date
        if not args.end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
            print(f"  Top-up to today: {end_date}")
        else:
            end_date = args.end_date
    elif not args.start_date:
        print("  Error: --start date is required")
        sys.exit(1)
    else:
        start_date = args.start_date
        # Default end_date to today if not provided
        end_date = args.end_date or datetime.now().strftime('%Y-%m-%d')
    
    # Parse indicators filter from args or config
    indicators_filter = None
    if args.indicators:
        # User specified indicators on command line
        indicators_filter = set(args.indicators)
        if args.flush_only:
            print(f"🗑️  Flush-only mode: Clearing {', '.join(sorted(indicators_filter))}\n")
        else:
            print(f"📌 Using {len(indicators_filter)} indicator(s) from command line: {', '.join(sorted(indicators_filter))}\n")
    elif config and 'indicators' in config and config['indicators']:
        # No indicators specified, load from config
        indicators_filter = set(config['indicators'])
        print(f"📌 Loaded {len(indicators_filter)} indicators from {args.config}")
        print(f"   Indicators: {', '.join(sorted(list(indicators_filter)[:10]))}{'...' if len(indicators_filter) > 10 else ''}\n")
    else:
        # No indicators specified and none in config, compute all
        print(f"📌 No indicators filter specified - computing all available indicators\n")
    
    # Run computation
    compute_and_save_indicators(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        frequency=args.frequency,
        auto_merge=not args.no_merge,
        flush_existing=args.flush,
        purge_tickers=args.purge,
        indicators_filter=indicators_filter,
        top_up=args.top_up,
        flush_only=args.flush_only,
        dry_run=args.dry_run
    )


if __name__ == '__main__':
    main()
