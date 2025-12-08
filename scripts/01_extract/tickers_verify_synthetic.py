"""Verify synthetic indicators data quality in BigQuery.

Analyzes synthetic_indicators table for completeness, gaps, duplicates, and data quality issues.
"""
import os
import sys
import argparse
from google.cloud import bigquery
import pandas as pd
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
SYNTHETIC_TABLE = os.environ.get('BQ_SYNTHETIC_TABLE', 'synthetic_indicators')
SYNTHETIC_STAGING = os.environ.get('BQ_SYNTHETIC_STAGING_TABLE', 'synthetic_indicators_staging')


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 80)
    print(f"   {text}")
    print("=" * 80)
    print()


def load_config(config_path: str = 'configs/tickers.yaml') -> dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config YAML file
    
    Returns:
        Configuration dictionary or None if file not found
    """
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"⚠️  Config file not found: {config_path}")
        return None


def get_group_tickers(config: dict, group_name: str) -> list:
    """Get tickers from a named group.
    
    Args:
        config: Configuration dictionary
        group_name: Name of the group (e.g., 'inflation', 'commodities')
    
    Returns:
        List of ticker symbols in the group
    """
    if not config or 'ticker_groups' not in config:
        print(f"❌ Error: No ticker_groups found in config")
        return []
    
    if group_name not in config['ticker_groups']:
        available_groups = list(config['ticker_groups'].keys())
        print(f"❌ Error: Group '{group_name}' not found")
        print(f"   Available groups: {', '.join(available_groups)}")
        return []
    
    return config['ticker_groups'][group_name]


def analyze_data_coverage(client: bigquery.Client, table_id: str, ticker: str = None, frequency: str = None):
    """Analyze data coverage and completeness."""
    print_header("Data Coverage Analysis")
    
    where_clause = []
    if ticker:
        where_clause.append(f"ticker = '{ticker}'")
    if frequency:
        where_clause.append(f"frequency = '{frequency}'")
    
    where_sql = "WHERE " + " AND ".join(where_clause) if where_clause else ""
    
    query = f"""
    SELECT 
        ticker,
        frequency,
        COUNT(*) as total_rows,
        MIN(date) as earliest_date,
        MAX(date) as latest_date,
        DATE_DIFF(MAX(date), MIN(date), DAY) + 1 as date_range_days,
        COUNT(DISTINCT date) as unique_dates,
        COUNT(*) / NULLIF(COUNT(DISTINCT date), 0) as bars_per_day
    FROM `{table_id}`
    {where_sql}
    GROUP BY ticker, frequency
    ORDER BY ticker, frequency
    """
    
    df = client.query(query).to_dataframe()
    
    if df.empty:
        print("❌ No data found!")
        return
    
    print("📊 Coverage Summary:\n")
    for _, row in df.iterrows():
        print(f"Ticker: {row['ticker']} | Frequency: {row['frequency']}")
        print(f"  Total rows: {row['total_rows']:,}")
        print(f"  Date range: {row['earliest_date']} to {row['latest_date']}")
        print(f"  Days covered: {row['date_range_days']}")
        print(f"  Unique dates: {row['unique_dates']}")
        print(f"  Avg bars/day: {row['bars_per_day']:.1f}")
        
        # Expected bars per day
        freq = row['frequency']
        expected_map = {
            'daily': 1, '1d': 1,
            'hourly': 24, '1h': 24,
            '15m': 96,
            '5m': 288
        }
        expected_bars = expected_map.get(freq)
        
        if expected_bars:
            actual = row['bars_per_day']
            coverage_pct = (actual / expected_bars) * 100
            print(f"  Expected bars/day: {expected_bars}")
            print(f"  Coverage: {coverage_pct:.1f}%")
            
            if coverage_pct < 80:
                print(f"  ⚠️  WARNING: Low coverage! Expected {expected_bars} bars/day, got {actual:.1f}")
            elif coverage_pct >= 99:
                print(f"  ✅ Excellent coverage!")
            else:
                print(f"  ⚠️  Partial coverage")
        print()


def find_date_gaps(client: bigquery.Client, table_id: str, ticker: str, frequency: str, min_gap_hours: int = 2, exclude_weekends: bool = False):
    """Find gaps in time series data.
    
    Args:
        client: BigQuery client
        table_id: Table to query
        ticker: Ticker symbol
        frequency: Data frequency
        min_gap_hours: Minimum gap size to report in hours
        exclude_weekends: If True, exclude weekend gaps (Fri->Mon) from results
    """
    print_header(f"Date Gaps Analysis - {ticker} ({frequency})")
    
    query = f"""
    WITH ordered_data AS (
        SELECT 
            timestamp,
            LAG(timestamp) OVER (ORDER BY timestamp) as prev_timestamp,
            TIMESTAMP_DIFF(timestamp, LAG(timestamp) OVER (ORDER BY timestamp), HOUR) as gap_hours
        FROM `{table_id}`
        WHERE ticker = '{ticker}' AND frequency = '{frequency}'
        ORDER BY timestamp
    )
    SELECT 
        prev_timestamp as gap_start,
        timestamp as gap_end,
        gap_hours
    FROM ordered_data
    WHERE gap_hours >= {min_gap_hours}
    ORDER BY gap_hours DESC
    LIMIT 20
    """
    
    df = client.query(query).to_dataframe()
    
    if df.empty:
        print(f"✅ No gaps >= {min_gap_hours} hours found!")
        return
    
    # Filter out weekend gaps if requested
    if exclude_weekends:
        original_count = len(df)
        # For stock market data, gaps are expected for:
        # - Regular weekends (Fri->Mon: 48-72 hours)
        # - 3-day weekends with holidays (Fri->Tue or Thu->Mon: 72-96 hours)
        # - 4-day holiday weekends (Thu->Tue: 96-120 hours)
        weekend_gaps = []
        non_weekend_gaps = []
        
        for idx, row in df.iterrows():
            gap_start = row['gap_start']
            gap_end = row['gap_end']
            gap_hours = row['gap_hours']
            
            start_weekday = gap_start.weekday()  # 0=Mon, 4=Fri, 6=Sun
            end_weekday = gap_end.weekday()
            
            # Classify as weekend/holiday gap if:
            # 1. Regular weekend: Fri->Mon (48-72h)
            # 2. 3-day weekend: Fri->Tue or Thu->Mon (72-96h)
            # 3. 4-day weekend: Thu->Tue (96-120h)
            # 4. Any gap ending on Monday/Tuesday that's 48-120 hours (covers most holidays)
            is_weekend_gap = (
                # Regular weekend or 3-day weekend
                (end_weekday in [0, 1] and 48 <= gap_hours <= 120) or
                # Specifically Friday to Monday/Tuesday
                (start_weekday == 4 and end_weekday in [0, 1] and 48 <= gap_hours <= 120) or
                # Thursday to Monday/Tuesday (holiday weekends)
                (start_weekday == 3 and end_weekday in [0, 1] and 72 <= gap_hours <= 120)
            )
            
            if is_weekend_gap:
                weekend_gaps.append(row)
            else:
                non_weekend_gaps.append(row)
        
        if weekend_gaps:
            print(f"ℹ️  Excluded {len(weekend_gaps)} weekend/holiday gaps (expected for stock market data)")
        
        if not non_weekend_gaps:
            print(f"✅ No unexpected gaps found (all {original_count} gaps are weekends/holidays)")
            return
        
        df = pd.DataFrame(non_weekend_gaps)
        print(f"⚠️  Found {len(df)} unexpected gaps >= {min_gap_hours} hours:\n")
    else:
        print(f"⚠️  Found {len(df)} gaps >= {min_gap_hours} hours:\n")
    
    for idx, row in df.iterrows():
        print(f"  Gap #{idx+1}: {row['gap_hours']:.1f} hours")
        print(f"    From: {row['gap_start']}")
        print(f"    To:   {row['gap_end']}")
        print()


def check_duplicates(client: bigquery.Client, table_id: str, ticker: str = None):
    """Check for duplicate records."""
    print_header("Duplicate Check")
    
    where_sql = f"WHERE ticker = '{ticker}'" if ticker else ""
    
    query = f"""
    SELECT 
        ticker,
        timestamp,
        frequency,
        COUNT(*) as duplicate_count
    FROM `{table_id}`
    {where_sql}
    GROUP BY ticker, timestamp, frequency
    HAVING COUNT(*) > 1
    ORDER BY duplicate_count DESC
    LIMIT 10
    """
    
    df = client.query(query).to_dataframe()
    
    if df.empty:
        print("✅ No duplicates found!")
        return
    
    print(f"❌ Found {len(df)} duplicate timestamp/ticker/frequency combinations:\n")
    for _, row in df.iterrows():
        print(f"  {row['ticker']} @ {row['timestamp']} ({row['frequency']}): {row['duplicate_count']} copies")
    print()


def analyze_indicator_coverage(client: bigquery.Client, table_id: str, ticker: str = None, frequency: str = None, config_indicators: list = None):
    """Analyze synthetic indicator coverage and quality."""
    print_header("Synthetic Indicator Coverage")
    
    where_clause = []
    if ticker:
        where_clause.append(f"ticker = '{ticker}'")
    if frequency:
        where_clause.append(f"frequency = '{frequency}'")
    
    where_sql = "WHERE " + " AND ".join(where_clause) if where_clause else ""
    
    # Count non-null values for each indicator
    query = f"""
    SELECT 
        ticker,
        frequency,
        COUNT(*) as total_rows,
        MIN(date) as earliest_date,
        MAX(date) as latest_date,
        -- Returns indicators
        COUNTIF(returns IS NOT NULL) as returns_count,
        COUNTIF(returns_5 IS NOT NULL) as returns_5_count,
        COUNTIF(log_returns IS NOT NULL) as log_returns_count,
        -- Volatility indicators
        COUNTIF(volatility_5 IS NOT NULL) as volatility_5_count,
        COUNTIF(volatility_10 IS NOT NULL) as volatility_10_count,
        COUNTIF(volatility_20 IS NOT NULL) as volatility_20_count,
        -- Momentum indicators
        COUNTIF(momentum_5 IS NOT NULL) as momentum_5_count,
        COUNTIF(momentum_10 IS NOT NULL) as momentum_10_count,
        COUNTIF(roc_5 IS NOT NULL) as roc_5_count,
        COUNTIF(roc_10 IS NOT NULL) as roc_10_count,
        COUNTIF(macd IS NOT NULL) as macd_count,
        COUNTIF(macd_signal IS NOT NULL) as macd_signal_count,
        COUNTIF(macd_hist IS NOT NULL) as macd_hist_count,
        -- Volume indicators
        COUNTIF(volume_ma_5 IS NOT NULL) as volume_ma_5_count,
        COUNTIF(volume_ma_10 IS NOT NULL) as volume_ma_10_count,
        COUNTIF(volume_ratio IS NOT NULL) as volume_ratio_count,
        COUNTIF(obv IS NOT NULL) as obv_count,
        -- Price range indicators
        COUNTIF(atr_14 IS NOT NULL) as atr_14_count,
        COUNTIF(high_low_ratio IS NOT NULL) as high_low_ratio_count,
        -- Bollinger Bands
        COUNTIF(bb_upper_20 IS NOT NULL) as bb_upper_20_count,
        COUNTIF(bb_middle_20 IS NOT NULL) as bb_middle_20_count,
        COUNTIF(bb_lower_20 IS NOT NULL) as bb_lower_20_count,
        COUNTIF(bb_width IS NOT NULL) as bb_width_count,
        -- SMA indicators
        COUNTIF(sma_10 IS NOT NULL) as sma_10_count,
        COUNTIF(sma_20 IS NOT NULL) as sma_20_count,
        COUNTIF(sma_50 IS NOT NULL) as sma_50_count,
        COUNTIF(sma_200 IS NOT NULL) as sma_200_count,
        -- EMA indicators
        COUNTIF(ema_12 IS NOT NULL) as ema_12_count,
        COUNTIF(ema_26 IS NOT NULL) as ema_26_count
    FROM `{table_id}`
    {where_sql}
    GROUP BY ticker, frequency
    ORDER BY ticker, frequency
    """
    
    df = client.query(query).to_dataframe()
    
    if df.empty:
        print("❌ No data found")
        return
    
    # Indicator categories - filter by config if provided
    all_indicator_groups = {
        "Returns": ["returns", "returns_5", "log_returns"],
        "Volatility": ["volatility_5", "volatility_10", "volatility_20"],
        "Momentum": ["momentum_5", "momentum_10", "roc_5", "roc_10", "macd", "macd_signal", "macd_hist"],
        "Volume": ["volume_ma_5", "volume_ma_10", "volume_ratio", "obv"],
        "Price Range": ["atr_14", "high_low_ratio"],
        "Bollinger Bands": ["bb_upper_20", "bb_middle_20", "bb_lower_20", "bb_width"],
        "SMA": ["sma_10", "sma_20", "sma_50", "sma_200"],
        "EMA": ["ema_12", "ema_26"]
    }
    
    # Filter indicators based on config if provided
    if config_indicators:
        indicator_groups = {}
        for group_name, indicators in all_indicator_groups.items():
            filtered = [ind for ind in indicators if ind in config_indicators]
            if filtered:
                indicator_groups[group_name] = filtered
    else:
        indicator_groups = all_indicator_groups
    
    for _, row in df.iterrows():
        print(f"\n📈 {row['ticker']} ({row['frequency']})")
        print(f"Date range: {row['earliest_date']} to {row['latest_date']}")
        print(f"Total rows: {row['total_rows']:,}\n")
        
        for group_name, indicators in indicator_groups.items():
            print(f"  {group_name}:")
            for indicator in indicators:
                col_name = f"{indicator}_count"
                if col_name in row.index:
                    count = row[col_name]
                    total = row['total_rows']
                    pct = (count / total * 100) if total > 0 else 0
                    
                    # Visual indicator
                    if pct >= 99:
                        status = "✅"
                    elif pct >= 80:
                        status = "⚠️ "
                    else:
                        status = "❌"
                    
                    print(f"    {status} {indicator:20s}: {count:6,} / {total:6,} ({pct:5.1f}%)")
            print()


def analyze_data_quality(client: bigquery.Client, table_id: str, ticker: str = None, frequency: str = None):
    """Analyze data quality metrics for synthetic indicators."""
    print_header("Data Quality Analysis")
    
    where_clause = []
    if ticker:
        where_clause.append(f"ticker = '{ticker}'")
    if frequency:
        where_clause.append(f"frequency = '{frequency}'")
    
    where_sql = "WHERE " + " AND ".join(where_clause) if where_clause else ""
    
    query = f"""
    SELECT 
        ticker,
        frequency,
        -- Check for suspicious values
        COUNTIF(returns > 1.0 OR returns < -1.0) as extreme_returns,
        COUNTIF(volatility_20 < 0) as negative_volatility,
        COUNTIF(atr_14 < 0) as negative_atr,
        COUNTIF(bb_width < 0) as negative_bb_width,
        COUNTIF(volume_ratio < 0) as negative_volume_ratio,
        COUNTIF(high_low_ratio < 0) as negative_high_low_ratio,
        -- Check for NaN or Inf
        COUNTIF(IS_NAN(returns)) as nan_returns,
        COUNTIF(IS_INF(returns)) as inf_returns,
        COUNTIF(IS_NAN(volatility_20)) as nan_volatility_20,
        COUNTIF(IS_INF(volatility_20)) as inf_volatility_20,
        -- Check for all-null rows
        COUNTIF(returns IS NULL AND log_returns IS NULL AND volatility_5 IS NULL 
                AND momentum_5 IS NULL AND sma_10 IS NULL) as completely_null_rows,
        COUNT(*) as total_rows
    FROM `{table_id}`
    {where_sql}
    GROUP BY ticker, frequency
    ORDER BY ticker, frequency
    """
    
    df = client.query(query).to_dataframe()
    
    if df.empty:
        print("❌ No data found")
        return
    
    for _, row in df.iterrows():
        print(f"\n🔍 {row['ticker']} ({row['frequency']})")
        print(f"Total rows: {row['total_rows']:,}\n")
        
        issues_found = False
        
        # Check for extreme values
        if row['extreme_returns'] > 0:
            print(f"  ⚠️  Extreme returns (>100% or <-100%): {row['extreme_returns']:,}")
            issues_found = True
        
        if row['negative_volatility'] > 0:
            print(f"  ❌ Negative volatility values: {row['negative_volatility']:,}")
            issues_found = True
        
        if row['negative_atr'] > 0:
            print(f"  ❌ Negative ATR values: {row['negative_atr']:,}")
            issues_found = True
        
        if row['negative_bb_width'] > 0:
            print(f"  ❌ Negative Bollinger Band width: {row['negative_bb_width']:,}")
            issues_found = True
        
        if row['negative_volume_ratio'] > 0:
            print(f"  ❌ Negative volume ratio: {row['negative_volume_ratio']:,}")
            issues_found = True
        
        if row['negative_high_low_ratio'] > 0:
            print(f"  ❌ Negative high/low ratio: {row['negative_high_low_ratio']:,}")
            issues_found = True
        
        # Check for NaN/Inf
        if row['nan_returns'] > 0 or row['inf_returns'] > 0:
            print(f"  ❌ NaN/Inf in returns: {row['nan_returns']:,} NaN, {row['inf_returns']:,} Inf")
            issues_found = True
        
        if row['nan_volatility_20'] > 0 or row['inf_volatility_20'] > 0:
            print(f"  ❌ NaN/Inf in volatility_20: {row['nan_volatility_20']:,} NaN, {row['inf_volatility_20']:,} Inf")
            issues_found = True
        
        if row['completely_null_rows'] > 0:
            pct = (row['completely_null_rows'] / row['total_rows'] * 100)
            print(f"  ⚠️  Completely null rows: {row['completely_null_rows']:,} ({pct:.1f}%)")
            issues_found = True
        
        if not issues_found:
            print("  ✅ No data quality issues detected!")
        print()


def analyze_specific_indicator(client: bigquery.Client, table_id: str, ticker: str, frequency: str, indicator: str):
    """Detailed analysis of a specific indicator for a ticker."""
    print_header(f"Detailed Analysis: {indicator.upper()} - {ticker} ({frequency})")
    
    # First, check if indicator exists
    query_check = f"""
    SELECT column_name 
    FROM `{PROJECT_ID}.{DATASET_ID}.INFORMATION_SCHEMA.COLUMNS`
    WHERE table_name = '{table_id.split('.')[-1]}'
    AND column_name = '{indicator}'
    """
    
    result = client.query(query_check).to_dataframe()
    if result.empty:
        print(f"❌ Indicator '{indicator}' not found in table!")
        print(f"\nAvailable indicators: returns, returns_5, log_returns, volatility_5, volatility_10, volatility_20,")
        print(f"  momentum_5, momentum_10, roc_5, roc_10, volume_ma_5, volume_ma_10, volume_ratio,")
        print(f"  atr_14, high_low_ratio, bb_upper_20, bb_middle_20, bb_lower_20, bb_width,")
        print(f"  sma_10, sma_20, sma_50, sma_200")
        return
    
    # Get overall statistics
    query_stats = f"""
    SELECT 
        COUNT(*) as total_rows,
        MIN(date) as overall_start,
        MAX(date) as overall_end,
        COUNTIF({indicator} IS NOT NULL) as non_null_count,
        COUNTIF({indicator} IS NULL) as null_count,
        MIN(CASE WHEN {indicator} IS NOT NULL THEN date END) as indicator_start,
        MAX(CASE WHEN {indicator} IS NOT NULL THEN date END) as indicator_end,
        MIN({indicator}) as min_value,
        MAX({indicator}) as max_value,
        AVG({indicator}) as avg_value,
        STDDEV({indicator}) as stddev_value
    FROM `{table_id}`
    WHERE ticker = '{ticker}' AND frequency = '{frequency}'
    """
    
    stats = client.query(query_stats).to_dataframe().iloc[0]
    
    if stats['total_rows'] == 0:
        print(f"❌ No data found for {ticker} ({frequency})")
        return
    
    # Print statistics
    print(f"📊 Coverage Summary:")
    print(f"  Total rows: {stats['total_rows']:,}")
    print(f"  Data range: {stats['overall_start']} to {stats['overall_end']}")
    print()
    
    coverage_pct = (stats['non_null_count'] / stats['total_rows'] * 100) if stats['total_rows'] > 0 else 0
    
    if stats['non_null_count'] > 0:
        print(f"✅ Indicator '{indicator}' Coverage:")
        print(f"  Non-NULL values: {stats['non_null_count']:,} / {stats['total_rows']:,} ({coverage_pct:.2f}%)")
        print(f"  NULL values: {stats['null_count']:,}")
        print(f"  First value: {stats['indicator_start']}")
        print(f"  Last value: {stats['indicator_end']}")
        print()
        print(f"📈 Value Statistics:")
        print(f"  Min: {stats['min_value']:.6f}")
        print(f"  Max: {stats['max_value']:.6f}")
        print(f"  Avg: {stats['avg_value']:.6f}")
        print(f"  StdDev: {stats['stddev_value']:.6f}")
        print()
    else:
        print(f"❌ Indicator '{indicator}' has NO non-NULL values!")
        return
    
    # Find NULL gaps (periods where indicator is NULL)
    query_gaps = f"""
    WITH numbered AS (
        SELECT 
            timestamp,
            date,
            {indicator},
            CASE WHEN {indicator} IS NULL THEN 1 ELSE 0 END as is_null,
            ROW_NUMBER() OVER (ORDER BY timestamp) as row_num
        FROM `{table_id}`
        WHERE ticker = '{ticker}' AND frequency = '{frequency}'
    ),
    grouped AS (
        SELECT 
            timestamp,
            date,
            is_null,
            row_num - ROW_NUMBER() OVER (PARTITION BY is_null ORDER BY timestamp) as grp
        FROM numbered
    ),
    gaps AS (
        SELECT 
            MIN(date) as gap_start,
            MAX(date) as gap_end,
            COUNT(*) as gap_length,
            DATE_DIFF(MAX(date), MIN(date), DAY) as gap_days
        FROM grouped
        WHERE is_null = 1
        GROUP BY grp
        HAVING COUNT(*) >= 2
        ORDER BY gap_length DESC
        LIMIT 20
    )
    SELECT * FROM gaps
    """
    
    gaps_df = client.query(query_gaps).to_dataframe()
    
    if not gaps_df.empty:
        print(f"⚠️  NULL Gaps Found ({len(gaps_df)} gaps with 2+ consecutive NULLs):")
        print()
        for idx, gap in gaps_df.iterrows():
            print(f"  Gap #{idx+1}: {gap['gap_length']} rows ({gap['gap_days']} days)")
            print(f"    From: {gap['gap_start']}")
            print(f"    To:   {gap['gap_end']}")
            print()
    else:
        print(f"✅ No significant NULL gaps found!")
        print()
    
    # Sample non-NULL values (first and last 5)
    query_sample = f"""
    (SELECT timestamp, date, {indicator}
     FROM `{table_id}`
     WHERE ticker = '{ticker}' AND frequency = '{frequency}' AND {indicator} IS NOT NULL
     ORDER BY timestamp ASC
     LIMIT 5)
    UNION ALL
    (SELECT timestamp, date, {indicator}
     FROM `{table_id}`
     WHERE ticker = '{ticker}' AND frequency = '{frequency}' AND {indicator} IS NOT NULL
     ORDER BY timestamp DESC
     LIMIT 5)
    ORDER BY timestamp ASC
    """
    
    sample_df = client.query(query_sample).to_dataframe()
    
    if not sample_df.empty:
        print(f"📅 Sample Values (First & Last 5):")
        pd.set_option('display.float_format', '{:.6f}'.format)
        print(sample_df.to_string(index=False))
        print()


def sample_data(client: bigquery.Client, table_id: str, ticker: str = None, frequency: str = None, limit: int = 10):
    """Show sample records - first 10 (oldest) and last 10 (newest) entries."""
    print_header("Sample Data")
    
    where_clause = []
    if ticker:
        where_clause.append(f"ticker = '{ticker}'")
    if frequency:
        where_clause.append(f"frequency = '{frequency}'")
    
    where_sql = "WHERE " + " AND ".join(where_clause) if where_clause else ""
    
    columns = """ticker, timestamp, frequency, returns, returns_5, log_returns, volatility_20, 
                 momentum_10, roc_10, atr_14, sma_10, sma_20, sma_50, bb_upper_20, bb_lower_20"""
    
    # Get first 10 (oldest)
    query_first = f"""
    SELECT {columns}
    FROM `{table_id}`
    {where_sql}
    ORDER BY timestamp ASC
    LIMIT {limit}
    """
    
    # Get last 10 (newest)
    query_last = f"""
    SELECT {columns}
    FROM `{table_id}`
    {where_sql}
    ORDER BY timestamp DESC
    LIMIT {limit}
    """
    
    df_first = client.query(query_first).to_dataframe()
    df_last = client.query(query_last).to_dataframe()
    
    if df_first.empty and df_last.empty:
        print("❌ No data found")
        return
    
    # Format output
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    pd.set_option('display.float_format', '{:.4f}'.format)
    
    if not df_first.empty:
        print(f"\n📅 OLDEST {len(df_first)} ENTRIES:")
        print(df_first.to_string(index=False))
        print()
    
    if not df_last.empty:
        print(f"\n📅 NEWEST {len(df_last)} ENTRIES:")
        # Reverse to show in chronological order (oldest to newest within this set)
        df_last_sorted = df_last.sort_values('timestamp')
        print(df_last_sorted.to_string(index=False))
    
    print()


def quick_verify_tickers(client: bigquery.Client, table_id: str, tickers: list, frequency: str, exclude_weekends_and_holidays: bool = False):
    """Quick verification of multiple tickers - shows summary table only."""
    print_header("Quick Ticker Verification")
    
    results = []
    for ticker in tickers:
        # Get basic stats
        query = f"""
        SELECT 
            '{ticker}' as ticker,
            COUNT(*) as total_rows,
            MIN(date) as start_date,
            MAX(date) as end_date,
            DATE_DIFF(MAX(date), MIN(date), DAY) + 1 as date_range_days,
            COUNT(DISTINCT date) as unique_dates
        FROM `{table_id}`
        WHERE ticker = '{ticker}'
            AND frequency = '{frequency}'
        """
        
        try:
            df = client.query(query).to_dataframe()
            if df.empty or df['total_rows'].iloc[0] == 0:
                results.append({
                    'ticker': ticker,
                    'status': '❌ NO DATA',
                    'rows': 0,
                    'start': 'N/A',
                    'end': 'N/A',
                    'days': 0,
                    'gaps': 0
                })
                continue
            
            row = df.iloc[0]
            
            # Check for gaps
            if exclude_weekends_and_holidays:
                # Exclude weekend and holiday gaps
                # Regular weekend: ~72 hours (Fri→Mon)
                # 3-day holiday weekend: ~96 hours (Fri→Tue)
                # Threshold of 100 hours catches most holidays while flagging real gaps
                gap_query = f"""
                WITH dates AS (
                    SELECT 
                        ticker,
                        date,
                        TIMESTAMP_DIFF(
                            date,
                            LAG(date) OVER (PARTITION BY ticker ORDER BY date),
                            HOUR
                        ) AS gap_hours
                    FROM `{table_id}`
                    WHERE ticker = '{ticker}' AND frequency = '{frequency}'
                )
                SELECT COUNT(*) as gap_count
                FROM dates
                WHERE gap_hours > 100  -- More than 4 days (excludes weekends + holidays)
                """
            else:
                gap_query = f"""
                WITH dates AS (
                    SELECT 
                        ticker,
                        date,
                        TIMESTAMP_DIFF(
                            date,
                            LAG(date) OVER (PARTITION BY ticker ORDER BY date),
                            HOUR
                        ) AS gap_hours
                    FROM `{table_id}`
                    WHERE ticker = '{ticker}' AND frequency = '{frequency}'
                )
                SELECT COUNT(*) as gap_count
                FROM dates
                WHERE gap_hours > 48  -- More than 2 days
                """
            
            gap_df = client.query(gap_query).to_dataframe()
            gap_count = gap_df['gap_count'].iloc[0] if not gap_df.empty else 0
            
            # Determine status
            if row['total_rows'] < 100:
                status = '⚠️  LOW DATA'
            elif gap_count > 5:
                status = '⚠️  GAPS'
            else:
                status = '✅ OK'
            
            results.append({
                'ticker': ticker,
                'status': status,
                'rows': int(row['total_rows']),
                'start': str(row['start_date']),
                'end': str(row['end_date']),
                'days': int(row['date_range_days']),
                'gaps': int(gap_count)
            })
            
        except Exception as e:
            results.append({
                'ticker': ticker,
                'status': f'❌ ERROR',
                'rows': 0,
                'start': 'N/A',
                'end': 'N/A',
                'days': 0,
                'gaps': 0
            })
    
    # Print summary table
    print(f"\n{'Ticker':<10} {'Status':<15} {'Rows':>8} {'Start Date':<12} {'End Date':<12} {'Days':>6} {'Gaps':>5}")
    print("=" * 80)
    
    for r in results:
        print(f"{r['ticker']:<10} {r['status']:<15} {r['rows']:>8,} {r['start']:<12} {r['end']:<12} {r['days']:>6} {r['gaps']:>5}")
    
    print()
    print(f"Total: {len(results)} tickers | " + 
          f"✅ {sum(1 for r in results if '✅' in r['status'])} OK | " +
          f"⚠️  {sum(1 for r in results if '⚠️' in r['status'])} Warning | " +
          f"❌ {sum(1 for r in results if '❌' in r['status'])} Error")
    print()


def main():
    """Main function to run verification checks."""
    parser = argparse.ArgumentParser(
        description="Verify synthetic indicators data quality in BigQuery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Verify all tickers from config
  python scripts/01_extract/tickers_verify_synthetic.py --frequency daily
  
  # Verify specific ticker (excluding weekend gaps)
  python scripts/01_extract/tickers_verify_synthetic.py --ticker SPY --frequency daily --exclude-weekends
  
  # Quick verify multiple tickers
  python scripts/01_extract/tickers_verify_synthetic.py --tickers CORN CANE COW UBC --frequency daily --quick
  
  # Quick verify ticker group
  python scripts/01_extract/tickers_verify_synthetic.py --group inflation --frequency daily --quick
  
  # Analyze specific indicator
  python scripts/01_extract/tickers_verify_synthetic.py --ticker SPY --frequency daily --indicator sma_20
  
  # Use custom config file
  python scripts/01_extract/tickers_verify_synthetic.py --config configs/my_tickers.yaml --frequency daily --exclude-weekends
        """
    )
    parser.add_argument(
        "--ticker",
        type=str,
        help="Filter by specific ticker (e.g., SPY). If not specified, loads all tickers from config file."
    )
    parser.add_argument(
        "--tickers",
        nargs='+',
        help="Quick verify multiple tickers (e.g., CORN CANE COW). Use with --quick flag."
    )
    parser.add_argument(
        "--group",
        help="Verify tickers from a named group (e.g., inflation, commodities, energy). Use with --quick flag."
    )
    parser.add_argument(
        "--config",
        default='configs/tickers.yaml',
        help='Path to tickers YAML config file (default: configs/tickers.yaml)'
    )
    parser.add_argument(
        "--frequency",
        type=str,
        help="Filter by frequency (e.g., 15m, 1h, daily)"
    )
    parser.add_argument(
        "--indicator",
        type=str,
        help="Analyze specific indicator (e.g., sma_10, volatility_20, returns). Requires --ticker and --frequency."
    )
    parser.add_argument(
        "--table",
        type=str,
        choices=['main', 'staging'],
        default='main',
        help="Which table to verify (default: main)"
    )
    parser.add_argument(
        "--skip-gaps",
        action="store_true",
        help="Skip date gaps analysis (can be slow)"
    )
    parser.add_argument(
        "--exclude-weekends-and-holidays",
        action="store_true",
        help="Exclude weekend and holiday gaps from analysis (recommended for stock market data)"
    )
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Only show sample data"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: lightweight verification with summary table (use with --tickers)"
    )
    
    args = parser.parse_args()
    
    # Validate environment
    if not PROJECT_ID:
        print("❌ Error: GCP_PROJECT_ID not set in environment")
        sys.exit(1)
    
    # Quick mode validation
    if args.quick:
        if not args.tickers and not args.group:
            print("❌ Error: --quick requires --tickers or --group")
            sys.exit(1)
        if not args.frequency:
            print("❌ Error: --quick requires --frequency")
            sys.exit(1)
    
    # Load config (always, for indicators list)
    config = load_config(args.config)
    
    # Get tickers - either from args or config
    tickers_to_verify = []
    if args.group:
        # Load tickers from group
        if not config:
            print("❌ Error: Cannot load groups without config file")
            sys.exit(1)
        tickers_to_verify = get_group_tickers(config, args.group)
        if not tickers_to_verify:
            sys.exit(1)
        print(f"📋 Group '{args.group}': {len(tickers_to_verify)} tickers - {', '.join(tickers_to_verify[:10])}{'...' if len(tickers_to_verify) > 10 else ''}")
    elif args.tickers:
        # Multiple tickers for quick mode
        tickers_to_verify = args.tickers
        print(f"📋 Verifying {len(tickers_to_verify)} tickers: {', '.join(tickers_to_verify)}")
    elif args.ticker:
        tickers_to_verify = [args.ticker]
        print(f"📋 Verifying ticker: {args.ticker}")
    else:
        # Load tickers from config
        if config and 'tickers' in config:
            tickers_to_verify = config['tickers']
            print(f"📋 Loaded {len(tickers_to_verify)} tickers from {args.config}")
            print(f"   Tickers: {', '.join(tickers_to_verify[:10])}{'...' if len(tickers_to_verify) > 10 else ''}")
        else:
            print(f"⚠️  No ticker specified and no tickers found in config")
            print(f"   Will verify all data in table")
    
    print_header("Synthetic Indicators Verification")
    print(f"Project: {PROJECT_ID}")
    print(f"Dataset: {DATASET_ID}")
    print(f"Table: {SYNTHETIC_TABLE if args.table == 'main' else SYNTHETIC_STAGING}")
    if args.frequency:
        print(f"Frequency filter: {args.frequency}")
    print()
    
    # Initialize BigQuery client
    try:
        client = bigquery.Client(project=PROJECT_ID)
        print("✅ Connected to BigQuery\n")
    except Exception as e:
        print(f"❌ Failed to connect: {str(e)}")
        sys.exit(1)
    
    # Construct table ID
    table_name = SYNTHETIC_TABLE if args.table == 'main' else SYNTHETIC_STAGING
    table_id = f"{PROJECT_ID}.{DATASET_ID}.{table_name}"
    
    # Run checks
    try:
        # Quick mode - lightweight multi-ticker verification
        if args.quick:
            quick_verify_tickers(client, table_id, tickers_to_verify, args.frequency, args.exclude_weekends_and_holidays)
        # Indicator-specific analysis mode
        elif args.indicator:
            if not args.ticker or not args.frequency:
                print("❌ Error: --indicator requires both --ticker and --frequency")
                sys.exit(1)
            analyze_specific_indicator(client, table_id, args.ticker, args.frequency, args.indicator)
        elif args.sample_only:
            ticker_filter = args.ticker if args.ticker else None
            sample_data(client, table_id, ticker_filter, args.frequency, limit=20)
        else:
            # If we have multiple tickers, verify each one
            if tickers_to_verify:
                for idx, ticker in enumerate(tickers_to_verify, 1):
                    print(f"\n{'='*80}")
                    print(f"   [{idx}/{len(tickers_to_verify)}] Verifying {ticker}")
                    print(f"{'='*80}\n")
                    
                    # Coverage analysis
                    analyze_data_coverage(client, table_id, ticker, args.frequency)
                
                    # Indicator coverage (use config indicators if available)
                    config_indicators = config.get('indicators') if config else None
                    analyze_indicator_coverage(client, table_id, ticker, args.frequency, config_indicators)
                    
                    # Data quality
                    analyze_data_quality(client, table_id, ticker, args.frequency)
                    
                    # Duplicate check
                    check_duplicates(client, table_id, ticker)
                    
                    # Gap analysis (if requested and frequency specified)
                    if not args.skip_gaps and args.frequency:
                        find_date_gaps(client, table_id, ticker, args.frequency, exclude_weekends=args.exclude_weekends)
                    
                    # Sample data
                    sample_data(client, table_id, ticker, args.frequency)
            else:
                # No ticker filter - verify all data
                analyze_data_coverage(client, table_id, None, args.frequency)
                analyze_indicator_coverage(client, table_id, None, args.frequency)
                analyze_data_quality(client, table_id, None, args.frequency)
                check_duplicates(client, table_id, None)
                sample_data(client, table_id, None, args.frequency)
        
        print_header("Verification Complete")
        print("✅ All checks finished")
        print()
        
    except Exception as e:
        print(f"\n❌ Error during verification: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()