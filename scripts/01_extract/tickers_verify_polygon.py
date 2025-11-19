"""Verify Polygon data quality in BigQuery.

Analyzes raw_ohlcv table for completeness, gaps, duplicates, and data quality issues.
"""
import os
import sys
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
TABLE_NAME = os.environ.get('BQ_TABLE', 'raw_ohlcv')
STAGING_TABLE_NAME = os.environ.get('BQ_STAGING_TABLE', 'raw_ohlcv_staging')
INDICATORS_TABLE = os.environ.get('BQ_INDICATORS_TABLE', 'technical_indicators')


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


def weekday_breakdown(client: bigquery.Client, table_id: str, ticker: str = None, frequency: str = None):
    """Analyze data distribution by day of week."""
    print_header("Weekday Breakdown")
    
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
        EXTRACT(DAYOFWEEK FROM date) as day_of_week,
        FORMAT_DATE('%A', date) as day_name,
        COUNT(*) as bar_count,
        COUNT(DISTINCT date) as unique_dates
    FROM `{table_id}`
    {where_sql}
    GROUP BY ticker, frequency, day_of_week, day_name
    ORDER BY ticker, frequency, day_of_week
    """
    
    df = client.query(query).to_dataframe()
    
    if df.empty:
        print("❌ No data found")
        return
    
    # Group by ticker and frequency
    for (ticker_name, freq), group in df.groupby(['ticker', 'frequency']):
        print(f"\n📅 {ticker_name} ({freq}):\n")
        total_bars = group['bar_count'].sum()
        
        for _, row in group.iterrows():
            day_name = row['day_name']
            bar_count = row['bar_count']
            unique_dates = row['unique_dates']
            pct = (bar_count / total_bars) * 100
            
            # Show visual bar
            bar_length = int(pct / 2)  # Scale to 50 chars max
            bar = '█' * bar_length
            
            print(f"  {day_name:9s}: {bar_count:6,} bars ({pct:5.1f}%) | {unique_dates:3} days | {bar}")
        
        print(f"\n  Total: {total_bars:,} bars")
    print()


def analyze_indicator_coverage(client: bigquery.Client, table_id: str, ticker: str = None, frequency: str = None):
    """Analyze technical indicator coverage."""
    print_header("Technical Indicator Coverage")
    
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
        COUNTIF(sma_10 IS NOT NULL) as sma_10_count,
        COUNTIF(sma_20 IS NOT NULL) as sma_20_count,
        COUNTIF(sma_50 IS NOT NULL) as sma_50_count,
        COUNTIF(sma_200 IS NOT NULL) as sma_200_count,
        COUNTIF(ema_10 IS NOT NULL) as ema_10_count,
        COUNTIF(ema_20 IS NOT NULL) as ema_20_count,
        COUNTIF(ema_50 IS NOT NULL) as ema_50_count,
        COUNTIF(ema_200 IS NOT NULL) as ema_200_count,
        COUNTIF(rsi_14 IS NOT NULL) as rsi_14_count,
        COUNTIF(macd_value IS NOT NULL) as macd_count
    FROM `{table_id}`
    {where_sql}
    GROUP BY ticker, frequency
    ORDER BY ticker, frequency
    """
    
    df = client.query(query).to_dataframe()
    
    if df.empty:
        print("❌ No data found!")
        return
    
    print("📊 Indicator Coverage Summary:\n")
    for _, row in df.iterrows():
        print(f"Ticker: {row['ticker']} | Frequency: {row['frequency']}")
        print(f"  Total rows: {row['total_rows']:,}")
        print(f"  Date range: {row['earliest_date']} to {row['latest_date']}")
        print()
        
        total = row['total_rows']
        indicators = [
            ('SMA-10', row['sma_10_count']),
            ('SMA-20', row['sma_20_count']),
            ('SMA-50', row['sma_50_count']),
            ('SMA-200', row['sma_200_count']),
            ('EMA-10', row['ema_10_count']),
            ('EMA-20', row['ema_20_count']),
            ('EMA-50', row['ema_50_count']),
            ('EMA-200', row['ema_200_count']),
            ('RSI-14', row['rsi_14_count']),
            ('MACD', row['macd_count']),
        ]
        
        print("  Indicator Coverage:")
        for ind_name, count in indicators:
            coverage = (count / total * 100) if total > 0 else 0
            status = "✅" if coverage >= 99 else "⚠️ " if coverage >= 80 else "❌"
            print(f"    {status} {ind_name:10s}: {count:6,} / {total:6,} ({coverage:5.1f}%)")
        print()


def sample_data(client: bigquery.Client, table_id: str, ticker: str, frequency: str, limit: int = 10, data_type: str = 'raw'):
    """Show sample data - first 10 (oldest) and last 10 (newest) entries."""
    print_header(f"Sample Data - {ticker} ({frequency}) [{data_type.upper()}]")
    
    # Define columns based on data type
    if data_type == 'raw':
        columns = "timestamp, date, open, high, low, close, volume"
    else:  # indicators
        columns = "timestamp, date, sma_10, sma_20, sma_50, ema_10, ema_20, rsi_14, macd_value, macd_signal"
    
    # Get first 10 (oldest)
    query_first = f"""
    SELECT {columns}
    FROM `{table_id}`
    WHERE ticker = '{ticker}' AND frequency = '{frequency}'
    ORDER BY timestamp ASC
    LIMIT {limit}
    """
    
    # Get last 10 (newest)
    query_last = f"""
    SELECT {columns}
    FROM `{table_id}`
    WHERE ticker = '{ticker}' AND frequency = '{frequency}'
    ORDER BY timestamp DESC
    LIMIT {limit}
    """
    
    df_first = client.query(query_first).to_dataframe()
    df_last = client.query(query_last).to_dataframe()
    
    if df_first.empty and df_last.empty:
        print("❌ No data found")
        return
    
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


def main():
    """Main verification function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Verify Polygon data in BigQuery',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Verify all tickers from config
  python scripts/01_extract/tickers_verify_polygon.py --frequency daily
  
  # Verify specific ticker
  python scripts/01_extract/tickers_verify_polygon.py --ticker SPY --frequency daily
  
  # Check for gaps (excluding weekends/holidays)
  python scripts/01_extract/tickers_verify_polygon.py --ticker SPY --frequency daily --check-gaps --exclude-weekends
  
  # Verify all tickers with gap checking
  python scripts/01_extract/tickers_verify_polygon.py --frequency daily --check-gaps --exclude-weekends
        """
    )
    
    parser.add_argument('--ticker', help='Filter by specific ticker (e.g., SPY). If not specified, verifies all tickers from config.')
    parser.add_argument('--frequency', help='Filter by frequency (e.g., daily, hourly, 15m)')
    parser.add_argument('--config', default='configs/tickers.yaml', help='Path to tickers config file (default: configs/tickers.yaml)')
    parser.add_argument('--data-type', choices=['raw', 'indicators', 'both'], default='raw', 
                        help='Type of data to verify: raw (OHLCV), indicators, or both (default: raw)')
    parser.add_argument('--check-gaps', action='store_true', help='Check for date/time gaps (raw data only)')
    parser.add_argument('--min-gap-hours', type=int, default=2, help='Minimum gap size to report in hours (default: 2)')
    parser.add_argument('--exclude-weekends', action='store_true', help='Exclude weekend gaps from gap analysis (recommended for stock market data)')
    
    args = parser.parse_args()
    
    if not PROJECT_ID:
        print("❌ Error: GCP_PROJECT_ID not set in environment")
        sys.exit(1)
    
    print_header("Polygon Data Verification")
    print(f"Project: {PROJECT_ID}")
    print(f"Dataset: {DATASET_ID}")
    print(f"Table: {TABLE_NAME}")
    
    # Load tickers from config if no specific ticker provided
    tickers_to_verify = []
    if args.ticker:
        tickers_to_verify = [args.ticker]
        print(f"Ticker: {args.ticker}")
    else:
        config = load_config(args.config)
        if config and 'tickers' in config:
            tickers_to_verify = config['tickers']
            print(f"Verifying all tickers from config: {len(tickers_to_verify)} tickers")
            print(f"  Tickers: {', '.join(tickers_to_verify[:10])}{'...' if len(tickers_to_verify) > 10 else ''}")
        else:
            print("⚠️  No tickers specified and config file not found or empty")
            print("   Use --ticker to specify a ticker or ensure config file exists")
    
    if args.frequency:
        print(f"Frequency filter: {args.frequency}")
    print()
    
    try:
        client = bigquery.Client(project=PROJECT_ID)
    except Exception as e:
        print(f"❌ Failed to connect to BigQuery: {str(e)}")
        sys.exit(1)
    
    # Run analyses based on data type
    if args.data_type in ['raw', 'both']:
        table_id = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_NAME}"
        print(f"\n{'='*80}")
        print("   RAW OHLCV DATA VERIFICATION")
        print(f"{'='*80}\n")
        
        # If verifying multiple tickers, run checks for each
        if len(tickers_to_verify) > 1:
            for idx, ticker in enumerate(tickers_to_verify, 1):
                print(f"\n[{idx}/{len(tickers_to_verify)}] Verifying {ticker}...")
                print("-" * 80)
                
                analyze_data_coverage(client, table_id, ticker, args.frequency)
                check_duplicates(client, table_id, ticker)
                
                if args.check_gaps and args.frequency:
                    find_date_gaps(client, table_id, ticker, args.frequency, args.min_gap_hours, args.exclude_weekends)
        else:
            # Single ticker verification
            ticker = tickers_to_verify[0] if tickers_to_verify else args.ticker
            
            analyze_data_coverage(client, table_id, ticker, args.frequency)
            check_duplicates(client, table_id, ticker)
            weekday_breakdown(client, table_id, ticker, args.frequency)
            
            if args.check_gaps and ticker and args.frequency:
                find_date_gaps(client, table_id, ticker, args.frequency, args.min_gap_hours, args.exclude_weekends)
            
            if ticker and args.frequency:
                sample_data(client, table_id, ticker, args.frequency, data_type='raw')
    
    if args.data_type in ['indicators', 'both']:
        indicators_table_id = f"{PROJECT_ID}.{DATASET_ID}.{INDICATORS_TABLE}"
        print(f"\n{'='*80}")
        print("   TECHNICAL INDICATORS VERIFICATION")
        print(f"{'='*80}\n")
        
        # If verifying multiple tickers, run checks for each
        if len(tickers_to_verify) > 1:
            for idx, ticker in enumerate(tickers_to_verify, 1):
                print(f"\n[{idx}/{len(tickers_to_verify)}] Verifying {ticker} indicators...")
                print("-" * 80)
                
                analyze_data_coverage(client, indicators_table_id, ticker, args.frequency)
                check_duplicates(client, indicators_table_id, ticker)
                analyze_indicator_coverage(client, indicators_table_id, ticker, args.frequency)
        else:
            # Single ticker verification
            ticker = tickers_to_verify[0] if tickers_to_verify else args.ticker
            
            analyze_data_coverage(client, indicators_table_id, ticker, args.frequency)
            check_duplicates(client, indicators_table_id, ticker)
            analyze_indicator_coverage(client, indicators_table_id, ticker, args.frequency)
            weekday_breakdown(client, indicators_table_id, ticker, args.frequency)
            
            if ticker and args.frequency:
                sample_data(client, indicators_table_id, ticker, args.frequency, data_type='indicators')
    
    print_header("Verification Complete")


if __name__ == '__main__':
    main()