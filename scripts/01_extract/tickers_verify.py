"""Master verification orchestrator that delegates to specialized verification scripts.

This script coordinates verification by delegating to:
- tickers_verify_synthetic.py: Verifies synthetic indicators, alignment, and data quality

For code reuse, all verification logic is in the specialized scripts.
"""
import os
import sys
import argparse
import subprocess
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
from google.cloud import bigquery
from pathlib import Path

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


def print_header(text: str):
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


def check_alignment(client: bigquery.Client, ticker: str, frequency: str, 
                     start_date: str = None, end_date: str = None) -> Dict:
    """Check timestamp alignment between raw and synthetic tables."""
    print_header("Timestamp Alignment Check")
    
    date_filter = ""
    if start_date and end_date:
        date_filter = f"AND DATE(timestamp) BETWEEN '{start_date}' AND '{end_date}'"
    elif start_date:
        date_filter = f"AND DATE(timestamp) >= '{start_date}'"
    elif end_date:
        date_filter = f"AND DATE(timestamp) <= '{end_date}'"
    
    query = f"""
    WITH raw_timestamps AS (
      SELECT ticker, frequency, timestamp
      FROM `{PROJECT_ID}.{DATASET_ID}.{RAW_TABLE}`
      WHERE ticker = '{ticker}' AND frequency = '{frequency}' {date_filter}
    ),
    synthetic_timestamps AS (
      SELECT ticker, frequency, timestamp
      FROM `{PROJECT_ID}.{DATASET_ID}.{SYNTHETIC_TABLE}`
      WHERE ticker = '{ticker}' AND frequency = '{frequency}' {date_filter}
    )
    SELECT
      COUNTIF(r.timestamp IS NOT NULL AND s.timestamp IS NOT NULL) as aligned,
      COUNTIF(r.timestamp IS NOT NULL AND s.timestamp IS NULL) as raw_only,
      COUNTIF(r.timestamp IS NULL AND s.timestamp IS NOT NULL) as synthetic_only
    FROM raw_timestamps r
    FULL OUTER JOIN synthetic_timestamps s
      ON r.timestamp = s.timestamp
    """
    
    result = client.query(query).to_dataframe().iloc[0]
    
    print(f"Ticker: {ticker} | Frequency: {frequency}")
    if start_date or end_date:
        print(f"Date range: {start_date or 'beginning'} to {end_date or 'end'}")
    print()
    print(f"✅ Aligned timestamps:        {result['aligned']:,}")
    print(f"📦 Raw only (missing synthetic): {result['raw_only']:,}")
    print(f"📈 Synthetic only (missing raw): {result['synthetic_only']:,}")
    
    # Only fail if raw data has timestamps missing from synthetic (data loss)
    if result['raw_only'] > 0:
        print(f"\n❌ ALIGNMENT ERROR: Raw data has timestamps missing from synthetic!")
        print(f"   This indicates synthetic indicators were not computed for all raw data.")
        print(f"   Run with --show-misalignment for details.")
        return {'aligned': False, 'stats': result}
    elif result['synthetic_only'] > 0:
        print(f"ℹ️  Note: Synthetic has {result['synthetic_only']:,} extra timestamps")
        return {'aligned': True, 'stats': result}
    else:
        print(f"\n✅ Perfect alignment! All timestamps match.")
        return {'aligned': True, 'stats': result}


def show_alignment_details(client: bigquery.Client, ticker: str, frequency: str,
                           start_date: str = None, end_date: str = None):
    """Show detailed misalignment information."""
    print_header("Alignment Mismatch Details")
    
    date_filter = ""
    if start_date and end_date:
        date_filter = f"AND DATE(timestamp) BETWEEN '{start_date}' AND '{end_date}'"
    elif start_date:
        date_filter = f"AND DATE(timestamp) >= '{start_date}'"
    elif end_date:
        date_filter = f"AND DATE(timestamp) <= '{end_date}'"
    
    # Raw only
    raw_only_query = f"""
    WITH raw_timestamps AS (
      SELECT timestamp, date
      FROM `{PROJECT_ID}.{DATASET_ID}.{RAW_TABLE}`
      WHERE ticker = '{ticker}' AND frequency = '{frequency}' {date_filter}
    ),
    synthetic_timestamps AS (
      SELECT timestamp
      FROM `{PROJECT_ID}.{DATASET_ID}.{SYNTHETIC_TABLE}`
      WHERE ticker = '{ticker}' AND frequency = '{frequency}' {date_filter}
    )
    SELECT r.timestamp, r.date
    FROM raw_timestamps r
    LEFT JOIN synthetic_timestamps s ON r.timestamp = s.timestamp
    WHERE s.timestamp IS NULL
    ORDER BY r.timestamp
    LIMIT 10
    """
    
    raw_only = client.query(raw_only_query).to_dataframe()
    if not raw_only.empty:
        print(f"📦 RAW ONLY (first 10 timestamps in raw but missing in synthetic):\n")
        for _, row in raw_only.iterrows():
            print(f"   {row['timestamp']} ({row['date']})")
        print()


def fetch_raw_data(client: bigquery.Client, ticker: str, frequency: str,
                   start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch raw OHLCV data from BigQuery."""
    print_header("Fetching Raw OHLCV Data")
    
    query = f"""
    SELECT 
        timestamp,
        open,
        high,
        low,
        close,
        volume
    FROM `{PROJECT_ID}.{DATASET_ID}.{RAW_TABLE}`
    WHERE ticker = '{ticker}'
      AND frequency = '{frequency}'
      AND DATE(timestamp) BETWEEN '{start_date}' AND '{end_date}'
    ORDER BY timestamp
    """
    
    print(f"Fetching {ticker} ({frequency}) from {start_date} to {end_date}...")
    df = client.query(query).to_dataframe()
    print(f"✅ Fetched {len(df):,} rows\n")
    
    return df


def compute_indicators_local(df: pd.DataFrame, indicators: List[str]) -> pd.DataFrame:
    """Compute synthetic indicators locally from OHLCV data."""
    print_header("Computing Indicators Locally")
    
    df = df.copy()
    computed = []
    
    # Returns
    if 'returns' in indicators:
        df['returns'] = df['close'].pct_change()
        computed.append('returns')
    
    if 'returns_5' in indicators:
        df['returns_5'] = df['close'].pct_change(periods=5)
        computed.append('returns_5')
    
    if 'log_returns' in indicators:
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        computed.append('log_returns')
    
    # Simple Moving Averages
    for period in [10, 20, 50, 200]:
        ind_name = f'sma_{period}'
        if ind_name in indicators:
            df[ind_name] = df['close'].rolling(window=period, min_periods=period).mean()
            computed.append(ind_name)
    
    # Exponential Moving Averages
    for period in [12, 26]:
        ind_name = f'ema_{period}'
        if ind_name in indicators:
            df[ind_name] = df['close'].ewm(span=period, adjust=False).mean()
            computed.append(ind_name)
    
    # MACD (Moving Average Convergence Divergence)
    if any(ind in indicators for ind in ['macd', 'macd_signal', 'macd_hist']):
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        
        if 'macd' in indicators:
            df['macd'] = macd
            computed.append('macd')
        if 'macd_signal' in indicators:
            df['macd_signal'] = macd.ewm(span=9, adjust=False).mean()
            computed.append('macd_signal')
        if 'macd_hist' in indicators:
            df['macd_hist'] = macd - macd.ewm(span=9, adjust=False).mean()
            computed.append('macd_hist')
    
    # Volatility (rolling std of returns)
    for period in [5, 10, 20]:
        ind_name = f'volatility_{period}'
        if ind_name in indicators:
            returns = df['close'].pct_change()
            df[ind_name] = returns.rolling(window=period, min_periods=period).std()
            computed.append(ind_name)
    
    # Bollinger Bands
    if any(ind in indicators for ind in ['bb_upper_20', 'bb_middle_20', 'bb_lower_20', 'bb_width']):
        sma_20 = df['close'].rolling(window=20, min_periods=20).mean()
        std_20 = df['close'].rolling(window=20, min_periods=20).std()
        
        if 'bb_upper_20' in indicators:
            df['bb_upper_20'] = sma_20 + (2 * std_20)
            computed.append('bb_upper_20')
        if 'bb_middle_20' in indicators:
            df['bb_middle_20'] = sma_20
            computed.append('bb_middle_20')
        if 'bb_lower_20' in indicators:
            df['bb_lower_20'] = sma_20 - (2 * std_20)
            computed.append('bb_lower_20')
        if 'bb_width' in indicators:
            df['bb_width'] = ((sma_20 + (2 * std_20)) - (sma_20 - (2 * std_20))) / sma_20
            computed.append('bb_width')
    
    # ATR (Average True Range)
    if 'atr_14' in indicators:
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = true_range.rolling(window=14, min_periods=14).mean()
        computed.append('atr_14')
    
    # Volume Moving Averages
    for period in [5, 10]:
        ind_name = f'volume_ma_{period}'
        if ind_name in indicators:
            df[ind_name] = df['volume'].rolling(window=period, min_periods=period).mean()
            computed.append(ind_name)
    
    # Momentum
    for period in [5, 10]:
        ind_name = f'momentum_{period}'
        if ind_name in indicators:
            df[ind_name] = df['close'] - df['close'].shift(period)
            computed.append(ind_name)
    
    # Rate of Change (ROC)
    for period in [5, 10]:
        ind_name = f'roc_{period}'
        if ind_name in indicators:
            df[ind_name] = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100
            computed.append(ind_name)
    
    # Simple ratios
    if 'volume_ratio' in indicators:
        volume_ma_10 = df['volume'].rolling(window=10, min_periods=10).mean()
        df['volume_ratio'] = df['volume'] / volume_ma_10
        computed.append('volume_ratio')
    
    if 'high_low_ratio' in indicators:
        df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        computed.append('high_low_ratio')
    
    print(f"✅ Computed {len(computed)} indicators: {', '.join(computed)}\n")
    
    return df


def fetch_synthetic_data(client: bigquery.Client, ticker: str, frequency: str,
                         start_date: str, end_date: str, indicators: List[str]) -> pd.DataFrame:
    """Fetch synthetic indicators from BigQuery."""
    print_header("Fetching Synthetic Indicators from BigQuery")
    
    indicator_cols = ', '.join(indicators)
    
    query = f"""
    SELECT 
        timestamp,
        {indicator_cols}
    FROM `{PROJECT_ID}.{DATASET_ID}.{SYNTHETIC_TABLE}`
    WHERE ticker = '{ticker}'
      AND frequency = '{frequency}'
      AND DATE(timestamp) BETWEEN '{start_date}' AND '{end_date}'
    ORDER BY timestamp
    """
    
    print(f"Fetching stored indicators for {ticker} ({frequency})...")
    df = client.query(query).to_dataframe()
    print(f"✅ Fetched {len(df):,} rows\n")
    
    return df


def compare_indicators(computed_df: pd.DataFrame, stored_df: pd.DataFrame, 
                       indicators: List[str], tolerance: float = 0.01) -> Dict:
    """Compare computed indicators with stored values."""
    print_header("Comparing Computed vs Stored Indicators")
    
    # Merge on timestamp
    merged = computed_df[['timestamp'] + indicators].merge(
        stored_df[['timestamp'] + indicators],
        on='timestamp',
        suffixes=('_computed', '_stored'),
        how='inner'
    )
    
    print(f"Comparing {len(merged):,} timestamps across {len(indicators)} indicators...\n")
    
    results = {}
    all_match = True
    
    for indicator in indicators:
        computed_col = f"{indicator}_computed"
        stored_col = f"{indicator}_stored"
        
        # Filter rows where both have non-null values
        valid_rows = merged[[computed_col, stored_col]].dropna()
        
        if len(valid_rows) == 0:
            print(f"⚠️  {indicator:20s}: No valid data to compare")
            results[indicator] = {'status': 'no_data', 'mismatches': 0}
            continue
        
        # Calculate relative difference
        diff = (valid_rows[computed_col] - valid_rows[stored_col]).abs()
        rel_diff = diff / valid_rows[stored_col].abs()
        
        # Count mismatches beyond tolerance
        mismatches = (rel_diff > tolerance).sum()
        mismatch_pct = (mismatches / len(valid_rows)) * 100
        
        max_diff = diff.max()
        max_rel_diff = rel_diff.max()
        
        if mismatches == 0:
            print(f"✅ {indicator:20s}: Perfect match ({len(valid_rows):,} values)")
            results[indicator] = {'status': 'match', 'mismatches': 0, 'total': len(valid_rows)}
        else:
            print(f"❌ {indicator:20s}: {mismatches:,} / {len(valid_rows):,} mismatches ({mismatch_pct:.1f}%)")
            print(f"   Max absolute diff: {max_diff:.6f}")
            print(f"   Max relative diff: {max_rel_diff:.2%}")
            results[indicator] = {
                'status': 'mismatch',
                'mismatches': mismatches,
                'total': len(valid_rows),
                'max_abs_diff': max_diff,
                'max_rel_diff': max_rel_diff
            }
            all_match = False
    
    print()
    
    # Show sample validation results
    print(f"\n🔍 Sample Validation Results (first 3 timestamps with all indicators):\n")
    sample_merged = merged[merged[list(merged.columns)[1:]].notna().all(axis=1)].head(3)
    
    if len(sample_merged) > 0:
        for idx, row in sample_merged.iterrows():
            print(f"  Timestamp: {row['timestamp']}")
            for ind in indicators:
                computed_col = f"{ind}_computed"
                stored_col = f"{ind}_stored"
                
                if computed_col in row and stored_col in row:
                    computed_val = row[computed_col]
                    stored_val = row[stored_col]
                    diff = abs(computed_val - stored_val)
                    rel_diff = diff / abs(stored_val) if stored_val != 0 else 0
                    
                    match_status = "✅" if rel_diff <= tolerance else "❌"
                    print(f"    {match_status} {ind:15s}: stored={stored_val:>10.4f}, computed={computed_val:>10.4f}, diff={rel_diff:>8.2%}")
            print()
    
    if all_match:
        print("✅ All indicators match perfectly!")
    else:
        print("⚠️  Some indicators have mismatches. Use --show-samples to see examples.")
    
    return results


def show_mismatch_samples(computed_df: pd.DataFrame, stored_df: pd.DataFrame,
                          indicators: List[str], tolerance: float = 0.01, n_samples: int = 5):
    """Show sample mismatches for debugging."""
    print_header("Sample Mismatches")
    
    merged = computed_df[['timestamp'] + indicators].merge(
        stored_df[['timestamp'] + indicators],
        on='timestamp',
        suffixes=('_computed', '_stored'),
        how='inner'
    )
    
    for indicator in indicators:
        computed_col = f"{indicator}_computed"
        stored_col = f"{indicator}_stored"
        
        valid_rows = merged[['timestamp', computed_col, stored_col]].dropna()
        
        if len(valid_rows) == 0:
            continue
        
        # Calculate differences
        valid_rows = valid_rows.copy()
        valid_rows['abs_diff'] = (valid_rows[computed_col] - valid_rows[stored_col]).abs()
        valid_rows['rel_diff'] = valid_rows['abs_diff'] / valid_rows[stored_col].abs()
        
        mismatches = valid_rows[valid_rows['rel_diff'] > tolerance]
        
        if len(mismatches) > 0:
            print(f"\n{indicator} - Top {min(n_samples, len(mismatches))} mismatches:\n")
            top_mismatches = mismatches.nlargest(n_samples, 'rel_diff')
            
            for _, row in top_mismatches.iterrows():
                print(f"  Timestamp: {row['timestamp']}")
                print(f"    Computed: {row[computed_col]:.6f}")
                print(f"    Stored:   {row[stored_col]:.6f}")
                print(f"    Abs diff: {row['abs_diff']:.6f}")
                print(f"    Rel diff: {row['rel_diff']:.2%}")
                print()


def export_combined_data(raw_df: pd.DataFrame, stored_df: pd.DataFrame, 
                         computed_df: pd.DataFrame, indicators: List[str],
                         output_file: str, ticker: str, frequency: str):
    """Export combined raw OHLCV and indicator data to Parquet file."""
    print_header("Exporting Combined Data")
    
    # Start with raw data
    combined = raw_df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    
    # Add stored indicators (clean names, no suffix)
    # Use stored values since they've been verified against computed
    stored_cols = stored_df[['timestamp'] + indicators].copy()
    combined = combined.merge(stored_cols, on='timestamp', how='left')
    
    # Add metadata columns
    combined['ticker'] = ticker
    combined['frequency'] = frequency
    combined['date'] = combined['timestamp'].dt.date
    
    # Reorder columns: metadata, OHLCV, then indicators
    metadata_cols = ['ticker', 'frequency', 'date', 'timestamp']
    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
    
    # All columns in clean order
    all_cols = metadata_cols + ohlcv_cols + indicators
    combined = combined[all_cols]
    
    # Show sample data (last 3 rows to ensure indicators are present)
    print(f"📝 Sample Exported Data (last 3 rows):\n")
    
    if len(combined) > 0:
        for idx, row in combined.tail(3).iterrows():
            print(f"  Timestamp: {row['timestamp']}")
            print(f"    OHLCV: O={row['open']:.2f}, H={row['high']:.2f}, L={row['low']:.2f}, C={row['close']:.2f}, V={row['volume']:.0f}")
            for ind in indicators:
                if ind in row:
                    val = row[ind]
                    if pd.notna(val):
                        print(f"    {ind:20s}: {val:>10.4f}")
                    else:
                        print(f"    {ind:20s}: NULL")
            print()
    
    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to Parquet
    combined.to_parquet(output_file, index=False, engine='pyarrow', compression='snappy')
    parquet_size = os.path.getsize(output_file) / 1024 / 1024
    
    # Also save as CSV
    csv_file = output_file.replace('.parquet', '.csv')
    combined.to_csv(csv_file, index=False)
    csv_size = os.path.getsize(csv_file) / 1024 / 1024
    
    print(f"✅ Exported {len(combined):,} rows to:")
    print(f"   Parquet: {output_file} ({parquet_size:.2f} MB)")
    print(f"   CSV:     {csv_file} ({csv_size:.2f} MB)")
    print(f"\n📊 Data Summary:")
    print(f"   Columns: {len(combined.columns)}")
    print(f"   Date range: {combined['date'].min()} to {combined['date'].max()}")
    print(f"\n💡 Load with:")
    print(f"   df = pd.read_parquet('{output_file}')")
    print(f"   df = pd.read_csv('{csv_file}')")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Verify synthetic indicator calculations by comparing with local computations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify all SMA indicators for SPY
  python tickers_verify.py --ticker SPY --frequency daily \\
    --start 2024-01-01 --end 2024-01-31 \\
    --indicators sma_50 sma_200
  
  # Check alignment only
  python tickers_verify.py --ticker SPY --frequency daily \\
    --start 2024-01-01 --end 2024-01-31 \\
    --check-alignment-only
  
  # Verify with sample mismatches
  python tickers_verify.py --ticker SPY --frequency daily \
    --start 2024-01-01 --end 2024-01-31 \
    --indicators sma_50 volatility_20 returns returns_5 \
    --show-samples
  
  # Export combined data to Parquet for offline analysis
  python tickers_verify.py --ticker SPY --frequency daily \
    --start 2024-01-01 --end 2024-01-31 \
    --indicators sma_50 sma_200 returns returns_5 log_returns \
    --export --output-file tickers_verification.parquet
        """
    )
    
    parser.add_argument('--ticker', required=True, help='Ticker symbol (e.g., SPY)')
    parser.add_argument('--frequency', required=True, help='Data frequency (e.g., 15m, 1h, daily)')
    parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--indicators', nargs='+', help='Indicators to verify (defaults to config file indicators)')
    parser.add_argument('--config', type=str, default='configs/tickers.yaml',
                       help='Path to config file (default: configs/tickers.yaml)')
    parser.add_argument('--tolerance', type=float, default=0.01, help='Relative difference tolerance (default: 0.01 = 1%%)')
    parser.add_argument('--check-alignment-only', action='store_true', help='Only check alignment, skip verification')
    parser.add_argument('--show-misalignment', action='store_true', help='Show detailed misalignment information')
    parser.add_argument('--show-samples', action='store_true', help='Show sample mismatches for debugging')
    parser.add_argument('--skip-alignment-check', action='store_true', help='Skip alignment check and proceed directly to verification')
    parser.add_argument('--export', action='store_true', help='Export combined data to Parquet file')
    parser.add_argument('--output-file', type=str, default='temp/tickers_verification_output.parquet',
                       help='Output file path for export (default: temp/tickers_verification_output.parquet)')
    parser.add_argument('--exclude-weekends', action='store_true',
                       help='Exclude weekend gaps from analysis (recommended for stock market data)')

    args = parser.parse_args()

    if not PROJECT_ID:
        print("❌ Error: GCP_PROJECT_ID environment variable not set")
        print("\nSet it with:")
        print("  export GCP_PROJECT_ID='your-project-id'")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("   Synthetic Indicators Verification")
    print("=" * 80)
    print(f"\nProject: {PROJECT_ID}")
    print(f"Dataset: {DATASET_ID}")
    print(f"Ticker: {args.ticker}")
    print(f"Frequency: {args.frequency}")
    print(f"Date range: {args.start} to {args.end}")
    if args.indicators:
        print(f"Indicators: {', '.join(args.indicators)}")
    print()
    
    # Initialize BigQuery client
    client = bigquery.Client(project=PROJECT_ID)
    
    # Check alignment
    if not args.skip_alignment_check:
        alignment_result = check_alignment(client, args.ticker, args.frequency, args.start, args.end)
        
        if args.show_misalignment and not alignment_result['aligned']:
            show_alignment_details(client, args.ticker, args.frequency, args.start, args.end)
        
        if not alignment_result['aligned']:
            print("\n❌ ERROR: Raw data has timestamps missing from synthetic indicators!")
            print("   This means synthetic indicators were not computed for all raw data.")
            print("   You must fix the synthetic indicators before verification.\n")
            sys.exit(1)
    
    if args.check_alignment_only:
        print("\n✅ Alignment check complete.")
        sys.exit(0)
    
    # Load indicators from config if not specified
    if not args.indicators:
        config = load_config(args.config)
        if config and 'indicators' in config:
            args.indicators = config['indicators']
            print(f"📋 Loaded {len(args.indicators)} indicators from config: {', '.join(args.indicators[:5])}{'...' if len(args.indicators) > 5 else ''}\n")
        else:
            print("\n❌ Error: --indicators is required for verification")
            print("   Either specify --indicators or ensure config file has 'indicators' list")
            print("   Use --check-alignment-only if you only want to check alignment")
            sys.exit(1)
    
    # Fetch raw data
    raw_df = fetch_raw_data(client, args.ticker, args.frequency, args.start, args.end)
    
    if raw_df.empty:
        print("❌ No raw data found for the specified parameters")
        sys.exit(1)
    
    # Compute indicators locally (no forward-filling - use only actual trading days)
    computed_df = compute_indicators_local(raw_df, args.indicators)
    
    # Fetch stored synthetic indicators
    stored_df = fetch_synthetic_data(client, args.ticker, args.frequency, 
                                      args.start, args.end, args.indicators)
    
    if stored_df.empty:
        print("❌ No synthetic indicator data found for the specified parameters")
        sys.exit(1)
    
    # Compare
    results = compare_indicators(computed_df, stored_df, args.indicators, args.tolerance)
    
    # Show samples if requested
    if args.show_samples:
        show_mismatch_samples(computed_df, stored_df, args.indicators, args.tolerance)
    
    # Export combined data if requested
    if args.export:
        if not args.output_file:
            # Create temp directory if it doesn't exist
            os.makedirs('temp', exist_ok=True)
            # Generate default filename in temp directory
            args.output_file = f"temp/{args.ticker.replace(':', '_')}_{args.frequency}_{args.start}_{args.end}.parquet"
        
        export_combined_data(
            raw_df, stored_df, computed_df, args.indicators,
            args.output_file, args.ticker, args.frequency
        )
    
    # Summary
    print_header("Verification Summary")
    
    total_indicators = len(results)
    matched = sum(1 for r in results.values() if r['status'] == 'match')
    mismatched = sum(1 for r in results.values() if r['status'] == 'mismatch')
    no_data = sum(1 for r in results.values() if r['status'] == 'no_data')
    
    print(f"Total indicators checked: {total_indicators}")
    print(f"✅ Matched: {matched}")
    print(f"❌ Mismatched: {mismatched}")
    print(f"⚠️  No data: {no_data}")
    print()
    
    if mismatched == 0 and no_data == 0:
        print("✅ All indicators verified successfully!")
        sys.exit(0)
    elif mismatched > 0:
        print("❌ Verification failed - some indicators have mismatches")
        sys.exit(1)
    else:
        print("⚠️  Verification incomplete - some indicators have no data")
        sys.exit(2)


if __name__ == '__main__':
    main()