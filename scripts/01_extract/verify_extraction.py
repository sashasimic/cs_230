"""Verify extracted raw stocks parquet data for correctness.

Usage:
    python scripts/verify_raw_stocks_data.py
    python scripts/verify_raw_stocks_data.py --file data/raw/stocks_raw.parquet
    python scripts/verify_raw_stocks_data.py --ticker XLU
"""

import os
import sys
import argparse
import pandas as pd

# Add project root to path (two levels up from scripts/01_extract/)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.logger import setup_logger

logger = setup_logger(__name__)


def verify_data(file_path: str, ticker: str = None):
    """
    Verify parquet data quality.
    
    Args:
        file_path: Path to parquet file
        ticker: Optional specific ticker to check
    """
    logger.info("=" * 70)
    logger.info("DATA VERIFICATION")
    logger.info("=" * 70)
    logger.info(f"File: {file_path}")
    
    # Check file exists
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False
    
    # Load data
    logger.info("Loading data...")
    df = pd.read_parquet(file_path)
    logger.info(f"✓ Loaded {len(df):,} rows × {len(df.columns)} columns\n")
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Basic shape info
    print("="*70)
    print("BASIC INFORMATION")
    print("="*70)
    print(f"Shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Number of days: {len(df)}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Data types
    print("\nData types:")
    print(df.dtypes.value_counts())
    
    # Check for weekends
    print("\n" + "="*70)
    print("WEEKEND CHECK")
    print("="*70)
    df['day_of_week'] = df['date'].dt.day_name()
    df['is_weekend'] = df['date'].dt.dayofweek >= 5
    
    print("Records by day of week:")
    print(df['day_of_week'].value_counts().sort_index())
    print(f"\nWeekend records: {df['is_weekend'].sum()}")
    print(f"Weekday records: {(~df['is_weekend']).sum()}")
    
    if df['is_weekend'].sum() > 0:
        print("\n⚠️  WARNING: Data contains weekend records (should be trading days only)")
    else:
        print("\n✓ Data contains only trading days (no weekends)")
    
    # Missing values
    print("\n" + "="*70)
    print("MISSING VALUES")
    print("="*70)
    missing_total = df.isnull().sum().sum()
    print(f"Total missing values: {missing_total}")
    
    if missing_total > 0:
        missing_by_col = df.isnull().sum()
        missing_by_col = missing_by_col[missing_by_col > 0].sort_values(ascending=False)
        print(f"\nColumns with missing values: {len(missing_by_col)}")
        for col, count in missing_by_col.head(10).items():
            pct = count / len(df) * 100
            print(f"  {col}: {count} ({pct:.2f}%)")
    else:
        print("✓ No missing values")
    
    # Extract tickers from column names
    ticker_cols = [col for col in df.columns if col != 'date' and not col.startswith('day_of_week') and not col.startswith('is_weekend')]
    tickers = sorted(list(set([col.split('_')[0] for col in ticker_cols])))
    
    print("\n" + "="*70)
    print(f"TICKERS ({len(tickers)} total)")
    print("="*70)
    print(", ".join(tickers))
    
    # Check each ticker has all OHLCV columns
    print("\n" + "="*70)
    print("COLUMN COMPLETENESS CHECK")
    print("="*70)
    expected_cols = ['open', 'high', 'low', 'close', 'volume', 'vwap']
    incomplete_tickers = []
    
    for t in tickers:
        t_cols = [col.split('_', 1)[1] for col in ticker_cols if col.startswith(f"{t}_")]
        missing_cols = [c for c in expected_cols if c not in t_cols]
        if missing_cols:
            incomplete_tickers.append((t, missing_cols))
    
    if incomplete_tickers:
        print(f"⚠️  {len(incomplete_tickers)} tickers have missing columns:")
        for t, missing in incomplete_tickers:
            print(f"  {t}: missing {missing}")
    else:
        print(f"✓ All {len(tickers)} tickers have complete OHLCV columns")
    
    # Specific ticker analysis
    if ticker:
        print("\n" + "="*70)
        print(f"DETAILED ANALYSIS: {ticker}")
        print("="*70)
        
        t_cols = [col for col in df.columns if col.startswith(f"{ticker}_")]
        if not t_cols:
            print(f"❌ Ticker {ticker} not found in data")
        else:
            print(f"Columns: {t_cols}")
            
            # Count total entries (rows with non-null values)
            ticker_df = df[t_cols]
            total_entries = ticker_df.notnull().all(axis=1).sum()
            print(f"\nTotal entries (complete rows): {total_entries:,} / {len(df):,} ({total_entries/len(df)*100:.1f}%)")
            
            print(f"\nSample data (first 10 rows):")
            print(df[['date'] + t_cols].head(10))
            
            print(f"\nStatistics:")
            print(df[t_cols].describe())
            
            print(f"\nMissing values:")
            for col in t_cols:
                missing = df[col].isnull().sum()
                print(f"  {col}: {missing}")
    
    # Data quality checks
    print("\n" + "="*70)
    print("DATA QUALITY CHECKS")
    print("="*70)
    
    issues = []
    
    # Check for duplicate dates
    dup_dates = df['date'].duplicated().sum()
    if dup_dates > 0:
        issues.append(f"Duplicate dates: {dup_dates}")
        print(f"❌ Found {dup_dates} duplicate dates")
    else:
        print("✓ No duplicate dates")
    
    # Check date sequence
    df_sorted = df.sort_values('date')
    date_gaps = (df_sorted['date'].diff() > pd.Timedelta(days=4)).sum()  # Allowing for long weekends
    if date_gaps > 0:
        print(f"⚠️  Found {date_gaps} gaps of >4 days between trading days")
        issues.append(f"Date gaps: {date_gaps}")
    else:
        print("✓ No unusual date gaps")
    
    # Check for negative values (shouldn't exist in prices/volume)
    for col in ticker_cols:
        if 'volume' in col.lower():
            negative = (df[col] < 0).sum()
            if negative > 0:
                issues.append(f"{col} has {negative} negative values")
                print(f"❌ {col} has {negative} negative values")
    
    # Check for zero prices (suspicious)
    for col in ticker_cols:
        if any(x in col.lower() for x in ['open', 'high', 'low', 'close']):
            zeros = (df[col] == 0).sum()
            if zeros > 0:
                issues.append(f"{col} has {zeros} zero values")
                print(f"⚠️  {col} has {zeros} zero values")
    
    # Final summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if not issues:
        print("✅ All checks passed! Data looks good.")
        return True
    else:
        print(f"⚠️  Found {len(issues)} issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Verify extracted parquet data')
    parser.add_argument('--file', default='data/raw/stocks_raw.parquet', help='Path to parquet file')
    parser.add_argument('--ticker', help='Specific ticker to analyze in detail')
    
    args = parser.parse_args()
    
    success = verify_data(args.file, args.ticker)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
