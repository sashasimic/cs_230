#!/usr/bin/env python
"""Verify extracted Google Trends data quality."""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def verify_google_trends(file_path: str = 'data/raw/google_trends.parquet') -> bool:
    """
    Verify Google Trends data quality.
    
    Args:
        file_path: Path to Google Trends parquet file
        
    Returns:
        True if all checks pass, False otherwise
    """
    logger.info("="*70)
    logger.info("Google Trends Data Verification")
    logger.info("="*70)
    logger.info(f"File: {file_path}")
    
    checks_passed = 0
    checks_failed = 0
    warnings = 0
    
    # Check 1: File exists
    logger.info("\n[1/8] Checking file existence...")
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error(f"  ❌ File not found: {file_path}")
        return False
    logger.info(f"  ✓ File exists ({file_path.stat().st_size / 1024:.1f} KB)")
    checks_passed += 1
    
    # Check 2: Load data
    logger.info("\n[2/8] Loading data...")
    try:
        df = pd.read_parquet(file_path)
        logger.info(f"  ✓ Loaded {len(df):,} rows")
        checks_passed += 1
    except Exception as e:
        logger.error(f"  ❌ Failed to load: {e}")
        return False
    
    # Check 3: Required columns
    logger.info("\n[3/8] Checking columns...")
    required_cols = ['date']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"  ❌ Missing required columns: {missing_cols}")
        checks_failed += 1
    else:
        logger.info(f"  ✓ Has 'date' column")
        checks_passed += 1
    
    # Get keyword columns (all except 'date')
    keyword_cols = [col for col in df.columns if col != 'date']
    logger.info(f"  Keywords: {', '.join(keyword_cols)}")
    
    # Check 4: Date column
    logger.info("\n[4/8] Checking date column...")
    try:
        df['date'] = pd.to_datetime(df['date'])
        logger.info(f"  ✓ Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"  ✓ Duration: {(df['date'].max() - df['date'].min()).days} days")
        checks_passed += 1
    except Exception as e:
        logger.error(f"  ❌ Date parsing failed: {e}")
        checks_failed += 1
    
    # Check 5: Date continuity
    logger.info("\n[5/8] Checking date continuity...")
    df_sorted = df.sort_values('date')
    date_diffs = df_sorted['date'].diff().dt.days
    gaps = date_diffs[date_diffs > 1]
    if len(gaps) > 0:
        logger.warning(f"  ⚠ Found {len(gaps)} gaps in dates")
        logger.warning(f"  Largest gap: {gaps.max()} days")
        warnings += 1
    else:
        logger.info("  ✓ No gaps in daily data")
        checks_passed += 1
    
    # Check 6: Value ranges (Google Trends is 0-100)
    logger.info("\n[6/8] Checking value ranges...")
    all_in_range = True
    for col in keyword_cols:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            if min_val < 0 or max_val > 100:
                logger.error(f"  ❌ {col}: values outside 0-100 range ({min_val} to {max_val})")
                all_in_range = False
            else:
                logger.info(f"  ✓ {col}: {min_val:.0f} to {max_val:.0f} (within 0-100)")
    
    if all_in_range:
        checks_passed += 1
    else:
        checks_failed += 1
    
    # Check 7: Missing values
    logger.info("\n[7/8] Checking for missing values...")
    missing_counts = df[keyword_cols].isnull().sum()
    total_missing = missing_counts.sum()
    if total_missing > 0:
        logger.warning(f"  ⚠ Found {total_missing} missing values:")
        for col in keyword_cols:
            if missing_counts[col] > 0:
                pct = (missing_counts[col] / len(df)) * 100
                logger.warning(f"    {col}: {missing_counts[col]} ({pct:.1f}%)")
        warnings += 1
    else:
        logger.info("  ✓ No missing values")
        checks_passed += 1
    
    # Check 8: Data statistics
    logger.info("\n[8/8] Data statistics:")
    for col in keyword_cols:
        if col in df.columns:
            stats = df[col].describe()
            logger.info(f"  {col}:")
            logger.info(f"    Mean:   {stats['mean']:.2f}")
            logger.info(f"    Median: {stats['50%']:.2f}")
            logger.info(f"    Std:    {stats['std']:.2f}")
            
            # Check for constant values
            if stats['std'] < 0.1:
                logger.warning(f"    ⚠ Very low variance (possibly constant)")
                warnings += 1
    
    checks_passed += 1
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("VERIFICATION SUMMARY")
    logger.info("="*70)
    logger.info(f"✓ Checks passed: {checks_passed}")
    if checks_failed > 0:
        logger.info(f"❌ Checks failed: {checks_failed}")
    if warnings > 0:
        logger.info(f"⚠ Warnings: {warnings}")
    
    success = checks_failed == 0
    if success:
        logger.info("\n✓ All checks passed!")
    else:
        logger.error("\n❌ Some checks failed")
    logger.info("="*70)
    
    return success


def main():
    parser = argparse.ArgumentParser(
        description='Verify Google Trends data quality',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--file',
        default='data/raw/google_trends.parquet',
        help='Path to Google Trends parquet file'
    )
    
    args = parser.parse_args()
    
    success = verify_google_trends(args.file)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
