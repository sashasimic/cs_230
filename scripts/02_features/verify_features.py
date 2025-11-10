"""
Verify processed feature data quality.

This script checks processed train/val/test data for:
- Correct feature/target counts
- No data leakage
- Proper scaling
- Missing values
- Target distribution

Usage:
    python scripts/verify_processed_data.py
    python scripts/verify_processed_data.py --data-dir data/processed_v2
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

# Add project root to path (two levels up from scripts/02_features/)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.logger import setup_logger

logger = setup_logger(__name__)


def load_processed_data(data_dir: str):
    """Load processed train/val/test data."""
    logger.info(f"Loading processed data from {data_dir}...")
    
    train_df = pd.read_parquet(os.path.join(data_dir, 'train.parquet'))
    val_df = pd.read_parquet(os.path.join(data_dir, 'val.parquet'))
    test_df = pd.read_parquet(os.path.join(data_dir, 'test.parquet'))
    
    logger.info(f"✓ Train: {len(train_df)} rows × {len(train_df.columns)} columns")
    logger.info(f"✓ Val:   {len(val_df)} rows × {len(val_df.columns)} columns")
    logger.info(f"✓ Test:  {len(test_df)} rows × {len(test_df.columns)} columns")
    
    return train_df, val_df, test_df


def check_column_consistency(train_df, val_df, test_df):
    """Check that all splits have same columns."""
    logger.info("\n" + "="*70)
    logger.info("COLUMN CONSISTENCY CHECK")
    logger.info("="*70)
    
    train_cols = set(train_df.columns)
    val_cols = set(val_df.columns)
    test_cols = set(test_df.columns)
    
    if train_cols == val_cols == test_cols:
        logger.info(f"✓ All splits have {len(train_cols)} identical columns")
        return True
    else:
        logger.error("❌ Column mismatch between splits!")
        logger.error(f"  Train only: {train_cols - val_cols - test_cols}")
        logger.error(f"  Val only: {val_cols - train_cols - test_cols}")
        logger.error(f"  Test only: {test_cols - train_cols - val_cols}")
        return False


def identify_columns(df):
    """Identify feature and target columns."""
    date_col = 'date'
    target_cols = [col for col in df.columns if col.startswith('target_')]
    feature_cols = [col for col in df.columns if col not in target_cols and col != date_col]
    
    return feature_cols, target_cols


def check_missing_values(train_df, val_df, test_df, feature_cols, target_cols):
    """Check for missing values."""
    logger.info("\n" + "="*70)
    logger.info("MISSING VALUES CHECK")
    logger.info("="*70)
    
    issues = []
    
    for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        feature_missing = df[feature_cols].isnull().sum().sum()
        target_missing = df[target_cols].isnull().sum().sum()
        
        logger.info(f"{name}:")
        logger.info(f"  Features: {feature_missing} missing values")
        logger.info(f"  Targets:  {target_missing} missing values")
        
        if feature_missing > 0:
            # Show which features have missing values
            missing_feats = df[feature_cols].isnull().sum()
            missing_feats = missing_feats[missing_feats > 0]
            logger.warning(f"  Features with missing: {list(missing_feats.index[:5])}")
            issues.append(f"{name} has {feature_missing} missing feature values")
        
        if target_missing > 0:
            issues.append(f"{name} has {target_missing} missing target values")
    
    if not issues:
        logger.info("\n✓ No missing values")
        return True
    else:
        logger.warning(f"\n⚠️  Found {len(issues)} missing value issues")
        return False


def check_feature_scaling(train_df, val_df, test_df, feature_cols):
    """Check that features are properly scaled."""
    logger.info("\n" + "="*70)
    logger.info("FEATURE SCALING CHECK")
    logger.info("="*70)
    
    # Check train features are roughly normalized
    train_mean = train_df[feature_cols].mean().mean()
    train_std = train_df[feature_cols].std().mean()
    
    logger.info(f"Train features:")
    logger.info(f"  Mean of means: {train_mean:.4f} (should be ~0)")
    logger.info(f"  Mean of stds:  {train_std:.4f} (should be ~1)")
    
    # Check val/test are in reasonable range
    val_mean = val_df[feature_cols].mean().mean()
    val_std = val_df[feature_cols].std().mean()
    test_mean = test_df[feature_cols].mean().mean()
    test_std = test_df[feature_cols].std().mean()
    
    logger.info(f"Val features:")
    logger.info(f"  Mean of means: {val_mean:.4f}")
    logger.info(f"  Mean of stds:  {val_std:.4f}")
    logger.info(f"Test features:")
    logger.info(f"  Mean of means: {test_mean:.4f}")
    logger.info(f"  Mean of stds:  {test_std:.4f}")
    
    # Check if scaling looks reasonable
    if abs(train_mean) < 0.1 and 0.8 < train_std < 1.2:
        logger.info("\n✓ Features appear properly scaled")
        return True
    else:
        logger.warning("\n⚠️  Features may not be properly scaled")
        return False


def check_data_leakage(train_df, val_df, test_df):
    """Check for data leakage (temporal overlap)."""
    logger.info("\n" + "="*70)
    logger.info("DATA LEAKAGE CHECK")
    logger.info("="*70)
    
    train_df['date'] = pd.to_datetime(train_df['date'])
    val_df['date'] = pd.to_datetime(val_df['date'])
    test_df['date'] = pd.to_datetime(test_df['date'])
    
    train_end = train_df['date'].max()
    val_start = val_df['date'].min()
    val_end = val_df['date'].max()
    test_start = test_df['date'].min()
    
    logger.info(f"Train: {train_df['date'].min()} to {train_end}")
    logger.info(f"Val:   {val_start} to {val_end}")
    logger.info(f"Test:  {test_start} to {test_df['date'].max()}")
    
    issues = []
    
    if train_end >= val_start:
        issues.append(f"Train end ({train_end}) >= Val start ({val_start})")
        logger.error(f"❌ {issues[-1]}")
    
    if val_end >= test_start:
        issues.append(f"Val end ({val_end}) >= Test start ({test_start})")
        logger.error(f"❌ {issues[-1]}")
    
    # Check for date overlaps
    train_dates = set(train_df['date'])
    val_dates = set(val_df['date'])
    test_dates = set(test_df['date'])
    
    train_val_overlap = train_dates & val_dates
    val_test_overlap = val_dates & test_dates
    train_test_overlap = train_dates & test_dates
    
    if train_val_overlap:
        issues.append(f"{len(train_val_overlap)} dates overlap between train and val")
        logger.error(f"❌ {issues[-1]}")
    
    if val_test_overlap:
        issues.append(f"{len(val_test_overlap)} dates overlap between val and test")
        logger.error(f"❌ {issues[-1]}")
    
    if train_test_overlap:
        issues.append(f"{len(train_test_overlap)} dates overlap between train and test")
        logger.error(f"❌ {issues[-1]}")
    
    if not issues:
        logger.info("\n✓ No data leakage detected")
        return True
    else:
        logger.error(f"\n❌ Found {len(issues)} data leakage issues!")
        return False


def check_target_distribution(train_df, val_df, test_df, target_cols):
    """Check target distribution."""
    logger.info("\n" + "="*70)
    logger.info("TARGET DISTRIBUTION")
    logger.info("="*70)
    
    for target in target_cols:
        logger.info(f"\n{target}:")
        
        train_stats = train_df[target].describe()
        val_stats = val_df[target].describe()
        test_stats = test_df[target].describe()
        
        logger.info(f"  Train: mean={train_stats['mean']:.4f}, std={train_stats['std']:.4f}, "
                   f"min={train_stats['min']:.4f}, max={train_stats['max']:.4f}")
        logger.info(f"  Val:   mean={val_stats['mean']:.4f}, std={val_stats['std']:.4f}, "
                   f"min={val_stats['min']:.4f}, max={val_stats['max']:.4f}")
        logger.info(f"  Test:  mean={test_stats['mean']:.4f}, std={test_stats['std']:.4f}, "
                   f"min={test_stats['min']:.4f}, max={test_stats['max']:.4f}")
        
        # Check for outliers (>3 std from mean)
        for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            target_data = df[target].dropna()
            mean = target_data.mean()
            std = target_data.std()
            outliers = ((target_data - mean).abs() > 3 * std).sum()
            if outliers > 0:
                logger.warning(f"  {name}: {outliers} outliers (>3 std)")


def check_feature_names(data_dir):
    """Check if feature_names.txt exists and is valid."""
    logger.info("\n" + "="*70)
    logger.info("FEATURE NAMES CHECK")
    logger.info("="*70)
    
    feature_names_path = os.path.join(data_dir, 'feature_names.txt')
    
    if not os.path.exists(feature_names_path):
        logger.warning("⚠️  feature_names.txt not found")
        return False
    
    with open(feature_names_path, 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    logger.info(f"✓ Found {len(feature_names)} feature names")
    logger.info(f"  First 5: {feature_names[:5]}")
    logger.info(f"  Last 5: {feature_names[-5:]}")
    
    return True


def check_scaler(data_dir):
    """Check if scaler.pkl exists."""
    logger.info("\n" + "="*70)
    logger.info("SCALER CHECK")
    logger.info("="*70)
    
    scaler_path = os.path.join(data_dir, 'scaler.pkl')
    
    if not os.path.exists(scaler_path):
        logger.warning("⚠️  scaler.pkl not found")
        return False
    
    import joblib
    scaler = joblib.load(scaler_path)
    
    logger.info(f"✓ Scaler loaded successfully")
    logger.info(f"  Type: {type(scaler)}")
    logger.info(f"  Keys: {list(scaler.keys())[:5]}...")
    
    return True


def check_weekday_coverage(train_df, val_df, test_df):
    """Check that data only contains weekdays and has full coverage."""
    logger.info("\n" + "="*70)
    logger.info("WEEKDAY COVERAGE CHECK")
    logger.info("="*70)
    
    # Combine all splits
    all_dfs = []
    for df in [train_df, val_df, test_df]:
        df_copy = df.copy()
        df_copy['date'] = pd.to_datetime(df_copy['date'])
        all_dfs.append(df_copy)
    
    # Combine
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df = combined_df.sort_values('date').reset_index(drop=True)
    
    # Add day of week info
    combined_df['day_of_week'] = combined_df['date'].dt.dayofweek  # 0=Mon, 6=Sun
    combined_df['day_name'] = combined_df['date'].dt.day_name()
    
    # Check day distribution
    logger.info("\nDay of week distribution:")
    day_counts = combined_df['day_name'].value_counts().reindex(
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
        fill_value=0
    )
    for day, count in day_counts.items():
        logger.info(f"  {day:9s}: {count:4d}")
    
    # Check for weekends
    weekends = combined_df[combined_df['day_of_week'] >= 5]
    weekend_count = len(weekends)
    
    if weekend_count > 0:
        logger.error(f"\n❌ Found {weekend_count} weekend days!")
        logger.error(f"  First 5 weekend dates: {list(weekends['date'].head().dt.date)}")
        return False
    else:
        logger.info(f"\n✓ No weekend days (Sat/Sun): 0")
    
    # Check coverage
    total_days = len(combined_df)
    date_start = combined_df['date'].min()
    date_end = combined_df['date'].max()
    years = (date_end - date_start).days / 365.25
    
    logger.info(f"\nDate range: {date_start.date()} to {date_end.date()}")
    logger.info(f"Time span: {years:.2f} years")
    logger.info(f"Total weekdays: {total_days}")
    
    # Expected weekdays in this range
    date_range = pd.date_range(start=date_start, end=date_end, freq='D')
    expected_weekdays = len(date_range[date_range.dayofweek < 5])
    
    logger.info(f"Expected weekdays: {expected_weekdays}")
    logger.info(f"Actual weekdays: {total_days}")
    coverage = (total_days / expected_weekdays) * 100
    logger.info(f"Coverage: {coverage:.1f}%")
    
    # Trading years
    trading_years = total_days / 252
    logger.info(f"Equivalent trading years: {trading_years:.2f} years")
    
    # Split breakdown
    logger.info(f"\nSplit breakdown:")
    logger.info(f"  Train: {len(train_df):4d} days ({len(train_df)/total_days*100:.1f}%)")
    logger.info(f"  Val:   {len(val_df):4d} days ({len(val_df)/total_days*100:.1f}%)")
    logger.info(f"  Test:  {len(test_df):4d} days ({len(test_df)/total_days*100:.1f}%)")
    logger.info(f"  Total: {total_days:4d} days (100.0%)")
    
    # Check if coverage is good
    if coverage >= 99.0:
        logger.info(f"\n✓ Excellent coverage: {coverage:.1f}%")
        return True
    elif coverage >= 95.0:
        logger.info(f"\n✓ Good coverage: {coverage:.1f}%")
        return True
    else:
        logger.warning(f"\n⚠️  Low coverage: {coverage:.1f}% (missing {expected_weekdays - total_days} weekdays)")
        return False


def main():
    parser = argparse.ArgumentParser(description='Verify processed feature data')
    parser.add_argument('--data-dir', default='data/processed', help='Processed data directory')
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("PROCESSED DATA VERIFICATION")
    logger.info("=" * 70)
    
    # Load data
    train_df, val_df, test_df = load_processed_data(args.data_dir)
    
    # Identify columns
    feature_cols, target_cols = identify_columns(train_df)
    logger.info(f"\nIdentified {len(feature_cols)} features and {len(target_cols)} targets")
    
    # Run checks
    checks_passed = []
    
    checks_passed.append(check_column_consistency(train_df, val_df, test_df))
    checks_passed.append(check_missing_values(train_df, val_df, test_df, feature_cols, target_cols))
    checks_passed.append(check_feature_scaling(train_df, val_df, test_df, feature_cols))
    checks_passed.append(check_data_leakage(train_df, val_df, test_df))
    checks_passed.append(check_weekday_coverage(train_df, val_df, test_df))
    check_target_distribution(train_df, val_df, test_df, target_cols)
    checks_passed.append(check_feature_names(args.data_dir))
    checks_passed.append(check_scaler(args.data_dir))
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    
    passed = sum(checks_passed)
    total = len(checks_passed)
    
    logger.info(f"Checks passed: {passed}/{total}")
    
    if all(checks_passed):
        logger.info("\n✅ All checks passed! Data is ready for training.")
        return 0
    else:
        logger.warning(f"\n⚠️  {total - passed} check(s) failed. Review issues above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
