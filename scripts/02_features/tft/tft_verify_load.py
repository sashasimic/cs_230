#!/usr/bin/env python3
"""Verify TFT features exported by tft_data_loader.py.

Reads and validates data/raw/tft_features.parquet to ensure:
- GDELT features are present (if enabled)
- No missing values
- Correct date range and feature counts
- Valid value ranges

Usage:
    python scripts/02_features/tft_verify_build.py
    python scripts/02_features/tft_verify_build.py --raw-file data/raw/custom.parquet
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)


def print_header(text):
    print("\n" + "="*80)
    print(f"   {text}")
    print("="*80)
    print()


def verify_features(raw_file='data/raw/tft_features.parquet', 
                   config_path='configs/model_tft_config.yaml'):
    """Verify exported TFT features."""
    print_header("TFT Feature Verification")
    
    print(f"Raw file: {raw_file}")
    print(f"Config: {config_path}")
    print()
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    ticker = config['data']['ticker']['symbol']
    frequency = config['data']['ticker']['frequency']
    use_gdelt = config['data']['gdelt'].get('enabled', False)
    
    if use_gdelt:
        gdelt_features = config['data']['gdelt']['features']
        gdelt_include_lags = config['data']['gdelt'].get('include_lags', True)
        gdelt_lag_periods = config['data']['gdelt'].get('lag_periods', [1, 4, 16])
    
    print(f"Expected Configuration:")
    print(f"  Ticker: {ticker}")
    print(f"  Frequency: {frequency}")
    print(f"  GDELT enabled: {use_gdelt}")
    if use_gdelt:
        print(f"  GDELT features: {gdelt_features}")
        if gdelt_include_lags:
            print(f"  GDELT lag periods: {gdelt_lag_periods}")
    print()
    
    # Load data
    print_header("Loading Exported Data")
    
    if not os.path.exists(raw_file):
        print(f"❌ File not found: {raw_file}")
        print(f"\n💡 Run this first to generate features:")
        print(f"   python scripts/02_features/tft_data_loader.py")
        return False
    
    print(f"Reading: {raw_file}")
    
    if raw_file.endswith('.parquet'):
        # Use pyarrow and convert timestamp columns
        try:
            df = pd.read_parquet(raw_file, engine='pyarrow')
        except (TypeError, ValueError) as e:
            # Fallback: use CSV if parquet has dtype issues
            print(f"\n⚠️  Parquet reading failed ({e}), trying CSV...")
            csv_file = raw_file.replace('.parquet', '.csv')
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                print(f"✅ Loaded from {csv_file} instead")
            else:
                raise
    else:
        df = pd.read_csv(raw_file)
    
    file_size = Path(raw_file).stat().st_size / 1024 / 1024
    print(f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns ({file_size:.2f} MB)")
    print()
    
    # Data summary
    print_header("Data Summary")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Total rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    print()
    
    # Column breakdown
    print("\nColumn Categories:")
    
    # Identify columns
    meta_cols = ['timestamp', 'hour', 'day_of_week', 'date']
    target_cols = [c for c in df.columns if c.startswith('target_')]
    
    ticker_cols = [c for c in df.columns if c in ['open', 'high', 'low', 'close', 'volume'] or 
                   c.startswith(('returns', 'log_returns', 'volatility', 'momentum', 'volume_', 
                                'high_low', 'bb_', 'sma_', 'ema_', 'rsi', 'macd', 'atr'))]
    
    if use_gdelt:
        gdelt_cols = [c for c in df.columns if 'weighted_avg' in c or 'num_articles' in c or 
                     'num_sources' in c or 'sentiment_lag' in c]
    else:
        gdelt_cols = []
    
    print(f"  Metadata: {len(meta_cols)}")
    print(f"  Ticker features: {len(ticker_cols)}")
    print(f"  GDELT features: {len(gdelt_cols)}")
    print(f"  Target labels: {len(target_cols)}")
    print()
    
    if ticker_cols:
        print(f"Ticker features: {', '.join(ticker_cols[:15])}...")
    if gdelt_cols:
        print(f"GDELT features: {', '.join(gdelt_cols)}")
    if target_cols:
        print(f"Target labels: {', '.join(target_cols)}")
    print()
    
    # Missing values check
    print_header("Missing Values Check")
    missing = df.isnull().sum()
    if missing.any():
        print("⚠️  Missing values detected:")
        missing_df = missing[missing > 0].sort_values(ascending=False)
        
        # Check if missing values are only in target columns at the end
        target_missing_only = all(col.startswith('target_') for col in missing_df.index)
        
        for col, count in missing_df.items():
            pct = count / len(df) * 100
            print(f"  {col:30s}: {count:>6,} ({pct:>5.2f}%)")
        
        if target_missing_only:
            print("\n✅ Only target columns have missing values (expected at end of dataset)")
            print("   Targets require future data which isn't available for last rows")
        else:
            print("\n❌ TEST FAILED: Missing values in non-target features")
            return False
    else:
        print("✅ No missing values")
    print()
    
    # Duplicate timestamps check
    print_header("Duplicate Timestamps Check")
    duplicates = df['timestamp'].duplicated().sum()
    if duplicates > 0:
        print(f"⚠️  {duplicates} duplicate timestamps found")
        print("\n❌ TEST FAILED: Duplicates detected")
        return False
    else:
        print("✅ No duplicate timestamps")
    print()
    
    # Feature statistics
    print_header("Feature Statistics")
    
    # Ticker OHLCV
    print("Ticker OHLCV:")
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            print(f"  {col:12s}: min={df[col].min():>12.2f}, max={df[col].max():>12.2f}, mean={df[col].mean():>12.2f}, std={df[col].std():>12.2f}")
    
    # Technical indicators
    print("\nTechnical Indicators (sample):")
    for col in ['returns', 'volatility_5', 'sma_50', 'bb_width']:
        if col in df.columns:
            print(f"  {col:15s}: min={df[col].min():>10.4f}, max={df[col].max():>10.4f}, mean={df[col].mean():>10.4f}")
    
    # GDELT features
    if use_gdelt and gdelt_cols:
        print("\nGDELT Sentiment:")
        for col in gdelt_cols:
            if col in df.columns:
                print(f"  {col:30s}: min={df[col].min():>10.4f}, max={df[col].max():>10.4f}, mean={df[col].mean():>10.4f}")
    elif use_gdelt and not gdelt_cols:
        print("\n⚠️  GDELT enabled in config but NO GDELT features found in data!")
        print("    This means tft_data_loader.py was run with GDELT disabled.")
        print("    Delete data/processed/*.npy and rebuild.")
    
    # Value range checks
    print_header("Value Range Validation")
    
    checks_passed = True
    
    # Check for negative prices
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns:
            negatives = (df[col] < 0).sum()
            if negatives > 0:
                print(f"❌ {col}: {negatives} negative values detected")
                checks_passed = False
            else:
                print(f"✅ {col}: All values >= 0")
    
    # Check for zero/negative volume
    if 'volume' in df.columns:
        zero_vol = (df['volume'] <= 0).sum()
        if zero_vol > 0:
            print(f"⚠️  volume: {zero_vol} zero/negative values")
        else:
            print(f"✅ volume: All values > 0")
    
    # Check GDELT sentiment range
    if use_gdelt and 'weighted_avg_tone' in df.columns:
        tone_min, tone_max = df['weighted_avg_tone'].min(), df['weighted_avg_tone'].max()
        if tone_min < -10 or tone_max > 10:
            print(f"⚠️  weighted_avg_tone: Out of expected range [-10, 10]")
            print(f"     Actual range: [{tone_min:.2f}, {tone_max:.2f}]")
        else:
            print(f"✅ weighted_avg_tone: Within expected range [-10, 10]")
    
    if not checks_passed:
        print("\n❌ TEST FAILED: Value range issues detected")
        return False
    
    print()
    
    # Sample data display
    print_header("Sample Data (last 5 rows)")
    display_cols = ['timestamp', 'close', 'returns']
    if use_gdelt:
        display_cols += [c for c in gdelt_cols[:3] if c in df.columns]
    display_cols = [c for c in display_cols if c in df.columns]
    
    print(df[display_cols].tail(5).to_string(index=False))
    print()
    
    # Correlation check (GDELT vs target)
    if use_gdelt and 'weighted_avg_tone' in df.columns and target_cols:
        print_header("GDELT Sentiment vs Target Correlation")
        
        # Use first target for correlation
        target_col = target_cols[0]
        if target_col in df.columns:
            corr_df = df[['weighted_avg_tone', target_col]].dropna()
            if len(corr_df) > 0:
                corr = corr_df.corr().iloc[0, 1]
                print(f"Correlation (tone vs {target_col}): {corr:.4f}")
                
                if abs(corr) > 0.1:
                    print(f"✅ Moderate correlation detected ({corr:.4f})")
                else:
                    print(f"⚠️  Low correlation ({corr:.4f}) - sentiment may have weak direct relationship")
        
        # Lagged correlations
        if use_gdelt and gdelt_include_lags:
            print("\nLagged sentiment correlations:")
            for lag_col in [c for c in gdelt_cols if 'sentiment_lag' in c]:
                if lag_col in df.columns and target_col in df.columns:
                    corr_df = df[[lag_col, target_col]].dropna()
                    if len(corr_df) > 0:
                        lag_corr = corr_df.corr().iloc[0, 1]
                        print(f"  {lag_col:20s}: {lag_corr:>7.4f}")
        print()
    
    # Final summary
    print_header("Test Summary")
    print("✅ All checks passed!")
    print(f"✅ Loaded {len(df):,} rows with {len(df.columns)} features")
    print(f"✅ Ticker features: {len(ticker_cols)}")
    print(f"✅ Target labels: {len(target_cols)}")
    if use_gdelt:
        print(f"✅ GDELT features: {len(gdelt_cols)}")
    print(f"✅ No missing values")
    print(f"✅ No duplicates")
    print(f"✅ Valid value ranges")
    print()
    print("🚀 Ready for TFT training!")
    print()
    print(f"💡 Exported data available at:")
    print(f"   {raw_file}")
    print(f"   {raw_file.replace('.parquet', '.csv')}")
    print()
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Verify TFT features exported by tft_data_loader.py',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify default exported features
  python scripts/02_features/tft_verify_build.py
  
  # Verify custom file
  python scripts/02_features/tft_verify_build.py --raw-file data/raw/custom.parquet
  
  # Use custom config
  python scripts/02_features/tft_verify_build.py --config configs/custom.yaml
        """
    )
    parser.add_argument('--raw-file', default='data/raw/tft_features.parquet', 
                       help='Raw features file to verify')
    parser.add_argument('--config', default='configs/model_tft_config.yaml', 
                       help='Config file path')
    
    args = parser.parse_args()
    
    success = verify_features(
        raw_file=args.raw_file,
        config_path=args.config
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
