#!/usr/bin/env python3
"""Diagnose NaN values in processed features."""

import pandas as pd
import numpy as np

print("="*70)
print("NaN DIAGNOSIS IN PROCESSED DATA")
print("="*70)

for split in ['train', 'val', 'test']:
    print(f"\n{'='*70}")
    print(f"{split.upper()} SPLIT")
    print("="*70)
    
    df = pd.read_parquet(f'data/processed/{split}.parquet')
    feature_cols = [c for c in df.columns if c not in ['date', 'target_1M', 'target_3M', 'target_6M']]
    
    total_features = len(feature_cols)
    total_samples = len(df)
    feature_nans = df[feature_cols].isna().sum().sum()
    
    print(f"Shape: {df.shape}")
    print(f"Feature NaN: {feature_nans:,}")
    
    if feature_nans > 0:
        # Show worst features
        nan_counts = df[feature_cols].isna().sum()
        worst = nan_counts[nan_counts > 0].sort_values(ascending=False).head(20)
        
        print(f"\nTop 20 features with NaN:")
        for feat, count in worst.items():
            pct = (count / total_samples) * 100
            print(f"  {feat:50s}: {count:5d}/{total_samples} ({pct:5.1f}%)")
        
        # Check if NaN at start
        first_50_nans = df[feature_cols].iloc[:50].isna().sum().sum()
        print(f"\nNaN in first 50 rows: {first_50_nans} ({first_50_nans/feature_nans*100:.1f}% of total)")
        
        # Group by type
        print("\nNaN by feature type:")
        ma_features = [c for c in feature_cols if '_sma_' in c or '_ma_dev_' in c]
        if ma_features:
            ma_nans = df[ma_features].isna().sum().sum()
            print(f"  Moving Averages: {len(ma_features)} features, {ma_nans} NaN")
        
        return_features = [c for c in feature_cols if '_return_' in c]
        if return_features:
            ret_nans = df[return_features].isna().sum().sum()
            print(f"  Returns: {len(return_features)} features, {ret_nans} NaN")

print("\n" + "="*70)
print("DONE")
print("="*70)
