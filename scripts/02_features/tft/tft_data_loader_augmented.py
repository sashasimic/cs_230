#!/usr/bin/env python3
"""
Augmented TFT Data Loader

Extends MultiTickerDataLoader to support data augmentation by varying inflation tickers.
Completely decoupled from the base data loader.

Usage:
    # With augmentation
    from tft_data_loader_augmented import AugmentedMultiTickerDataLoader
    loader = AugmentedMultiTickerDataLoader('configs/model_tft_config.yaml')
    splits = loader.prepare_data()
    
    # Without augmentation (use original)
    from tft_data_loader import MultiTickerDataLoader
    loader = MultiTickerDataLoader('configs/model_tft_config.yaml')
    splits = loader.prepare_data()
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
from datetime import datetime

# Import base data loader (handle both relative and absolute imports)
try:
    from .tft_data_loader import MultiTickerDataLoader
except ImportError:
    from tft_data_loader import MultiTickerDataLoader


class AugmentedMultiTickerDataLoader(MultiTickerDataLoader):
    """
    Augmented data loader that creates multiple samples per date by varying inflation tickers.
    
    Pipeline:
    1. Fetch data from BigQuery
    2. Pivot to wide format (close_SPY, close_DBC, etc.)
    3. **AUGMENT**: Create N samples per date (one per inflation ticker)
    4. Create sequences
    5. Split temporally
    6. Normalize
    """
    
    def __init__(self, config_path: str = 'configs/model_tft_config.yaml', export_temp: bool = False):
        """Initialize augmented data loader."""
        # Call parent init
        super().__init__(config_path, export_temp)
        
        # Verify config has augmentation setup
        augment_groups = self.config['data'].get('ticker_augment_groups', [])
        if not augment_groups:
            raise ValueError(
                "Augmented data loader requires 'data.ticker_augment_groups' in config. "
                "Use MultiTickerDataLoader for configs without augmentation."
            )
        
        # Augmentation is always enabled for this loader
        aug_config = self.config['data'].get('augmentation', {})
        self.augmentation_enabled = aug_config.get('enabled', True)  # Default True for augmented loader
        
        # Load ticker groupings for augmentation
        self._load_ticker_groupings()
        
        print(f"\n{'='*80}")
        print("  AUGMENTED DATA LOADER")
        print(f"{'='*80}")
        print(f"Augmentation: {'ENABLED' if self.augmentation_enabled else 'DISABLED'}")
        print(f"Tickers loaded: {len(self.tickers)} total")
        print(f"{'='*80}\n")
    
    def _load_ticker_groupings(self):
        """Load augmentation config from ticker groups."""
        augment_groups = self.config['data'].get('ticker_augment_groups', [])
        ticker_groups = self.config['data'].get('ticker_groups', {})
        
        # Validate augment_groups exists
        if not augment_groups:
            raise ValueError(
                "data.ticker_augment_groups is required for augmented data loader. "
                "Add 'ticker_augment_groups' list referencing data.ticker_groups."
            )
        
        # Resolve tickers from groups (flattened structure)
        self.augment_across = []
        self.ticker_to_category = {}  # Map ticker -> group name
        
        for group_name in augment_groups:
            if group_name not in ticker_groups:
                raise ValueError(
                    f"Group '{group_name}' not found in data.ticker_groups. "
                    f"Available groups: {list(ticker_groups.keys())}"
                )
            
            group_config = ticker_groups[group_name]
            group_tickers = group_config.get('tickers', [])
            
            # Add tickers to augmentation list
            self.augment_across.extend(group_tickers)
            
            # Map each ticker to its group name (used as category)
            for ticker in group_tickers:
                self.ticker_to_category[ticker] = group_name
        
        if not self.augment_across:
            raise ValueError(
                f"No tickers found in groups: {augment_groups}. "
                "Check data.ticker_groups configuration."
            )
        
        # Economy tickers = all tickers minus augment_across
        self.economy_group = [t for t in self.tickers if t not in self.augment_across]
        
        print(f"\nAugmentation config:")
        print(f"  Augment groups: {augment_groups}")
        print(f"  Augment across: {len(self.augment_across)} tickers - {self.augment_across}")
        print(f"  Economy context: {len(self.economy_group)} tickers")
        print(f"  Group mapping: {dict(list(self.ticker_to_category.items())[:3])}...")  # Show first 3
        if self.augmentation_enabled:
            print(f"  Expected augmentation: {len(self.augment_across)}x per date")
    
    def _augment_dataframe(self, df: pd.DataFrame, split: str = 'all') -> pd.DataFrame:
        """
        Augment DataFrame by creating variants with different inflation tickers.
        
        For each date:
        - Keep economy ticker features + GDELT constant
        - Create one sample per inflation ticker
        - Add inflation_ticker and inflation_category metadata
        
        Args:
            df: DataFrame in pivoted format
            split: Split name (for logging)
        
        Returns:
            Augmented DataFrame
        """
        if not self.augmentation_enabled:
            return df
        
        print(f"\n🔄 Augmenting {split} set...")
        print(f"   Original: {len(df):,} samples")
        
        # Identify inflation vs economy columns
        inflation_cols = []
        for ticker in self.augment_across:
            inflation_cols.extend([c for c in df.columns if c.endswith(f'_{ticker}')])
        
        economy_cols = [c for c in df.columns if c not in inflation_cols]
        
        print(f"   Inflation columns: {len(inflation_cols)}")
        print(f"   Economy columns: {len(economy_cols)}")
        
        # Group by date
        date_col = 'date' if 'date' in df.columns else 'timestamp'
        if date_col not in df.columns:
            print(f"   ⚠️  No date column, skipping augmentation")
            return df
        
        augmented = []
        
        for date in df[date_col].unique():
            date_row = df[df[date_col] == date].iloc[0]
            
            # Create one sample per ticker in augment_across
            for inf_ticker in self.augment_across:
                # Check if this ticker has data for this date
                # Use 'close' as the indicator - if close is NaN, skip this sample
                close_col = f"close_{inf_ticker}"
                if close_col not in df.columns or pd.isna(date_row[close_col]):
                    continue  # Skip this ticker for this date
                
                new_row = date_row[economy_cols].copy()
                
                # Copy this inflation ticker's features
                for feature in self.raw_features + self.synthetic_features:
                    src_col = f"{feature}_{inf_ticker}"
                    if src_col in df.columns:
                        new_row[f"{feature}_inflation"] = date_row[src_col]
                
                # Add metadata
                new_row['inflation_ticker'] = inf_ticker
                new_row['inflation_category'] = self.ticker_to_category.get(inf_ticker, 'unknown')
                augmented.append(new_row)
        
        df_aug = pd.DataFrame(augmented)
        print(f"   Augmented: {len(df_aug):,} samples ({len(df_aug)/len(df):.1f}x)\n")
        
        return df_aug
    
    def prepare_data(self) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Augmented data preparation pipeline.
        
        Returns:
            Dictionary with 'train', 'val', 'test' splits
        """
        # 1. Fetch raw data (calls parent's fetch_data)
        df = self.fetch_data()
        
        # 2. Pivot to wide format
        print(f"\n{'='*80}")
        print("  PIVOTING TO WIDE FORMAT")
        print(f"{'='*80}\n")
        df_pivoted = self._pivot_to_wide(df)
        
        # 3. Apply group feature aggregation if configured
        feature_type = self.config.get('model', {}).get('feature_type', 'raw')
        print(f"\n🔍 DEBUG: feature_type = '{feature_type}' (checking if == 'group_signals')")
        if feature_type == 'group_signals':
            print(f"\n🔄 Applying group-level feature aggregation...")
            
            # Initialize aggregator  
            from group_features import TickerGroupFeatureAggregator
            aggregator = TickerGroupFeatureAggregator(self.config_path)
            
            # Compute group features from ticker data
            group_features_df = aggregator.compute_group_features(df_pivoted)
            
            # Merge group features with other data
            # Keep 'timestamp' column for consistency
            merged = pd.DataFrame({'timestamp': df_pivoted['timestamp']})
            
            # Add group features
            for col in group_features_df.columns:
                merged[col] = group_features_df[col]
            
            # Preserve target basket for Y labels
            if 'target_basket_close' in df_pivoted.columns:
                merged['target_basket_close'] = df_pivoted['target_basket_close']
            
            # Preserve individual agriculture ticker columns for augmentation
            # These are needed for creating inflation_ticker variants
            for ticker in self.augment_across:
                for feat in ['close', 'volume', 'sma_50', 'sma_200']:
                    col_name = f"{feat}_{ticker}"
                    if col_name in df_pivoted.columns:
                        merged[col_name] = df_pivoted[col_name]
            
            # Replace pivoted data with merged data
            df_pivoted = merged
            
            print(f"   ✅ Generated {len(group_features_df.columns)} group features")
        
        # 4. Add time features
        df_pivoted = self.add_time_features(df_pivoted)
        
        # Add 'date' column as alias for 'timestamp' (needed for augmentation/sequence creation)
        if 'timestamp' in df_pivoted.columns and 'date' not in df_pivoted.columns:
            df_pivoted['date'] = df_pivoted['timestamp']
        
        # 5. Apply augmentation (if enabled)
        if self.augmentation_enabled:
            print(f"\n{'='*80}")
            print("  APPLYING AUGMENTATION")
            print(f"{'='*80}")
            df_augmented = self._augment_dataframe(df_pivoted, split='all')
        else:
            print(f"\n⏭️  Skipping augmentation (disabled)\n")
            df_augmented = df_pivoted
        
        # 6. Create sequences (convert to numpy arrays)
        print(f"\n{'='*80}")
        print("  CREATING SEQUENCES")
        print(f"{'='*80}")
        X_raw, y_raw, ts, df_seq = self._create_sequences_from_dataframe(df_augmented)
        
        # 7. Temporal split
        print(f"\n{'='*80}")
        print("  TEMPORAL SPLIT")
        print(f"{'='*80}\n")
        
        n_samples = len(X_raw)
        train_ratio = self.config['data']['train_ratio']
        val_ratio = self.config['data']['val_ratio']
        
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        X_train_raw = X_raw[:train_end]
        X_val_raw = X_raw[train_end:val_end]
        X_test_raw = X_raw[val_end:]
        
        y_train = y_raw[:train_end]
        y_val = y_raw[train_end:val_end]
        y_test = y_raw[val_end:]
        
        ts_train = ts[:train_end]
        ts_val = ts[train_end:val_end]
        ts_test = ts[val_end:]
        
        # Split static features
        static_train = self.static_sequences[:train_end]
        static_val = self.static_sequences[train_end:val_end]
        static_test = self.static_sequences[val_end:]
        
        print(f"  train: {len(X_train_raw):,} samples ({len(X_train_raw)/n_samples*100:.1f}%)")
        print(f"  val  : {len(X_val_raw):,} samples ({len(X_val_raw)/n_samples*100:.1f}%)")
        print(f"  test : {len(X_test_raw):,} samples ({len(X_test_raw)/n_samples*100:.1f}%)")
        
        # 8. Normalize
        print(f"\n{'='*80}")
        print("  NORMALIZING")
        print(f"{'='*80}\n")
        
        X_train_norm, X_val_norm, X_test_norm, y_train_norm, y_val_norm, y_test_norm = self._normalize_splits(
            X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test
        )
        
        splits = {
            'train': (X_train_norm, y_train_norm, ts_train, static_train),
            'val': (X_val_norm, y_val_norm, ts_val, static_val),
            'test': (X_test_norm, y_test_norm, ts_test, static_test)
        }
        
        # Export
        self._export_raw_validation(df_seq)
        self.save_processed_data(splits, df_seq)
        
        return splits
    
    def _pivot_to_wide(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pivot from long to wide format.
        
        Input:
            ticker | timestamp | close | volume
            SPY    | 2024-01-01| 450   | 1000
            DBC    | 2024-01-01| 20    | 500
        
        Output:
            date       | close_SPY | close_DBC | volume_SPY | volume_DBC
            2024-01-01 | 450       | 20        | 1000       | 500
        """
        print(f"Pivoting {len(df):,} rows...")
        
        # Group by date
        dates = df['date'].unique()
        feature_cols = self.raw_features + self.synthetic_features
        
        # Build pivoted data
        pivot_dict = {'date': dates}
        
        for ticker in self.tickers:
            ticker_data = df[df['ticker'] == ticker].set_index('date')
            for feature in feature_cols:
                if feature in ticker_data.columns:
                    pivot_dict[f"{feature}_{ticker}"] = ticker_data[feature].reindex(dates).values
        
        df_pivoted = pd.DataFrame(pivot_dict)
        
        # Add timestamp
        df_pivoted['timestamp'] = pd.to_datetime(df_pivoted['date'])
        
        print(f"✅ Pivoted: {df_pivoted.shape}")
        print(f"   Columns: {len(df_pivoted.columns)}\n")
        
        return df_pivoted
    
    def _create_sequences_from_dataframe(self, df: pd.DataFrame) -> Tuple:
        """
        Create sequences from DataFrame.
        
        Handles augmented data with inflation_ticker and inflation_category columns.
        """
        # Identify columns
        metadata_cols = ['date', 'timestamp', 'inflation_ticker', 'inflation_category']
        feature_cols = [c for c in df.columns if c not in metadata_cols]
        
        print(f"\nCreating sequences...")
        print(f"  Samples: {len(df):,}")
        print(f"  Features: {len(feature_cols)}")
        print(f"  Lookback: {self.lookback}")
        print(f"  Horizons: {self.horizons}")
        
        # Extract arrays
        features = df[feature_cols].values
        dates = df['date'].values
        
        # Extract static features (constant per sequence)
        static_cols = ['inflation_ticker', 'inflation_category']
        static_data = df[static_cols].values
        
        # Create sliding windows
        X_list, y_list, ts_list, static_list = [], [], [], []
        
        # Find target column once (not in the loop!)
        target_idx = self._find_target_index(feature_cols)
        print(f"   Using target column index: {target_idx} ({feature_cols[target_idx]})")
        
        max_horizon = max(self.horizons)
        for i in range(len(features) - self.lookback - max_horizon):
            # Input sequence
            X_list.append(features[i:i + self.lookback])
            
            # Multi-horizon targets
            targets = [features[i + self.lookback + h - 1, target_idx] for h in self.horizons]
            y_list.append(targets)
            
            # Timestamp
            ts_list.append(dates[i + self.lookback])
            
            # Static features (from the first row of the sequence, they're constant)
            static_list.append(static_data[i])
        
        X = np.array(X_list)
        y = np.array(y_list)
        ts = np.array(ts_list)
        static = np.array(static_list)  # [num_sequences, 2] - ticker and category strings
        
        # Store features for normalization
        self.final_features = feature_cols
        self.static_features = static_cols
        
        print(f"\n✅ Created {len(X):,} sequences")
        print(f"   X: {X.shape}")
        print(f"   y: {y.shape}")
        print(f"   Static: {static.shape}\n")
        
        # Store static features for later use
        self.static_sequences = static
        
        return X, y, ts, df
    
    def save_processed_data(self, splits: Dict[str, Tuple], df_raw: pd.DataFrame):
        """Override to save static features along with X, y, ts."""
        import shutil
        import pickle
        from pathlib import Path
        
        output_dir = Path('data/processed')
        
        # Clear existing data
        if output_dir.exists():
            print(f"\n🗑️  Clearing existing processed data...")
            shutil.rmtree(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get feature list
        all_features = self.final_features
        time_varying_known = self.config['model'].get('time_varying_known', [])
        time_varying_unknown = [f for f in all_features if f not in time_varying_known]
        
        print(f"\n" + "="*80)
        print(f"   Saving Processed Data (NumPy Arrays)")
        print("="*80)
        print(f"\n💾 Output directory: {output_dir}/")
        print(f"   Features: {len(all_features)} total")
        print(f"     - {len(time_varying_known)} time-varying known (future known)")
        print(f"     - {len(time_varying_unknown)} time-varying unknown (past only)")
        print(f"   Static features: {self.static_features}")
        print(f"   Prediction horizons: {self.horizons}")
        print(f"   Format: NumPy arrays (.npy) - optimized for PyTorch/TensorFlow")
        print()
        
        # Save each split as separate X, y, ts, static arrays
        for split_name, data in splits.items():
            X, y, ts, static = data
            n_samples, lookback, n_features = X.shape
            n_horizons = y.shape[1] if len(y.shape) > 1 else 1
            
            print(f"  💾 Saving {split_name.upper()} split ({n_samples:,} sequences)...")
            
            # Save arrays
            X_file = output_dir / f'X_{split_name}.npy'
            y_file = output_dir / f'y_{split_name}.npy'
            ts_file = output_dir / f'ts_{split_name}.npy'
            static_file = output_dir / f'static_{split_name}.npy'
            
            print(f"     Saving X_{split_name}.npy {X.shape}...", end='', flush=True)
            np.save(X_file, X)
            print(f" Done!")
            
            print(f"     Saving y_{split_name}.npy {y.shape}...", end='', flush=True)
            np.save(y_file, y)
            print(f" Done!")
            
            print(f"     Saving ts_{split_name}.npy {ts.shape}...", end='', flush=True)
            np.save(ts_file, ts)
            print(f" Done!")
            
            print(f"     Saving static_{split_name}.npy {static.shape}...", end='', flush=True)
            np.save(static_file, static)
            print(f" Done!")
            
            # Calculate file sizes
            X_size_mb = X_file.stat().st_size / 1024 / 1024
            y_size_mb = y_file.stat().st_size / 1024 / 1024
            ts_size_mb = ts_file.stat().st_size / 1024 / 1024
            static_size_mb = static_file.stat().st_size / 1024 / 1024
            total_split_mb = X_size_mb + y_size_mb + ts_size_mb + static_size_mb
            
            print(f"  ✅ {split_name.upper():5s}:")
            print(f"       Sequences: {n_samples:,}")
            print(f"       X_{split_name}.npy: {X.shape} = {X_size_mb:.2f} MB")
            print(f"       y_{split_name}.npy: {y.shape} = {y_size_mb:.2f} MB")
            print(f"       ts_{split_name}.npy: {ts.shape} = {ts_size_mb:.2f} MB")
            print(f"       static_{split_name}.npy: {static.shape} = {static_size_mb:.2f} MB")
            print(f"       Total: {total_split_mb:.2f} MB")
            print()
        
        # Save scalers
        scalers_file = output_dir / 'scalers.pkl'
        with open(scalers_file, 'wb') as f:
            pickle.dump(self.scalers, f)
        print(f"  💾 Saved scalers: {scalers_file}")
        
        # Save feature names
        feature_file = output_dir / 'feature_names.txt'
        with open(feature_file, 'w') as f:
            f.write("TIME-VARYING KNOWN FEATURES:\n")
            for feat in time_varying_known:
                f.write(f"  - {feat}\n")
            f.write("\nTIME-VARYING UNKNOWN FEATURES:\n")
            for feat in time_varying_unknown:
                f.write(f"  - {feat}\n")
            f.write("\nSTATIC FEATURES:\n")
            for feat in self.static_features:
                f.write(f"  - {feat}\n")
        print(f"  💾 Saved feature names: {feature_file}")
        
        print(f"\n" + "="*80)
        print(f"✅ ALL DATA SAVED TO: {output_dir}/")
        print("="*80 + "\n")
    
    def _find_target_index(self, feature_cols: List[str]) -> int:
        """Find target column index for augmented data."""
        target = self.target_col
        
        # For augmented data, use returns from the inflation ticker
        if self.augmentation_enabled:
            # Look for returns_inflation or close_inflation columns
            target_candidates = [
                'returns_inflation',
                'target_basket_returns_inflation',
                'close_inflation'
            ]
            
            for candidate in target_candidates:
                if candidate in feature_cols:
                    return feature_cols.index(candidate)
            
            # Fallback: any column with 'returns' and '_inflation'
            inflation_return_cols = [c for c in feature_cols if 'returns' in c and c.endswith('_inflation')]
            if inflation_return_cols:
                return feature_cols.index(inflation_return_cols[0])
        
        # Non-augmented: exact match
        if target in feature_cols:
            return feature_cols.index(target)
        
        # Debug: print available columns
        print(f"   \n   Available columns: {feature_cols[:20]}...") 
        raise ValueError(f"Target '{target}' not found in features. Looking for inflation target column.")


if __name__ == '__main__':
    """Test augmented data loader."""
    print("Testing Augmented Data Loader...\n")
    
    loader = AugmentedMultiTickerDataLoader('configs/model_tft_config.yaml')
    splits = loader.prepare_data()
    
    print("\n" + "="*80)
    print("  TEST COMPLETE")
    print("="*80)
    for split_name, (X, y, ts) in splits.items():
        print(f"{split_name}: X={X.shape}, y={y.shape}, ts={len(ts)}")
    print("="*80)