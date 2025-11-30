#!/usr/bin/env python3
"""TFT Pipeline - Uses Model-Agnostic Modules (NEW CODE)

This replaces the old tft/tft_data_loader.py and tft/tft_data_loader_augmented.py.
It uses the new model-agnostic base modules and adds TFT-specific functionality.

Key difference: 'tft' and 'tft-augmented' both use this same class.
Augmentation is enabled/disabled via config, not via separate classes.
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import pickle
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler

# Add current directory to path for importing model-agnostic modules
sys.path.insert(0, str(Path(__file__).parent))

# Import NEW model-agnostic modules
from data_loader import MultiTickerDataLoader
from data_grouping import TickerGroupFeatureAggregator  
from data_augmentation import DataAugmenter


class TFTDataPipeline:
    """TFT data pipeline using model-agnostic modules.
    
    Architecture:
    - Core: MultiTickerDataLoader (fetch, preprocess) 
    - Helper: TickerGroupFeatureAggregator (group features) 
    - Helper: DataAugmenter (augmentation) 
    - TFT-specific: This class (sequences, normalize, save)
    
    Replaces:
    - tft/tft_data_loader.py (old base)
    - tft/tft_data_loader_augmented.py (old augmented)
    """
    
    def __init__(self, config_path: str, export_temp: bool = False):
        """Initialize TFT pipeline.
        
        Args:
            config_path: Path to config YAML
            export_temp: Export debug data (unused, kept for compatibility)
        """
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize model-agnostic base loader
        self.loader = MultiTickerDataLoader(config_path, export_temp)
        
        # TFT-specific config
        self.lookback = self.config['data']['lookback_window']
        self.horizons = self.config['data']['prediction_horizons']
        
        # Check what features/augmentation to use
        self.use_group_features = self.config.get('model', {}).get('feature_type') == 'group_signals'
        self.use_augmentation = self.config['data'].get('augmentation', {}).get('enabled', False)
        
        # Print configuration
        print(f"\n{'='*80}")
        print("   TFT DATA PIPELINE (NEW - Model-Agnostic Base)")
        print(f"{'='*80}")
        print(f"Configuration:")
        print(f"  - Group features: {'ENABLED' if self.use_group_features else 'DISABLED'}")
        print(f"  - Augmentation: {'ENABLED' if self.use_augmentation else 'DISABLED'}")
        print(f"  - Lookback: {self.lookback} periods")
        print(f"  - Horizons: {self.horizons}")
        print(f"{'='*80}\n")
    
    def prepare_data(self) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Full TFT data preparation pipeline.
        
        Returns:
            Dictionary with train/val/test splits
        """
        # 1. Fetch data using NEW model-agnostic loader
        print("\n[1/7] Fetching data...")
        df = self.loader.fetch_data()
        df = self.loader.add_time_features(df)
        
        # 2. Create sequences (includes pivoting and optional group aggregation)
        print("\n[2/7] Creating sequences...")
        X, y, ts, static, df_pivoted = self.create_sequences(df)
        
        # Convert to numpy arrays immediately
        X = np.array(X)
        y = np.array(y)
        ts = np.array(ts)
        static = np.array(static)
        
        print(f"✅ Created {len(X)} sequences")
        print(f"   X shape: {X.shape}")
        print(f"   y shape: {y.shape}")
        print(f"   static shape: {static.shape}")
        
        # Log static features
        if len(static) > 0 and static.shape[1] >= 2:
            print(f"   \n📊 Static Features:")
            print(f"      - inflation_ticker: {len(np.unique(static[:, 0]))} unique")
            print(f"      - inflation_category: {len(np.unique(static[:, 1]))} unique")
        else:
            print(f"   static: placeholder (no augmentation)")
        
        # 3. Split data temporally
        print("\n[3/7] Splitting data...")
        n_samples = len(X)
        train_ratio = self.config['data']['train_ratio']
        val_ratio = self.config['data']['val_ratio']
        
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        X_train = X[:train_end]
        X_val = X[train_end:val_end]
        X_test = X[val_end:]
        
        y_train = y[:train_end]
        y_val = y[train_end:val_end]
        y_test = y[val_end:]
        
        ts_train = ts[:train_end]
        ts_val = ts[train_end:val_end]
        ts_test = ts[val_end:]
        
        static_train = static[:train_end]
        static_val = static[train_end:val_end]
        static_test = static[val_end:]
        
        print(f"  train: {len(X_train):,} samples ({len(X_train)/n_samples*100:.1f}%)")
        print(f"  val  : {len(X_val):,} samples ({len(X_val)/n_samples*100:.1f}%)")
        print(f"  test : {len(X_test):,} samples ({len(X_test)/n_samples*100:.1f}%)")
        
        # 4. Normalize per-split (fit on train only)
        print("\n[4/7] Normalizing...")
        X_train_norm, X_val_norm, X_test_norm, y_train_norm, y_val_norm, y_test_norm = self._normalize_splits(
            X_train, X_val, X_test, y_train, y_val, y_test
        )
        
        splits = {
            'train': (X_train_norm, y_train_norm, ts_train, static_train),
            'val': (X_val_norm, y_val_norm, ts_val, static_val),
            'test': (X_test_norm, y_test_norm, ts_test, static_test)
        }
        
        # 5. Export raw validation data
        print("\n[5/7] Exporting raw data...")
        self._export_raw_validation(df_pivoted)
        
        # 6. Save processed data
        print("\n[6/7] Saving processed data...")
        self.save_processed_data(splits, df_pivoted)
        
        print("\n[7/7] ✅ Complete!\n")
        
        return splits
    
    def create_sequences(self, df: pd.DataFrame) -> Tuple[List, List, List, List, pd.DataFrame]:
        """Create lookback sequences and multi-horizon targets.
        
        This includes:
        1. Pivoting data (one row per timestamp)
        2. Optional group feature aggregation
        3. Optional augmentation
        4. Sequence creation with multi-horizon targets
        
        Returns:
            X: [num_samples, lookback, num_features]
            y: [num_samples, num_horizons]
            ts: [num_samples] timestamps
            static: [num_samples, 2] static features (inflation_ticker, inflation_category)
            df_pivoted: Pivoted dataframe for export
        """
        print(f"\n{'='*80}")
        print("  PIVOTING TO WIDE FORMAT")
        print(f"{'='*80}\n")
        
        # Pivot: one row per timestamp
        print(f"Pivoting {len(df):,} rows...")
        df_pivoted = self._pivot_to_wide(df)
        print(f"✅ Pivoted: {df_pivoted.shape}")
        print(f"   Columns: {len(df_pivoted.columns)}\n")
        
        # Apply group feature aggregation if configured
        feature_type = self.config.get('model', {}).get('feature_type', 'raw')
        print(f"🔍 DEBUG: feature_type = '{feature_type}' (checking if == 'group_signals')\n")
        
        if feature_type == 'group_signals':
            print(f"🔄 Applying group-level feature aggregation...\n")
            aggregator = TickerGroupFeatureAggregator(self.config_path)
            group_features_df = aggregator.compute_group_features(df_pivoted)
            
            # Merge group features with timestamp (use concat to avoid fragmentation)
            merge_dfs = [pd.DataFrame({'timestamp': df_pivoted['timestamp']}), group_features_df]
            
            # Preserve target basket for Y labels
            if 'target_basket_close' in df_pivoted.columns:
                merge_dfs.append(pd.DataFrame({'target_basket_close': df_pivoted['target_basket_close']}))
            
            # Only preserve augmentation group tickers (not all tickers)
            # This prevents feature explosion when using group_signals
            if self.use_augmentation:
                augment_config = self.config['data'].get('ticker_augment_groups', [])
                if augment_config:
                    # Get tickers from augmentation groups
                    augment_tickers = set()
                    ticker_groups = self.config['data'].get('ticker_groups', {})
                    for group_name in augment_config:
                        if group_name in ticker_groups:
                            augment_tickers.update(ticker_groups[group_name].get('tickers', []))
                    
                    ticker_cols = {}
                    for ticker in augment_tickers:
                        for feat in ['close', 'volume', 'sma_50', 'sma_200']:
                            col_name = f"{feat}_{ticker}"
                            if col_name in df_pivoted.columns:
                                ticker_cols[col_name] = df_pivoted[col_name]
                    
                    if ticker_cols:
                        merge_dfs.append(pd.DataFrame(ticker_cols))
                        print(f"   ⚠️  Preserved {len(augment_tickers)} augmentation tickers ({len(ticker_cols)} features) for data augmentation")
            else:
                print(f"   ✅ Using group signals only (no individual tickers preserved)")
            
            # Concat all at once
            df_pivoted = pd.concat(merge_dfs, axis=1)
            print(f"   ✅ Generated {len(group_features_df.columns)} group features")
        
        # Add time features
        df_pivoted = self.loader.add_time_features(df_pivoted)
        
        # Log final feature list
        feature_cols = [col for col in df_pivoted.columns if col not in ['timestamp', 'date']]
        print(f"\n✅ Final feature set: {len(feature_cols)} features")
        if feature_type != 'group_signals':
            print(f"   Feature categories:")
            
            # Augmentation context features
            augment_feats = [f for f in feature_cols if f.endswith('_inflation')]
            
            # Ticker features (exclude augmentation context)
            ticker_feats = [f for f in feature_cols 
                           if any(x in f for x in ['close_', 'volume_', 'sma_']) 
                           and not f.endswith('_inflation')]
            
            # GDELT features
            gdelt_feats = [f for f in feature_cols 
                          if 'sentiment' in f or f in ['weighted_avg_tone', 'weighted_avg_polarity', 'num_articles', 'num_sources']]
            
            # Time features
            time_feats = [f for f in feature_cols 
                         if any(x in f for x in ['hour', 'day', 'month', 'weekend'])]
            
            # Target basket
            target_feats = [f for f in feature_cols if 'target_basket' in f]
            
            # Count
            print(f"   - Ticker features: {len(ticker_feats)} (individual ticker OHLCV)")
            print(f"   - Augmentation context: {len(augment_feats)} ({', '.join(augment_feats)})")
            print(f"   - GDELT sentiment: {len(gdelt_feats)}")
            print(f"   - Time features: {len(time_feats)}")
            print(f"   - Target basket: {len(target_feats)} ({', '.join(target_feats)})")
            print(f"   Total: {len(ticker_feats)} + {len(augment_feats)} + {len(gdelt_feats)} + {len(time_feats)} + {len(target_feats)} = {len(feature_cols)}")
        print()
        
        # Add 'date' column as alias for 'timestamp' (use assign to avoid fragmentation)
        if 'timestamp' in df_pivoted.columns and 'date' not in df_pivoted.columns:
            df_pivoted = df_pivoted.assign(date=df_pivoted['timestamp'])
        
        # Apply augmentation if enabled
        if self.use_augmentation:
            print(f"\n{'='*80}")
            print("  APPLYING AUGMENTATION")
            print(f"{'='*80}\n")
            augmenter = DataAugmenter(self.config_path)
            df_pivoted = augmenter.augment(df_pivoted, all_tickers=self.loader.tickers)
            
            # Log all columns after augmentation
            print(f"\n✅ Augmentation complete. Columns in augmented dataframe ({len(df_pivoted.columns)}):")
            print(f"\n   Meta columns (2):")
            meta_cols = [c for c in df_pivoted.columns if c in ['timestamp', 'date', 'inflation_ticker', 'inflation_category']]
            for col in sorted(meta_cols):
                print(f"      - {col}")
            
            print(f"\n   Feature columns ({len(df_pivoted.columns) - len(meta_cols)}):")
            feature_cols = [c for c in df_pivoted.columns if c not in meta_cols]
            
            # Group by prefix for readability
            ticker_cols = sorted([c for c in feature_cols if any(x in c for x in ['close_', 'volume_', 'sma_']) and not c.endswith('_inflation')])
            augment_cols = sorted([c for c in feature_cols if c.endswith('_inflation')])
            gdelt_cols = sorted([c for c in feature_cols if 'sentiment' in c or c in ['weighted_avg_tone', 'weighted_avg_polarity', 'num_articles', 'num_sources']])
            target_cols = sorted([c for c in feature_cols if 'target_basket' in c])
            time_cols = sorted([c for c in feature_cols if any(x in c for x in ['hour', 'day', 'month', 'weekend'])])
            
            if ticker_cols:
                print(f"\n      Ticker features ({len(ticker_cols)}):")
                # Show first 10 and last 5
                for col in ticker_cols[:10]:
                    print(f"         - {col}")
                if len(ticker_cols) > 15:
                    print(f"         ... ({len(ticker_cols) - 15} more) ...")
                for col in ticker_cols[-5:]:
                    print(f"         - {col}")
            
            if augment_cols:
                print(f"\n      Augmentation context ({len(augment_cols)}):")
                for col in augment_cols:
                    print(f"         - {col}")
            
            if target_cols:
                print(f"\n      Target basket ({len(target_cols)}):")
                for col in target_cols:
                    print(f"         - {col}")
            
            if gdelt_cols:
                print(f"\n      GDELT sentiment ({len(gdelt_cols)}):")
                for col in gdelt_cols:
                    print(f"         - {col}")
            
            if time_cols:
                print(f"\n      Time features ({len(time_cols)}):")
                for col in time_cols:
                    print(f"         - {col}")
            
            print()
        
        # Create sequences
        print(f"\n{'='*80}")
        print("  CREATING SEQUENCES")
        print(f"{'='*80}\n")
        
        X, y, ts, static = self._create_sequences_from_dataframe(df_pivoted)
        
        return X, y, ts, static, df_pivoted
    
    # ========== Helper Methods ==========
    
    def _pivot_to_wide(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pivot long format to wide format (one row per timestamp)."""
        if 'ticker' not in df.columns:
            return df
        
        # Get unique tickers and timestamps
        tickers = sorted(df['ticker'].unique())
        timestamps = sorted(df['timestamp'].unique())
        
        # Start with base dataframe
        result_dfs = [pd.DataFrame({'timestamp': timestamps})]
        
        # Pivot ticker-specific features (build all columns, then concat once)
        ticker_features = self.loader.raw_features + self.loader.synthetic_features
        ticker_dfs = {}
        for ticker in tickers:
            ticker_data = df[df['ticker'] == ticker].set_index('timestamp')
            ticker_cols = {}
            for feat in ticker_features:
                if feat in df.columns:
                    col_name = f'{feat}_{ticker}'
                    ticker_cols[col_name] = pd.Series(timestamps).map(ticker_data[feat]).values
            if ticker_cols:
                ticker_dfs[ticker] = pd.DataFrame(ticker_cols, index=range(len(timestamps)))
        
        if ticker_dfs:
            result_dfs.extend(ticker_dfs.values())
        
        # Add shared features (GDELT) - build dict, then create DataFrame
        if self.loader.use_gdelt:
            gdelt_features = self.loader.gdelt_features.copy()
            if self.loader.gdelt_include_lags:
                for lag in self.loader.gdelt_lag_periods:
                    gdelt_features.append(f'sentiment_lag_{lag}')
            
            gdelt_cols = {}
            for feat in gdelt_features:
                if feat in df.columns:
                    feat_values = df.groupby('timestamp')[feat].first()
                    gdelt_cols[feat] = pd.Series(timestamps).map(feat_values).values
            
            if gdelt_cols:
                result_dfs.append(pd.DataFrame(gdelt_cols, index=range(len(timestamps))))
        
        # Add target basket
        if 'target_basket_close' in df.columns:
            basket_values = df.groupby('timestamp')['target_basket_close'].first()
            target_df = pd.DataFrame({
                'target_basket_close': pd.Series(timestamps).map(basket_values).values
            }, index=range(len(timestamps)))
            result_dfs.append(target_df)
        
        # Concat all at once (much faster than iterative assignment)
        grouped = pd.concat(result_dfs, axis=1)
        
        return grouped
    
    def _create_sequences_from_dataframe(self, df: pd.DataFrame) -> Tuple[List, List, List, List]:
        """Create sequences from pivoted dataframe.
        
        Returns:
            X, y, ts, static: Feature sequences, targets, timestamps, and static features
        """
        # Determine target column
        if 'target_basket_close' in df.columns:
            target_col = 'target_basket_close'
        elif 'close_inflation' in df.columns:
            target_col = 'close_inflation'
        else:
            # Find first close column
            close_cols = [c for c in df.columns if c.startswith('close_')]
            target_col = close_cols[0] if close_cols else 'close'
        
        print(f"\n{'='*80}")
        print("  Y LABEL COMPUTATION (Multi-Horizon Forward Returns)")
        print(f"{'='*80}")
        
        # Show which tickers compose the target
        target_config = self.config['data'].get('target', {})
        target_tickers = target_config.get('basket_tickers', [])
        target_group = target_config.get('group', 'unknown')
        aggregation = target_config.get('aggregation', 'mean')
        
        print(f"   Target group: {target_group}")
        print(f"   Target tickers: {', '.join(target_tickers)} ({len(target_tickers)} tickers)")
        print(f"   Aggregation: {aggregation} of basket tickers")
        print(f"   Target column: {target_col}")
        print(f"\n   Prediction horizons: {self.horizons} periods ahead")
        print(f"   Formula: y[h] = (price[t+h] - price[t]) / price[t]")
        print(f"\n   Example:")
        print(f"     If current {target_group} basket price = $100")
        for h in self.horizons:
            print(f"       y[{h}d] = (basket_price[t+{h}] - $100) / $100")
        print(f"\n   This gives % returns for {target_group} basket at each horizon")
        print(f"{'='*80}\n")
        
        # Get feature columns (exclude timestamp, date, static features)
        feature_cols = [c for c in df.columns if c not in ['timestamp', 'date', 'inflation_ticker', 'inflation_category']]
        feature_data = df[feature_cols].values
        target_prices = df[target_col].values if target_col in df.columns else df['close'].values
        timestamps = df['timestamp' if 'timestamp' in df.columns else 'date'].values
        
        # Extract static features if available
        has_static = 'inflation_ticker' in df.columns and 'inflation_category' in df.columns
        if has_static:
            # Create ticker -> ID mapping
            unique_tickers = df['inflation_ticker'].unique()
            ticker_to_id = {ticker: idx for idx, ticker in enumerate(unique_tickers)}
            
            # Create category -> ID mapping
            unique_categories = df['inflation_category'].unique()
            category_to_id = {cat: idx for idx, cat in enumerate(unique_categories)}
            
            static_ticker = df['inflation_ticker'].map(ticker_to_id).values
            static_category = df['inflation_category'].map(category_to_id).values
        
        max_horizon = max(self.horizons)
        X, y, ts, static = [], [], [], []
        
        print(f"   Creating sequences...")
        print(f"   Lookback: {self.lookback} periods")
        print(f"   Max horizon: {max_horizon} periods")
        print(f"   Available samples: {len(df)} rows")
        print(f"   Valid range: [{self.lookback}, {len(df) - max_horizon}]\n")
        
        for i in range(self.lookback, len(df) - max_horizon):
            X.append(feature_data[i - self.lookback:i])
            
            current_price = target_prices[i]
            targets = [(target_prices[i + h] - current_price) / current_price for h in self.horizons]
            y.append(targets)
            
            ts.append(timestamps[i])
            
            # Add static features for this sequence
            if has_static:
                static.append([static_ticker[i], static_category[i]])
            else:
                static.append([0, 0])  # Placeholder
        
        # Show sample Y labels
        if len(y) > 0:
            print(f"   Sample Y labels (first sequence):")
            print(f"     Current price: ${target_prices[self.lookback]:.2f}")
            for idx, h in enumerate(self.horizons):
                future_price = target_prices[self.lookback + h]
                return_pct = y[0][idx] * 100
                print(f"     {h}d ahead: ${future_price:.2f} → {return_pct:+.2f}% return")
        
        return X, y, ts, static
    
    def _normalize_splits(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Normalize splits (fit on train only to prevent leakage)."""
        print(f"\n{'='*80}")
        print("  NORMALIZING")
        print(f"{'='*80}\n")
        
        # Normalize features
        n_features = X_train.shape[2]
        print(f"  Normalizing {n_features} features")
        
        # Skip time features
        skip_features = ['is_weekend', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos']
        
        # Normalize each feature
        for feat_idx in range(n_features):
            # Reshape to 2D for scaler
            train_feat = X_train[:, :, feat_idx].reshape(-1, 1)
            
            scaler = StandardScaler()
            scaler.fit(train_feat)
            
            # Transform all splits
            X_train[:, :, feat_idx] = scaler.transform(train_feat).reshape(X_train.shape[0], X_train.shape[1])
            X_val[:, :, feat_idx] = scaler.transform(X_val[:, :, feat_idx].reshape(-1, 1)).reshape(X_val.shape[0], X_val.shape[1])
            X_test[:, :, feat_idx] = scaler.transform(X_test[:, :, feat_idx].reshape(-1, 1)).reshape(X_test.shape[0], X_test.shape[1])
        
        print(f"  ✅ Normalized {n_features} features")
        
        # Normalize targets (flatten all horizons → fit single scaler → reshape)
        # This matches decoder v1's normalization strategy
        print(f"\n  Normalizing targets (y) - fitting scaler on multi-horizon returns...\n")
        y_scaler = StandardScaler()
        
        # Flatten all horizons together for fitting
        y_train_flat = y_train.flatten().reshape(-1, 1)  # (n_samples * n_horizons, 1)
        y_scaler.fit(y_train_flat)
        
        print(f"     Scaler mean: {y_scaler.mean_[0]:.6f}, std: {y_scaler.scale_[0]:.6f}")
        print(f"     Fitted on {y_train_flat.shape[0]} values (all horizons combined)")
        
        # Transform each split (flatten → transform → reshape)
        y_train_norm = y_scaler.transform(y_train.flatten().reshape(-1, 1)).reshape(y_train.shape)
        y_val_norm = y_scaler.transform(y_val.flatten().reshape(-1, 1)).reshape(y_val.shape)
        y_test_norm = y_scaler.transform(y_test.flatten().reshape(-1, 1)).reshape(y_test.shape)
        
        self.target_scaler = y_scaler
        
        print(f"  ✅ Normalized targets (single scaler for all horizons)")
        
        return X_train, X_val, X_test, y_train_norm, y_val_norm, y_test_norm
    
    def save_processed_data(self, splits, df_raw):
        """Save processed data to data/processed/."""
        output_dir = Path('data/processed')
        
        # Clear existing
        if output_dir.exists():
            print(f"🗑️  Clearing existing processed data...")
            shutil.rmtree(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*80}")
        print("   Saving Processed Data (NumPy Arrays)")
        print(f"{'='*80}\n")
        print(f"💾 Output directory: {output_dir}/")
        
        # Save each split
        for split_name, (X, y, ts, static) in splits.items():
            print(f"\n  💾 Saving {split_name.upper()} split ({len(X):,} sequences)...")
            print(f"      X: {X.shape}, y: {y.shape}, static: {static.shape}")
            np.save(output_dir / f'X_{split_name}.npy', X)
            np.save(output_dir / f'static_{split_name}.npy', static)
            np.save(output_dir / f'y_{split_name}.npy', y)
            np.save(output_dir / f'ts_{split_name}.npy', ts)
        
        # Save scalers
        if hasattr(self, 'target_scaler'):
            with open(output_dir / 'scalers.pkl', 'wb') as f:
                pickle.dump({'target_scaler': self.target_scaler}, f)
        
        # Save feature names
        feature_cols = [c for c in df_raw.columns if c not in ['timestamp', 'date', 'inflation_ticker', 'inflation_category']]
        with open(output_dir / 'feature_names.txt', 'w') as f:
            for feat in feature_cols:
                f.write(f"{feat}\n")
        
        # Save metadata.yaml
        from datetime import datetime
        
        # Get date range from timestamps
        all_timestamps = np.concatenate([splits['train'][2], splits['val'][2], splits['test'][2]])
        start_date = pd.Timestamp(all_timestamps.min()).strftime('%Y-%m-%d')
        end_date = pd.Timestamp(all_timestamps.max()).strftime('%Y-%m-%d')
        
        metadata = {
            'generated_at': datetime.now().isoformat(),
            'dataset_version': 'unknown',  # Will be set by generate_dataset.py
            'model_type': self.config.get('model', {}).get('type', 'tft-augmented'),
            'data': {
                'start_date': start_date,
                'end_date': end_date,
                'lookback_window': self.lookback,
                'prediction_horizons': self.horizons,
                'num_features': len(feature_cols),
                'train_samples': len(splits['train'][0]),
                'val_samples': len(splits['val'][0]),
                'test_samples': len(splits['test'][0]),
            },
            'features': {
                'feature_type': self.config.get('model', {}).get('feature_type', 'individual'),
                'use_group_features': self.use_group_features,
                'use_augmentation': self.use_augmentation,
                'static_features': 2 if 'inflation_ticker' in df_raw.columns else 0,
            },
            'normalization': {
                'method': 'StandardScaler',
                'fit_on': 'train_split_only',
            }
        }
        
        with open(output_dir / 'metadata.yaml', 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
        
        print(f"\n{'='*80}")
        print(f"✅ ALL DATA SAVED TO: {output_dir}/")
        print(f"{'='*80}\n")
    
    def _export_raw_validation(self, df_raw):
        """Export raw validation data to data/raw/."""
        print(f"💾 Exporting validation data (raw, non-normalized)...")
        raw_dir = Path('data/raw')
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Export as CSV only (parquet not needed)
        df_raw.to_csv(raw_dir / 'tft_features.csv', index=False)
        
        print(f"  ✅ CSV: {raw_dir}/tft_features.csv")