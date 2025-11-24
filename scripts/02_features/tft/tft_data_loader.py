#!/usr/bin/env python3
"""
TFT Data Loader

Fetches  OHLCV and synthetic indicators from BigQuery,
prepares time series data for TFT multi-horizon quantile forecasting.
"""

import os
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from google.cloud import bigquery
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader


class MultiTickerDataLoader:
    """Load and prepare multi-ticker data from BigQuery for TFT training."""
    
    def __init__(self, config_path: str = 'configs/model_tft_config.yaml', export_temp: bool = False):
        """Initialize data loader with configuration.
        
        Args:
            config_path: Path to config YAML file
            export_temp: If True, export raw data to temp/ directory (for debugging)
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.project_id = os.getenv('GCP_PROJECT_ID')
        if not self.project_id:
            raise ValueError("GCP_PROJECT_ID environment variable not set")
        
        self.client = bigquery.Client(project=self.project_id)
        
        # Extract config
        self.tickers = self.config['data']['tickers']['symbols']
        self.frequency = self.config['data']['tickers']['frequency']
        self.start_date = self.config['data']['start_date']
        self.end_date = self.config['data']['end_date']
        self.lookback = self.config['data']['lookback_window']
        self.horizons = self.config['data']['prediction_horizons']
        
        self.raw_features = self.config['data']['tickers']['raw_features']
        self.synthetic_features = self.config['data']['tickers']['synthetic_features']
        self.target_col = self.config['data']['target']
        
        # GDELT config
        self.use_gdelt = self.config['data']['gdelt'].get('enabled', False)
        if self.use_gdelt:
            self.gdelt_frequency = self.config['data']['gdelt'].get('frequency', self.frequency)  # Use GDELT-specific frequency
            self.gdelt_topic_groups = self.config['data']['gdelt'].get('topic_groups', ['inflation_prices'])  # Topic groups to load
            self.gdelt_features = self.config['data']['gdelt']['features']
            self.gdelt_normalize_counts = self.config['data']['gdelt'].get('normalize_counts', True)
            self.gdelt_include_lags = self.config['data']['gdelt'].get('include_lags', True)
            self.gdelt_lag_periods = self.config['data']['gdelt'].get('lag_periods', [1, 4, 16])
        
        # Normalization
        self.normalize = self.config['data']['normalize']
        self.norm_method = self.config['data']['normalization_method']
        self.scalers = {}
        
        # Weekend filtering
        self.skip_weekends = self.config['data'].get('skip_weekends', False)
        
        # Forward filling config
        self.forward_fill_config = self.config['data'].get('forward_fill', {})
        self.forward_fill_enabled = self.forward_fill_config.get('enabled', True)
        self.forward_fill_log_stats = self.forward_fill_config.get('log_stats', True)
        self.forward_fill_max_limit = self.forward_fill_config.get('max_fill_limit', 5)
        
        # Export settings
        self.export_temp = export_temp
        
    def fetch_ticker_data(self) -> pd.DataFrame:
        """Fetch combined raw OHLCV and synthetic indicators from BigQuery for all tickers."""
        dataset_id = self.config['bigquery']['dataset_id']
        raw_table = self.config['bigquery']['ticker']['raw_table']
        synthetic_table = self.config['bigquery']['ticker']['synthetic_table']
        
        print(f"Fetching data for {len(self.tickers)} ticker(s): {', '.join(self.tickers)}")
        print(f"Frequency: {self.frequency}")
        print(f"Date range: {self.start_date} to {self.end_date}")
        
        # Build feature lists for SQL
        raw_cols = ', '.join([f'r.{col}' for col in self.raw_features])
        synthetic_cols = ', '.join([f's.{col}' for col in self.synthetic_features])
        
        # Build ticker list for SQL IN clause
        ticker_list = "', '".join(self.tickers)
        
        query = f"""
        SELECT 
            r.ticker,
            r.timestamp,
            r.date,
            {raw_cols},
            {synthetic_cols}
        FROM `{self.project_id}.{dataset_id}.{raw_table}` r
        INNER JOIN `{self.project_id}.{dataset_id}.{synthetic_table}` s
            ON r.ticker = s.ticker
            AND r.timestamp = s.timestamp
            AND r.frequency = s.frequency
        WHERE r.ticker IN ('{ticker_list}')
            AND r.frequency = '{self.frequency}'
            AND DATE(r.timestamp) BETWEEN '{self.start_date}' AND '{self.end_date}'
        ORDER BY r.ticker, r.timestamp
        """
        
        df = self.client.query(query).to_dataframe()
        print(f"✅ Fetched {len(df):,} total rows")
        for ticker in self.tickers:
            ticker_rows = len(df[df['ticker'] == ticker])
            print(f"   {ticker}: {ticker_rows:,} rows")
        
        return df
    
    def fetch_gdelt_data(self) -> pd.DataFrame:
        """Fetch GDELT sentiment data from BigQuery for specified topic groups."""
        if not self.use_gdelt:
            return None
        
        dataset_id = self.config['bigquery']['dataset_id']
        gdelt_table = self.config['bigquery']['gdelt']['table']
        
        print(f"Fetching GDELT sentiment data ({self.gdelt_frequency})...")
        print(f"  Topic groups: {', '.join(self.gdelt_topic_groups)}")
        
        # Build feature list for SQL
        gdelt_cols = ', '.join([f'g.{col}' for col in self.gdelt_features])
        
        # Build topic group filter
        topic_group_list = "', '".join(self.gdelt_topic_groups)
        
        query = f"""
        SELECT 
            g.timestamp,
            g.topic_group_id,
            {gdelt_cols}
        FROM `{self.project_id}.{dataset_id}.{gdelt_table}` g
        WHERE g.frequency = '{self.gdelt_frequency}'
            AND g.topic_group_id IN ('{topic_group_list}')
            AND DATE(g.timestamp) BETWEEN '{self.start_date}' AND '{self.end_date}'
        ORDER BY g.timestamp, g.topic_group_id
        """
        
        df = self.client.query(query).to_dataframe()
        
        if len(df) == 0:
            print(f"⚠️  Warning: No GDELT data found for topic groups: {', '.join(self.gdelt_topic_groups)}")
            print(f"   Make sure data is loaded for these topic groups at frequency '{self.gdelt_frequency}'")
            return None
        
        print(f"✅ Fetched {len(df):,} GDELT rows")
        
        # Show breakdown by topic group
        for topic_group in self.gdelt_topic_groups:
            count = len(df[df['topic_group_id'] == topic_group])
            print(f"   {topic_group}: {count:,} rows")
        
        # If multiple topic groups, aggregate them (average sentiment across groups)
        if len(self.gdelt_topic_groups) > 1:
            print(f"  Aggregating {len(self.gdelt_topic_groups)} topic groups (averaging sentiment)...")
            # Group by timestamp and average the sentiment features
            agg_dict = {col: 'mean' for col in self.gdelt_features}
            df = df.groupby('timestamp').agg(agg_dict).reset_index()
            print(f"  ✅ Aggregated to {len(df):,} rows")
        else:
            # Single topic group - just drop the topic_group_id column
            df = df.drop(columns=['topic_group_id'])
        
        return df
    
    def fetch_agriculture_basket(self) -> pd.DataFrame:
        """Fetch agriculture basket (WEAT, SOYB, RJA) for target computation."""
        dataset_id = self.config['bigquery']['dataset_id']
        raw_table = self.config['bigquery']['ticker']['raw_table']
        
        agriculture_tickers = ['WEAT', 'SOYB', 'RJA']
        ticker_list = "', '".join(agriculture_tickers)
        
        print(f"\nFetching agriculture basket for target: {', '.join(agriculture_tickers)}...")
        
        query = f"""
        SELECT 
            timestamp,
            ticker,
            close
        FROM `{self.project_id}.{dataset_id}.{raw_table}`
        WHERE ticker IN ('{ticker_list}')
            AND frequency = '{self.frequency}'
            AND DATE(timestamp) BETWEEN '{self.start_date}' AND '{self.end_date}'
        ORDER BY timestamp, ticker
        """
        
        df = self.client.query(query).to_dataframe()
        
        if len(df) == 0:
            print("⚠️  Warning: No agriculture basket data found!")
            print(f"   Make sure WEAT, SOYB, RJA are loaded for frequency '{self.frequency}'")
            return None
        
        # Pivot to get one column per ticker
        df_pivot = df.pivot(index='timestamp', columns='ticker', values='close')
        
        # Compute average close price across available tickers (skip NaN)
        # This ensures we average only available tickers if some are missing
        df_pivot['agriculture_basket_close'] = df_pivot[agriculture_tickers].mean(axis=1, skipna=True)
        
        # Count how many tickers contributed to each average
        df_pivot['num_tickers_available'] = df_pivot[agriculture_tickers].notna().sum(axis=1)
        
        # Keep only the average column
        result = df_pivot[['agriculture_basket_close']].reset_index()
        
        print(f"✅ Fetched {len(result):,} agriculture basket rows")
        for ticker in agriculture_tickers:
            if ticker in df_pivot.columns:
                count = df_pivot[ticker].notna().sum()
                print(f"   {ticker}: {count:,} rows")
        
        # Show statistics on ticker availability
        ticker_counts = df_pivot['num_tickers_available'].value_counts().sort_index()
        print(f"\n   Ticker availability per timestamp:")
        for num_tickers, count in ticker_counts.items():
            print(f"      {int(num_tickers)} ticker(s): {count:,} timestamps ({count/len(result)*100:.1f}%)")
        
        return result
    
    def compute_basket_target(self, ticker_df: pd.DataFrame, basket_df: pd.DataFrame) -> pd.DataFrame:
        """Join agriculture basket close prices to ticker data for target computation."""
        if basket_df is None:
            print("⚠️  Warning: No agriculture basket data - cannot compute target!")
            return ticker_df
        
        print("\nJoining agriculture basket for target computation...")
        
        # Left join to preserve all ticker timestamps
        df = ticker_df.merge(basket_df, on='timestamp', how='left')
        
        # Forward fill missing basket values
        missing_before = df['agriculture_basket_close'].isnull().sum()
        if missing_before > 0:
            df = self.forward_fill_with_stats(df, ['agriculture_basket_close'], context="Agriculture basket")
        
        print(f"✅ Joined agriculture basket, shape: {df.shape}")
        
        return df
    
    def filter_weekends(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove weekend data (Saturday/Sunday) if configured."""
        if not self.skip_weekends:
            return df
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        initial_rows = len(df)
        # Filter out weekends (dayofweek: 5=Saturday, 6=Sunday)
        df = df[df['timestamp'].dt.dayofweek < 5].copy()
        removed_rows = initial_rows - len(df)
        
        if removed_rows > 0:
            print(f"🗓️  Filtered out {removed_rows:,} weekend rows ({removed_rows/initial_rows*100:.1f}%)")
            print(f"   Remaining: {len(df):,} weekday rows")
        
        return df
    
    def forward_fill_with_stats(self, df: pd.DataFrame, columns: List[str], context: str = "") -> pd.DataFrame:
        """Forward fill missing values with detailed statistics logging.
        
        Args:
            df: DataFrame to fill
            columns: List of column names to forward fill
            context: Description for logging (e.g., 'GDELT features', 'All features')
        
        Returns:
            DataFrame with forward filled values
        """
        if not self.forward_fill_enabled:
            return df
        
        df = df.copy()
        fill_stats = {}
        total_filled = 0
        
        for col in columns:
            if col not in df.columns:
                continue
            
            # Count missing before
            missing_before = df[col].isnull().sum()
            if missing_before == 0:
                continue
            
            # Forward fill with limit
            df[col] = df[col].ffill(limit=self.forward_fill_max_limit)
            
            # Count missing after
            missing_after = df[col].isnull().sum()
            filled_count = missing_before - missing_after
            total_filled += filled_count
            
            fill_stats[col] = {
                'missing_before': missing_before,
                'filled': filled_count,
                'still_missing': missing_after,
                'pct_filled': (filled_count / missing_before * 100) if missing_before > 0 else 0
            }
        
        # Log statistics if enabled
        if self.forward_fill_log_stats and fill_stats:
            print(f"\n🔧 Forward Fill Statistics{' (' + context + ')' if context else ''}:")
            print(f"   Max consecutive fills: {self.forward_fill_max_limit} periods")
            print(f"   Total values filled: {total_filled:,}")
            print(f"\n   {'Column':<30} {'Missing':<10} {'Filled':<10} {'Still Missing':<15} {'% Filled':<10}")
            print(f"   {'-'*85}")
            
            for col, stats in fill_stats.items():
                print(f"   {col:<30} {stats['missing_before']:<10,} {stats['filled']:<10,} "
                      f"{stats['still_missing']:<15,} {stats['pct_filled']:<10.1f}%")
            
            # Warn about columns that still have missing values
            still_missing_cols = [col for col, stats in fill_stats.items() if stats['still_missing'] > 0]
            if still_missing_cols:
                print(f"\n   ⚠️  Warning: {len(still_missing_cols)} column(s) still have missing values after forward fill:")
                for col in still_missing_cols:
                    print(f"      - {col}: {fill_stats[col]['still_missing']:,} missing")
        
        return df
    
    def join_gdelt_features(self, ticker_df: pd.DataFrame, gdelt_df: pd.DataFrame) -> pd.DataFrame:
        """Join GDELT features with ticker data."""
        if gdelt_df is None or not self.use_gdelt:
            return ticker_df
        
        print("Joining GDELT sentiment features...")
        
        # Left join to preserve all ticker timestamps
        df = ticker_df.merge(gdelt_df, on='timestamp', how='left')
        
        # Forward fill missing GDELT values (for gaps in sentiment data)
        df = self.forward_fill_with_stats(df, self.gdelt_features, context="GDELT features")
        
        # Normalize article/source counts if configured
        if self.gdelt_normalize_counts:
            if 'num_articles' in df.columns:
                df['num_articles'] = np.log1p(df['num_articles'])  # Log transform
            if 'num_sources' in df.columns:
                df['num_sources'] = np.log1p(df['num_sources'])  # Log transform
            print("✅ Normalized GDELT article/source counts (log1p)")
        
        # Add lagged sentiment features if configured
        if self.gdelt_include_lags and 'weighted_avg_tone' in df.columns:
            lag_cols = []
            for lag in self.gdelt_lag_periods:
                col_name = f'sentiment_lag_{lag}'
                df[col_name] = df['weighted_avg_tone'].shift(lag)
                lag_cols.append(col_name)
            
            # Forward fill NaN from initial lags
            df = self.forward_fill_with_stats(df, lag_cols, context="GDELT lagged features")
            print(f"✅ Added lagged sentiment features: {self.gdelt_lag_periods}")
        
        print(f"✅ Joined GDELT features, final shape: {df.shape}")
        
        return df
    
    def fetch_data(self) -> pd.DataFrame:
        """Fetch and combine all data sources."""
        # Fetch ticker data (OHLCV + indicators)
        ticker_df = self.fetch_ticker_data()
        
        # Fetch and join GDELT data if enabled
        if self.use_gdelt:
            gdelt_df = self.fetch_gdelt_data()
            df = self.join_gdelt_features(ticker_df, gdelt_df)
        else:
            df = ticker_df
            print("⚠️  GDELT features disabled in config")
        
        # Fetch and join agriculture basket for target computation
        basket_df = self.fetch_agriculture_basket()
        df = self.compute_basket_target(df, basket_df)
        
        # Filter weekends if configured (after joining all data)
        df = self.filter_weekends(df)
        
        # Check for remaining missing values
        missing = df.isnull().sum()
        if missing.any():
            missing_cols = missing[missing > 0].index.tolist()
            print(f"\n⚠️  Warning: {len(missing_cols)} column(s) have missing values:")
            for col in missing_cols[:10]:  # Show first 10
                print(f"   - {col}: {missing[col]:,} missing ({missing[col]/len(df)*100:.2f}%)")
            if len(missing_cols) > 10:
                print(f"   ... and {len(missing_cols)-10} more")
            
            # Forward fill remaining missing values with stats
            df = self.forward_fill_with_stats(df, missing_cols, context="Remaining features")
            
            # Backward fill for any remaining (at start of series)
            still_missing = df.isnull().sum()
            if still_missing.any():
                still_missing_cols = still_missing[still_missing > 0].index.tolist()
                print(f"\n🔙 Backward filling {len(still_missing_cols)} column(s) for initial NaNs...")
                df = df.bfill()
                print(f"✅ Backward fill complete")
        
        return df
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features (future-known)."""
        df = df.copy()
        
        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract time features (these are future-known)
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)  # Saturday=5, Sunday=6
        
        # Cyclical encoding (helps model learn periodicity)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        print(f"✅ Added time features: hour, day_of_week, month, cyclical encodings")
        print(f"   Total columns: {len(df.columns)}")
        
        return df
    
    def normalize_data(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Normalize features using standard or minmax scaling."""
        if not self.normalize:
            return df
        
        df = df.copy()
        
        # Get all features from config (time-varying known + unknown)
        time_varying_known = self.config['model'].get('time_varying_known', [])
        time_varying_unknown = self.config['model']['time_varying_unknown']
        all_features = list(time_varying_known) + list(time_varying_unknown)
        
        # Time features should NOT be normalized (they're already in good ranges)
        time_features_to_skip = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 
                                  'month_sin', 'month_cos', 'is_weekend',
                                  'hour', 'day_of_week', 'month', 'day_of_month']
        
        for col in all_features:
            if col not in df.columns:
                continue
            
            # Skip time features (already in good ranges)
            if col in time_features_to_skip:
                continue
            
            if fit:
                # Fit scaler on training data
                if self.norm_method == 'standard':
                    scaler = StandardScaler()
                else:
                    scaler = MinMaxScaler()
                
                df[col] = scaler.fit_transform(df[[col]])
                self.scalers[col] = scaler
            else:
                # Use existing scaler (for validation/test)
                if col in self.scalers:
                    df[col] = self.scalers[col].transform(df[[col]])
        
        return df
    
    def create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Create lookback sequences and multi-horizon targets.
        Groups by timestamp to ensure each unique date appears only once.
        
        Returns:
            X: [num_samples, lookback, num_features]
            y: [num_samples, num_horizons]
            timestamps: [num_samples]
            grouped: Pivoted dataframe (one row per date with ticker-specific columns)
        """
        # Combine all features: time-varying known + unknown + time features
        time_varying_known = self.config['model'].get('time_varying_known', [])
        time_varying_unknown = self.config['model']['time_varying_unknown']
        
        # Time features (if they exist in the dataframe)
        time_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos', 'is_weekend']
        available_time_features = [f for f in time_features if f in df.columns]
        
        # Combine all features: time_varying_known first, then time_varying_unknown
        all_features = list(time_varying_known) + list(time_varying_unknown)
        
        # Note: Don't filter here - pivoting will create ticker-specific columns
        print(f"📊 Config features: {len(all_features)} total ({len(time_varying_known)} known + {len(time_varying_unknown)} unknown)")
        
        # Pivot data: one row per timestamp with ticker-specific columns
        print(f"\n⚠️  Pivoting by timestamp to prevent data leakage...")
        print(f"   Before: {len(df):,} rows (multiple tickers per date)")
        
        # Identify ticker-specific vs shared features
        ticker_features = ['close', 'volume', 'sma_50', 'sma_200']
        gdelt_features = ['weighted_avg_tone', 'weighted_avg_polarity', 'num_articles', 
                         'num_sources', 'sentiment_lag_1', 'sentiment_lag_7', 'sentiment_lag_30']
        time_features = ['month_sin', 'month_cos', 'is_weekend']
        
        # Get unique tickers (sorted for consistency)
        if 'ticker' in df.columns:
            tickers = sorted(df['ticker'].unique())
            print(f"   Found {len(tickers)} tickers: {', '.join(tickers)}")
            
            # Start with timestamps
            timestamps = sorted(df['timestamp'].unique())
            grouped = pd.DataFrame({'timestamp': timestamps})
            
            # Pivot ticker-specific features
            for ticker in tickers:
                ticker_data = df[df['ticker'] == ticker].set_index('timestamp')
                for feat in ticker_features:
                    if feat in df.columns:
                        col_name = f'{feat}_{ticker}'
                        grouped[col_name] = grouped['timestamp'].map(ticker_data[feat])
            
            # Add shared features (GDELT and time) - take first value for each timestamp
            for feat in gdelt_features + time_features:
                if feat in df.columns:
                    feat_values = df.groupby('timestamp')[feat].first()
                    grouped[feat] = grouped['timestamp'].map(feat_values)
            
            # Add agriculture basket
            if 'agriculture_basket_close' in df.columns:
                basket_values = df.groupby('timestamp')['agriculture_basket_close'].first()
                grouped['agriculture_basket_close'] = grouped['timestamp'].map(basket_values)
            
            # Update all_features to include ticker-specific columns
            new_features = []
            for feat in all_features:
                if feat in ticker_features:
                    # Replace with ticker-specific versions
                    new_features.extend([f'{feat}_{ticker}' for ticker in tickers])
                else:
                    # Keep GDELT and time features as-is
                    new_features.append(feat)
            all_features = new_features
            
            # Save final features for metadata export
            self.final_features = all_features
            
            print(f"   ✅ Created {len(ticker_features) * len(tickers)} ticker-specific features")
            print(f"   Total features: {len(all_features)} = {len(ticker_features)*len(tickers)} ticker + {len(gdelt_features)} GDELT + {len(time_features)} time")
        else:
            # No ticker column, just group by timestamp (shouldn't happen)
            grouped = df.groupby('timestamp').first().reset_index()
            print(f"   ⚠️  No ticker column found, using first value per timestamp")
        
        # Store final feature list for metadata
        self.final_features = all_features
        
        print(f"   After: {len(grouped):,} rows (one per unique date)")
        print(f"   ✅ Each date now appears exactly once")
        
        feature_data = grouped[all_features].values
        
        # Get target prices for computing true multi-horizon returns
        # Use agriculture basket if available, otherwise use ticker close price
        if 'agriculture_basket_close' in grouped.columns:
            target_prices = grouped['agriculture_basket_close'].values
            print(f"🌾 Using agriculture basket (WEAT+SOYB+RJA avg) as target")
        else:
            target_prices = grouped['close'].values
            print(f"📊 Using ticker close price as target")
        
        timestamps = grouped['timestamp'].values
        
        max_horizon = max(self.horizons)
        
        X, y, ts = [], [], []
        
        # Create sequences (use grouped data length)
        for i in range(self.lookback, len(grouped) - max_horizon):
            # Input: lookback window of features
            X.append(feature_data[i - self.lookback:i])
            
            # Target: TRUE k-period forward returns from current price
            # Formula: (price[t+k] - price[t]) / price[t]
            current_price = target_prices[i]
            targets = [(target_prices[i + h] - current_price) / current_price for h in self.horizons]
            y.append(targets)
            
            # Timestamp of prediction point
            ts.append(timestamps[i])
        
        return np.array(X), np.array(y), np.array(ts), grouped
    
    def split_data(self, X: np.ndarray, y: np.ndarray, ts: np.ndarray
                   ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Split data into train/val/test sets."""
        n_samples = len(X)
        
        train_ratio = self.config['data']['train_ratio']
        val_ratio = self.config['data']['val_ratio']
        
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        splits = {
            'train': (X[:train_end], y[:train_end], ts[:train_end]),
            'val': (X[train_end:val_end], y[train_end:val_end], ts[train_end:val_end]),
            'test': (X[val_end:], y[val_end:], ts[val_end:])
        }
        
        print(f"\nData splits:")
        for split_name, (x, _, _) in splits.items():
            print(f"  {split_name:5s}: {len(x):,} samples ({x.shape[0]/n_samples*100:.1f}%)")
        
        return splits
    
    def save_processed_data(self, splits: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]], 
                           df_raw: pd.DataFrame):
        """Save train/val/test splits as numpy arrays (.npy) in data/processed."""
        import shutil
        import pickle
        from pathlib import Path
        
        output_dir = Path('data/processed')
        
        # Clear existing data
        if output_dir.exists():
            print(f"\n🗑️  Clearing existing processed data...")
            shutil.rmtree(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get feature list from config (same as create_sequences)
        # Use final_features if available (after pivoting), otherwise use config
        if hasattr(self, 'final_features'):
            all_features = self.final_features
            time_varying_known = self.config['model'].get('time_varying_known', [])
            time_varying_unknown = [f for f in all_features if f not in time_varying_known]
        else:
            time_varying_known = self.config['model'].get('time_varying_known', [])
            time_varying_unknown = self.config['model']['time_varying_unknown']
            all_features = list(time_varying_known) + list(time_varying_unknown)
        
        print(f"\n" + "="*80)
        print(f"   Saving Processed Data (NumPy Arrays)")
        print("="*80)
        print(f"\n💾 Output directory: {output_dir}/")
        print(f"   Features: {len(all_features)} total")
        print(f"     - {len(time_varying_known)} time-varying known (future known)")
        print(f"     - {len(time_varying_unknown)} time-varying unknown (past only)")
        print(f"   Prediction horizons: {self.horizons}")
        print(f"   Format: NumPy arrays (.npy) - optimized for PyTorch/TensorFlow")
        print()
        
        # Save each split as separate X, y, ts arrays
        for split_name, (X, y, ts) in splits.items():
            n_samples, lookback, n_features = X.shape
            n_horizons = y.shape[1] if len(y.shape) > 1 else 1
            
            print(f"  💾 Saving {split_name.upper()} split ({n_samples:,} sequences)...")
            
            # Save arrays (memory efficient - no copies)
            X_file = output_dir / f'X_{split_name}.npy'
            y_file = output_dir / f'y_{split_name}.npy'
            ts_file = output_dir / f'ts_{split_name}.npy'
            
            print(f"     Saving X_{split_name}.npy {X.shape}...", end='', flush=True)
            np.save(X_file, X)
            print(f" Done!")
            
            print(f"     Saving y_{split_name}.npy {y.shape}...", end='', flush=True)
            np.save(y_file, y)
            print(f" Done!")
            
            print(f"     Saving ts_{split_name}.npy {ts.shape}...", end='', flush=True)
            np.save(ts_file, ts)
            print(f" Done!")
            
            # Calculate file sizes
            X_size_mb = X_file.stat().st_size / 1024 / 1024
            y_size_mb = y_file.stat().st_size / 1024 / 1024
            ts_size_mb = ts_file.stat().st_size / 1024 / 1024
            total_split_mb = X_size_mb + y_size_mb + ts_size_mb
            
            print(f"  ✅ {split_name.upper():5s}:")
            print(f"       Sequences: {n_samples:,}")
            print(f"       X_{split_name}.npy: {X.shape} = {X_size_mb:.2f} MB")
            print(f"       y_{split_name}.npy: {y.shape} = {y_size_mb:.2f} MB")
            print(f"       ts_{split_name}.npy: {ts.shape} = {ts_size_mb:.2f} MB")
            print(f"       Total: {total_split_mb:.2f} MB")
            print()
        
        # Save scalers for inverse transform
        scalers_file = output_dir / 'scalers.pkl'
        with open(scalers_file, 'wb') as f:
            pickle.dump(self.scalers, f)
        
        scalers_size_kb = scalers_file.stat().st_size / 1024
        print(f"  💾 SCALERS: {scalers_file.name}")
        print(f"       File size: {scalers_size_kb:.2f} KB")
        print(f"       Contains: {len(self.scalers)} feature scalers + target scaler")
        print()
        
        # Save metadata
        metadata = {
            'created_at': datetime.now().isoformat(),
            'tickers': self.tickers,  # List of ticker symbols
            'frequency': self.frequency,
            'lookback_window': self.lookback,
            'prediction_horizons': self.horizons,
            'raw_features': self.raw_features,
            'synthetic_features': self.synthetic_features,
            'target': self.target_col,
            'normalization': self.norm_method if self.normalize else 'none',
            'train_samples': len(splits['train'][0]),
            'val_samples': len(splits['val'][0]),
            'test_samples': len(splits['test'][0]),
            'total_features': len(all_features),
            'features': all_features,  # Actual feature list (after pivoting)
            'time_varying_known': time_varying_known,
            'time_varying_unknown': time_varying_unknown,
            'array_shapes': {
                'X': f"[samples, {self.lookback}, {len(all_features)}]",
                'y': f"[samples, {len(self.horizons)}]",
                'ts': '[samples]'
            }
        }
        
        metadata_file = output_dir / 'metadata.yaml'
        with open(metadata_file, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)
        
        metadata_size_kb = metadata_file.stat().st_size / 1024
        
        print(f"  📋 METADATA: {metadata_file.name}")
        print(f"       File size: {metadata_size_kb:.2f} KB")
        print(f"       Contains: config, shapes, feature lists, sample counts")
        print()
        
        # Save feature names
        feature_file = output_dir / 'feature_names.txt'
        with open(feature_file, 'w') as f:
            f.write("TIME-VARYING KNOWN FEATURES:\n")
            for feat in time_varying_known:
                f.write(f"  - {feat}\n")
            f.write("\nTIME-VARYING UNKNOWN FEATURES:\n")
            for feat in time_varying_unknown:
                f.write(f"  - {feat}\n")
            f.write("\n# Usage Example:\n")
            f.write("# import numpy as np\n")
            f.write("# X_train = np.load('data/processed/X_train.npy')\n")
            f.write("# y_train = np.load('data/processed/y_train.npy')\n")
            f.write("# ts_train = np.load('data/processed/ts_train.npy')\n")
        
        feature_file_size_kb = feature_file.stat().st_size / 1024
        
        print(f"  📄 FEATURES: {feature_file.name}")
        print(f"       File size: {feature_file_size_kb:.2f} KB")
        print(f"       Contains: {len(time_varying_known)} known + {len(time_varying_unknown)} unknown features")
        print()
        
        # Calculate total directory size
        total_size_mb = sum(f.stat().st_size for f in output_dir.glob('*')) / 1024 / 1024
        
        print(f"" + "="*80)
        print(f"✅ All processed data saved successfully!")
        print(f"   Location: {output_dir}/")
        print(f"   Total size: {total_size_mb:.2f} MB")
        print(f"   Files: X_*.npy, y_*.npy, ts_*.npy (3 splits), scalers.pkl, metadata.yaml")
        print(f"\n💡 Load with: X = np.load('data/processed/X_train.npy')")
        print("="*80)
    
    def _export_raw_validation(self, df: pd.DataFrame):
        """Export raw validation data to data/raw (always called)."""
        print(f"\n💾 Exporting validation data (raw, non-normalized)...")
        
        raw_dir = Path('data/raw')
        raw_dir.mkdir(parents=True, exist_ok=True)
        self._export_raw_with_targets(df, raw_dir)
    
    def _export_raw_with_targets(self, df_raw: pd.DataFrame, output_dir: Path):
        """Export raw (non-normalized) pivoted data with computed target labels.
        
        df_raw is the grouped/pivoted dataframe with one row per date and ticker-specific columns.
        """
        df_raw = df_raw.copy()
        
        # Add target labels (TRUE k-period forward returns)
        # Use agriculture basket if available, otherwise use first ticker close
        if 'agriculture_basket_close' in df_raw.columns:
            target_price_col = 'agriculture_basket_close'
            print(f"  🌾 Computing targets from agriculture basket")
        else:
            # Find first ticker close column
            close_cols = [c for c in df_raw.columns if c.startswith('close_')]
            target_price_col = close_cols[0] if close_cols else 'close'
            print(f"  📊 Computing targets from {target_price_col}")
        
        # Formula: (price[t+k] - price[t]) / price[t]
        for horizon in self.horizons:
            col_name = f'target_{horizon}periods_ahead'
            df_raw[col_name] = (df_raw[target_price_col].shift(-horizon) - df_raw[target_price_col]) / df_raw[target_price_col]
        
        # Add helpful time columns if not present
        if 'hour' not in df_raw.columns:
            df_raw['hour'] = pd.to_datetime(df_raw['timestamp']).dt.hour
        if 'day_of_week' not in df_raw.columns:
            df_raw['day_of_week'] = pd.to_datetime(df_raw['timestamp']).dt.day_name()
        
        # Reorder columns for readability: timestamp, time info, then all features, then targets
        time_cols = ['timestamp', 'hour', 'day_of_week']
        target_cols = [c for c in df_raw.columns if c.startswith('target_')]
        feature_cols = [c for c in df_raw.columns if c not in time_cols and c not in target_cols]
        df_raw = df_raw[time_cols + feature_cols + target_cols]
        
        # Export to CSV and Parquet (clean naming)
        csv_file = output_dir / 'tft_features.csv'
        parquet_file = output_dir / 'tft_features.parquet'
        
        df_raw.to_csv(csv_file, index=False, float_format='%.8f')  # 8 decimals for small indicator values
        df_raw.to_parquet(parquet_file, index=False)
        
        print(f"  ✅ CSV:     {csv_file}")
        print(f"  ✅ Parquet: {parquet_file}")
        print(f"     Format: Raw prices (not normalized)")
        print(f"     Columns: timestamp, OHLCV, indicators, targets")
        print(f"     Targets: {', '.join(target_cols)}")
        print(f"     Period: {df_raw['timestamp'].iloc[0]} to {df_raw['timestamp'].iloc[-1]}")
        print(f"     Rows: {len(df_raw):,}")
        print(f"\n  📋 Sample (first row):")
        print(f"     Time: {df_raw['timestamp'].iloc[0]}")
        
        # After pivoting, close columns are ticker-specific (e.g., close_SPY, close_QQQ)
        close_cols = [col for col in df_raw.columns if col.startswith('close_')]
        if close_cols:
            # Show first ticker's close price as example
            first_close_col = close_cols[0]
            print(f"     {first_close_col}: ${df_raw[first_close_col].iloc[0]:.2f}")
        
        if len(target_cols) > 0:
            for tc in target_cols:
                val = df_raw[tc].iloc[0]
                if pd.notna(val):
                    print(f"     {tc}: ${val:.2f}")
    
    def _export_normalized_data(self, df: pd.DataFrame, X: np.ndarray, y: np.ndarray, 
                                ts: np.ndarray, base_name: str):
        """Export normalized data and sequence summaries (original behavior)."""
        
        # 1. Export normalized dataframe
        normalized_file = f"{base_name}_normalized.parquet"
        df.to_parquet(normalized_file, index=False)
        
        print(f"\n  ✅ Normalized data: {normalized_file}")
        print(f"     Format: Normalized features (for model training)")
        print(f"     Shape: {df.shape} (rows × columns)")
        
        # 2. Export sequence summary
        summary_file = f"{base_name}_sequences_summary.csv"
        
        summary_data = []
        for i in [0, len(X)//2, len(X)-1]:  # First, middle, last
            summary_data.append({
                'sequence_id': i,
                'timestamp': pd.Timestamp(ts[i]),
                'target_1h': y[i, 0] if len(y.shape) > 1 else y[i],
                'target_2h': y[i, 1] if len(y.shape) > 1 and y.shape[1] > 1 else None,
                'target_4h': y[i, 2] if len(y.shape) > 1 and y.shape[1] > 2 else None,
                'input_shape': str(X[i].shape),
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_file, index=False)
        
        print(f"  ✅ Sequences summary: {summary_file}")
        print(f"     Total sequences: {len(X):,}")
        print(f"     Input shape: {X[0].shape} (lookback × features)")
        print(f"     Target shape: {y[0].shape} (horizons)")
    
    def prepare_data(self) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Full data preparation pipeline."""
        # Fetch data
        df = self.fetch_data()
        
        # Add time features (before normalization)
        df = self.add_time_features(df)
        
        # Create sequences from RAW data (pivoting happens inside)
        print(f"\n" + "="*80)
        print(f"   Creating Time-Series Sequences")
        print("="*80)
        print(f"\n  Lookback window: {self.lookback} periods")
        print(f"  Prediction horizons: {self.horizons} periods")
        print(f"  Processing {len(df):,} timesteps...")
        X_raw, y_raw, ts, df_raw = self.create_sequences(df)
        print(f"\n✅ Created {len(X_raw):,} sequences")
        print(f"   Input shape: ({len(X_raw)}, {self.lookback}, {X_raw.shape[2]})")
        print(f"   Targets shape: ({y_raw.shape[0]}, {y_raw.shape[1]})")
        
        # Split data FIRST (before normalization)
        print(f"\n" + "="*80)
        print(f"   Splitting Data Chronologically")
        print("="*80)
        print()
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
        
        print(f"  train: {len(X_train_raw):,} samples ({len(X_train_raw)/n_samples*100:.1f}%)")
        print(f"  val  : {len(X_val_raw):,} samples ({len(X_val_raw)/n_samples*100:.1f}%)")
        print(f"  test : {len(X_test_raw):,} samples ({len(X_test_raw)/n_samples*100:.1f}%)")
        
        # Normalize ONLY on training data (both features and targets)
        print(f"\n" + "="*80)
        print(f"   Normalizing Train/Val/Test Splits")
        print("="*80)
        print(f"\n⚠️  Fitting scalers on TRAIN set only (prevents data leakage)...")
        X_train_norm, X_val_norm, X_test_norm, y_train_norm, y_val_norm, y_test_norm = self._normalize_splits(
            X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test
        )
        
        splits = {
            'train': (X_train_norm, y_train_norm, ts_train),
            'val': (X_val_norm, y_val_norm, ts_val),
            'test': (X_test_norm, y_test_norm, ts_test)
        }
        
        # Always export raw validation data to data/raw
        self._export_raw_validation(df_raw)
        
        # Export debug data to temp (optional)
        if self.export_temp:
            print(f"\n📁 Exporting debug data to temp/ (export_temp=True)...")
            # Note: Exporting normalized version
            df_norm = self.normalize_data(df, fit=True)
            self._export_normalized_data(df_norm, X_raw, y_raw, ts, f"temp/tft_data_{self.ticker.replace(':', '_')}_{self.frequency}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Save processed parquet files for inspection (always)
        self.save_processed_data(splits, df_raw)
        
        return splits
    
    def _normalize_splits(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Normalize train/val/test splits. Fit ONLY on training data."""
        from sklearn.preprocessing import StandardScaler
        
        # CRITICAL: Use final_features (after pivoting) instead of config!
        # After pivoting, features like 'close' become 'close_SPY', 'close_QQQ', etc.
        if not hasattr(self, 'final_features'):
            raise ValueError("final_features not set! create_sequences() must be called first.")
        
        all_features = self.final_features
        
        # Time features should NOT be normalized (already in good ranges: -1 to 1)
        time_features_to_skip = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 
                                  'month_sin', 'month_cos', 'is_weekend']
        
        # Get indices of features to normalize
        features_to_normalize = []
        for i, feat in enumerate(all_features):
            if feat not in time_features_to_skip:
                features_to_normalize.append(i)
        
        skipped_features = [f for f in all_features if f in time_features_to_skip]
        print(f"  Normalizing {len(features_to_normalize)}/{len(all_features)} features")
        if skipped_features:
            print(f"  Skipping: {', '.join(skipped_features)}")
        else:
            print(f"  Skipping: (none - all features will be normalized)")
        
        # Normalize each feature across the sequence dimension
        X_train_norm = X_train.copy()
        X_val_norm = X_val.copy()
        X_test_norm = X_test.copy()
        
        import time
        n_features = len(features_to_normalize)
        print(f"  Processing {n_features} features...\n")
        start_time = time.time()
        
        for i, feat_idx in enumerate(features_to_normalize, 1):
            feat_name = all_features[feat_idx]
            
            # Reshape to (n_samples * lookback, 1) for fitting
            train_values = X_train[:, :, feat_idx].reshape(-1, 1)
            
            # Fit scaler on training data only
            scaler = StandardScaler()
            scaler.fit(train_values)
            
            # Transform all splits
            X_train_norm[:, :, feat_idx] = scaler.transform(
                X_train[:, :, feat_idx].reshape(-1, 1)
            ).reshape(X_train.shape[0], X_train.shape[1])
            
            X_val_norm[:, :, feat_idx] = scaler.transform(
                X_val[:, :, feat_idx].reshape(-1, 1)
            ).reshape(X_val.shape[0], X_val.shape[1])
            
            X_test_norm[:, :, feat_idx] = scaler.transform(
                X_test[:, :, feat_idx].reshape(-1, 1)
            ).reshape(X_test.shape[0], X_test.shape[1])
            
            # Store scaler
            self.scalers[feat_name] = scaler
            
            # Progress update (log every feature)
            elapsed = time.time() - start_time
            eta = (elapsed / i) * (n_features - i) if i < n_features else 0
            pct = (i / n_features) * 100
            
            # Shorten feature name if too long (e.g., ticker-specific features)
            display_name = feat_name if len(feat_name) <= 30 else feat_name[:27] + '...'
            print(f"    [{i:2d}/{n_features}] {display_name:<30} mean={scaler.mean_[0]:>12.4f} std={scaler.scale_[0]:>12.4f} ({pct:5.1f}%, ETA: {eta:3.0f}s)")
        
        total_time = time.time() - start_time
        print(f"\n  ✅ Normalized {n_features} features in {total_time:.1f}s ({total_time/n_features:.2f}s per feature)")
        
        # Normalize targets (y) - FIT on actual multi-horizon target distribution
        # NOTE: Do NOT use the 1-period 'returns' scaler! Multi-horizon returns have different variance.
        print(f"\n  Normalizing targets (y) - fitting scaler on multi-horizon returns...")
        
        # Show raw statistics BEFORE normalization (diagnostic)
        print(f"\n  📊 RAW TARGET STATS (before normalization):")
        print(f"     y_train mean: {y_train.mean():.6f}, std: {y_train.std():.6f}")
        print(f"     y_train range: [{y_train.min():.6f}, {y_train.max():.6f}]")
        print(f"     y_val mean: {y_val.mean():.6f}, std: {y_val.std():.6f}")
        print(f"     y_val range: [{y_val.min():.6f}, {y_val.max():.6f}]")
        
        # Always fit a NEW scaler specifically for targets
        # (Multi-horizon returns have different statistics than 1-period returns)
        target_scaler = StandardScaler()
        # Fit on training targets (all horizons combined)
        target_scaler.fit(y_train.reshape(-1, 1))
        self.scalers['target'] = target_scaler  # Store under 'target' key, not self.target_col
        
        print(f"\n  ✅ Target scaler fitted on {y_train.size} samples")
        print(f"     Scaler mean: {target_scaler.mean_[0]:.6f}, std: {target_scaler.scale_[0]:.6f}")
        
        # Transform targets for all splits (flatten, transform, reshape)
        # y shape: (n_samples, n_horizons) -> flatten -> transform -> reshape back
        y_train_norm = target_scaler.transform(y_train.flatten().reshape(-1, 1)).reshape(y_train.shape)
        y_val_norm = target_scaler.transform(y_val.flatten().reshape(-1, 1)).reshape(y_val.shape)
        y_test_norm = target_scaler.transform(y_test.flatten().reshape(-1, 1)).reshape(y_test.shape)
        
        print(f"\n  📊 NORMALIZED TARGET STATS (after normalization):")
        print(f"     y_train_norm mean: {y_train_norm.mean():.6f}, std: {y_train_norm.std():.6f}")
        print(f"     y_train_norm range: [{y_train_norm.min():.3f}, {y_train_norm.max():.3f}]")
        print(f"     y_val_norm mean: {y_val_norm.mean():.6f}, std: {y_val_norm.std():.6f}")
        print(f"     y_val_norm range: [{y_val_norm.min():.3f}, {y_val_norm.max():.3f}]")
        
        # CRITICAL: Check for NaN/Inf after normalization
        print(f"\n  🔍 DATA QUALITY CHECKS:")
        
        try:
            # Ensure numeric dtype
            X_train_arr = np.asarray(X_train_norm, dtype=np.float64)
            X_val_arr = np.asarray(X_val_norm, dtype=np.float64)
            y_train_arr = np.asarray(y_train_norm, dtype=np.float64)
            y_val_arr = np.asarray(y_val_norm, dtype=np.float64)
            
            # Check X (features)
            train_nan = np.isnan(X_train_arr).sum()
            train_inf = np.isinf(X_train_arr).sum()
            val_nan = np.isnan(X_val_arr).sum()
            val_inf = np.isinf(X_val_arr).sum()
            
            print(f"     X_train: NaN={train_nan:,}, Inf={train_inf:,}")
            print(f"     X_val:   NaN={val_nan:,}, Inf={val_inf:,}")
            
            # Check y (targets)
            y_train_nan = np.isnan(y_train_arr).sum()
            y_train_inf = np.isinf(y_train_arr).sum()
            y_val_nan = np.isnan(y_val_arr).sum()
            y_val_inf = np.isinf(y_val_arr).sum()
            
            print(f"     y_train: NaN={y_train_nan:,}, Inf={y_train_inf:,}")
            print(f"     y_val:   NaN={y_val_nan:,}, Inf={y_val_inf:,}")
            
            # If any NaN/Inf found, identify which features
            if train_nan > 0 or train_inf > 0:
                print(f"\n  ⚠️  WARNING: Found NaN/Inf in training data!")
            nan_features = []
            inf_features = []
            for i, feat_name in enumerate(all_features):
                feat_data = X_train_arr[:, :, i]
                if np.isnan(feat_data).any():
                    nan_count = np.isnan(feat_data).sum()
                    nan_features.append(f"{feat_name} ({nan_count:,} NaN)")
                if np.isinf(feat_data).any():
                    inf_count = np.isinf(feat_data).sum()
                    inf_features.append(f"{feat_name} ({inf_count:,} Inf)")
            
            if nan_features:
                print(f"\n  🔴 Features with NaN:")
                for feat in nan_features[:10]:  # Show first 10
                    print(f"     - {feat}")
                if len(nan_features) > 10:
                    print(f"     ... and {len(nan_features) - 10} more")
            
            if inf_features:
                print(f"\n  🔴 Features with Inf:")
                for feat in inf_features[:10]:  # Show first 10
                    print(f"     - {feat}")
                if len(inf_features) > 10:
                    print(f"     ... and {len(inf_features) - 10} more")
            
                raise ValueError(
                    f"Data contains NaN ({train_nan:,}) or Inf ({train_inf:,}) after normalization. "
                    "This will cause model training to fail. Check the features listed above."
                )
            
            print(f"     ✅ No NaN/Inf detected!")
            
        except (TypeError, ValueError) as e:
            if 'isnan' in str(e) or 'dtype' in str(e):
                print(f"     ⚠️  Could not validate data types (arrays may have mixed types)")
                print(f"     Skipping NaN/Inf check - model will fail at runtime if data is bad")
            else:
                raise
        
        return X_train_norm, X_val_norm, X_test_norm, y_train_norm, y_val_norm, y_test_norm


class TFTDataset(Dataset):
    """PyTorch Dataset for TFT training."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, timestamps: np.ndarray):
        """
        Args:
            X: [num_samples, lookback, num_features]
            y: [num_samples, num_horizons]
            timestamps: [num_samples]
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.timestamps = timestamps
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def create_data_loaders(config_path: str = 'configs/model_tft_config.yaml',
                        export_temp: bool = False,
                        force_refresh: bool = False
                        ) -> Dict[str, DataLoader]:
    """Create PyTorch DataLoaders for train/val/test.
    
    Args:
        config_path: Path to config YAML
        export_temp: If True, export raw data to temp/ directory (for debugging)
        force_refresh: If True, always fetch from BigQuery (ignore cached numpy arrays)
    """
    from pathlib import Path
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check if pre-processed numpy arrays exist (unless force_refresh is True)
    processed_dir = Path('data/processed')
    # Use new naming convention: X_train.npy (not train_X.npy)
    train_X_file = processed_dir / 'X_train.npy'
    train_y_file = processed_dir / 'y_train.npy'
    train_ts_file = processed_dir / 'ts_train.npy'
    
    use_cached = (train_X_file.exists() and train_y_file.exists() and train_ts_file.exists()) and not force_refresh
    
    if use_cached:
        # Load pre-processed data from numpy arrays
        print("\n📂 Loading pre-processed data from disk (skipping BigQuery)...")
        splits = {}
        for split_name in ['train', 'val', 'test']:
            # Use new naming convention: X_train.npy
            X = np.load(processed_dir / f'X_{split_name}.npy', allow_pickle=True)
            y = np.load(processed_dir / f'y_{split_name}.npy', allow_pickle=True)
            ts = np.load(processed_dir / f'ts_{split_name}.npy', allow_pickle=True)  # Timestamps are datetime objects
            
            # Ensure arrays are float type (not object)
            X = X.astype(np.float32)
            y = y.astype(np.float32)
            
            splits[split_name] = (X, y, ts)
            print(f"  ✅ {split_name}: X{X.shape}, y{y.shape}, ts{ts.shape}")
        
        # Load scalers
        scalers_file = processed_dir / 'scalers.pkl'
        if scalers_file.exists():
            import pickle
            with open(scalers_file, 'rb') as f:
                scalers = pickle.load(f)
            print(f"  ✅ Loaded scalers from {scalers_file}")
        else:
            print(f"  ⚠️  No scalers file found, will use default normalization")
            scalers = None
    else:
        # Prepare data from BigQuery (original behavior)
        print("\n📂 No pre-processed data found, fetching from BigQuery...")
        loader = MultiTickerDataLoader(config_path, export_temp=export_temp)
        splits = loader.prepare_data()

        # Ensure arrays are float type (not object) - convert in-place
        for split_name in splits:
            X, y, ts = splits[split_name]
            splits[split_name] = (X.astype(np.float32), y.astype(np.float32), ts)

        # Note: Arrays already saved by prepare_data() with new naming convention
        # No need to save again
        print(f"\n✅ Using arrays from {processed_dir}/ (already saved by prepare_data)")

        # Load scalers (already saved by prepare_data)
        scalers_file = processed_dir / 'scalers.pkl'
        if scalers_file.exists():
            import pickle
            with open(scalers_file, 'rb') as f:
                scalers = pickle.load(f)
            print(f"  ✅ Loaded scalers from {scalers_file}")
        else:
            scalers = loader.scalers if hasattr(loader, 'scalers') else None
    
    # Create datasets
    datasets = {
        split_name: TFTDataset(X, y, ts)
        for split_name, (X, y, ts) in splits.items()
    }
    
    # Create dataloaders
    batch_size = config['training']['batch_size']
    num_workers = config['hardware']['num_workers']
    pin_memory = config['hardware']['pin_memory']
    
    dataloaders = {
        'train': DataLoader(
            datasets['train'],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        ),
        'val': DataLoader(
            datasets['val'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        ),
        'test': DataLoader(
            datasets['test'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    }
    
    return dataloaders, scalers


if __name__ == '__main__':
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='TFT Data Loader - Fetch and process multi-ticker data for training')
    parser.add_argument('--force-refresh', '--reload', action='store_true', dest='force_refresh',
                       help='Force reload from BigQuery (ignore cached .npy files)')
    parser.add_argument('--export-temp', action='store_true',
                       help='Export debug data to temp/ directory')
    args = parser.parse_args()
    
    # Test data loading
    print("="*80)
    print("   TFT Data Loader Test")
    print("="*80)
    
    if args.force_refresh:
        print("\n🔄 Force refresh enabled - will fetch from BigQuery...")
    
    loaders, scalers = create_data_loaders(
        export_temp=args.export_temp,
        force_refresh=args.force_refresh
    )
    
    print(f"\n✅ Data loaders created successfully")
    print(f"\nScalers: {list(scalers.keys())}")
    
    # Test batch
    for batch_X, batch_y in loaders['train']:
        print(f"\nSample batch:")
        print(f"  Input shape:  {batch_X.shape}  # [batch, lookback, features]")
        print(f"  Target shape: {batch_y.shape}  # [batch, horizons]")
        break
    
    print(f"\n💡 Command-line options:")
    print(f"   --force-refresh    Force reload from BigQuery (ignore cache)")
    print(f"   --export-temp      Export debug data to temp/ directory")
    print(f"\n   Example: python scripts/02_features/tft_data_loader.py --force-refresh")
    
    print(f"\n📂 Files created:")
    print(f"\n   data/raw/ (validation data - raw prices):")
    print(f"     - tft_features.csv")
    print(f"     - tft_features.parquet")
    print(f"\n   data/processed/ (training data - normalized):")
    print(f"     - X_train.npy, y_train.npy, ts_train.npy")
    print(f"     - X_val.npy, y_val.npy, ts_val.npy")
    print(f"     - X_test.npy, y_test.npy, ts_test.npy")
    print(f"     - scalers.pkl, metadata.yaml, feature_names.txt")
    print(f"     - feature_names.txt")