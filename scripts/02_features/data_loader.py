#!/usr/bin/env python3
"""Model-Agnostic Time Series Data Loader

Core data loading functionality for time series models.
Fetches data from BigQuery, applies transformations, and prepares for training.

Model-agnostic - can be used with TFT, LSTM, Transformer, or any time series model.

Usage:
    from scripts.02_features.data_loader import MultiTickerDataLoader
    
    loader = MultiTickerDataLoader('configs/model_config.yaml')
    data = loader.fetch_data()
    data = loader.add_time_features(data)
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


class MultiTickerDataLoader:
    """Load and prepare multi-ticker time series data from BigQuery.
    
    Core functionality:
    - Fetch ticker data (OHLCV + indicators) from BigQuery
    - Fetch GDELT sentiment data
    - Compute target baskets
    - Add time features
    - Handle missing values
    - Normalize features
    
    Model-agnostic - works with any time series model.
    """
    
    def __init__(self, config_path: str, export_temp: bool = False):
        """Initialize data loader with configuration.
        
        Args:
            config_path: Path to config YAML file
            export_temp: If True, export raw data to temp/ directory (for debugging)
        """
        self.config_path = config_path  # Save for later use
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.project_id = os.getenv('GCP_PROJECT_ID')
        if not self.project_id:
            raise ValueError("GCP_PROJECT_ID environment variable not set")
        
        self.client = bigquery.Client(project=self.project_id)
        
        # Data configuration
        self.tickers = self.config['data'].get('tickers', [])
        if not self.tickers:
            raise ValueError("Config must have 'tickers' list in data section")
        
        print(f"Loading {len(self.tickers)} tickers from config")
        
        self.frequency = self.config['data']['frequency']
        self.start_date = self.config['data']['start_date']
        self.end_date = self.config['data']['end_date']
        self.lookback = self.config['data']['lookback_window']
        self.horizons = self.config['data']['prediction_horizons']
        
        self.raw_features = self.config['data'].get('ticker_raw_features', ['close', 'volume'])
        self.synthetic_features = self.config['data'].get('ticker_synthetic_features', ['sma_50', 'sma_200'])
        
        # GDELT config
        self.use_gdelt = self.config['data']['gdelt'].get('enabled', False)
        if self.use_gdelt:
            self.gdelt_frequency = self.config['data']['gdelt'].get('frequency', self.frequency)
            self.gdelt_topic_groups = self.config['data']['gdelt'].get('topic_groups', ['inflation_prices'])
            self.gdelt_features = self.config['data']['gdelt']['features']
            self.gdelt_normalize_counts = self.config['data']['gdelt'].get('normalize_counts', True)
            self.gdelt_include_lags = self.config['data']['gdelt'].get('include_lags', True)
            self.gdelt_lag_periods = self.config['data']['gdelt'].get('lag_periods', [1, 7, 30])
        
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
    
    # ========== Data Fetching Methods ==========
    
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
    
    def fetch_target_basket(self) -> pd.DataFrame:
        """Fetch target basket tickers for Y label computation."""
        dataset_id = self.config['bigquery']['dataset_id']
        raw_table = self.config['bigquery']['ticker']['raw_table']
        
        # Read target config
        target_config = self.config['data'].get('target', {})
        target_tickers = target_config.get('basket_tickers', ['WEAT', 'SOYB', 'RJA'])  # Default fallback
        target_group = target_config.get('group', 'agriculture')
        aggregation = target_config.get('aggregation', 'mean')
        
        ticker_list = "', '".join(target_tickers)
        
        print(f"\nFetching {target_group} basket for target: {', '.join(target_tickers)}...")
        
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
            print(f"⚠️  Warning: No {target_group} basket data found!")
            print(f"   Make sure {', '.join(target_tickers)} are loaded for frequency '{self.frequency}'")
            return None
        
        # Pivot to get one column per ticker
        df_pivot = df.pivot(index='timestamp', columns='ticker', values='close')
        
        # Compute equal-weighted index from mean of RETURNS (proper equal weighting)
        # This ensures each ticker contributes equally regardless of price level
        # Method: Compute percentage returns, average them, then reconstruct index level
        
        if aggregation == 'mean':
            # Step 1: Compute percentage returns for each ticker
            returns = df_pivot[target_tickers].pct_change()
            
            # Step 2: Average returns across tickers (equal weighting)
            avg_returns = returns.mean(axis=1, skipna=True)
            
            # Step 3: Reconstruct cumulative index level (start at 100)
            df_pivot['target_basket_close'] = 100 * (1 + avg_returns).cumprod()
            
            print(f"   📊 Using equal-weighted aggregation (mean of PERCENTAGE RETURNS)")
            print(f"      Each ticker contributes equally regardless of price level")
        
        elif aggregation == 'median':
            # For median, still use price-based aggregation (less common)
            df_pivot['target_basket_close'] = df_pivot[target_tickers].median(axis=1, skipna=True)
            print(f"   📊 Using median aggregation (absolute prices)")
        
        else:
            # Default to mean of returns
            returns = df_pivot[target_tickers].pct_change()
            avg_returns = returns.mean(axis=1, skipna=True)
            df_pivot['target_basket_close'] = 100 * (1 + avg_returns).cumprod()
            print(f"   📊 Using equal-weighted aggregation (mean of PERCENTAGE RETURNS) [default]")
        
        # Count how many tickers contributed to each aggregation
        df_pivot['num_tickers_available'] = df_pivot[target_tickers].notna().sum(axis=1)
        
        # Keep only the aggregated column
        result = df_pivot[['target_basket_close']].reset_index()
        
        print(f"✅ Fetched {len(result):,} {target_group} basket rows")
        for ticker in target_tickers:
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
        """Join target basket close prices to ticker data for target computation."""
        if basket_df is None:
            print("⚠️  Warning: No target basket data - cannot compute target!")
            return ticker_df
        
        target_group = self.config['data'].get('target', {}).get('group', 'agriculture')
        print(f"\nJoining {target_group} basket for target computation...")
        
        # Left join to preserve all ticker timestamps
        df = ticker_df.merge(basket_df, on='timestamp', how='left')
        
        # Forward fill missing basket values
        missing_before = df['target_basket_close'].isnull().sum()
        if missing_before > 0:
            df = self.forward_fill_with_stats(df, ['target_basket_close'], context=f"{target_group.title()} basket")
        
        print(f"✅ Joined {target_group} basket, shape: {df.shape}")
        
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
            print(f"\n🗓️  Filtered {removed_rows:,} weekend rows ({removed_rows/initial_rows*100:.1f}%)")
            print(f"   Remaining: {len(df):,} rows")
        
        return df
    
    # ========== Utility Methods ==========
    
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
        """Fetch and combine all data sources (orchestrator method)."""
        # Fetch ticker data (OHLCV + indicators)
        ticker_df = self.fetch_ticker_data()
        
        # Fetch and join GDELT data if enabled
        if self.use_gdelt:
            gdelt_df = self.fetch_gdelt_data()
            df = self.join_gdelt_features(ticker_df, gdelt_df)
        else:
            df = ticker_df
            print("⚠️  GDELT features disabled in config")
        
        # Fetch and join target basket for Y label computation
        basket_df = self.fetch_target_basket()
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
        """Add time-based features - SIMPLIFIED to match decoder baseline.
        
        Args:
            df: DataFrame with 'timestamp' column
        
        Returns:
            DataFrame with added time features (month_sin, month_cos, is_weekend only)
        
        Note:
            Matches decoder transformer exactly with only 3 time features:
            - month_sin, month_cos: Cyclical month encoding
            - is_weekend: Binary weekend indicator
            
            Removed from baseline: day_of_week, day_of_month, month, day_sin, day_cos, hour features
        """
        if 'timestamp' not in df.columns:
            raise ValueError("DataFrame must have 'timestamp' column")
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # SIMPLIFIED: Only 3 time features to match decoder
        # Extract month and day_of_week for computing derived features
        month = df['timestamp'].dt.month
        day_of_week = df['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday
        
        time_features = {
            'month_sin': np.sin(2 * np.pi * month / 12),
            'month_cos': np.cos(2 * np.pi * month / 12),
            'is_weekend': (day_of_week >= 5).astype(int)  # Saturday=5, Sunday=6
        }
        
        # Add all time features at once using assign
        df = df.assign(**time_features)
        
        feature_list = list(time_features.keys())
        print(f"✅ Added time features: {', '.join(feature_list)}")
        print(f"   Total columns: {len(df.columns)}")
        
        return df

# ============================================================================
# MODEL-AGNOSTIC CORE
# ============================================================================
# This module provides core data loading functionality that is independent
# of any specific model architecture.
#
# What's included:
# - BigQuery data fetching (tickers, GDELT, target baskets)
# - Time feature engineering
# - Missing value handling (forward/backward fill)
# - Weekend filtering
# - GDELT sentiment integration
#
# What's NOT included (model-specific):
# - Sequence creation (varies by model: TFT uses multi-horizon, LSTM uses single)
# - Normalization (should be done per-split to avoid data leakage)
# - Train/val/test splitting (temporal vs random depends on model)
# - PyTorch/TensorFlow dataset wrappers
#
# Usage with helpers:
#   from scripts.02_features.data_loader import MultiTickerDataLoader
#   from scripts.02_features.data_grouping import TickerGroupFeatureAggregator
#   from scripts.02_features.data_augmentation import DataAugmenter
#
#   # 1. Load core data
#   loader = MultiTickerDataLoader('configs/model_config.yaml')
#   data = loader.fetch_data()
#   data = loader.add_time_features(data)
#
#   # 2. Optional: Apply group aggregation
#   if config['model']['feature_type'] == 'group_signals':
#       aggregator = TickerGroupFeatureAggregator(config_path)
#       data = aggregator.compute_group_features(data)
#
#   # 3. Optional: Apply augmentation
#   if config['data']['augmentation']['enabled']:
#       augmenter = DataAugmenter(config_path)
#       data = augmenter.augment(data)
#
#   # 4. Model-specific processing (normalization, sequences, etc.)
#   #    Implement in model-specific modules
# ============================================================================