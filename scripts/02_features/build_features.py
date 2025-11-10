"""
Phase 2: Build features from raw OHLCV data.

This script transforms raw stock data into ML-ready features with:
- Flexible input/output ticker selection
- Technical indicators (returns, volatility, MA, RSI, momentum)
- Weighted multi-horizon targets (1M, 3M, 6M)
- Train/val/test splits

Usage:
    # Use default config
    python scripts/build_features.py
    
    # Use custom config
    python scripts/build_features.py --config configs/features_custom.yaml
    
    # Override output directory
    python scripts/build_features.py --output-dir data/processed_v2
    
    # Skip sequences (for MLP models)
    python scripts/build_features.py --no-sequences
"""

import os
import sys
import argparse
import yaml
import warnings
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Add project root to path (two levels up from scripts/02_features/)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.logger import setup_logger

logger = setup_logger(__name__)

warnings.filterwarnings('ignore')


def load_config(config_path: str) -> dict:
    """Load feature engineering configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_raw_data(file_path: str) -> pd.DataFrame:
    """Load raw parquet file."""
    logger.info(f"Loading raw data from {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw data file not found: {file_path}")
    
    df = pd.read_parquet(file_path)
    
    # Handle different date column names (date, DATE, Date)
    date_col = None
    for possible_name in ['date', 'DATE', 'Date']:
        if possible_name in df.columns:
            date_col = possible_name
            break
    
    if date_col is None:
        raise ValueError(f"No date column found in {file_path}. Columns: {df.columns.tolist()}")
    
    # Normalize to lowercase 'date'
    if date_col != 'date':
        df = df.rename(columns={date_col: 'date'})
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    logger.info(f"✓ Loaded {len(df)} rows × {len(df.columns)} columns")
    logger.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    
    return df


def load_data_sources(config: dict) -> pd.DataFrame:
    """Load and merge multiple data sources."""
    logger.info("Loading data sources...")
    
    data_sources_config = config['data_sources']
    dfs = []
    source_num = 1
    total_sources = sum([1 for k in ['stocks', 'google_trends', 'gdelt'] 
                         if data_sources_config.get(k, {}).get('enabled', k == 'stocks')])
    
    # Load stocks data
    if data_sources_config.get('stocks', {}).get('enabled', True):
        stocks_path = data_sources_config['stocks']['path']
        logger.info(f"\n[{source_num}/{total_sources}] Loading stocks data from {stocks_path}")
        stocks_df = load_raw_data(stocks_path)
        dfs.append(stocks_df)
        source_num += 1
    
    # Load Google Trends data
    if data_sources_config.get('google_trends', {}).get('enabled', False):
        trends_path = data_sources_config['google_trends']['path']
        logger.info(f"\n[{source_num}/{total_sources}] Loading Google Trends data from {trends_path}")
        if os.path.exists(trends_path):
            trends_df = load_raw_data(trends_path)
            dfs.append(trends_df)
        else:
            logger.warning(f"Google Trends data not found at {trends_path}, skipping")
        source_num += 1
    
    # Load GDELT data
    if data_sources_config.get('gdelt', {}).get('enabled', False):
        gdelt_path = data_sources_config['gdelt']['path']
        logger.info(f"\n[{source_num}/{total_sources}] Loading GDELT data from {gdelt_path}")
        if os.path.exists(gdelt_path):
            gdelt_df = load_raw_data(gdelt_path)
            dfs.append(gdelt_df)
        else:
            logger.warning(f"GDELT data not found at {gdelt_path}, skipping")
        source_num += 1
    
    # Merge data sources on date
    if len(dfs) == 0:
        raise ValueError("No data sources enabled in config")
    elif len(dfs) == 1:
        merged_df = dfs[0]
    else:
        logger.info("\nMerging data sources on date...")
        merged_df = dfs[0]
        for df in dfs[1:]:
            merged_df = pd.merge(merged_df, df, on='date', how='outer')
        merged_df = merged_df.sort_values('date').reset_index(drop=True)
        logger.info(f"✓ Merged {len(merged_df)} rows × {len(merged_df.columns)} columns")
    
    return merged_df


def filter_date_range(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Filter data to specified date range."""
    date_range_config = config.get('date_range', {})
    start_date = date_range_config.get('start_date')
    end_date = date_range_config.get('end_date')
    
    if start_date is None and end_date is None:
        logger.info("No date range filter specified, using all available data")
        return df
    
    original_len = len(df)
    original_start = df['date'].min()
    original_end = df['date'].max()
    
    # Apply filters
    if start_date is not None:
        start_date = pd.to_datetime(start_date)
        df = df[df['date'] >= start_date]
    
    if end_date is not None:
        end_date = pd.to_datetime(end_date)
        df = df[df['date'] <= end_date]
    
    df = df.reset_index(drop=True)
    
    logger.info("\nFiltering to date range...")
    logger.info(f"  Original: {original_start.date()} to {original_end.date()} ({original_len} rows)")
    logger.info(f"  Filtered: {df['date'].min().date()} to {df['date'].max().date()} ({len(df)} rows)")
    logger.info(f"  Removed: {original_len - len(df)} rows")
    
    return df


def add_google_trends_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Add Google Trends-specific features."""
    if not config['data_sources'].get('google_trends', {}).get('enabled', False):
        return df
    
    trends_config = config['data_sources']['google_trends']['features']
    
    # Find Google Trends columns (exclude 'date' and stock columns)
    trends_cols = [col for col in df.columns 
                   if col not in ['date'] 
                   and not any(x in col for x in ['_open', '_high', '_low', '_close', '_volume', '_vwap'])]
    
    if not trends_cols:
        logger.warning("No Google Trends columns found")
        return df
    
    logger.info(f"\nAdding Google Trends features for {len(trends_cols)} keywords...")
    
    for keyword in trends_cols:
        # Lagged values
        if trends_config.get('lags', {}).get('enabled', True):
            for lag in trends_config['lags']['periods']:
                df[f"{keyword}_lag_{lag}d"] = df[keyword].shift(lag)
        
        # Rolling averages
        if trends_config.get('rolling_average', {}).get('enabled', True):
            for window in trends_config['rolling_average']['windows']:
                df[f"{keyword}_ma_{window}d"] = df[keyword].rolling(window=window, min_periods=1).mean()
        
        # Rate of change
        if trends_config.get('rate_of_change', {}).get('enabled', True):
            for period in trends_config['rate_of_change']['periods']:
                df[f"{keyword}_roc_{period}d"] = (df[keyword] - df[keyword].shift(period)) / (df[keyword].shift(period) + 1e-10)
        
        # Z-score (normalized deviation)
        if trends_config.get('z_score', {}).get('enabled', True):
            window = trends_config['z_score']['window']
            rolling_mean = df[keyword].rolling(window=window, min_periods=1).mean()
            rolling_std = df[keyword].rolling(window=window, min_periods=1).std()
            df[f"{keyword}_zscore_{window}d"] = (df[keyword] - rolling_mean) / (rolling_std + 1e-10)
        
        # Volatility (rolling standard deviation of changes)
        if trends_config.get('volatility', {}).get('enabled', False):
            changes = df[keyword].diff()
            for window in trends_config['volatility']['windows']:
                df[f"{keyword}_volatility_{window}d"] = changes.rolling(window=window, min_periods=1).std()
        
        # RSI (Relative Strength Index)
        if trends_config.get('rsi', {}).get('enabled', False):
            period = trends_config['rsi']['period']
            delta = df[keyword].diff()
            
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=period, min_periods=1).mean()
            avg_loss = loss.rolling(window=period, min_periods=1).mean()
            
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            df[f"{keyword}_rsi_{period}d"] = rsi
    
    logger.info(f"✓ Google Trends features expanded to {len(df.columns)} columns")
    
    return df


def add_gdelt_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Add GDELT news sentiment-specific features."""
    if not config['data_sources'].get('gdelt', {}).get('enabled', False):
        return df
    
    gdelt_config = config['data_sources']['gdelt']
    gdelt_columns = gdelt_config['columns']
    features_config = gdelt_config['features']
    
    # Check if GDELT columns exist
    gdelt_cols_found = [col for col in gdelt_columns if col in df.columns]
    if not gdelt_cols_found:
        logger.warning("No GDELT columns found")
        return df
    
    logger.info(f"\nAdding GDELT features for {len(gdelt_cols_found)} columns...")
    
    for col in gdelt_cols_found:
        # Lagged values
        if features_config.get('lags', {}).get('enabled', True):
            for lag in features_config['lags']['periods']:
                df[f"{col}_lag_{lag}d"] = df[col].shift(lag)
        
        # Rolling averages
        if features_config.get('rolling_average', {}).get('enabled', True):
            for window in features_config['rolling_average']['windows']:
                df[f"{col}_ma_{window}d"] = df[col].rolling(window=window, min_periods=1).mean()
        
        # Rate of change
        if features_config.get('rate_of_change', {}).get('enabled', True):
            for period in features_config['rate_of_change']['periods']:
                df[f"{col}_roc_{period}d"] = (df[col] - df[col].shift(period)) / (df[col].shift(period).abs() + 1e-10)
        
        # Z-score (normalized deviation)
        if features_config.get('z_score', {}).get('enabled', True):
            window = features_config['z_score']['window']
            rolling_mean = df[col].rolling(window=window, min_periods=1).mean()
            rolling_std = df[col].rolling(window=window, min_periods=1).std()
            df[f"{col}_zscore_{window}d"] = (df[col] - rolling_mean) / (rolling_std + 1e-10)
        
        # Volatility (rolling standard deviation of changes)
        if features_config.get('volatility', {}).get('enabled', True):
            changes = df[col].diff()
            for window in features_config['volatility']['windows']:
                df[f"{col}_volatility_{window}d"] = changes.rolling(window=window, min_periods=1).std()
        
        # RSI (Relative Strength Index)
        if features_config.get('rsi', {}).get('enabled', True):
            period = features_config['rsi']['period']
            delta = df[col].diff()
            
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=period, min_periods=1).mean()
            avg_loss = loss.rolling(window=period, min_periods=1).mean()
            
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            df[f"{col}_rsi_{period}d"] = rsi
    
    logger.info(f"✓ GDELT features expanded to {len(df.columns)} columns")
    
    return df


def compute_returns(df: pd.DataFrame, ticker: str, periods: List[int], log_returns: bool = False) -> pd.DataFrame:
    """Compute returns over multiple periods."""
    close_col = f"{ticker}_close"
    
    if close_col not in df.columns:
        logger.warning(f"Column {close_col} not found, skipping returns for {ticker}")
        return df
    
    for period in periods:
        if log_returns:
            col_name = f"{ticker}_log_return_{period}d"
            df[col_name] = np.log(df[close_col] / df[close_col].shift(period))
        else:
            col_name = f"{ticker}_return_{period}d"
            df[col_name] = (df[close_col] - df[close_col].shift(period)) / df[close_col].shift(period)
    
    return df


def compute_volatility(df: pd.DataFrame, ticker: str, windows: List[int]) -> pd.DataFrame:
    """Compute rolling volatility."""
    close_col = f"{ticker}_close"
    
    if close_col not in df.columns:
        return df
    
    # First compute daily returns
    returns = df[close_col].pct_change()
    
    for window in windows:
        col_name = f"{ticker}_volatility_{window}d"
        df[col_name] = returns.rolling(window=window).std()
    
    return df


def compute_moving_average(df: pd.DataFrame, ticker: str, windows: List[int], ma_type: str = 'sma') -> pd.DataFrame:
    """Compute moving averages."""
    close_col = f"{ticker}_close"
    
    if close_col not in df.columns:
        return df
    
    for window in windows:
        if ma_type == 'sma':
            col_name = f"{ticker}_sma_{window}d"
            df[col_name] = df[close_col].rolling(window=window).mean()
        elif ma_type == 'ema':
            col_name = f"{ticker}_ema_{window}d"
            df[col_name] = df[close_col].ewm(span=window, adjust=False).mean()
        
        # Also compute price deviation from MA
        deviation_col = f"{ticker}_ma_dev_{window}d"
        df[deviation_col] = (df[close_col] - df[col_name]) / df[col_name]
    
    return df


def compute_rsi(df: pd.DataFrame, ticker: str, period: int = 14) -> pd.DataFrame:
    """Compute Relative Strength Index."""
    close_col = f"{ticker}_close"
    
    if close_col not in df.columns:
        return df
    
    # Calculate price changes
    delta = df[close_col].diff()
    
    # Separate gains and losses
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    
    # Calculate average gains and losses
    avg_gains = gains.rolling(window=period).mean()
    avg_losses = losses.rolling(window=period).mean()
    
    # Calculate RS and RSI
    rs = avg_gains / (avg_losses + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    col_name = f"{ticker}_rsi_{period}d"
    df[col_name] = rsi
    
    return df


def compute_momentum(df: pd.DataFrame, ticker: str, periods: List[int]) -> pd.DataFrame:
    """Compute momentum indicators."""
    close_col = f"{ticker}_close"
    
    if close_col not in df.columns:
        return df
    
    for period in periods:
        col_name = f"{ticker}_momentum_{period}d"
        df[col_name] = df[close_col] - df[close_col].shift(period)
    
    return df


def add_technical_indicators(df: pd.DataFrame, tickers: List[str], config: dict) -> pd.DataFrame:
    """Add technical indicators for all tickers."""
    logger.info(f"Computing technical indicators for {len(tickers)} tickers...")
    
    indicators_config = config['data_sources']['stocks']['technical_indicators']
    
    for ticker in tickers:
        # Returns
        if indicators_config['returns']['enabled']:
            df = compute_returns(df, ticker, indicators_config['returns']['periods'], log_returns=False)
        
        # Log returns
        if indicators_config['log_returns']['enabled']:
            df = compute_returns(df, ticker, indicators_config['log_returns']['periods'], log_returns=True)
        
        # Volatility
        if indicators_config['volatility']['enabled']:
            df = compute_volatility(df, ticker, indicators_config['volatility']['windows'])
        
        # Moving averages
        if indicators_config['moving_average']['enabled']:
            for ma_type in indicators_config['moving_average']['types']:
                df = compute_moving_average(df, ticker, indicators_config['moving_average']['windows'], ma_type)
        
        # RSI
        if indicators_config['rsi']['enabled']:
            df = compute_rsi(df, ticker, indicators_config['rsi']['period'])
        
        # Momentum
        if indicators_config['momentum']['enabled']:
            df = compute_momentum(df, ticker, indicators_config['momentum']['periods'])
    
    logger.info(f"✓ Features expanded to {len(df.columns)} columns")
    
    return df


def compute_weighted_target(df: pd.DataFrame, target_tickers: List[Dict], horizon_periods: int, 
                            target_type: str = 'returns') -> pd.Series:
    """Compute weighted average target across multiple tickers."""
    target = pd.Series(0.0, index=df.index)
    
    for ticker_config in target_tickers:
        ticker = ticker_config['ticker']
        weight = ticker_config['weight']
        close_col = f"{ticker}_close"
        
        if close_col not in df.columns:
            logger.warning(f"Target ticker {ticker} not found, skipping")
            continue
        
        # Compute forward return
        if target_type == 'returns':
            ticker_target = (df[close_col].shift(-horizon_periods) - df[close_col]) / df[close_col]
        else:  # log_returns
            ticker_target = np.log(df[close_col].shift(-horizon_periods) / df[close_col])
        
        target += weight * ticker_target
    
    return target


def create_targets(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Create target variables for multiple horizons."""
    logger.info("Creating target variables...")
    
    output_config = config['output_targets']
    horizons = output_config['horizons']
    target_tickers = output_config['target_tickers']
    target_type = output_config['target_type']
    separate_horizons = output_config['separate_horizons']
    
    # Verify weights sum to 1.0
    ticker_weight_sum = sum(t['weight'] for t in target_tickers)
    if not np.isclose(ticker_weight_sum, 1.0):
        logger.warning(f"Target ticker weights sum to {ticker_weight_sum:.3f}, normalizing to 1.0")
        for t in target_tickers:
            t['weight'] /= ticker_weight_sum
    
    if separate_horizons:
        # Create separate target column for each horizon
        for horizon in horizons:
            col_name = f"target_{horizon['name']}"
            df[col_name] = compute_weighted_target(df, target_tickers, horizon['periods'], target_type)
            logger.info(f"  Created {col_name} ({horizon['periods']} days forward)")
    else:
        # Create single weighted target across horizons
        horizon_weight_sum = sum(h['weight'] for h in horizons)
        if not np.isclose(horizon_weight_sum, 1.0):
            logger.warning(f"Horizon weights sum to {horizon_weight_sum:.3f}, normalizing to 1.0")
            for h in horizons:
                h['weight'] /= horizon_weight_sum
        
        target = pd.Series(0.0, index=df.index)
        for horizon in horizons:
            horizon_target = compute_weighted_target(df, target_tickers, horizon['periods'], target_type)
            target += horizon['weight'] * horizon_target
        
        df['target'] = target
        logger.info(f"  Created weighted target (1M: {horizons[0]['weight']}, 3M: {horizons[1]['weight']}, 6M: {horizons[2]['weight']})")
    
    return df


def select_feature_columns(df: pd.DataFrame, config: dict) -> List[str]:
    """Select feature columns based on config."""
    stocks_config = config['data_sources']['stocks']
    tickers = stocks_config['tickers']
    columns = stocks_config['columns']
    
    feature_cols = []
    
    # Add raw OHLCV columns for stocks
    for ticker in tickers:
        for col in columns:
            col_name = f"{ticker}_{col}"
            if col_name in df.columns:
                feature_cols.append(col_name)
    
    # Add stock technical indicator columns
    for col in df.columns:
        if any(indicator in col for indicator in ['_return_', '_log_return_', '_volatility_', 
                                                   '_sma_', '_ema_', '_ma_dev_', '_rsi_', '_momentum_']):
            if any(col.startswith(f"{ticker}_") for ticker in tickers):
                feature_cols.append(col)
    
    # Add Google Trends features (if enabled)
    if config['data_sources'].get('google_trends', {}).get('enabled', False):
        trends_features = config['data_sources']['google_trends']['features']
        
        for col in df.columns:
            # Include raw trends values
            if trends_features.get('raw_values', True):
                # Raw keyword columns (no suffix)
                if col not in ['date'] and not any(x in col for x in ['_open', '_high', '_low', '_close', '_volume', '_vwap', '_lag_', '_ma_', '_roc_', '_zscore_', '_volatility_', '_rsi_']):
                    # Check if it's a trends column by seeing if it has any derived features
                    if any(f"{col}_lag_" in c or f"{col}_ma_" in c for c in df.columns):
                        feature_cols.append(col)
            
            # Include derived Google Trends features
            if any(suffix in col for suffix in ['_lag_', '_ma_', '_roc_', '_zscore_', '_volatility_', '_rsi_']) and not any(col.startswith(f"{t}_") for t in tickers):
                feature_cols.append(col)
    
    # Add GDELT features (if enabled)
    if config['data_sources'].get('gdelt', {}).get('enabled', False):
        gdelt_config = config['data_sources']['gdelt']
        gdelt_columns = gdelt_config['columns']
        gdelt_features = gdelt_config['features']
        
        for col in df.columns:
            # Include raw GDELT values
            if gdelt_features.get('raw_values', True):
                if col in gdelt_columns:
                    feature_cols.append(col)
            
            # Include derived GDELT features (lags, MA, ROC, z-score, volatility, RSI)
            if any(gdelt_col in col for gdelt_col in gdelt_columns):
                if any(suffix in col for suffix in ['_lag_', '_ma_', '_roc_', '_zscore_', '_volatility_', '_rsi_']):
                    feature_cols.append(col)
    
    # Remove duplicates while preserving order
    feature_cols = list(dict.fromkeys(feature_cols))
    
    # Count features by source
    stock_features = sum(1 for c in feature_cols if any(c.startswith(f'{t}_') for t in tickers))
    
    # Get GDELT column names for counting
    gdelt_columns = config['data_sources'].get('gdelt', {}).get('columns', [])
    gdelt_features_count = sum(1 for c in feature_cols if any(gdelt_col in c for gdelt_col in gdelt_columns))
    
    trends_features_count = len(feature_cols) - stock_features - gdelt_features_count
    
    logger.info(f"Selected {len(feature_cols)} feature columns")
    logger.info(f"  Stock features: {stock_features}")
    logger.info(f"  Google Trends features: {trends_features_count}")
    logger.info(f"  GDELT features: {gdelt_features_count}")
    
    return feature_cols


def handle_missing_values(df: pd.DataFrame, method: str, max_gap: int) -> pd.DataFrame:
    """Handle missing values in features."""
    logger.info(f"Handling missing values (method: {method})...")
    
    initial_missing = df.isnull().sum().sum()
    logger.info(f"  Initial missing values: {initial_missing}")
    
    if method == 'forward_fill':
        df = df.fillna(method='ffill', limit=max_gap)
    elif method == 'interpolate':
        df = df.interpolate(method='linear', limit=max_gap)
    elif method == 'drop':
        df = df.dropna()
    
    final_missing = df.isnull().sum().sum()
    logger.info(f"  Final missing values: {final_missing}")
    
    return df


def remove_correlated_features(df: pd.DataFrame, feature_cols: List[str], threshold: float) -> List[str]:
    """Remove highly correlated features."""
    logger.info(f"Removing features with correlation > {threshold}...")
    
    # Compute correlation matrix
    corr_matrix = df[feature_cols].corr().abs()
    
    # Find upper triangle
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features to drop
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    logger.info(f"  Dropping {len(to_drop)} highly correlated features")
    
    return [col for col in feature_cols if col not in to_drop]


def scale_features(df: pd.DataFrame, feature_cols: List[str], method: str, 
                   per_ticker: bool, fit: bool = True, scaler: Optional[Dict] = None) -> Tuple[pd.DataFrame, Dict]:
    """Scale features."""
    if method == 'none':
        return df, {}
    
    logger.info(f"Scaling features (method: {method}, per_ticker: {per_ticker})...")
    
    if scaler is None:
        scaler = {}
    
    if per_ticker:
        # Scale each ticker's features independently
        tickers = list(set([col.split('_')[0] for col in feature_cols]))
        
        for ticker in tickers:
            ticker_cols = [col for col in feature_cols if col.startswith(f"{ticker}_")]
            
            if not ticker_cols:
                continue
            
            if fit:
                if method == 'standard':
                    scaler[ticker] = StandardScaler()
                elif method == 'minmax':
                    scaler[ticker] = MinMaxScaler()
                elif method == 'robust':
                    scaler[ticker] = RobustScaler()
                
                df[ticker_cols] = scaler[ticker].fit_transform(df[ticker_cols])
            else:
                if ticker in scaler:
                    df[ticker_cols] = scaler[ticker].transform(df[ticker_cols])
    else:
        # Scale all features together
        if fit:
            if method == 'standard':
                scaler['all'] = StandardScaler()
            elif method == 'minmax':
                scaler['all'] = MinMaxScaler()
            elif method == 'robust':
                scaler['all'] = RobustScaler()
            
            df[feature_cols] = scaler['all'].fit_transform(df[feature_cols])
        else:
            if 'all' in scaler:
                df[feature_cols] = scaler['all'].transform(df[feature_cols])
    
    return df, scaler


def split_data(df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train/val/test sets."""
    split_config = config['split']
    method = split_config['method']
    
    logger.info(f"Splitting data (method: {method})...")
    
    if method == 'time':
        # Time-based split (chronological)
        n = len(df)
        
        if split_config['dates']['train_end']:
            # Use specific dates
            train_end = pd.to_datetime(split_config['dates']['train_end'])
            val_end = pd.to_datetime(split_config['dates']['val_end'])
            
            train_df = df[df['date'] <= train_end]
            val_df = df[(df['date'] > train_end) & (df['date'] <= val_end)]
            test_df = df[df['date'] > val_end]
        else:
            # Use ratios
            train_ratio = split_config['train_ratio']
            val_ratio = split_config['val_ratio']
            
            train_end_idx = int(n * train_ratio)
            val_end_idx = int(n * (train_ratio + val_ratio))
            
            train_df = df.iloc[:train_end_idx]
            val_df = df.iloc[train_end_idx:val_end_idx]
            test_df = df.iloc[val_end_idx:]
    else:
        # Random split
        train_df = df.sample(frac=split_config['train_ratio'], random_state=42)
        remaining = df.drop(train_df.index)
        val_frac = split_config['val_ratio'] / (split_config['val_ratio'] + split_config['test_ratio'])
        val_df = remaining.sample(frac=val_frac, random_state=42)
        test_df = remaining.drop(val_df.index)
    
    logger.info(f"  Train: {len(train_df)} samples ({train_df['date'].min()} to {train_df['date'].max()})")
    logger.info(f"  Val:   {len(val_df)} samples ({val_df['date'].min()} to {val_df['date'].max()})")
    logger.info(f"  Test:  {len(test_df)} samples ({test_df['date'].min()} to {test_df['date'].max()})")
    
    # Check minimum samples
    min_samples = split_config['min_samples']
    if len(train_df) < min_samples['train']:
        raise ValueError(f"Train set too small: {len(train_df)} < {min_samples['train']}")
    if len(val_df) < min_samples['val']:
        raise ValueError(f"Val set too small: {len(val_df)} < {min_samples['val']}")
    if len(test_df) < min_samples['test']:
        raise ValueError(f"Test set too small: {len(test_df)} < {min_samples['test']}")
    
    return train_df, val_df, test_df


def save_processed_data(df: pd.DataFrame, output_path: str, feature_cols: List[str], target_cols: List[str]):
    """Save processed data to file."""
    # Select only feature and target columns (+ date for reference)
    cols_to_save = ['date'] + feature_cols + target_cols
    df_to_save = df[cols_to_save].copy()
    
    # Drop rows with NaN in targets
    df_to_save = df_to_save.dropna(subset=target_cols)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if output_path.endswith('.parquet'):
        df_to_save.to_parquet(output_path, index=False)
    else:
        df_to_save.to_csv(output_path, index=False)
    
    file_size = os.path.getsize(output_path) / 1024**2
    logger.info(f"  Saved: {output_path} ({file_size:.2f} MB, {len(df_to_save)} rows)")


def report_statistics(df: pd.DataFrame, feature_cols: List[str], target_cols: List[str], split_name: str):
    """Report feature and target statistics."""
    logger.info(f"\n{split_name} Statistics:")
    logger.info(f"  Features: {len(feature_cols)}")
    logger.info(f"  Targets: {target_cols}")
    logger.info(f"  Samples: {len(df)}")
    
    # Target statistics
    for target_col in target_cols:
        if target_col in df.columns:
            target_data = df[target_col].dropna()
            logger.info(f"  {target_col}: mean={target_data.mean():.4f}, std={target_data.std():.4f}, "
                       f"min={target_data.min():.4f}, max={target_data.max():.4f}")


def main():
    parser = argparse.ArgumentParser(description='Build features from raw OHLCV data')
    parser.add_argument('--config', default='configs/features.yaml', help='Feature config file')
    parser.add_argument('--output-dir', help='Override output directory')
    parser.add_argument('--no-sequences', action='store_true', help='Skip sequence creation')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    if args.output_dir:
        config['data']['output_dir'] = args.output_dir
    
    if args.no_sequences:
        config['sequences']['create_sequences'] = False
    
    logger.info("=" * 70)
    logger.info("PHASE 2: FEATURE ENGINEERING")
    logger.info("=" * 70)
    
    # Load raw data from multiple sources
    df = load_data_sources(config)
    
    # Filter to specified date range
    df = filter_date_range(df, config)
    
    # Add Google Trends features (if enabled)
    df = add_google_trends_features(df, config)
    
    # Add GDELT features (if enabled)
    df = add_gdelt_features(df, config)
    
    # Add stock technical indicators
    tickers = config['data_sources']['stocks']['tickers']
    df = add_technical_indicators(df, tickers, config)
    
    # Create targets
    df = create_targets(df, config)
    
    # Select feature columns
    feature_cols = select_feature_columns(df, config)
    
    # Get target columns
    if config['output_targets']['separate_horizons']:
        target_cols = [f"target_{h['name']}" for h in config['output_targets']['horizons']]
    else:
        target_cols = ['target']
    
    # Handle missing values
    df = handle_missing_values(df, 
                               config['feature_engineering']['missing_values']['method'],
                               config['feature_engineering']['missing_values']['max_gap'])
    
    # Remove correlated features
    if config['feature_engineering']['remove_correlated']['enabled']:
        feature_cols = remove_correlated_features(df, feature_cols, 
                                                  config['feature_engineering']['remove_correlated']['threshold'])
    
    # Split data
    train_df, val_df, test_df = split_data(df, config)
    
    # Scale features (fit on train only)
    scaling_method = config['feature_engineering']['scaling']['method']
    per_ticker = config['feature_engineering']['scaling']['per_ticker']
    
    train_df, scaler = scale_features(train_df, feature_cols, scaling_method, per_ticker, fit=True)
    val_df, _ = scale_features(val_df, feature_cols, scaling_method, per_ticker, fit=False, scaler=scaler)
    test_df, _ = scale_features(test_df, feature_cols, scaling_method, per_ticker, fit=False, scaler=scaler)
    
    # Report statistics
    if config['quality_checks']['report_stats']:
        report_statistics(train_df, feature_cols, target_cols, "TRAIN")
        report_statistics(val_df, feature_cols, target_cols, "VAL")
        report_statistics(test_df, feature_cols, target_cols, "TEST")
    
    # Save processed data
    output_dir = config['data']['output_dir']
    output_format = config['data']['output_format']
    ext = f".{output_format}"
    
    logger.info(f"\nSaving processed data to {output_dir}...")
    save_processed_data(train_df, os.path.join(output_dir, f"train{ext}"), feature_cols, target_cols)
    save_processed_data(val_df, os.path.join(output_dir, f"val{ext}"), feature_cols, target_cols)
    save_processed_data(test_df, os.path.join(output_dir, f"test{ext}"), feature_cols, target_cols)
    
    # Save scaler
    import joblib
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    logger.info(f"  Saved scaler: {scaler_path}")
    
    # Save feature names
    feature_names_path = os.path.join(output_dir, 'feature_names.txt')
    with open(feature_names_path, 'w') as f:
        f.write('\n'.join(feature_cols))
    logger.info(f"  Saved feature names: {feature_names_path}")
    
    logger.info("\n" + "=" * 70)
    logger.info("FEATURE ENGINEERING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"✓ Features: {len(feature_cols)}")
    logger.info(f"✓ Targets: {target_cols}")
    logger.info(f"✓ Train samples: {len(train_df)}")
    logger.info(f"✓ Val samples: {len(val_df)}")
    logger.info(f"✓ Test samples: {len(test_df)}")
    logger.info(f"✓ Output: {output_dir}")
    logger.info("=" * 70)
    logger.info("\n✓ Phase 2 complete! Ready for Phase 3 (model training)")


if __name__ == '__main__':
    main()
