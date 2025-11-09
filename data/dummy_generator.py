"""Generate dummy time series data for testing."""

import numpy as np
import pandas as pd
import os
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def generate_dummy_data(
    n_samples: int = 10000,
    n_features: int = 10,
    sequence_length: int = 30,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    output_dir: str = 'data/dummy',
    seed: int = 42,
    multi_horizon: bool = True,
    shuffle: bool = False
) -> None:
    """
    Generate dummy time series data for inflation forecasting.
    
    Creates features and multi-horizon targets:
    - target_1m: 1-month ahead inflation
    - target_3m: 3-month ahead inflation  
    - target_6m: 6-month ahead inflation
    
    Args:
        n_samples: Number of time steps
        n_features: Number of features
        sequence_length: Length of sequences
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        output_dir: Output directory
        seed: Random seed
        multi_horizon: If True, generate 3 horizon targets; else single target
    """
    np.random.seed(seed)
    
    logger.info(f"Generating dummy data: {n_samples} samples, {n_features} features")
    
    # Generate timestamps
    timestamps = pd.date_range(
        start='2020-01-01',
        periods=n_samples,
        freq='D'
    )
    
    # Generate features with various patterns
    features = []
    for i in range(n_features):
        # Each feature has different characteristics
        if i % 3 == 0:
            # Trend + noise
            feature = np.linspace(0, 10, n_samples) + np.random.randn(n_samples) * 0.5
        elif i % 3 == 1:
            # Seasonal + noise
            t = np.arange(n_samples)
            feature = 5 * np.sin(2 * np.pi * t / 365) + np.random.randn(n_samples) * 0.5
        else:
            # Random walk
            feature = np.cumsum(np.random.randn(n_samples) * 0.1)
        
        features.append(feature)
    
    features = np.column_stack(features)
    
    # Generate multi-horizon targets for inflation forecasting
    if multi_horizon:
        # Create base inflation signal
        base_inflation = (
            0.3 * features[:, 0] +
            0.2 * features[:, 1] +
            0.15 * features[:, 2 % n_features] +
            np.random.randn(n_samples) * 0.2
        )
        
        # Normalize to realistic inflation range (e.g., -2% to 8%)
        base_inflation = 2.0 + 3.0 * (base_inflation - base_inflation.mean()) / base_inflation.std()
        
        # Generate 1-month, 3-month, and 6-month ahead targets
        target_1m = np.roll(base_inflation, -1)  # 1-month ahead
        target_3m = np.roll(base_inflation, -3)  # 3-month ahead
        target_6m = np.roll(base_inflation, -6)  # 6-month ahead
        
        # Add some decorrelation between horizons
        target_3m += np.random.randn(n_samples) * 0.1
        target_6m += np.random.randn(n_samples) * 0.15
        
        # Handle edge cases (last values)
        target_1m[-1] = target_1m[-2]
        target_3m[-3:] = target_3m[-4]
        target_6m[-6:] = target_6m[-7]
        
    else:
        # Single target (backward compatible)
        lag = 5
        target = np.zeros(n_samples)
        target[lag:] = (
            0.3 * features[:-lag, 0] +
            0.2 * features[:-lag, 1] +
            0.15 * features[:-lag, 2] +
            np.random.randn(n_samples - lag) * 0.3
        )
        target[:lag] = target[lag]
    
    # Create DataFrame
    df = pd.DataFrame(
        features,
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    if multi_horizon:
        df['target_1m'] = target_1m
        df['target_3m'] = target_3m
        df['target_6m'] = target_6m
    else:
        df['target'] = target
    
    df['timestamp'] = timestamps
    
    # Optionally shuffle (breaks temporal order!)
    if shuffle:
        logger.info("Shuffling data before split (temporal order broken)")
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    else:
        logger.info("Maintaining temporal order (sequential split)")
    
    # Split data
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    logger.info(f"Split ratios: train={train_ratio:.0%}, val={val_ratio:.0%}, test={1-(train_ratio+val_ratio):.0%}")
    
    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, 'train.csv')
    val_path = os.path.join(output_dir, 'val.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    logger.info(f"Saved train data: {len(train_df)} samples to {train_path}")
    logger.info(f"Saved val data: {len(val_df)} samples to {val_path}")
    logger.info(f"Saved test data: {len(test_df)} samples to {test_path}")
    
    # Print statistics
    logger.info(f"\nData statistics:")
    logger.info(f"Feature range: [{features.min():.2f}, {features.max():.2f}]")
    
    if multi_horizon:
        logger.info(f"Target 1M range: [{target_1m.min():.2f}, {target_1m.max():.2f}], "
                   f"mean: {target_1m.mean():.2f}, std: {target_1m.std():.2f}")
        logger.info(f"Target 3M range: [{target_3m.min():.2f}, {target_3m.max():.2f}], "
                   f"mean: {target_3m.mean():.2f}, std: {target_3m.std():.2f}")
        logger.info(f"Target 6M range: [{target_6m.min():.2f}, {target_6m.max():.2f}], "
                   f"mean: {target_6m.mean():.2f}, std: {target_6m.std():.2f}")
    else:
        logger.info(f"Target range: [{target.min():.2f}, {target.max():.2f}]")
        logger.info(f"Target mean: {target.mean():.2f}, std: {target.std():.2f}")


if __name__ == '__main__':
    generate_dummy_data()
