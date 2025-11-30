"""Data augmentation for time series models.

Creates training variants by swapping target group tickers while keeping
economy context constant. Model-agnostic - can be used with TFT, LSTM, etc.

Example:
    For agriculture inflation prediction:
    - Original: 1,393 dates
    - Augmented: 6,965 samples (1,393 × 5 agriculture tickers)
    - Each sample: economy features + one agriculture ticker as target
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import yaml


class DataAugmenter:
    """Augments time series data by creating variants with different target tickers.
    
    Strategy:
    - Keep economy context features constant (e.g., SPY, bonds, dollar)
    - Swap target group tickers (e.g., DBA, WEAT, SOYB for agriculture)
    - Create N samples per date (one per target ticker)
    - Add metadata: inflation_ticker, inflation_category
    
    Usage:
        augmenter = DataAugmenter('configs/model_config.yaml')
        augmented_df = augmenter.augment(pivoted_df)
    """
    
    def __init__(self, config_path: str):
        """Initialize augmenter from config.
        
        Args:
            config_path: Path to model config with augmentation settings
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load configuration
        augment_groups = self.config['data'].get('ticker_augment_groups', [])
        ticker_groups = self.config['data'].get('ticker_groups', {})
        
        if not augment_groups:
            raise ValueError(
                "data.ticker_augment_groups is required. "
                "Specify which ticker groups to augment across."
            )
        
        # Resolve tickers from groups
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
            
            self.augment_across.extend(group_tickers)
            
            # Map each ticker to its group name (used as category)
            for ticker in group_tickers:
                self.ticker_to_category[ticker] = group_name
        
        if not self.augment_across:
            raise ValueError(
                f"No tickers found in groups: {augment_groups}. "
                "Check data.ticker_groups configuration."
            )
        
        # Store feature lists for augmentation
        self.raw_features = self.config['data'].get('ticker_raw_features', ['close', 'volume'])
        self.synthetic_features = self.config['data'].get('ticker_synthetic_features', ['sma_50', 'sma_200'])
        
        print(f"\nData Augmenter initialized:")
        print(f"  - Augment across: {len(self.augment_across)} tickers - {self.augment_across}")
        print(f"  - Expected augmentation: {len(self.augment_across)}x per date")
    
    def augment(self, df: pd.DataFrame, all_tickers: List[str] = None) -> pd.DataFrame:
        """Augment DataFrame by creating variants with different target tickers.
        
        For each date:
        - Keep economy ticker features constant
        - Create one sample per target ticker in augment_across
        - Add inflation_ticker and inflation_category metadata
        
        Args:
            df: DataFrame in pivoted format (wide) with columns like close_SPY, close_DBA, etc.
            all_tickers: List of all tickers (used to determine economy vs target columns)
        
        Returns:
            Augmented DataFrame with N× samples
        """
        print(f"\n🔄 Augmenting data...")
        print(f"   Original: {len(df):,} samples")
        
        # Identify target vs economy columns
        target_cols = []
        for ticker in self.augment_across:
            target_cols.extend([c for c in df.columns if c.endswith(f'_{ticker}')])
        
        economy_cols = [c for c in df.columns if c not in target_cols]
        
        print(f"   Target columns: {len(target_cols)}")
        print(f"   Economy columns: {len(economy_cols)}")
        
        # Find date column
        date_col = 'date' if 'date' in df.columns else 'timestamp'
        if date_col not in df.columns:
            print(f"   ⚠️  No date column found, skipping augmentation")
            return df
        
        augmented = []
        
        # For each date, create one sample per target ticker
        for date in df[date_col].unique():
            date_row = df[df[date_col] == date].iloc[0]
            
            for target_ticker in self.augment_across:
                # Check if this ticker has data for this date
                close_col = f"close_{target_ticker}"
                if close_col not in df.columns or pd.isna(date_row[close_col]):
                    continue  # Skip if no data
                
                # Start with economy features
                new_row = date_row[economy_cols].copy()
                
                # Copy this ticker's features as "inflation" target
                for feature in self.raw_features + self.synthetic_features:
                    src_col = f"{feature}_{target_ticker}"
                    if src_col in df.columns:
                        new_row[f"{feature}_inflation"] = date_row[src_col]
                
                # Add metadata
                new_row['inflation_ticker'] = target_ticker
                new_row['inflation_category'] = self.ticker_to_category.get(target_ticker, 'unknown')
                
                augmented.append(new_row)
        
        df_aug = pd.DataFrame(augmented)
        print(f"   Augmented: {len(df_aug):,} samples ({len(df_aug)/len(df):.1f}x)\n")
        
        return df_aug