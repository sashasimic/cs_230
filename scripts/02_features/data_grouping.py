"""Group-level feature aggregation for time series models.

Transforms per-ticker features into compact, interpretable group-level signals.
Principle: aggregate in signal space (returns, trends) not raw price space.

Model-agnostic - can be used with TFT, LSTM, or any other model.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import yaml


class TickerGroupFeatureAggregator:
    """Aggregates ticker-level features into group-level signals.
    
    Reduces ~99 per-ticker features to ~45 group-level features that are:
    - More interpretable ("equity momentum" vs individual ticker moves)
    - More stable (less noise)
    - More efficient for the model to process
    
    Usage:
        aggregator = TickerGroupFeatureAggregator('configs/model_config.yaml')
        group_features = aggregator.compute_group_features(ticker_df)
    """
    
    def __init__(self, config_path: str):
        """Initialize with config.
        
        Args:
            config_path: Path to model config with ticker_groups
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load ticker groups from config
        self.ticker_groups = self.config['data']['ticker_groups']
        
        # Load features to generate per group from config
        self.group_features = self.config['data'].get('ticker_group_features', [
            'ret_mean',         # Mean return (basket return)
            'ret_std',          # Return dispersion
            'trend50_mean',     # Short-term trend strength
            'breadth_above50',  # Short-term breadth
            'vol_rel_mean'      # Relative volume
        ])
        
        print(f"\nTicker Group Feature Aggregator initialized:")
        print(f"  - {len(self.ticker_groups)} ticker groups")
        print(f"  - {len(self.group_features)} features per group")
        print(f"  - Total output features: {len(self.ticker_groups) * len(self.group_features)}")
    
    def compute_group_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute group-level features from ticker data.
        
        Args:
            df: DataFrame with columns like 'close_SPY', 'volume_SPY', 'sma_50_SPY', etc.
                Index should be dates.
        
        Returns:
            DataFrame with group-level features like 'equity_largecap_ret_mean', etc.
        """
        out = pd.DataFrame(index=df.index)
        
        for group_name, group_config in self.ticker_groups.items():
            tickers = group_config.get('tickers', [])
            if not tickers:
                continue
            
            # Skip if tickers not in data
            close_cols = [f"close_{t}" for t in tickers]
            available_cols = [c for c in close_cols if c in df.columns]
            if not available_cols:
                print(f"  ⚠️  Skipping {group_name}: no ticker data found")
                continue
            
            # Use only available tickers
            available_tickers = [t for t in tickers if f"close_{t}" in df.columns]
            
            # Compute all possible features, then filter by config
            computed = {}
            
            # ===== 1. Return features =====
            close_data = df[[f"close_{t}" for t in available_tickers]]
            returns = close_data.pct_change()
            
            computed['ret_mean'] = returns.mean(axis=1)  # Mean return (basket return)
            computed['ret_std'] = returns.std(axis=1)    # Return dispersion
            
            # ===== 2. Trend features (relative to SMAs) =====
            has_sma50 = all(f"sma_50_{t}" in df.columns for t in available_tickers)
            
            if has_sma50:
                sma50_data = df[[f"sma_50_{t}" for t in available_tickers]]
                # % above 50-day SMA
                trend50 = (close_data.values / sma50_data.values) - 1.0
                trend50 = pd.DataFrame(trend50, index=df.index, columns=available_tickers)
                
                computed['trend50_mean'] = trend50.mean(axis=1)
                computed['breadth_above50'] = (trend50 > 0).mean(axis=1)
            else:
                # Fallback: use 20-day return as proxy
                computed['trend50_mean'] = close_data.pct_change(20).mean(axis=1)
                computed['breadth_above50'] = (close_data.pct_change(20) > 0).mean(axis=1)
            
            # Optional: trend200_mean (if configured)
            if 'trend200_mean' in self.group_features:
                has_sma200 = all(f"sma_200_{t}" in df.columns for t in available_tickers)
                if has_sma200:
                    sma200_data = df[[f"sma_200_{t}" for t in available_tickers]]
                    trend200 = (close_data.values / sma200_data.values) - 1.0
                    trend200 = pd.DataFrame(trend200, index=df.index, columns=available_tickers)
                    computed['trend200_mean'] = trend200.mean(axis=1)
                else:
                    computed['trend200_mean'] = close_data.pct_change(60).mean(axis=1)
            
            # ===== 3. Volume features =====
            vol_cols = [f"volume_{t}" for t in available_tickers]
            has_volume = all(c in df.columns for c in vol_cols)
            
            if has_volume:
                vol_data = df[vol_cols]
                vol_ma60 = vol_data.rolling(60, min_periods=10).mean()
                vol_rel = vol_data / vol_ma60
                computed['vol_rel_mean'] = vol_rel.mean(axis=1)
            else:
                computed['vol_rel_mean'] = pd.Series(1.0, index=df.index)
            
            # ===== 4. Assign only configured features to output =====
            for feature in self.group_features:
                if feature in computed:
                    out[f"{group_name}_{feature}"] = computed[feature]
        
        # Fill NaNs with forward fill then zero
        out = out.ffill().fillna(0)
        
        print(f"\n✅ Generated {out.shape[1]} group features:")
        for group_name in self.ticker_groups.keys():
            group_cols = [c for c in out.columns if c.startswith(group_name)]
            if group_cols:
                print(f"  - {group_name}: {len(group_cols)} features")
        
        return out