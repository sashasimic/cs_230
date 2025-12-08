"""Baseline Benchmark Forecasting Models

Provides standard industry baselines for time series forecasting:
1. Naïve forecast (random walk / no-change)
2. Moving average forecasts (simple and exponential)
3. Random walk with drift

These are the gold standard benchmarks that deep learning models must beat
to demonstrate value in financial forecasting tasks.
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional


def compute_naive_forecast(
    targets: torch.Tensor,
    horizons: list = None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Naïve Forecast: Predict 0% return (random walk assumption).
    
    This is the industry standard baseline for return forecasting.
    Assumes that the best predictor of tomorrow's return is 0%
    (i.e., no change from current price).
    
    Args:
        targets: [batch, horizons] - Ground truth returns
        horizons: Optional list of horizon values for labeling
    
    Returns:
        predictions: [batch, horizons] - All zeros
        metrics: Dictionary of benchmark metrics
    """
    # Predict 0% return for all horizons
    predictions = torch.zeros_like(targets)
    
    # Compute metrics
    mae = torch.abs(predictions - targets).mean().item()
    mse = ((predictions - targets) ** 2).mean().item()
    rmse = np.sqrt(mse)
    
    # Directional accuracy
    pred_direction = torch.sign(predictions)
    target_direction = torch.sign(targets)
    dir_acc = (pred_direction == target_direction).float().mean().item() * 100
    
    metrics = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'dir_acc': dir_acc
    }
    
    # Per-horizon metrics
    num_horizons = targets.shape[1]
    for h_idx in range(num_horizons):
        h_mae = torch.abs(predictions[:, h_idx] - targets[:, h_idx]).mean().item()
        h_rmse = torch.sqrt(((predictions[:, h_idx] - targets[:, h_idx]) ** 2).mean()).item()
        
        horizon_label = f"H{horizons[h_idx]}" if (horizons and h_idx < len(horizons)) else f"H{h_idx+1}"
        metrics[f"{horizon_label}_MAE"] = h_mae
        metrics[f"{horizon_label}_RMSE"] = h_rmse
    
    return predictions, metrics


def compute_ma_forecast(
    historical_returns: np.ndarray,
    targets: torch.Tensor,
    window_size: int = 5,
    horizons: list = None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Moving Average Forecast: Predict MA of recent returns.
    
    Args:
        historical_returns: [batch, lookback] - Historical return sequence
        targets: [batch, horizons] - Ground truth returns
        window_size: MA window (e.g., 5, 10, 21 days)
        horizons: Optional list of horizon values for labeling
    
    Returns:
        predictions: [batch, horizons] - MA forecast
        metrics: Dictionary of benchmark metrics
    """
    batch_size = historical_returns.shape[0]
    num_horizons = targets.shape[1]
    
    ma_values = []
    for i in range(batch_size):
        # Take last `window_size` returns
        recent = historical_returns[i, -window_size:]
        ma = np.mean(recent)
        ma_values.append(ma)
    
    ma_values = np.array(ma_values)
    
    # Predict same MA for all horizons
    predictions = torch.tensor(
        np.tile(ma_values[:, None], (1, num_horizons)),
        dtype=targets.dtype,
        device=targets.device
    )
    
    # Compute metrics
    mae = torch.abs(predictions - targets).mean().item()
    mse = ((predictions - targets) ** 2).mean().item()
    rmse = np.sqrt(mse)
    
    # Directional accuracy
    pred_direction = torch.sign(predictions)
    target_direction = torch.sign(targets)
    dir_acc = (pred_direction == target_direction).float().mean().item() * 100
    
    metrics = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'dir_acc': dir_acc
    }
    
    # Per-horizon metrics
    for h_idx in range(num_horizons):
        h_mae = torch.abs(predictions[:, h_idx] - targets[:, h_idx]).mean().item()
        h_rmse = torch.sqrt(((predictions[:, h_idx] - targets[:, h_idx]) ** 2).mean()).item()
        
        horizon_label = f"H{horizons[h_idx]}" if (horizons and h_idx < len(horizons)) else f"H{h_idx+1}"
        metrics[f"{horizon_label}_MAE"] = h_mae
        metrics[f"{horizon_label}_RMSE"] = h_rmse
    
    return predictions, metrics


def compute_ema_forecast(
    historical_returns: np.ndarray,
    targets: torch.Tensor,
    alpha: float = 0.3,
    horizons: list = None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Exponential Moving Average Forecast.
    
    Gives more weight to recent observations.
    
    Args:
        historical_returns: [batch, lookback] - Historical return sequence
        targets: [batch, horizons] - Ground truth returns
        alpha: Smoothing factor (0 < alpha < 1, higher = more recent weight)
        horizons: Optional list of horizon values for labeling
    
    Returns:
        predictions: [batch, horizons] - EMA forecast
        metrics: Dictionary of benchmark metrics
    """
    batch_size = historical_returns.shape[0]
    num_horizons = targets.shape[1]
    
    ema_values = []
    for i in range(batch_size):
        # Compute EMA
        ema = historical_returns[i, 0]
        for ret in historical_returns[i, 1:]:
            ema = alpha * ret + (1 - alpha) * ema
        ema_values.append(ema)
    
    ema_values = np.array(ema_values)
    
    # Predict same EMA for all horizons
    predictions = torch.tensor(
        np.tile(ema_values[:, None], (1, num_horizons)),
        dtype=targets.dtype,
        device=targets.device
    )
    
    # Compute metrics
    mae = torch.abs(predictions - targets).mean().item()
    mse = ((predictions - targets) ** 2).mean().item()
    rmse = np.sqrt(mse)
    
    # Directional accuracy
    pred_direction = torch.sign(predictions)
    target_direction = torch.sign(targets)
    dir_acc = (pred_direction == target_direction).float().mean().item() * 100
    
    metrics = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'dir_acc': dir_acc
    }
    
    # Per-horizon metrics
    for h_idx in range(num_horizons):
        h_mae = torch.abs(predictions[:, h_idx] - targets[:, h_idx]).mean().item()
        h_rmse = torch.sqrt(((predictions[:, h_idx] - targets[:, h_idx]) ** 2).mean()).item()
        
        horizon_label = f"H{horizons[h_idx]}" if (horizons and h_idx < len(horizons)) else f"H{h_idx+1}"
        metrics[f"{horizon_label}_MAE"] = h_mae
        metrics[f"{horizon_label}_RMSE"] = h_rmse
    
    return predictions, metrics


def compute_all_benchmarks(
    targets: torch.Tensor,
    historical_returns: Optional[np.ndarray] = None,
    horizons: list = None,
    ma_windows: list = [5, 10, 21],
    ema_alpha: float = 0.3
) -> Dict[str, Dict[str, float]]:
    """
    Compute all benchmark baselines at once.
    
    Args:
        targets: [batch, horizons] - Ground truth returns
        historical_returns: [batch, lookback] - Historical returns (if available)
        horizons: Optional list of horizon values for labeling
        ma_windows: List of MA window sizes to try
        ema_alpha: EMA smoothing factor
    
    Returns:
        Dictionary of {benchmark_name: metrics_dict}
    """
    benchmarks = {}
    
    # 1. Naïve forecast (always available)
    _, naive_metrics = compute_naive_forecast(targets, horizons)
    benchmarks['naive'] = naive_metrics
    
    # 2. Moving averages (if historical data available)
    if historical_returns is not None:
        for window in ma_windows:
            if window <= historical_returns.shape[1]:
                _, ma_metrics = compute_ma_forecast(
                    historical_returns, targets, window_size=window, horizons=horizons
                )
                benchmarks[f'ma_{window}'] = ma_metrics
        
        # 3. EMA
        _, ema_metrics = compute_ema_forecast(
            historical_returns, targets, alpha=ema_alpha, horizons=horizons
        )
        benchmarks['ema'] = ema_metrics
    
    return benchmarks


def print_benchmark_comparison(
    model_metrics: Dict[str, float],
    benchmark_metrics: Dict[str, Dict[str, float]],
    model_name: str = "Model"
):
    """
    Print a comparison table of model vs benchmarks.
    
    Args:
        model_metrics: Model's test metrics
        benchmark_metrics: Dict of {benchmark_name: metrics}
        model_name: Name of the model
    """
    print("\n" + "="*80)
    print("   Benchmark Comparison (Test Set)")
    print("="*80)
    
    # Header
    print(f"\n{'Benchmark':<20} {'MAE':>10} {'RMSE':>10} {'Dir Acc':>10} {'vs Naïve':>15}")
    print("-" * 80)
    
    # Naïve baseline (reference)
    naive_mae = benchmark_metrics.get('naive', {}).get('mae', 0)
    
    # Print model
    model_mae = model_metrics.get('mae', 0)
    model_rmse = model_metrics.get('rmse', 0)
    model_dir = model_metrics.get('dir_acc', 0)
    
    if naive_mae > 0:
        improvement = ((naive_mae - model_mae) / naive_mae) * 100
        status = f"↓{improvement:+.1f}%" if improvement > 0 else f"↑{-improvement:.1f}%"
    else:
        status = "N/A"
    
    print(f"{model_name:<20} {model_mae:>10.6f} {model_rmse:>10.6f} {model_dir:>9.2f}% {status:>15}")
    
    print("-" * 80)
    
    # Print benchmarks
    for bench_name, metrics in sorted(benchmark_metrics.items()):
        bench_mae = metrics.get('mae', 0)
        bench_rmse = metrics.get('rmse', 0)
        bench_dir = metrics.get('dir_acc', 0)
        
        display_name = bench_name.replace('_', ' ').upper()
        
        # Only naive is the baseline, compare others to naive
        if bench_name == 'naive':
            status = "(baseline)"
        elif naive_mae > 0:
            improvement = ((naive_mae - bench_mae) / naive_mae) * 100
            status = f"↓{improvement:+.1f}%" if improvement > 0 else f"↑{-improvement:.1f}%"
        else:
            status = "N/A"
        
        print(f"{display_name:<20} {bench_mae:>10.6f} {bench_rmse:>10.6f} {bench_dir:>9.2f}% {status:>15}")
    
    print("="*80)
    print("\n💡 Interpretation:")
    print("   ↓ = Model beats benchmark (lower MAE is better)")
    print("   ↑ = Benchmark beats model (model not adding value)")
    print("   Naïve = Predict 0% return (random walk baseline)")
    print("   MA/EMA = Technical trading strategies\n")