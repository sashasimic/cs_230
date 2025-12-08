#!/usr/bin/env python3
"""
Metrics Module

Common metric computation functions for model evaluation:
- MAE, MSE, RMSE
- Directional accuracy
- Per-horizon metrics
"""

import torch
import numpy as np
from typing import Dict, Optional, List, Union


def compute_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    horizons: Optional[List[int]] = None
) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Args:
        predictions: [batch, horizons] or [batch]
        targets: [batch, horizons] or [batch]
        horizons: Optional list of actual horizon values for labeling
    
    Returns:
        Dictionary of metrics (includes per-horizon metrics if multi-horizon)
    """
    with torch.no_grad():
        # Convert to tensors if numpy arrays
        if isinstance(predictions, np.ndarray):
            predictions = torch.from_numpy(predictions)
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets)
        
        # Ensure same device
        if predictions.device != targets.device:
            targets = targets.to(predictions.device)
        
        # Overall MAE
        mae = torch.abs(predictions - targets).mean().item()
        
        # Overall MSE
        mse = ((predictions - targets) ** 2).mean().item()
        
        # Directional accuracy
        pred_direction = torch.sign(predictions)
        target_direction = torch.sign(targets)
        dir_acc = (pred_direction == target_direction).float().mean().item() * 100
        
        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': np.sqrt(mse),
            'dir_acc': dir_acc
        }
        
        # Check if multi-horizon
        if len(predictions.shape) == 2 and predictions.shape[1] > 1:
            # Compute per-horizon metrics
            num_horizons = predictions.shape[1]
            for h_idx in range(num_horizons):
                h_mae = torch.abs(predictions[:, h_idx] - targets[:, h_idx]).mean().item()
                h_rmse = torch.sqrt(((predictions[:, h_idx] - targets[:, h_idx]) ** 2).mean()).item()
                h_dir_acc = (torch.sign(predictions[:, h_idx]) == torch.sign(targets[:, h_idx])).float().mean().item() * 100
                
                # Use actual horizon values for labeling
                horizon_label = f"H{horizons[h_idx]}" if (horizons and h_idx < len(horizons)) else f"H{h_idx+1}"
                
                metrics[f"{horizon_label}_MAE"] = h_mae
                metrics[f"{horizon_label}_RMSE"] = h_rmse
                metrics[f"{horizon_label}_dir_acc"] = h_dir_acc
        
        return metrics


def compute_mape(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float = 1e-8
) -> float:
    """
    Compute Mean Absolute Percentage Error.
    
    Args:
        predictions: Model predictions
        targets: True values
        epsilon: Small value to avoid division by zero
    
    Returns:
        MAPE as percentage
    """
    with torch.no_grad():
        # Avoid division by zero
        mask = torch.abs(targets) > epsilon
        if mask.sum() == 0:
            return 0.0
        
        ape = torch.abs((targets[mask] - predictions[mask]) / targets[mask])
        mape = ape.mean().item() * 100
        return mape


def compute_quantile_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    quantiles: List[float]
) -> Dict[str, float]:
    """
    Compute quantile loss for multi-quantile predictions.
    
    Args:
        predictions: [batch, horizons, quantiles] or [batch, quantiles]
        targets: [batch, horizons] or [batch]
        quantiles: List of quantile values (e.g., [0.1, 0.5, 0.9])
    
    Returns:
        Dictionary with per-quantile losses and total loss
    """
    with torch.no_grad():
        # Expand targets to match predictions shape if needed
        if len(targets.shape) < len(predictions.shape):
            targets = targets.unsqueeze(-1).expand_as(predictions)
        
        losses = {}
        total_loss = 0.0
        
        for i, q in enumerate(quantiles):
            # Extract predictions for this quantile
            if len(predictions.shape) == 3:
                # [batch, horizons, quantiles]
                pred_q = predictions[:, :, i]
                target_q = targets[:, :, 0] if len(targets.shape) == 3 else targets
            else:
                # [batch, quantiles]
                pred_q = predictions[:, i]
                target_q = targets[:, 0] if len(targets.shape) == 2 else targets
            
            # Quantile loss
            errors = target_q - pred_q
            q_loss = torch.max((q - 1) * errors, q * errors).mean().item()
            
            losses[f'q{int(q*100)}_loss'] = q_loss
            total_loss += q_loss
        
        losses['total_quantile_loss'] = total_loss
        return losses


def compute_calibration(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    quantiles: List[float],
    return_counts: bool = False
) -> Union[Dict[str, float], tuple]:
    """
    Compute calibration metrics for quantile predictions.
    
    Calibration measures how well predicted quantiles match actual coverage.
    For a well-calibrated model, the q-th quantile should contain q% of observations.
    
    Args:
        predictions: [batch, quantiles] - predicted quantiles
        targets: [batch] - actual values
        quantiles: List of quantile values
        return_counts: If True, also return observation counts
    
    Returns:
        Dictionary of calibration metrics (and optionally counts)
    """
    with torch.no_grad():
        metrics = {}
        counts = {}
        
        for i, q in enumerate(quantiles):
            pred_q = predictions[:, i]
            
            # For lower quantiles, check how many targets are above prediction
            if q <= 0.5:
                actual_coverage = (targets >= pred_q).float().mean().item()
                expected_coverage = 1 - q
            else:
                # For upper quantiles, check how many targets are below prediction
                actual_coverage = (targets <= pred_q).float().mean().item()
                expected_coverage = q
            
            # Calibration error (ideal = 0)
            calibration_error = abs(actual_coverage - expected_coverage)
            
            metrics[f'q{int(q*100)}_calibration'] = calibration_error
            metrics[f'q{int(q*100)}_coverage'] = actual_coverage
            
            if return_counts:
                counts[f'q{int(q*100)}_count'] = (targets <= pred_q).sum().item()
        
        # Average calibration error
        metrics['avg_calibration_error'] = np.mean([v for k, v in metrics.items() if 'calibration' in k])
        
        if return_counts:
            return metrics, counts
        return metrics


def compute_prediction_intervals(
    predictions: torch.Tensor,
    lower_quantile: float = 0.1,
    upper_quantile: float = 0.9
) -> tuple:
    """
    Compute prediction intervals from quantile predictions.
    
    Args:
        predictions: [batch, quantiles] - must include the specified quantiles
        lower_quantile: Lower bound quantile
        upper_quantile: Upper bound quantile
    
    Returns:
        (lower_bounds, upper_bounds, interval_widths)
    """
    # This assumes predictions are ordered by quantile
    # In practice, you'd need to know which column corresponds to which quantile
    # For now, assume standard ordering [0.1, 0.5, 0.9]
    
    lower_idx = 0  # Would need mapping from quantile to index
    upper_idx = -1  # Would need mapping from quantile to index
    
    lower_bounds = predictions[:, lower_idx]
    upper_bounds = predictions[:, upper_idx]
    interval_widths = upper_bounds - lower_bounds
    
    return lower_bounds, upper_bounds, interval_widths


def compute_skill_score(
    model_mse: float,
    baseline_mse: float
) -> float:
    """
    Compute skill score relative to baseline.
    
    Skill score = 1 - (model_error / baseline_error)
    - Score > 0: Model beats baseline
    - Score = 0: Model equals baseline
    - Score < 0: Model worse than baseline
    
    Args:
        model_mse: MSE of the model
        baseline_mse: MSE of the baseline
    
    Returns:
        Skill score value
    """
    if baseline_mse == 0:
        return 0.0
    return 1 - (model_mse / baseline_mse)