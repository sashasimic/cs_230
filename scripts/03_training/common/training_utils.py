#!/usr/bin/env python3
"""
Training Utilities

Common training functions used across different models:
- Gradient computation and statistics
- Training epoch functions
- Learning rate scheduling
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, Any


def compute_grad_norm(model: nn.Module) -> float:
    """Compute total gradient norm across all model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Total gradient norm (L2)
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def compute_layer_grad_stats(model: nn.Module) -> Dict[str, Dict[str, float]]:
    """Compute gradient statistics per layer.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary mapping layer names to gradient statistics
    """
    layer_stats = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            grad_mean = param.grad.data.mean().item()
            grad_std = param.grad.data.std().item() if param.grad.data.numel() > 1 else 0.0
            grad_max = param.grad.data.abs().max().item()
            
            # Group by layer type
            layer_type = _get_layer_type(name)
            
            if layer_type not in layer_stats:
                layer_stats[layer_type] = {'norm': 0.0, 'max': 0.0, 'std': 0.0}
            
            # Aggregate: sum norms, max for max/std
            layer_stats[layer_type]['norm'] += grad_norm
            layer_stats[layer_type]['max'] = max(layer_stats[layer_type]['max'], grad_max)
            layer_stats[layer_type]['std'] = max(layer_stats[layer_type]['std'], grad_std)
    
    return layer_stats


def _get_layer_type(param_name: str) -> str:
    """Extract layer type from parameter name.
    
    Args:
        param_name: Full parameter name (e.g., 'transformer_encoder.layers.0.self_attn.in_proj_weight')
        
    Returns:
        Layer type string for grouping
    """
    if 'variable_selection' in param_name or 'variable_grns' in param_name:
        return 'VSN'
    elif 'feature_projection' in param_name and 'weight' in param_name:
        return 'Input'
    elif 'static_enrichment' in param_name or 'position_wise_grn' in param_name:
        return 'GRN'
    elif 'lstm_encoder' in param_name:
        if 'weight_ih_l0' in param_name or 'weight_hh_l0' in param_name:
            return 'LSTM_L0'
        elif 'weight_ih_l1' in param_name or 'weight_hh_l1' in param_name:
            return 'LSTM_L1'
        elif 'weight_ih_l2' in param_name or 'weight_hh_l2' in param_name:
            return 'LSTM_L2'
        else:
            return 'LSTM_Other'
    elif 'transformer_encoder.layers' in param_name:
        parts = param_name.split('.')
        layer_idx = parts[2] if len(parts) > 2 else '?'
        if 'self_attn' in param_name:
            return f'Attention_L{layer_idx}'
        elif 'linear1' in param_name or 'linear2' in param_name:
            return f'Feedforward_L{layer_idx}'
        elif 'norm1' in param_name:
            return f'Attention_L{layer_idx}'
        elif 'norm2' in param_name:
            return f'Feedforward_L{layer_idx}'
        else:
            return f'Encoder_L{layer_idx}_Other'
    elif 'enc_norm' in param_name:
        return 'EncoderNorm'
    elif 'future_decoder' in param_name or 'future_in_proj' in param_name or 'future_out_proj' in param_name:
        return 'FutureDecoder'
    elif 'quantile_outputs' in param_name:
        return 'Output'
    elif 'pos_encoder' in param_name:
        return 'PosEnc'
    else:
        return 'Other'


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int = 0,
    clip_norm: float = 1.0,
    has_static_features: bool = False
) -> Dict[str, Any]:
    """Train for one epoch and return detailed metrics.
    
    Args:
        model: Model to train
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on
        epoch: Current epoch number
        clip_norm: Gradient clipping threshold
        has_static_features: Whether the model uses static features
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0.0
    unclipped_grad_norms = []
    clipped_grad_norms = []
    layer_grad_stats = None
    
    total_batches = len(dataloader)
    print(f"\n  Training: 0/{total_batches} batches", end='', flush=True)
    
    for batch_idx, batch in enumerate(dataloader):
        # Always 3 items: (X, y, static) - static may be empty placeholder
        batch_X, batch_y, batch_static = batch
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        batch_static = batch_static.to(device) if has_static_features else None
        
        optimizer.zero_grad()
        
        # Forward pass - handle different model types
        if hasattr(model, 'forward'):
            # Check if model expects specific kwargs
            import inspect
            sig = inspect.signature(model.forward)
            params = sig.parameters
            
            if 'static_features' in params:
                # TFT-style model
                predictions = model(batch_X, static_features=batch_static, y_future=batch_y, teacher_forcing=True)
            else:
                # Simple model (LSTM, etc.)
                predictions = model(batch_X)
        else:
            predictions = model(batch_X)
        
        loss = criterion(predictions, batch_y)
        loss.backward()
        
        # Compute unclipped gradient norm
        unclipped_norm = compute_grad_norm(model)
        unclipped_grad_norms.append(unclipped_norm)
        
        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        
        # Compute clipped gradient norm
        clipped_norm = compute_grad_norm(model)
        clipped_grad_norms.append(clipped_norm)
        
        # Get detailed layer stats for first batch only
        if batch_idx == 0:
            layer_grad_stats = compute_layer_grad_stats(model)
        
        optimizer.step()
        total_loss += loss.item()
        
        # Progress update every 10 batches or at end
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
            print(f"\r  Training: {batch_idx + 1}/{total_batches} batches (loss: {total_loss / (batch_idx + 1):.4f})", end='', flush=True)
    
    print()  # New line after progress
    avg_loss = total_loss / len(dataloader)
    
    # Gradient statistics
    avg_unclipped = float(np.mean(unclipped_grad_norms))
    max_unclipped = float(np.max(unclipped_grad_norms))
    avg_clipped = float(np.mean(clipped_grad_norms))
    max_clipped = float(np.max(clipped_grad_norms))
    
    return {
        'loss': avg_loss,
        'avg_unclipped': avg_unclipped,
        'max_unclipped': max_unclipped,
        'avg_clipped': avg_clipped,
        'max_clipped': max_clipped,
        'layer_grad_stats': layer_grad_stats
    }


def create_optimizer(
    model: nn.Module,
    config: Dict[str, Any]
) -> torch.optim.Optimizer:
    """Create optimizer from config.
    
    Args:
        model: Model to optimize
        config: Training configuration dictionary
        
    Returns:
        Optimizer instance
    """
    training_config = config.get('training', {})
    optimizer_type = training_config.get('optimizer', 'adam').lower()
    learning_rate = training_config.get('learning_rate', 0.001)
    weight_decay = training_config.get('weight_decay', 0.0)
    
    if optimizer_type == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_type == 'adamw':
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_type == 'sgd':
        momentum = training_config.get('momentum', 0.9)
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any]
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Create learning rate scheduler from config.
    
    Args:
        optimizer: Optimizer to schedule
        config: Training configuration dictionary
        
    Returns:
        Scheduler instance or None if not configured
    """
    training_config = config.get('training', {})
    scheduler_config = training_config.get('lr_scheduler', {})
    
    if not scheduler_config.get('enabled', False):
        return None
    
    scheduler_type = scheduler_config.get('type', 'reduce_on_plateau')
    
    if scheduler_type == 'reduce_on_plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_config.get('factor', 0.5),
            patience=scheduler_config.get('patience', 5),
            min_lr=scheduler_config.get('min_lr', 1e-6)
        )
    elif scheduler_type == 'cosine':
        T_max = scheduler_config.get('T_max', 100)
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max
        )
    elif scheduler_type == 'step':
        step_size = scheduler_config.get('step_size', 30)
        gamma = scheduler_config.get('gamma', 0.1)
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")