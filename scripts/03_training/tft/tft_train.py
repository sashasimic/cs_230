#!/usr/bin/env python3
"""
Core TFT Training Logic

Shared training function used by both:
- tft_train_local.py (local training)
- train_vertex.py (cloud training)
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


class SimplifiedTFT(nn.Module):
    """Simplified Temporal Fusion Transformer for multi-horizon forecasting."""
    
    def __init__(self, config: dict, num_features: int = None):
        super().__init__()
        
        # Extract config
        time_varying_known = config['model'].get('time_varying_known', [])
        time_varying_unknown = config['model']['time_varying_unknown']
        # Use actual feature count from data if provided (accounts for pivoting)
        # Otherwise fallback to config count (for backward compatibility)
        if num_features is not None:
            self.num_features = num_features
        else:
            self.num_features = len(time_varying_known) + len(time_varying_unknown)
        self.hidden_size = config['model']['hidden_size']
        self.lstm_layers = config['model']['lstm_layers']
        self.attention_heads = config['model']['attention_heads']
        self.dropout = config['model']['dropout']
        self.num_horizons = len(config['data']['prediction_horizons'])
        
        # Input projection
        self.input_projection = nn.Linear(self.num_features, self.hidden_size)
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.lstm_layers,
            batch_first=True,
            dropout=self.dropout if self.lstm_layers > 1 else 0
        )
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.attention_heads,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Post-attention layer norm and feedforward
        self.attention_norm = nn.LayerNorm(self.hidden_size)
        self.feedforward = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size * 4, self.hidden_size),
            nn.Dropout(self.dropout)
        )
        self.ff_norm = nn.LayerNorm(self.hidden_size)
        
        # Output heads for each horizon
        self.output_heads = nn.ModuleList([
            nn.Linear(self.hidden_size, 1)
            for _ in range(self.num_horizons)
        ])
        
        # Initialize weights
        self._init_weights()
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass.
        
        Args:
            x: [batch, lookback, features]
        
        Returns:
            predictions: [batch, horizons]
        """
        # Input projection
        x = self.input_projection(x)  # [batch, lookback, hidden]
        
        # LSTM encoding
        lstm_out, _ = self.lstm(x)  # [batch, lookback, hidden]
        
        # Multi-head attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Residual connection + layer norm
        x = self.attention_norm(lstm_out + attn_out)
        
        # Feedforward + residual
        ff_out = self.feedforward(x)
        x = self.ff_norm(x + ff_out)  # [batch, lookback, hidden]
        
        # Use last timestep for prediction
        x = x[:, -1, :]  # [batch, hidden]
        
        # Generate predictions for each horizon
        predictions = []
        for horizon_head in self.output_heads:
            horizon_pred = horizon_head(x)  # [batch, 1]
            predictions.append(horizon_pred)
        
        # Stack: [batch, horizons]
        predictions = torch.cat(predictions, dim=1)
        
        return predictions
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)


def compute_grad_norm(model: nn.Module) -> float:
    """Compute total gradient norm across all model parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def compute_layer_grad_stats(model: nn.Module) -> dict:
    """Compute gradient statistics per layer."""
    layer_stats = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            grad_mean = param.grad.data.mean().item()
            grad_std = param.grad.data.std().item() if param.grad.data.numel() > 1 else 0.0
            grad_max = param.grad.data.abs().max().item()
            
            # Group by layer type
            if 'vsn' in name:
                layer_type = 'VSN'
            elif 'input_projection' in name:
                layer_type = 'Input'
            elif 'lstm' in name:
                if 'weight_ih_l0' in name or 'weight_hh_l0' in name or 'bias_ih_l0' in name or 'bias_hh_l0' in name:
                    layer_type = 'LSTM_L0'
                elif 'weight_ih_l1' in name or 'weight_hh_l1' in name or 'bias_ih_l1' in name or 'bias_hh_l1' in name:
                    layer_type = 'LSTM_L1'
                elif 'weight_ih_l2' in name or 'weight_hh_l2' in name or 'bias_ih_l2' in name or 'bias_hh_l2' in name:
                    layer_type = 'LSTM_L2'
                else:
                    layer_type = 'LSTM_Other'
            elif 'attention' in name:
                layer_type = 'Attention'
            elif 'feedforward' in name:
                layer_type = 'Feedforward'
            elif 'output_heads' in name:
                layer_type = 'Output'
            else:
                layer_type = 'Other'
            
            if layer_type not in layer_stats:
                layer_stats[layer_type] = {'norms': [], 'means': [], 'stds': [], 'maxs': []}
            
            layer_stats[layer_type]['norms'].append(grad_norm)
            layer_stats[layer_type]['means'].append(grad_mean)
            layer_stats[layer_type]['stds'].append(grad_std)
            layer_stats[layer_type]['maxs'].append(grad_max)
    
    # Aggregate statistics
    aggregated = {}
    for layer_type, stats in layer_stats.items():
        aggregated[layer_type] = {
            'avg_norm': np.mean(stats['norms']),
            'max_norm': np.max(stats['norms']),
            'avg_std': np.mean(stats['stds'])
        }
    
    return aggregated


def compute_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Args:
        predictions: [batch, horizons]
        targets: [batch, horizons]
    
    Returns:
        Dictionary of metrics
    """
    with torch.no_grad():
        # MAE
        mae = torch.abs(predictions - targets).mean().item()
        
        # MSE
        mse = ((predictions - targets) ** 2).mean().item()
        
        # Directional accuracy
        pred_direction = torch.sign(predictions)
        target_direction = torch.sign(targets)
        dir_acc = (pred_direction == target_direction).float().mean().item() * 100
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': np.sqrt(mse),
            'dir_acc': dir_acc
        }


def train_epoch(model: nn.Module, dataloader, criterion, optimizer, device, epoch: int = 0) -> dict:
    """Train for one epoch and return detailed metrics."""
    model.train()
    total_loss = 0
    grad_norms = []
    layer_grad_stats = None
    
    for batch_idx, (batch_X, batch_y) in enumerate(dataloader):
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        
        # Track gradients before clipping
        grad_norm = compute_grad_norm(model)
        grad_norms.append(grad_norm)
        
        # Clip gradients to prevent explosion/vanishing
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Get detailed layer stats for first batch only (after clipping)
        if batch_idx == 0:
            layer_grad_stats = compute_layer_grad_stats(model)
        
        optimizer.step()
        total_loss += loss.item()
    
    return {
        'loss': total_loss / len(dataloader),
        'avg_grad_norm': np.mean(grad_norms),
        'max_grad_norm': np.max(grad_norms),
        'min_grad_norm': np.min(grad_norms),
        'layer_grad_stats': layer_grad_stats
    }


def train(config_path: str, dataloaders: Optional[Dict] = None, scalers: Optional[Dict] = None):
    """
    Core TFT training function.
    
    Args:
        config_path: Path to model config YAML
        dataloaders: Optional pre-loaded DataLoaders (if None, will load from data/processed/)
        scalers: Optional pre-loaded scalers
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data if not provided
    if dataloaders is None:
        print("\n📂 Loading data from data/processed/...")
        import importlib.util
        data_loader_path = project_root / 'scripts' / '02_features' / 'tft' / 'tft_data_loader.py'
        spec = importlib.util.spec_from_file_location('tft_data_loader', data_loader_path)
        data_loader_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(data_loader_module)
        
        dataloaders, scalers = data_loader_module.create_data_loaders(
            config_path=config_path,
            force_refresh=False
        )
        print("✅ Data loaded!")
    
    # Print detailed feature information
    print("\n" + "="*80)
    print("   Feature Configuration")
    print("="*80)
    
    # Get sample batch to determine actual feature count
    sample_batch = next(iter(dataloaders['train']))
    batch_X, batch_y = sample_batch
    num_features = batch_X.shape[2]
    lookback = batch_X.shape[1]
    num_horizons = batch_y.shape[1]
    
    print(f"\n📊 Data Dimensions:")
    print(f"  Lookback window: {lookback} timesteps")
    print(f"  Number of features: {num_features}")
    print(f"  Prediction horizons: {num_horizons}")
    
    # Load and display feature metadata
    time_varying_known = config['model'].get('time_varying_known', [])
    time_varying_unknown = config['model'].get('time_varying_unknown', [])
    
    print(f"\n🔑 TIME-VARYING KNOWN Features ({len(time_varying_known)}):")
    for i, feat in enumerate(time_varying_known, 1):
        print(f"  {i:2d}. {feat}")
    
    print(f"\n📊 TIME-VARYING UNKNOWN Features ({len(time_varying_unknown)}):")
    for i, feat in enumerate(time_varying_unknown, 1):
        print(f"  {i:2d}. {feat}")
    
    total_config_features = len(time_varying_known) + len(time_varying_unknown)
    print(f"\n➡️  Total Features (config): {total_config_features}")
    print(f"➡️  Total Features (actual data): {num_features}")
    
    # Load and print actual final features being used
    # Note: Config shows 14 base features, but data has more after pivoting (e.g., close_SPY, close_QQQ)
    try:
        # Load from metadata.yaml (should exist in data/processed/)
        metadata_path = Path('data/processed/metadata.yaml')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = yaml.safe_load(f)
                if 'features' in metadata:
                    actual_features = metadata['features']
                    print(f"\n📋 ACTUAL FEATURES USED IN TRAINING ({len(actual_features)}):")
                    print(f"   (Ticker-specific features created through pivoting)")
                    print(f"")
                    for i, feat in enumerate(actual_features, 1):
                        print(f"   {i:2d}. {feat}")
                else:
                    print(f"\n⚠️  'features' key not found in metadata.yaml")
        else:
            print(f"\n⚠️  metadata.yaml not found at {metadata_path}")
            print(f"   Cannot display individual feature names")
            print(f"   Config features (14) are expanded to {num_features} after pivoting")
    except Exception as e:
        print(f"\n⚠️  Could not load actual feature names: {e}")
        import traceback
        traceback.print_exc()
    
    # Output targets
    horizons_config = config['data']['prediction_horizons']
    print(f"\n🎯 Output Targets ({len(horizons_config)} horizons):")
    for i, h in enumerate(horizons_config, 1):
        print(f"  {i}. Horizon {h} (target_{h}_periods_ahead)")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Model hyperparameters from config
    model_config = config['model']
    training_config = config['training']
    
    print("\n" + "="*80)
    print("   Model Configuration")
    print("="*80)
    print(f"\nArchitecture: Temporal Fusion Transformer")
    print(f"  Hidden size: {model_config['hidden_size']}")
    print(f"  LSTM layers: {model_config['lstm_layers']}")
    print(f"  Attention heads: {model_config['attention_heads']}")
    print(f"  Dropout: {model_config['dropout']}")
    print(f"\nTraining:")
    print(f"  Epochs: {training_config['epochs']}")
    print(f"  Batch size: {training_config['batch_size']}")
    print(f"  Learning rate: {training_config['learning_rate']}")
    print(f"  Early stopping patience: {training_config['early_stopping']['patience']}")
    
    # Setup TensorBoard
    writer = None
    if config.get('logging', {}).get('tensorboard', False):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Check for Vertex AI TensorBoard directory (set automatically by Vertex AI)
        tensorboard_log_dir = os.getenv('AIP_TENSORBOARD_LOG_DIR')
        
        if tensorboard_log_dir:
            # Vertex AI managed TensorBoard - logs auto-sync
            log_dir = tensorboard_log_dir
            writer = SummaryWriter(str(log_dir))
            print(f"\n📊 TensorBoard (Vertex AI): {log_dir}")
            print(f"   Logs will auto-sync to TensorBoard instance")
        else:
            # Local or manual TensorBoard - use local paths
            log_dir = Path(config.get('logging', {}).get('log_dir', 'logs/tensorboard')) / timestamp
            log_dir.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(str(log_dir))
            print(f"\n📊 TensorBoard logs → {log_dir}")
            print(f"   View with: tensorboard --logdir {log_dir.parent}")
    
    # Initialize TFT model
    print("\n" + "="*80)
    print("   Initializing TFT Model")
    print("="*80)
    
    # Pass actual feature count from data (after pivoting)
    model = SimplifiedTFT(config, num_features=num_features).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Setup optimizer and loss
    optimizer = optim.Adam(
        model.parameters(),
        lr=training_config['learning_rate'],
        weight_decay=training_config.get('weight_decay', 0.0)  # L2 regularization
    )
    criterion = nn.MSELoss()
    
    # Create output directory
    output_dir = Path('models/tft')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print("\n" + "="*80)
    print("   Training Loop")
    print("="*80)
    
    best_val_loss = float('inf')
    best_val_mae = float('inf')
    patience_counter = 0
    
    for epoch in range(training_config['epochs']):
        # Training phase
        train_results = train_epoch(model, dataloaders['train'], criterion, optimizer, device, epoch=epoch+1)
        train_loss = train_results['loss']
        avg_grad_norm = train_results['avg_grad_norm']
        max_grad_norm = train_results['max_grad_norm']
        min_grad_norm = train_results['min_grad_norm']
        layer_grad_stats = train_results['layer_grad_stats']
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in dataloaders['val']:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                
                val_loss += loss.item()
                val_batches += 1
                
                all_preds.append(predictions)
                all_targets.append(batch_y)
        
        val_loss = val_loss / val_batches
        
        # Compute metrics
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = compute_metrics(all_preds, all_targets)
        
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{training_config['epochs']}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}")
        print(f"  Dir Acc: {metrics['dir_acc']:.2f}%")
        print(f"  Grad Norm: avg={avg_grad_norm:.4f}, max={max_grad_norm:.4f}, min={min_grad_norm:.4f}")
        
        # Print layer-wise gradient stats
        if layer_grad_stats:
            print(f"  Layer Gradients (batch 1):")
            for layer_name in ['VSN', 'Input', 'LSTM_L0', 'LSTM_L1', 'LSTM_L2', 'Attention', 'Feedforward', 'Output']:
                if layer_name in layer_grad_stats:
                    stats = layer_grad_stats[layer_name]
                    print(f"    {layer_name:12s}: norm={stats['avg_norm']:.4f}, max={stats['max_norm']:.4f}, std={stats['avg_std']:.4f}")
        
        # Log to TensorBoard
        if writer is not None:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Metrics/MAE', metrics['mae'], epoch)
            writer.add_scalar('Metrics/RMSE', metrics['rmse'], epoch)
            writer.add_scalar('Metrics/DirectionalAccuracy', metrics['dir_acc'], epoch)
            writer.add_scalar('LR', training_config['learning_rate'], epoch)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_mae = metrics['mae']
            patience_counter = 0
            
            # Save checkpoint
            checkpoint_path = output_dir / 'tft_best.pt'
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mae': metrics['mae'],
                'val_dir_acc': metrics['dir_acc'],
                'config': config,
                'scalers': scalers
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"  ✅ Saved checkpoint: {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= training_config['early_stopping']['patience']:
                print(f"\n⏹️  Early stopping triggered (patience: {patience_counter})")
                break
    
    # Close TensorBoard writer
    if writer is not None:
        writer.close()
    
    print("\n" + "="*80)
    print("   Training Complete")
    print("="*80)
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Best validation MAE: {best_val_mae:.4f}")
    print(f"  Model saved: {output_dir / 'tft_best.pt'}")
    if writer is not None and not os.getenv('CLOUD_ML_JOB_ID'):
        print(f"\n📊 View TensorBoard: tensorboard --logdir logs/tensorboard")
    print("="*80)


if __name__ == '__main__':
    # Allow running directly for testing
    import argparse
    
    parser = argparse.ArgumentParser(description='Core TFT training function')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/model_tft_config.yaml',
        help='Path to model config YAML'
    )
    args = parser.parse_args()
    
    train(args.config)