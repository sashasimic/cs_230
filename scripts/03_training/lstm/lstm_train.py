#!/usr/bin/env python3
"""
LSTM Multi-Horizon Time Series Forecasting

Compatible with decoder_transformer dataset format:
- Input X: [batch, lookback, features]
- Output y: [batch, num_horizons]

Supports both local and Vertex AI training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import yaml
import os
from pathlib import Path
from typing import Optional, Dict, Tuple
import time

# Try to import TensorBoard (optional for local training)
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except Exception as e:
    TENSORBOARD_AVAILABLE = False
    print("⚠️  TensorBoard not available (import failed)")
    print("   Training will continue without TensorBoard logging")
    # Uncomment to see full error:
    # print(f"   Error: {e}")


class LSTMMultiHorizon(nn.Module):
    """
    LSTM model for multi-horizon time series forecasting.
    
    Architecture:
    - LSTM layers process historical sequence [batch, lookback, features]
    - Dense layers map LSTM output to multiple horizons [batch, num_horizons]
    
    Note: Uses forward-only LSTM (no bidirectional) to prevent data leakage.
    
    Args:
        num_features: Input feature dimension
        hidden_dim: LSTM hidden dimension
        num_layers: Number of LSTM layers
        num_horizons: Number of prediction horizons
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_horizons: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_horizons = num_horizons
        
        # LSTM layers (forward-only for forecasting)
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False  # Never use bidirectional for forecasting!
        )
        
        # Dropout after LSTM
        self.dropout = nn.Dropout(dropout)
        
        # Dense layers for multi-horizon prediction
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_horizons)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, features]
        Returns:
            predictions: [batch, num_horizons]
        """
        # LSTM forward pass
        # output: [batch, seq_len, hidden_dim]
        # h_n: [num_layers, batch, hidden_dim]
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last time step output
        last_output = lstm_out[:, -1, :]  # [batch, hidden_dim]
        
        # Apply dropout
        x = self.dropout(last_output)
        
        # Dense layers for prediction
        x = self.fc1(x)          # [batch, hidden_dim]
        x = self.relu(x)
        x = self.dropout(x)
        predictions = self.fc2(x)  # [batch, num_horizons]
        
        return predictions


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    lossFun: nn.Module,
    device: torch.device,
    clip_norm: Optional[float] = None
) -> Tuple[float, float, float, float, float]:
    """Train for one epoch.
    
    Returns:
        train_loss, avg_unclipped_norm, max_unclipped_norm, avg_clipped_norm, max_clipped_norm
    """
    model.train()
    total_loss = 0.0
    unclipped_norms = []
    clipped_norms = []
    max_unclipped = 0.0
    max_clipped = 0.0
    
    total_batches = len(train_loader)
    
    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = lossFun(predictions, y_batch)
        
        # Backward pass
        loss.backward()
        
        # Compute gradient norms before clipping
        total_norm = sum(
            p.grad.detach().norm(2).item() ** 2
            for p in model.parameters() if p.grad is not None
        ) ** 0.5

        unclipped_norms.append(total_norm)
        max_unclipped = max(max_unclipped, total_norm)
        
        # Gradient clipping
        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            
            # Compute gradient norms after clipping
            total_norm_clipped = sum(
                p.grad.detach().norm(2).item() ** 2
                for p in model.parameters() if p.grad is not None
            ) ** 0.5
            
            clipped_norms.append(total_norm_clipped)
            max_clipped = max(max_clipped, total_norm_clipped)
        else:
            clipped_norms.append(total_norm)
            max_clipped = max_unclipped
        
        # Update parameters using p.grad
        optimizer.step()
        
        # Accumulate loss from this batch
        total_loss += loss.item()
        
        # Progress indicator
        if (batch_idx + 1) % max(1, total_batches // 3) == 0 or batch_idx == total_batches - 1:
            print(f"\r  Training: {batch_idx + 1}/{total_batches} batches (loss: {loss.item():.4f})", end='', flush=True)
    
    print()  # New line after progress
    avg_loss = total_loss / len(train_loader)
    avg_unclipped = np.mean(unclipped_norms)
    avg_clipped = np.mean(clipped_norms)
    
    return avg_loss, avg_unclipped, max_unclipped, avg_clipped, max_clipped


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    lossFun: nn.Module,
    device: torch.device,
    horizons: Optional[list] = None
) -> Tuple[float, float, float, float, Dict[str, float]]:
    """Evaluate model on validation set.
    
    Args:
        horizons: List of horizon values [7, 14, 30] for logging
    
    Returns:
        val_loss, mae, rmse, directional_accuracy, per_horizon_metrics
    """
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    total_batches = len(val_loader)
    print(f"  Validation: 0/{total_batches} batches", end='', flush=True)
    
    with torch.no_grad():
        for batch_idx, (X_batch, y_batch) in enumerate(val_loader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            predictions = model(X_batch)
            loss = lossFun(predictions, y_batch)
            
            total_loss += loss.item()
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
    
    print(f"\r  Validation: {total_batches}/{total_batches} batches", end='')
    print()  # New line
    
    # Concatenate all batches
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Compute overall metrics
    val_loss = total_loss / len(val_loader)
    mae = np.mean(np.abs(all_predictions - all_targets))
    rmse = np.sqrt(np.mean((all_predictions - all_targets) ** 2))
    
    # Compute per-horizon metrics
    per_horizon_metrics = {}
    num_horizons = all_predictions.shape[1]
    for h_idx in range(num_horizons):
        h_mae = np.mean(np.abs(all_predictions[:, h_idx] - all_targets[:, h_idx]))
        h_rmse = np.sqrt(np.mean((all_predictions[:, h_idx] - all_targets[:, h_idx]) ** 2))
        
        # Use actual horizon values for labeling (e.g., H7, H14, H30)
        horizon_label = f"H{horizons[h_idx]}" if (horizons and h_idx < len(horizons)) else f"H{h_idx+1}"
        
        per_horizon_metrics[f"{horizon_label}_MAE"] = h_mae
        per_horizon_metrics[f"{horizon_label}_RMSE"] = h_rmse
    
    # Directional accuracy (for first horizon only)
    pred_diff = np.diff(all_predictions[:, 0])  # Direction of predictions
    true_diff = np.diff(all_targets[:, 0])      # True direction
    correct_direction = np.sum((pred_diff * true_diff) > 0)
    dir_acc = correct_direction / len(pred_diff) if len(pred_diff) > 0 else 0.0
    
    return val_loss, mae, rmse, dir_acc, per_horizon_metrics


def train(
    config_path: str,
    dataloaders: Optional[Dict] = None,
    scalers: Optional[Dict] = None,
    dataset_version: Optional[str] = None
):
    """
    Core LSTM training function.
    
    Args:
        config_path: Path to model config YAML
        dataloaders: Optional pre-loaded DataLoaders (if None, will load from data/processed/)
        scalers: Optional pre-loaded scalers
        dataset_version: Optional dataset version (e.g., 'v1', 'v3'). Data always loaded from data/processed/ (copied by *_train_local.py or downloaded by train_vertex.py)
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seeds for reproducibility
    seed = config.get('training', {}).get('random_seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print("\n" + "="*80)
    print("   LSTM Multi-Horizon Forecasting")
    print("="*80)
    print(f"\n🎲 Random seed: {seed}")
    
    # Load data if not provided
    if dataloaders is None:
        # Always use data/processed/ (unified behavior for local and Vertex AI)
        # Local: Copied from versioned dataset by *_train_local.py
        # Vertex AI: Downloaded from GCS by train_vertex.py
        data_path = Path('data/processed')
        print(f"\n📂 Loading data from: {data_path}...")
        
        # Verify path exists
        if not data_path.exists():
            if dataset_version:
                raise FileNotFoundError(
                    f"Data directory not found: {data_path}\n"
                    f"For local training, run:\n"
                    f"  python scripts/03_training/lstm/lstm_train_local.py --dataset-version {dataset_version}"
                )
            else:
                raise FileNotFoundError(f"Data directory not found: {data_path}")
        
        # Load preprocessed arrays
        train_X_np = np.load(data_path / 'X_train.npy', allow_pickle=True)
        train_y_np = np.load(data_path / 'y_train.npy', allow_pickle=True)
        val_X_np = np.load(data_path / 'X_val.npy', allow_pickle=True)
        val_y_np = np.load(data_path / 'y_val.npy', allow_pickle=True)
        test_X_np = np.load(data_path / 'X_test.npy', allow_pickle=True)
        test_y_np = np.load(data_path / 'y_test.npy', allow_pickle=True)
        
        # Handle object dtype (sometimes happens with numpy saves)
        if train_X_np.dtype == object:
            train_X_np = train_X_np.item() if train_X_np.shape == () else np.array(train_X_np.tolist())
        if train_y_np.dtype == object:
            train_y_np = train_y_np.item() if train_y_np.shape == () else np.array(train_y_np.tolist())
        if val_X_np.dtype == object:
            val_X_np = val_X_np.item() if val_X_np.shape == () else np.array(val_X_np.tolist())
        if val_y_np.dtype == object:
            val_y_np = val_y_np.item() if val_y_np.shape == () else np.array(val_y_np.tolist())
        if test_X_np.dtype == object:
            test_X_np = test_X_np.item() if test_X_np.shape == () else np.array(test_X_np.tolist())
        if test_y_np.dtype == object:
            test_y_np = test_y_np.item() if test_y_np.shape == () else np.array(test_y_np.tolist())
        
        # Convert to tensors
        train_X = torch.FloatTensor(train_X_np.astype(np.float32))
        train_y = torch.FloatTensor(train_y_np.astype(np.float32))
        val_X = torch.FloatTensor(val_X_np.astype(np.float32))
        val_y = torch.FloatTensor(val_y_np.astype(np.float32))
        test_X = torch.FloatTensor(test_X_np.astype(np.float32))
        test_y = torch.FloatTensor(test_y_np.astype(np.float32))
        
        # Create simple datasets
        train_dataset = TensorDataset(train_X, train_y)
        val_dataset = TensorDataset(val_X, val_y)
        test_dataset = TensorDataset(test_X, test_y)
        
        # Create DataLoaders
        batch_size = config['training']['batch_size']
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        print("✅ Data loaded!")
    else:
        train_loader = dataloaders['train']
        val_loader = dataloaders['val']
        test_loader = dataloaders['test']
        train_X = None  # Will get from dataset
    
    # Setup output directory
    output_dir = Path('models/lstm')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get dimensions from dataset
    if train_X is not None:
        num_features = train_X.shape[-1]
        lookback = train_X.shape[1]
        num_horizons = train_y.shape[1]
    else:
        train_dataset = train_loader.dataset
        sample_X, sample_y = train_dataset[0]
        num_features = sample_X.shape[-1]
        lookback = sample_X.shape[0]
        num_horizons = sample_y.shape[0]
    
    # Print configuration
    print("\n" + "="*80)
    print("   LSTM Configuration")
    print("="*80)
    
    # Get prediction horizons - read from dataset metadata if available, otherwise from config
    horizons = None
    if dataset_version:
        # Read horizons from data/processed/metadata.yaml (populated by local copy or Vertex AI download)
        try:
            metadata_path = Path('data/processed/metadata.yaml')
            
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    dataset_metadata = yaml.safe_load(f)
                    horizons = dataset_metadata.get('prediction_horizons', None)
                    if horizons:
                        print(f"\n✅ Using prediction horizons from dataset metadata: {horizons}")
        except Exception as e:
            print(f"\n⚠️  Could not read horizons from metadata: {e}")
    
    # Fallback to config if not found in metadata
    if horizons is None:
        horizons = config['data'].get('prediction_horizons', [])
        print(f"\n📄 Using prediction horizons from config: {horizons}")
    
    print(f"\n📊 Data Dimensions:")
    print(f"  Lookback window: {lookback} timesteps")
    print(f"  Number of features: {num_features}")
    print(f"  Number of horizons: {num_horizons}")
    if horizons:
        print(f"  Prediction horizons: {horizons} days ahead")
    
    # Display date range and sample counts from metadata (actual data)
    try:
        # Always use data/processed/metadata.yaml (populated by local copy or Vertex AI download)
        metadata_path = Path('data/processed/metadata.yaml')
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = yaml.safe_load(f)
                start_date = metadata.get('start_date', 'N/A')
                end_date = metadata.get('end_date', 'N/A')
                train_samples = metadata.get('train_samples', 0)
                val_samples = metadata.get('val_samples', 0)
                test_samples = metadata.get('test_samples', 0)
                total = train_samples + val_samples + test_samples
                
                print(f"\n📅 Date Range (from actual data):")
                print(f"  Start: {start_date}")
                print(f"  End: {end_date}")
                if total > 0:
                    print(f"  Total sequences: {total:,} (train: {train_samples}, val: {val_samples}, test: {test_samples})")
        else:
            # Fallback to config if metadata doesn't exist
            if 'data' in config:
                data_cfg = config['data']
                start_date = data_cfg.get('start_date', 'N/A')
                end_date = data_cfg.get('end_date', 'N/A')
                print(f"\n📅 Date Range (from config):")
                print(f"  Start: {start_date}")
                print(f"  End: {end_date}")
    except Exception as e:
        print(f"\n⚠️  Could not load date range from metadata: {e}")
        # Fallback to config
        if 'data' in config:
            data_cfg = config['data']
            start_date = data_cfg.get('start_date', 'N/A')
            end_date = data_cfg.get('end_date', 'N/A')
            print(f"\n📅 Date Range (from config):")
            print(f"  Start: {start_date}")
            print(f"  End: {end_date}")
    
    print(f"\n🖥️  Device: {device}")
    
    print("\n" + "="*80)
    print("   Model Architecture")
    print("="*80)
    
    model_cfg = config['model']
    hidden_dim = model_cfg['hidden_dim']
    num_layers = model_cfg['num_layers']
    dropout = model_cfg['dropout']
    
    print(f"\nArchitecture: LSTM Multi-Horizon (Forward-Only)")
    print(f"  hidden_dim: {hidden_dim}")
    print(f"  num_layers: {num_layers}")
    print(f"  dropout: {dropout}")
    
    train_cfg = config['training']
    print(f"\nTraining:")
    print(f"  Epochs: {train_cfg['epochs']}")
    print(f"  Batch size: {train_cfg['batch_size']}")
    print(f"  Learning rate: {train_cfg['learning_rate']}")
    print(f"  Gradient clip: {train_cfg.get('gradient_clip_norm', 'None')}")
    
    # Initialize model
    model = LSTMMultiHorizon(
        num_features=num_features,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_horizons=num_horizons,
        dropout=dropout
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n🔧 Model Parameters:")
    print(f"   Total: {total_params:,}")
    print(f"   Trainable: {trainable_params:,}")
    
    # Setup TensorBoard (optional)
    writer = None
    log_dir = None
    job_name = os.getenv('JOB_NAME', f'lstm-local-{time.strftime("%Y%m%d_%H%M%S")}')
    
    if TENSORBOARD_AVAILABLE:
        # Check for Vertex AI TensorBoard directory (set automatically by Vertex AI)
        tensorboard_log_dir = os.getenv('AIP_TENSORBOARD_LOG_DIR')
        
        if tensorboard_log_dir:
            # Vertex AI managed TensorBoard - logs auto-sync in real-time
            log_dir = tensorboard_log_dir
            writer = SummaryWriter(str(log_dir))
            print(f"\n📊 TensorBoard (Vertex AI): {log_dir}")
            print(f"   Logs will auto-sync to TensorBoard instance")
            
        else:
            # Local or manual TensorBoard - use local paths
            log_dir = Path('logs/tensorboard') / job_name
            log_dir.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(str(log_dir))
            print(f"\n📊 TensorBoard logs → {log_dir}")
            print(f"   View with: tensorboard --logdir logs/tensorboard")
    else:
        print(f"\n📊 TensorBoard disabled (not available)")
    
    # Loss and optimizer
    lossFun = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=train_cfg['learning_rate'],
        weight_decay=train_cfg.get('weight_decay', 0.0)
    )
    
    # Training loop configuration
    best_val_loss = float('inf')
    best_val_mae = float('inf')
    patience_counter = 0
    epochs = train_cfg['epochs']
    clip_norm = train_cfg.get('gradient_clip_norm', None)
    early_stopping_patience = train_cfg.get('early_stopping', {}).get('patience', 10)
    
    print("\n" + "="*80)
    print("   TRAINING CONFIGURATION")
    print("="*80)
    print(f"\n🚀 Starting training for {epochs} epochs")
    print(f"   Device: {device}")
    print(f"   Batch size: {train_cfg['batch_size']}")
    print(f"   Learning rate: {train_cfg['learning_rate']}")
    print(f"   Gradient clipping: {clip_norm}")
    print(f"   Early stopping patience: {early_stopping_patience}")
    print(f"\n⏱️  Estimated time:")
    print(f"   Per epoch: 0.5-2 minutes")
    print(f"   Total ({epochs} epochs): {epochs * 1 / 60:.1f} hours")
    print(f"   Training batches per epoch: {len(train_loader)}")
    print("\n" + "="*80)
    
    training_start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        train_loss, avg_unclipped, max_unclipped, avg_clipped, max_clipped = train_epoch(
            model, train_loader, optimizer, lossFun, device, clip_norm
        )
        
        val_loss, mae, rmse, dir_acc, per_horizon_metrics = evaluate(
            model, val_loader, lossFun, device, horizons
        )
        
        epoch_time = time.time() - epoch_start_time
        total_elapsed = time.time() - training_start_time
        
        print(f"\nEpoch {epoch+1}/{epochs} - {epoch_time/60:.1f} min (total: {total_elapsed/60:.1f} min)")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val   Loss: {val_loss:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}")
        print(f"  Dir Acc (H1): {dir_acc * 100:.2f}%")
        
        # Log per-horizon metrics
        if per_horizon_metrics:
            horizon_strs = []
            for key, value in sorted(per_horizon_metrics.items()):
                if 'MAE' in key:
                    horizon_strs.append(f"{key}={value:.6f}")
            if horizon_strs:
                print(f"  Per-Horizon MAE: {', '.join(horizon_strs)}")
        
        print(f"  Grad Norm (unclipped): avg={avg_unclipped:.4f}, max={max_unclipped:.4f}")
        print(f"  Grad Norm (clipped):   avg={avg_clipped:.4f}, max={max_clipped:.4f}")
        
        # Log to TensorBoard (if available)
        if writer is not None:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Metrics/MAE', mae, epoch)
            writer.add_scalar('Metrics/RMSE', rmse, epoch)
            writer.add_scalar('Metrics/DirectionalAccuracy', dir_acc, epoch)
            writer.add_scalar('Gradients/Unclipped_Avg', avg_unclipped, epoch)
            writer.add_scalar('Gradients/Clipped_Avg', avg_clipped, epoch)
            writer.add_scalar('LR', train_cfg['learning_rate'], epoch)
            
            # Log per-horizon metrics to TensorBoard
            for metric_name, metric_value in per_horizon_metrics.items():
                writer.add_scalar(f'PerHorizon/{metric_name}', metric_value, epoch)
        
        print("")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_mae = mae
            patience_counter = 0
            
            # Save checkpoint
            checkpoint_path = output_dir / 'lstm_best.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mae': mae,
                'val_dir_acc': dir_acc,
                'config': config
            }, checkpoint_path)
            print(f"  ✅ Saved checkpoint: {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"\n⏹️  Early stopping triggered (patience: {early_stopping_patience})")
                break
    
    # Close TensorBoard writer (if available)
    if writer is not None:
        writer.close()
        print(f"\n📊 TensorBoard logs finalized")
    
    # Final evaluation on test set
    print("\n" + "="*80)
    print("   Final Test Set Evaluation")
    print("="*80)
    
    # Load best model (weights_only=False is safe for our own checkpoint)
    checkpoint = torch.load(output_dir / 'lstm_best.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_mae, test_rmse, test_dir_acc, test_per_horizon = evaluate(
        model, test_loader, lossFun, device, horizons
    )
    
    print(f"\n📊 Test Set Results:")
    print(f"  Test Loss: {test_loss:.6f}")
    print(f"  Test MAE: {test_mae:.6f}")
    print(f"  Test RMSE: {test_rmse:.6f}")
    print(f"  Test Dir Acc (H1): {test_dir_acc * 100:.2f}%")
    
    # Log per-horizon test metrics
    if test_per_horizon:
        horizon_strs = []
        for key, value in sorted(test_per_horizon.items()):
            if 'MAE' in key:
                horizon_strs.append(f"{key}={value:.6f}")
        if horizon_strs:
            print(f"  Per-Horizon MAE: {', '.join(horizon_strs)}")
    
    print("\n" + "="*80)
    print("   Training Complete")
    print("="*80)
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Best validation MAE: {best_val_mae:.4f}")
    print(f"  Model saved: {output_dir / 'lstm_best.pt'}")
    if not os.getenv('CLOUD_ML_JOB_ID'):
        print(f"\n📊 View TensorBoard: tensorboard --logdir logs/tensorboard")
    print("="*80)
    
    return model, best_val_loss