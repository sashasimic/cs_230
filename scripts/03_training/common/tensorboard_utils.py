#!/usr/bin/env python3
"""
TensorBoard Utilities for Model Training

Centralized TensorBoard logging functions to keep training scripts clean and maintainable.
Provides functions for:
- Writer initialization
- Dataset and model metadata logging
- Hyperparameter logging (HParams dashboard)
- Gradient and weight histogram logging
- Attention visualization
- Metric logging helpers
"""

import os
import torch
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List


def initialize_tensorboard_writer(
    config: Dict[str, Any],
    model_name: str,
    eval_suffix: str = ""
) -> Optional[Any]:
    """
    Initialize TensorBoard SummaryWriter with appropriate directory.
    
    Args:
        config: Training configuration dict
        model_name: Name of the model (e.g., 'decoder', 'lstm', 'tft')
        eval_suffix: Optional suffix for eval mode (e.g., '_tf' or '_ar')
    
    Returns:
        SummaryWriter instance or None if TensorBoard disabled
    """
    writer = None
    
    tensorboard_enabled = config.get('logging', {}).get('tensorboard', False)
    print(f"   TensorBoard enabled in config: {tensorboard_enabled}")
    
    if not tensorboard_enabled:
        print(f"   ⚠️  TensorBoard is DISABLED in config - returning None")
        return None
    
    try:
        # Lazy import - only load when actually needed
        from torch.utils.tensorboard import SummaryWriter
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Check for Vertex AI TensorBoard directory (set automatically by Vertex AI)
        tensorboard_log_dir = os.getenv('AIP_TENSORBOARD_LOG_DIR')
        print(f"   AIP_TENSORBOARD_LOG_DIR env var: {tensorboard_log_dir}")
        
        if tensorboard_log_dir:
            # Vertex AI managed TensorBoard - logs auto-sync
            log_dir = tensorboard_log_dir
            writer = SummaryWriter(str(log_dir))
            print(f"\n📊 TensorBoard (Vertex AI): {log_dir}")
            print(f"   Logs will auto-sync to TensorBoard instance")
        else:
            # Local or manual TensorBoard - use local paths
            base_log_dir = Path(config.get('logging', {}).get('log_dir', 'logs/tensorboard'))
            log_dir = base_log_dir / f"{model_name}{eval_suffix}" / timestamp
            log_dir.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(str(log_dir))
            print(f"\n📊 TensorBoard logs → {log_dir}")
            print(f"   View with: tensorboard --logdir {base_log_dir}")
            
    except Exception as e:
        print(f"\n⚠️  TensorBoard not available: {type(e).__name__}")
        print(f"   Error details: {str(e)}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        print("   Training will continue without TensorBoard logging")
        writer = None
    
    return writer


def log_dataset_info(
    writer: Any,
    dataset_version: Optional[str],
    start_date: str,
    end_date: str,
    horizons: List[int],
    lookback: int,
    num_features: int,
    num_horizons: int,
    train_samples: int,
    val_samples: int,
    test_samples: int
) -> None:
    """
    Log dataset information to TensorBoard Text tab.
    
    Args:
        writer: TensorBoard SummaryWriter
        dataset_version: Version string or None
        start_date: Dataset start date
        end_date: Dataset end date
        horizons: List of prediction horizons
        lookback: Lookback window size
        num_features: Number of features
        num_horizons: Number of prediction horizons
        train_samples: Number of training samples
        val_samples: Number of validation samples
        test_samples: Number of test samples
    """
    if writer is None:
        return
    
    try:
        dataset_info = f"""
# Dataset Information

**Version**: {dataset_version if dataset_version else 'N/A'}  
**Date Range**: {start_date} to {end_date}  
**Prediction Horizons**: {horizons}  

## Data Dimensions
- **Lookback Window**: {lookback} timesteps
- **Features**: {num_features}
- **Horizons**: {num_horizons}

## Sample Counts
- **Train**: {train_samples:,}
- **Validation**: {val_samples:,}
- **Test**: {test_samples:,}
- **Total**: {train_samples + val_samples + test_samples:,}
"""
        writer.add_text('Experiment/Dataset', dataset_info, 0)
    except Exception as e:
        print(f"   ⚠️  Failed to log dataset info: {e}")


def log_model_info(
    writer: Any,
    model_type: str,
    config: Dict[str, Any],
    total_params: int,
    trainable_params: int,
    additional_info: Optional[Dict[str, str]] = None
) -> None:
    """
    Log model architecture information to TensorBoard Text tab.
    
    Args:
        writer: TensorBoard SummaryWriter
        model_type: Type of model (e.g., 'Decoder Transformer', 'LSTM', 'TFT')
        config: Model configuration dict
        total_params: Total number of parameters
        trainable_params: Number of trainable parameters
        additional_info: Optional dict of additional info to display
    """
    if writer is None:
        return
    
    try:
        model_cfg = config.get('model', {})
        
        # Build architecture details section
        arch_details = []
        common_keys = ['hidden_dim', 'd_model', 'num_layers', 'n_layers', 'num_heads', 'n_heads', 
                      'feedforward_dim', 'd_ff', 'dropout', 'num_lstm_layers', 'num_attention_heads']
        
        for key in common_keys:
            if key in model_cfg:
                # Convert key to display name
                display_name = key.replace('_', ' ').title().replace('D Model', 'Hidden Dimension')
                display_name = display_name.replace('D Ff', 'Feedforward Dimension')
                display_name = display_name.replace('N Layers', 'Num Layers')
                display_name = display_name.replace('N Heads', 'Num Heads')
                arch_details.append(f"- **{display_name}**: {model_cfg[key]}")
        
        arch_section = "\n".join(arch_details) if arch_details else "- Configuration details not available"
        
        # Build additional info section
        additional_section = ""
        if additional_info:
            additional_section = "\n## Additional Configuration\n"
            for key, value in additional_info.items():
                additional_section += f"- **{key}**: {value}\n"
        
        model_info = f"""
# Model Architecture: {model_type}

## Architecture Details
{arch_section}

## Parameters
- **Total Parameters**: {total_params:,}
- **Trainable Parameters**: {trainable_params:,}
{additional_section}
"""
        writer.add_text('Experiment/Model', model_info, 0)
    except Exception as e:
        print(f"   ⚠️  Failed to log model info: {e}")


def log_experiment_metadata(
    writer: Any,
    dataset_version: Optional[str],
    start_date: str,
    end_date: str,
    horizons: List[int],
    lookback: int,
    num_features: int,
    num_horizons: int,
    train_samples: int,
    val_samples: int,
    test_samples: int,
    model_type: str,
    config: Dict[str, Any],
    total_params: int,
    trainable_params: int,
    additional_model_info: Optional[Dict[str, str]] = None
) -> None:
    """
    Log both dataset and model information in one call.
    
    Convenience function that calls both log_dataset_info and log_model_info.
    """
    log_dataset_info(
        writer, dataset_version, start_date, end_date, horizons,
        lookback, num_features, num_horizons,
        train_samples, val_samples, test_samples
    )
    
    log_model_info(
        writer, model_type, config, total_params, trainable_params,
        additional_model_info
    )
    
    if writer is not None:
        print("   ✅ Logged dataset and model info to TensorBoard")


def log_training_hyperparameters(
    writer: Any,
    training_config: Dict[str, Any],
    model_config: Dict[str, Any]
) -> None:
    """
    Log training hyperparameters as text to TensorBoard for easy comparison.
    
    Args:
        writer: TensorBoard SummaryWriter
        training_config: Training configuration dict
        model_config: Model configuration dict
    """
    if writer is None:
        return
    
    try:
        # Build architecture section with conditional components
        arch_lines = [
            f"- **Hidden Size**: {model_config.get('hidden_size', 'N/A')}"
        ]
        
        # Only show LSTM if enabled
        if model_config.get('use_lstm', False):
            arch_lines.append(f"- **LSTM Layers**: {model_config.get('lstm_layers', 1)}")
        
        arch_lines.extend([
            f"- **Attention Layers**: {model_config.get('attention_layers', 1)}",
            f"- **Attention Heads**: {model_config.get('attention_heads', 'N/A')}"
        ])
        
        # Show optional components if enabled
        if model_config.get('use_variable_selection', False):
            arch_lines.append("- **Variable Selection Network**: Enabled")
        if model_config.get('use_static_enrichment', False):
            arch_lines.append("- **Static Enrichment**: Enabled")
        if model_config.get('use_position_wise_grn', False):
            arch_lines.append("- **Position-wise GRN**: Enabled")
        
        arch_section = "\n".join(arch_lines)
        
        # Extract key hyperparameters
        hparam_text = f"""
# Training Hyperparameters

## Architecture
{arch_section}

## Regularization
- **Dropout**: {model_config.get('dropout', 'N/A')}
- **Weight Decay (L2)**: {training_config.get('weight_decay', 0.0)}

## Training
- **Learning Rate**: {training_config.get('learning_rate', 'N/A')}
- **Batch Size**: {training_config.get('batch_size', 'N/A')}
- **Epochs**: {training_config.get('epochs', 'N/A')}
- **Gradient Clip Norm**: {training_config.get('gradient_clip_norm', 1.0)}

## Learning Rate Scheduler
- **Enabled**: {training_config.get('lr_scheduler', {}).get('enabled', False)}
- **Type**: {training_config.get('lr_scheduler', {}).get('type', 'N/A')}
- **Factor**: {training_config.get('lr_scheduler', {}).get('factor', 'N/A')}
- **Patience**: {training_config.get('lr_scheduler', {}).get('patience', 'N/A')}

## Early Stopping
- **Enabled**: {training_config.get('early_stopping', {}).get('enabled', False)}
- **Patience**: {training_config.get('early_stopping', {}).get('patience', 'N/A')}
"""
        writer.add_text('Experiment/Hyperparameters', hparam_text, 0)
        print("   ✅ Logged training hyperparameters to TensorBoard")
    except Exception as e:
        print(f"   ⚠️  Failed to log training hyperparameters: {e}")


def log_hyperparameters(
    writer: Any,
    hparams: Dict[str, Any],
    metrics: Dict[str, float]
) -> None:
    """
    Log hyperparameters to TensorBoard HParams dashboard.
    
    Args:
        writer: TensorBoard SummaryWriter
        hparams: Dictionary of hyperparameters
        metrics: Dictionary of metrics (must start with 'hparam/')
    """
    if writer is None:
        return
    
    try:
        writer.add_hparams(hparams, metrics)
        print("\n✅ Logged hyperparameters to TensorBoard HParams dashboard")
    except Exception as e:
        print(f"\n⚠️  Failed to log hyperparameters: {e}")


def log_gradients_and_weights(
    writer: Any,
    model: torch.nn.Module,
    epoch: int
) -> None:
    """
    Log gradient and weight histograms to TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter
        model: PyTorch model
        epoch: Current epoch number
    """
    if writer is None:
        return
    
    try:
        for name, param in model.named_parameters():
            if param.grad is not None:
                # Log gradient distributions
                writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
            # Log weight distributions
            writer.add_histogram(f'Weights/{name}', param.data, epoch)
    except Exception as e:
        print(f"  ⚠️  Failed to log histograms: {e}")


def visualize_attention_weights(
    model: torch.nn.Module,
    data_loader: Any,
    device: torch.device,
    writer: Any,
    epoch: int,
    num_samples: int = 2
) -> None:
    """
    Visualize attention weights from transformer encoder.
    
    Args:
        model: The transformer model
        data_loader: DataLoader to get samples from
        device: Device to run on
        writer: TensorBoard writer
        epoch: Current epoch number
        num_samples: Number of samples to visualize
    """
    if writer is None:
        return
    
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        
        model.eval()
        with torch.no_grad():
            # Get a batch
            for X_batch, _ in data_loader:
                X_batch = X_batch.to(device)
                
                batch_size, seq_len, num_features = X_batch.shape
                
                # Take first samples
                sample = X_batch[0:num_samples]
                
                # Project to d_model space
                x_proj = model.input_projection(sample)
                x_proj = model.pos_encoder(x_proj)
                
                # Visualize the norm of each timestep's representation
                norms = torch.norm(x_proj, dim=-1).cpu().numpy()
                
                for i in range(min(num_samples, norms.shape[0])):
                    fig, ax = plt.subplots(figsize=(12, 2))
                    
                    # Plot as heatmap
                    im = ax.imshow(norms[i:i+1], aspect='auto', cmap='viridis')
                    ax.set_xlabel('Timestep')
                    ax.set_ylabel('Sample')
                    ax.set_title(f'Attention Pattern (Feature Norm) - Sample {i+1}')
                    plt.colorbar(im, ax=ax)
                    
                    # Log to TensorBoard
                    writer.add_figure(f'Attention/Sample_{i+1}', fig, epoch)
                    plt.close(fig)
                
                # Only visualize first batch
                break
                
    except Exception as e:
        print(f"  ⚠️  Failed to visualize attention: {e}")
    finally:
        model.train()


def log_epoch_metrics(
    writer: Any,
    epoch: int,
    train_loss: float,
    val_loss: float,
    mae: float,
    rmse: float,
    dir_acc: float,
    per_horizon_metrics: Dict[str, float],
    learning_rate: float,
    avg_grad_norm_unclipped: float,
    avg_grad_norm_clipped: float
) -> None:
    """
    Log all epoch metrics to TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter
        epoch: Current epoch number
        train_loss: Training loss
        val_loss: Validation loss
        mae: Mean Absolute Error
        rmse: Root Mean Squared Error
        dir_acc: Directional accuracy
        per_horizon_metrics: Dict of per-horizon metrics
        learning_rate: Current learning rate
        avg_grad_norm_unclipped: Average gradient norm before clipping
        avg_grad_norm_clipped: Average gradient norm after clipping
    """
    if writer is None:
        return
    
    try:
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Metrics/MAE', mae, epoch)
        writer.add_scalar('Metrics/RMSE', rmse, epoch)
        writer.add_scalar('Metrics/DirectionalAccuracy', dir_acc, epoch)
        writer.add_scalar('Gradients/Unclipped_Avg', avg_grad_norm_unclipped, epoch)
        writer.add_scalar('Gradients/Clipped_Avg', avg_grad_norm_clipped, epoch)
        writer.add_scalar('LR', learning_rate, epoch)
        
        # Log per-horizon metrics
        for metric_name, metric_value in per_horizon_metrics.items():
            writer.add_scalar(f'PerHorizon/{metric_name}', metric_value, epoch)
    except Exception as e:
        print(f"  ⚠️  Failed to log epoch metrics: {e}")
