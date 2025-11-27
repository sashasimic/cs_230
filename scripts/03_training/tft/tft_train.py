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

# Fix for Mac threading issues - must be set before importing torch
if sys.platform == 'darwin':  # Mac OS
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

# Set PyTorch to single-threaded mode on Mac
if sys.platform == 'darwin':
    torch.set_num_threads(1)

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import TensorBoard utilities
common_path = Path(__file__).parent.parent / 'common'
if str(common_path) not in sys.path:
    sys.path.insert(0, str(common_path))

try:
    import tensorboard_utils as tb_utils
    print(f"\n✅ TensorBoard utilities loaded from: {tb_utils.__file__}")
except ImportError as e:
    print(f"\n⚠️  CRITICAL: Failed to import TensorBoard utilities: {e}")
    print(f"   Looked in: {common_path}")
    raise


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer (same as decoder transformer)."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        import math
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer("pe", pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            x with positional encoding added
        """
        return x + self.pe[:, :x.size(1), :]


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN) - core TFT building block.
    
    Applies non-linear processing with gating and residual connections.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.0, context_dim: int = None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        
        # Primary path
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        
        # Context path (optional)
        if context_dim is not None:
            self.context_fc = nn.Linear(context_dim, hidden_dim, bias=False)
        
        # Output path with gating
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
        # Residual connection (if dimensions match)
        if input_dim != output_dim:
            self.skip = nn.Linear(input_dim, output_dim)
        else:
            self.skip = None
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, x: torch.Tensor, context: torch.Tensor = None):
        """Forward pass with optional context.
        
        Args:
            x: [batch, ..., input_dim]
            context: [batch, ..., context_dim] (optional)
        """
        # Skip connection
        if self.skip is not None:
            residual = self.skip(x)
        else:
            residual = x
        
        # Primary path
        hidden = self.elu(self.fc1(x))
        
        # Add context if provided
        if context is not None and self.context_dim is not None:
            hidden = hidden + self.context_fc(context)
        
        # Gated output
        gate = self.sigmoid(self.gate(hidden))
        output = self.fc2(self.dropout(hidden))
        output = gate * output
        
        # Add residual and normalize
        output = self.layer_norm(output + residual)
        
        return output


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network (VSN) - learns which features are important."""
    
    def __init__(self, input_dim: int, num_vars: int, hidden_dim: int, dropout: float = 0.0, context_dim: int = None):
        super().__init__()
        self.input_dim = input_dim
        self.num_vars = num_vars
        self.hidden_dim = hidden_dim
        
        # Individual variable GRNs
        self.variable_grns = nn.ModuleList([
            GatedResidualNetwork(input_dim, hidden_dim, hidden_dim, dropout, context_dim)
            for _ in range(num_vars)
        ])
        
        # Variable selection weights
        flattened_dim = num_vars * hidden_dim
        self.selection_grn = GatedResidualNetwork(
            flattened_dim, hidden_dim, num_vars, dropout, context_dim
        )
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, variables: torch.Tensor, context: torch.Tensor = None):
        """Select important variables.
        
        Args:
            variables: [batch, time, num_vars, input_dim] or [batch, num_vars, input_dim]
            context: Optional context [batch, time, context_dim] or [batch, context_dim]
        
        Returns:
            selected: Weighted combination of variables
            weights: Variable importance weights
        """
        # Handle both 3D and 4D inputs
        is_temporal = len(variables.shape) == 4
        
        if is_temporal:
            batch, time, num_vars, _ = variables.shape
            # Flatten time dimension
            variables = variables.reshape(batch * time, num_vars, -1)
            if context is not None:
                context = context.reshape(batch * time, -1)
        else:
            batch, num_vars, _ = variables.shape
        
        # Process each variable
        processed_vars = []
        for i in range(num_vars):
            processed = self.variable_grns[i](variables[:, i], context)
            processed_vars.append(processed)
        
        # Stack: [batch, num_vars, hidden_dim]
        processed_vars = torch.stack(processed_vars, dim=1)
        
        # Flatten for selection
        flattened = processed_vars.reshape(variables.shape[0], -1)
        
        # Compute selection weights
        weights = self.selection_grn(flattened, context)
        weights = self.softmax(weights)  # [batch, num_vars]
        
        # Apply weights
        weights = weights.unsqueeze(-1)  # [batch, num_vars, 1]
        selected = (processed_vars * weights).sum(dim=1)  # [batch, hidden_dim]
        
        # Reshape back if temporal
        if is_temporal:
            selected = selected.reshape(batch, time, -1)
            weights = weights.reshape(batch, time, num_vars, 1)
        
        return selected, weights


class TemporalFusionTransformer(nn.Module):
    """Full Temporal Fusion Transformer implementation.
    
    Based on: "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
    https://arxiv.org/abs/1912.09363
    """
    
    def __init__(self, config: dict, num_features: int = None):
        super().__init__()
        
        # Extract config
        time_varying_known = config['model'].get('time_varying_known', [])
        time_varying_unknown = config['model']['time_varying_unknown']
        static_features = config['model'].get('static_features', [])
        
        # Use actual feature count from data if provided (accounts for pivoting)
        if num_features is not None:
            self.num_features = num_features
        else:
            self.num_features = len(time_varying_known) + len(time_varying_unknown)
        
        self.hidden_size = config['model']['hidden_size']
        self.lstm_layers = config['model']['lstm_layers']
        self.attention_heads = config['model']['attention_heads']
        self.dropout = config['model']['dropout']
        self.num_horizons = len(config['data']['prediction_horizons'])
        self.quantiles = config['model'].get('quantiles', [0.5])
        self.num_quantiles = len(self.quantiles)
        
        # For simplicity, treat all features as time-varying unknown
        # In production, you'd separate known vs unknown based on config
        self.num_time_varying = self.num_features
        
        print(f"   Initializing TFT with:")
        print(f"   - {self.num_features} input features")
        print(f"   - {self.num_horizons} prediction horizons")
        print(f"   - {self.num_quantiles} quantiles: {self.quantiles}")
        
        # ===== 1. Variable Selection Network (VSN) =====
        # Learns which input features are important
        self.variable_selection = VariableSelectionNetwork(
            input_dim=1,  # Each feature is scalar
            num_vars=self.num_features,
            hidden_dim=self.hidden_size,
            dropout=self.dropout
        )
        self.vsn_norm = nn.LayerNorm(self.hidden_size)  # Normalize VSN output
        
        # ===== 2. LSTM Encoder =====
        # Processes historical sequence
        self.lstm_encoder = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.lstm_layers,
            batch_first=True,
            dropout=self.dropout if self.lstm_layers > 1 else 0
        )
        self.lstm_norm = nn.LayerNorm(self.hidden_size)  # Normalize LSTM output
        
        # ===== 3. Static Enrichment (using GRN) =====
        # Enriches temporal features with static context
        self.static_enrichment = GatedResidualNetwork(
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.hidden_size,
            dropout=self.dropout
        )
        
        # ===== 4. Temporal Self-Attention =====
        # Multi-head attention over time (with causal masking for forecasting)
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.attention_heads,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Register causal mask as buffer (won't be trained)
        # This prevents attention from looking at future timesteps
        lookback = config['data'].get('lookback_window', config['data'].get('lookback', 192))
        self.register_buffer(
            'causal_mask',
            self._generate_causal_mask(lookback)
        )
        
        # Attention output processing
        self.attention_gate = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.attention_sigmoid = nn.Sigmoid()
        self.attention_norm = nn.LayerNorm(self.hidden_size)
        
        # ===== 5. Position-wise Feed-Forward =====
        self.position_wise_grn = GatedResidualNetwork(
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.hidden_size,
            dropout=self.dropout
        )
        
        # ===== 6. Quantile Output Heads =====
        # Separate head for each (horizon, quantile) combination
        self.quantile_outputs = nn.ModuleList([
            nn.Linear(self.hidden_size, self.num_quantiles)
            for _ in range(self.num_horizons)
        ])
        
        # Initialize weights
        self._init_weights()
    
    def _generate_causal_mask(self, size: int) -> torch.Tensor:
        """Generate causal mask to prevent attention to future positions.
        
        Returns:
            mask: [size, size] with 0 for allowed, -inf for masked
            [[  0, -inf, -inf],
             [  0,   0, -inf],
             [  0,   0,   0]]
        """
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, x: torch.Tensor):
        """
        TFT Forward pass.
        
        Args:
            x: [batch, lookback, features]
        
        Returns:
            predictions: [batch, horizons, quantiles] or [batch, horizons] if single quantile
        """
        batch_size, lookback, num_features = x.shape
        
        # ===== 1. Variable Selection =====
        # Reshape to [batch, time, num_vars, 1] for VSN
        x_reshaped = x.unsqueeze(-1)  # [batch, lookback, features, 1]
        
        # Apply variable selection to learn feature importance
        selected_features, var_weights = self.variable_selection(x_reshaped)
        # selected_features: [batch, lookback, hidden_size]
        selected_features = self.vsn_norm(selected_features)  # Normalize
        
        # ===== 2. LSTM Encoding =====
        # Encode the full sequence
        lstm_output, (hidden, cell) = self.lstm_encoder(selected_features)
        # lstm_output: [batch, lookback, hidden_size]
        lstm_output = self.lstm_norm(lstm_output)  # Normalize
        temporal_features = lstm_output
        
        # ===== 3. Static Enrichment =====
        # Apply GRN to enrich features (no actual static features in our case)
        enriched = self.static_enrichment(temporal_features)
        # enriched: [batch, lookback, hidden_size]
        
        # ===== 4. Temporal Self-Attention =====
        # Apply multi-head attention over time with causal masking
        attn_output, attn_weights = self.temporal_attention(
            enriched, enriched, enriched,
            attn_mask=self.causal_mask  # Prevent looking at future
        )
        # attn_output: [batch, lookback, hidden_size]
        
        # Gated residual connection
        gate_input = torch.cat([enriched, attn_output], dim=-1)
        gate = self.attention_sigmoid(self.attention_gate(gate_input))
        gated_output = gate * attn_output + (1 - gate) * enriched
        gated_output = self.attention_norm(gated_output)
        
        # ===== 5. Position-wise Processing =====
        # Apply GRN to each timestep
        processed = self.position_wise_grn(gated_output)
        # processed: [batch, lookback, hidden_size]
        
        # ===== 6. Quantile Predictions =====
        # Use last timestep for multi-horizon forecasting
        final_repr = processed[:, -1, :]  # [batch, hidden_size]
        
        # Generate quantile predictions for each horizon
        all_predictions = []
        for horizon_idx in range(self.num_horizons):
            quantile_preds = self.quantile_outputs[horizon_idx](final_repr)
            # quantile_preds: [batch, num_quantiles]
            all_predictions.append(quantile_preds)
        
        # Stack: [batch, horizons, quantiles]
        predictions = torch.stack(all_predictions, dim=1)
        
        # If single quantile (median), squeeze last dimension for compatibility
        if self.num_quantiles == 1:
            predictions = predictions.squeeze(-1)  # [batch, horizons]
        
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
            if 'variable_selection' in name or 'variable_grns' in name:
                layer_type = 'VSN'
            elif 'static_enrichment' in name or 'position_wise_grn' in name:
                layer_type = 'GRN'
            elif 'lstm_encoder' in name:
                if 'weight_ih_l0' in name or 'weight_hh_l0' in name or 'bias_ih_l0' in name or 'bias_hh_l0' in name:
                    layer_type = 'LSTM_L0'
                elif 'weight_ih_l1' in name or 'weight_hh_l1' in name or 'bias_ih_l1' in name or 'bias_hh_l1' in name:
                    layer_type = 'LSTM_L1'
                elif 'weight_ih_l2' in name or 'weight_hh_l2' in name or 'bias_ih_l2' in name or 'bias_hh_l2' in name:
                    layer_type = 'LSTM_L2'
                else:
                    layer_type = 'LSTM_Other'
            elif 'temporal_attention' in name or 'attention_gate' in name or 'attention_norm' in name:
                layer_type = 'Attention'
            elif 'quantile_outputs' in name:
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


def compute_metrics(predictions: torch.Tensor, targets: torch.Tensor, horizons: list = None) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Args:
        predictions: [batch, horizons]
        targets: [batch, horizons]
        horizons: Optional list of actual horizon values for labeling
    
    Returns:
        Dictionary of metrics (includes per-horizon metrics if horizons provided)
    """
    with torch.no_grad():
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
        
        # Compute per-horizon metrics
        num_horizons = predictions.shape[1]
        for h_idx in range(num_horizons):
            h_mae = torch.abs(predictions[:, h_idx] - targets[:, h_idx]).mean().item()
            h_rmse = torch.sqrt(((predictions[:, h_idx] - targets[:, h_idx]) ** 2).mean()).item()
            
            # Use actual horizon values for labeling
            horizon_label = f"H{horizons[h_idx]}" if (horizons and h_idx < len(horizons)) else f"H{h_idx+1}"
            
            metrics[f"{horizon_label}_MAE"] = h_mae
            metrics[f"{horizon_label}_RMSE"] = h_rmse
        
        return metrics


def train_epoch(model: nn.Module, dataloader, criterion, optimizer, device, epoch: int = 0, clip_norm: float = 1.0) -> dict:
    """Train for one epoch and return detailed metrics (matches decoder transformer)."""
    model.train()
    total_loss = 0.0
    unclipped_grad_norms = []
    clipped_grad_norms = []
    layer_grad_stats = None
    
    total_batches = len(dataloader)
    print(f"\n  Training: 0/{total_batches} batches", end='', flush=True)
    
    for batch_idx, (batch_X, batch_y) in enumerate(dataloader):
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
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
        
        # Get detailed layer stats for first batch only (after clipping)
        if batch_idx == 0:
            layer_grad_stats = compute_layer_grad_stats(model)
        
        optimizer.step()
        total_loss += loss.item()
        
        # Progress update every 10 batches or at end
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
            print(f"\r  Training: {batch_idx + 1}/{total_batches} batches (loss: {total_loss / (batch_idx + 1):.4f})", end='', flush=True)
    
    print()  # New line after progress
    avg_loss = total_loss / len(dataloader)
    
    # Unclipped stats
    avg_unclipped = float(np.mean(unclipped_grad_norms))
    max_unclipped = float(np.max(unclipped_grad_norms))
    
    # Clipped stats
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


def train(config_path: str, dataloaders: Optional[Dict] = None, scalers: Optional[Dict] = None, dataset_version: Optional[str] = None):
    """
    Core TFT training function.
    
    Args:
        config_path: Path to model config YAML
        dataloaders: Optional pre-loaded DataLoaders (if None, will load from data/processed/)
        scalers: Optional pre-loaded scalers
        dataset_version: Optional dataset version (e.g., 'v1', 'v3'). Data always loaded from data/processed/ (copied by *_train_local.py or downloaded by train_vertex.py)
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data if not provided
    if dataloaders is None:
        # Always use data/processed/ (unified behavior for local and Vertex AI)
        # Local: Copied from versioned dataset by *_train_local.py
        # Vertex AI: Downloaded from GCS by train_vertex.py
        if dataset_version:
            data_path = Path('data/processed')
            print(f"\n📂 Loading data from: {data_path}...")
            
            if not data_path.exists():
                raise FileNotFoundError(
                    f"Data directory not found: {data_path}\n"
                    f"For local training, run:\n"
                    f"  python scripts/03_training/tft/tft_train_local.py --dataset-version {dataset_version}"
                )
            
            # Load preprocessed arrays directly
            import numpy as np
            from torch.utils.data import TensorDataset, DataLoader
            
            train_X_np = np.load(data_path / 'X_train.npy', allow_pickle=True)
            train_y_np = np.load(data_path / 'y_train.npy', allow_pickle=True)
            val_X_np = np.load(data_path / 'X_val.npy', allow_pickle=True)
            val_y_np = np.load(data_path / 'y_val.npy', allow_pickle=True)
            test_X_np = np.load(data_path / 'X_test.npy', allow_pickle=True)
            test_y_np = np.load(data_path / 'y_test.npy', allow_pickle=True)
            
            # Handle object dtype
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
            
            # Convert to PyTorch tensors
            train_X = torch.tensor(train_X_np, dtype=torch.float32)
            train_y = torch.tensor(train_y_np, dtype=torch.float32)
            val_X = torch.tensor(val_X_np, dtype=torch.float32)
            val_y = torch.tensor(val_y_np, dtype=torch.float32)
            test_X = torch.tensor(test_X_np, dtype=torch.float32)
            test_y = torch.tensor(test_y_np, dtype=torch.float32)
            
            # Create datasets and dataloaders
            batch_size = config['training']['batch_size']
            train_dataset = TensorDataset(train_X, train_y)
            val_dataset = TensorDataset(val_X, val_y)
            test_dataset = TensorDataset(test_X, test_y)
            
            dataloaders = {
                'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
                'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
                'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            }
            scalers = None  # Scalers not available when loading from numpy
            print("✅ Data loaded!")
        else:
            # Use existing create_data_loaders for default behavior
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
    
    # Display date range from metadata (actual data) and sample counts
    try:
        # Always use data/processed/metadata.yaml (populated by local copy or Vertex AI download)
        metadata_path = Path('data/processed/metadata.yaml')
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = yaml.safe_load(f)
                start_date = metadata.get('start_date', 'N/A')
                end_date = metadata.get('end_date', 'N/A')
                print(f"\n📅 Date Range (from actual data):")
                print(f"  Start: {start_date}")
                print(f"  End: {end_date}")
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
    
    # Get sample counts from data loaders
    train_samples = len(dataloaders['train'].dataset)
    val_samples = len(dataloaders['val'].dataset)
    test_samples = len(dataloaders['test'].dataset)
    total = train_samples + val_samples + test_samples
    print(f"  Total sequences: {total:,} (train: {train_samples:,}, val: {val_samples:,}, test: {test_samples:,})")
    
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
        # Always use data/processed/metadata.yaml (populated by local copy or Vertex AI download)
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
    
    # Output targets - read from dataset metadata if available, otherwise from config
    horizons_config = None
    if dataset_version:
        # Read horizons from data/processed/metadata.yaml (populated by local copy or Vertex AI download)
        try:
            metadata_path = Path('data/processed/metadata.yaml')
            
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    dataset_metadata = yaml.safe_load(f)
                    horizons_config = dataset_metadata.get('prediction_horizons', None)
                    if horizons_config:
                        print(f"\n✅ Using prediction horizons from dataset metadata: {horizons_config}")
        except Exception as e:
            print(f"\n⚠️  Could not read horizons from metadata: {e}")
    
    # Fallback to config if not found in metadata
    if horizons_config is None:
        horizons_config = config['data']['prediction_horizons']
        print(f"\n📄 Using prediction horizons from config: {horizons_config}")
    
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
    print(f"\nArchitecture: Temporal Fusion Transformer (TFT)")
    print(f"  Components:")
    print(f"    - Variable Selection Network (VSN)")
    print(f"    - LSTM Encoder ({model_config['lstm_layers']} layers)")
    print(f"    - Gated Residual Networks (GRN)")
    print(f"    - Temporal Self-Attention ({model_config['attention_heads']} heads)")
    print(f"    - Quantile Output Heads")
    print(f"  Hidden size: {model_config['hidden_size']}")
    print(f"  Dropout: {model_config['dropout']}")
    print(f"\nTraining:")
    print(f"  Epochs: {training_config['epochs']}")
    print(f"  Batch size: {training_config['batch_size']}")
    print(f"  Learning rate: {training_config['learning_rate']}")
    print(f"  Early stopping patience: {training_config['early_stopping']['patience']}")
    
    # Setup TensorBoard
    print(f"\n🔍 Initializing TensorBoard...")
    print(f"   Config tensorboard enabled: {config.get('logging', {}).get('tensorboard', False)}")
    writer = tb_utils.initialize_tensorboard_writer(config, 'tft', '')
    
    if writer is None:
        print(f"\n⚠️  CRITICAL: TensorBoard writer is None!")
        print(f"   All TensorBoard logging will be skipped.")
    else:
        print(f"\n✅ TensorBoard writer initialized successfully")
    
    # Initialize TFT model
    print("\n" + "="*80)
    print("   Initializing Temporal Fusion Transformer")
    print("="*80)
    
    # Pass actual feature count from data (after pivoting)
    model = TemporalFusionTransformer(config, num_features=num_features).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Log experiment metadata to TensorBoard
    additional_model_info = {
        'Model Type': 'Temporal Fusion Transformer',
        'Hidden Size': model_config['hidden_size'],
        'LSTM Layers': model_config['lstm_layers'],
        'Attention Heads': model_config['attention_heads'],
        'Dropout': model_config['dropout'],
        'Total Parameters': f"{total_params:,}",
        'Trainable Parameters': f"{trainable_params:,}"
    }
    
    tb_utils.log_experiment_metadata(
        writer, dataset_version, start_date, end_date, horizons_config,
        lookback, num_features, num_horizons,
        train_samples, val_samples, test_samples,
        'Temporal Fusion Transformer', config, total_params, trainable_params,
        additional_model_info
    )
    print(f"\u2705 Logged dataset and model info to TensorBoard\n")
    
    # Setup optimizer and loss
    optimizer = optim.Adam(
        model.parameters(),
        lr=training_config['learning_rate'],
        weight_decay=training_config.get('weight_decay', 0.0)  # L2 regularization
    )
    criterion = nn.MSELoss()
    
    # Setup learning rate scheduler (if enabled)
    scheduler = None
    if training_config.get('lr_scheduler', {}).get('enabled', False):
        scheduler_config = training_config['lr_scheduler']
        if scheduler_config['type'] == 'reduce_on_plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 5),
                min_lr=scheduler_config.get('min_lr', 0.00001)
            )
            print(f"\n📉 Learning Rate Scheduler: ReduceLROnPlateau")
            print(f"   Mode: min (reduce on validation loss plateau)")
            print(f"   Factor: {scheduler_config.get('factor', 0.5)}")
            print(f"   Patience: {scheduler_config.get('patience', 5)}")
            print(f"   Min LR: {scheduler_config.get('min_lr', 0.00001)}")
    
    # Create output directory
    output_dir = Path('models/tft')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    # Training loop configuration
    best_val_loss = float('inf')
    best_val_mae = float('inf')
    patience_counter = 0
    epochs = training_config['epochs']
    clip_norm = training_config.get('gradient_clip_norm', 1.0)
    early_stopping_patience = training_config.get('early_stopping', {}).get('patience', 10)
    
    print("\n" + "="*80)
    print("   TRAINING CONFIGURATION")
    print("="*80)
    print(f"\n🚀 Starting training for {epochs} epochs")
    print(f"   Device: {device}")
    print(f"   Batch size: {training_config['batch_size']}")
    print(f"   Learning rate: {training_config['learning_rate']}")
    print(f"   Gradient clipping: {clip_norm}")
    print(f"   Early stopping patience: {early_stopping_patience}")
    print(f"\n⏱️  Estimated time:")
    print(f"   Per epoch: 2-5 minutes")
    print(f"   Total ({epochs} epochs): {epochs * 0.06:.1f} hours")
    print(f"   Training batches per epoch: {len(dataloaders['train'])}")
    print("\n" + "="*80)
    
    import time
    training_start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # Training phase
        train_results = train_epoch(
            model, dataloaders['train'], criterion, optimizer, device, 
            epoch=epoch+1, clip_norm=clip_norm
        )
        train_loss = train_results['loss']
        avg_unclipped = train_results['avg_unclipped']
        max_unclipped = train_results['max_unclipped']
        avg_clipped = train_results['avg_clipped']
        max_clipped = train_results['max_clipped']
        layer_grad_stats = train_results['layer_grad_stats']
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        all_preds = []
        all_targets = []
        
        total_val_batches = len(dataloaders['val'])
        print(f"  Validation: 0/{total_val_batches} batches", end='', flush=True)
        
        with torch.no_grad():
            for batch_idx, (batch_X, batch_y) in enumerate(dataloaders['val']):
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                
                val_loss += loss.item()
                val_batches += 1
                
                all_preds.append(predictions)
                all_targets.append(batch_y)
        
        print()  # New line after validation progress
        val_loss = val_loss / val_batches
        
        # Compute metrics
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = compute_metrics(all_preds, all_targets, horizons=horizons_config)
        
        epoch_time = time.time() - epoch_start_time
        total_elapsed = time.time() - training_start_time
        
        print(f"\nEpoch {epoch+1}/{training_config['epochs']} - {epoch_time/60:.1f} min (total: {total_elapsed/60:.1f} min)")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val   Loss: {val_loss:.6f}, MAE: {metrics['mae']:.6f}, RMSE: {metrics['rmse']:.6f}")
        print(f"  Dir Acc (H1): {metrics['dir_acc']:.2f}%")
        
        # Print per-horizon MAE
        horizon_strs = []
        for key, value in metrics.items():
            if 'MAE' in key and key != 'mae':  # Skip overall MAE
                horizon_strs.append(f"{key}={value:.6f}")
        if horizon_strs:
            print(f"  Per-Horizon MAE: {', '.join(horizon_strs)}")
        
        print(f"  Grad Norm (unclipped): avg={avg_unclipped:.4f}, max={max_unclipped:.4f}")
        print(f"  Grad Norm (clipped):   avg={avg_clipped:.4f}, max={max_clipped:.4f}")
        
        # Print layer-wise gradient stats every 10 epochs
        if (epoch + 1) % 10 == 0 and layer_grad_stats:
            print(f"  Layer Gradients (batch 1):")
            for layer_name in ['VSN', 'Input', 'LSTM_L0', 'LSTM_L1', 'LSTM_L2', 'Attention', 'Feedforward', 'Output']:
                if layer_name in layer_grad_stats:
                    stats = layer_grad_stats[layer_name]
                    print(f"    {layer_name:12s}: norm={stats['avg_norm']:.4f}, max={stats['max_norm']:.4f}, std={stats['avg_std']:.4f}")
        
        # Log to TensorBoard
        try:
            # Prepare per-horizon metrics dict (exclude overall metrics)
            per_horizon_metrics = {k: v for k, v in metrics.items() 
                                  if k not in ['mae', 'mse', 'rmse', 'dir_acc']}
            
            current_lr = optimizer.param_groups[0]['lr']
            tb_utils.log_epoch_metrics(
                writer, epoch, train_loss, val_loss, 
                metrics['mae'], metrics['rmse'], metrics['dir_acc'],
                per_horizon_metrics, current_lr,
                avg_unclipped, avg_clipped
            )
            # Flush to ensure data is written immediately
            if writer is not None:
                writer.flush()
        except Exception as e:
            print(f"\n⚠️  ERROR logging epoch metrics: {e}")
            import traceback
            traceback.print_exc()
        
        # Log gradient and weight histograms to TensorBoard (every 10 epochs)
        if (epoch + 1) % 10 == 0:
            try:
                tb_utils.log_gradients_and_weights(writer, model, epoch)
                if writer is not None:
                    writer.flush()
                print(f"  ✅ Logged histograms")
            except Exception as e:
                print(f"  ⚠️  ERROR logging histograms: {e}")
                import traceback
                traceback.print_exc()
        
        # Learning rate scheduler step (if enabled)
        if scheduler is not None:
            scheduler.step(val_loss)
        
        print("")  # Empty line before next epoch
        
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
    
    # Final evaluation on test set
    print("\n" + "="*80)
    print("   Final Test Set Evaluation")
    print("="*80)
    
    # Load best model
    checkpoint_path = output_dir / 'tft_best.pt'
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on test set
    model.eval()
    test_preds = []
    test_targets = []
    test_loss = 0.0
    test_batches = 0
    
    print(f"  Evaluating on test set...")
    with torch.no_grad():
        for batch_X, batch_y in dataloaders['test']:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            
            test_loss += loss.item()
            test_batches += 1
            
            test_preds.append(predictions)
            test_targets.append(batch_y)
    
    test_loss = test_loss / test_batches
    test_preds = torch.cat(test_preds, dim=0)
    test_targets = torch.cat(test_targets, dim=0)
    test_metrics = compute_metrics(test_preds, test_targets, horizons=horizons_config)
    
    print(f"\n📊 Test Set Results:")
    print(f"  Test Loss: {test_loss:.6f}")
    print(f"  Test MAE: {test_metrics['mae']:.6f}")
    print(f"  Test RMSE: {test_metrics['rmse']:.6f}")
    print(f"  Test Dir Acc (H1): {test_metrics['dir_acc']:.2f}%")
    
    # Log per-horizon test metrics
    horizon_strs = []
    for key, value in sorted(test_metrics.items()):
        if 'MAE' in key and key != 'mae':  # Skip overall MAE
            horizon_strs.append(f"{key}={value:.6f}")
    if horizon_strs:
        print(f"  Per-Horizon MAE: {', '.join(horizon_strs)}")
    
    # Log hyperparameters to TensorBoard HParams
    hparams = {
        'model': 'TFT',
        'hidden_size': model_config['hidden_size'],
        'lstm_layers': model_config['lstm_layers'],
        'attention_heads': model_config['attention_heads'],
        'dropout': model_config['dropout'],
        'lookback': lookback,
        'batch_size': training_config['batch_size'],
        'learning_rate': training_config['learning_rate'],
        'weight_decay': training_config.get('weight_decay', 0.0),
        'gradient_clip': training_config.get('gradient_clip_norm', 1.0),
    }
    
    hparam_metrics = {
        'hparam/best_val_loss': best_val_loss,
        'hparam/best_val_mae': best_val_mae,
        'hparam/test_loss': test_loss,
        'hparam/test_mae': test_metrics['mae'],
        'hparam/test_rmse': test_metrics['rmse'],
        'hparam/test_dir_acc': test_metrics['dir_acc'],
    }
    
    tb_utils.log_hyperparameters(writer, hparams, hparam_metrics)
    
    # Close TensorBoard writer
    if writer is not None:
        writer.close()
    
    print("\n" + "="*80)
    print("   Training Complete")
    print("="*80)
    print(f"  Best validation loss: {best_val_loss:.6f}")
    print(f"  Best validation MAE: {best_val_mae:.6f}")
    print(f"  Model saved: {checkpoint_path}")
    if writer is not None and not os.getenv('CLOUD_ML_JOB_ID'):
        print(f"\n📊 View TensorBoard: tensorboard --logdir logs/tft")
    print("="*80)
    
    return model, best_val_loss


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