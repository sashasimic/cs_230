#!/usr/bin/env python3
"""
Decoder-Only Autoregressive Transformer Training Logic

Past encoder: causal transformer over historical window.
Future decoder: small autoregressive GRU over forecast horizons.

Shared training function used by both:
- dec_train_local.py (local training)
- train_vertex.py (cloud training)
"""

import os
import sys
import yaml
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
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


# FinCast imports are handled in fincast_extension.py
# No fallback imports needed here


class DecoderOnlyTransformerAR(nn.Module):
    """
    Decoder-only autoregressive transformer for time series forecasting.

    Architecture:
      - Past encoder: causal TransformerEncoder over historical window
      - Future decoder: small GRU that autoregressively generates horizon steps

    Inputs:
      - x: [batch, lookback, features]
      - y_future (optional during training): [batch, num_horizons]

    Outputs:
      - predictions: [batch, num_horizons]
    """
    
    def __init__(self, config: dict, num_features: int = None):
        super().__init__()
        
        # ---- Config extraction ----
        time_varying_known = config['model'].get('time_varying_known', [])
        time_varying_unknown = config['model']['time_varying_unknown']
        
        if num_features is not None:
            self.num_features = num_features
        else:
            self.num_features = len(time_varying_known) + len(time_varying_unknown)
        
        self.d_model = config['model']['d_model']      # hidden dim
        self.n_heads = config['model']['n_heads']
        self.n_layers = config['model']['n_layers']
        self.d_ff = config['model']['d_ff']            # feedforward dim
        self.dropout = config['model']['dropout']
        self.num_horizons = len(config['data']['prediction_horizons'])
        
        # Allow either 'lookback' or 'lookback_window' in config
        data_cfg = config['data']
        self.lookback = data_cfg.get('lookback', data_cfg.get('lookback_window'))
        if self.lookback is None:
            raise ValueError("Config must define data.lookback or data.lookback_window")
        
        # ---- Past encoder ----
        # Project features -> d_model
        self.input_projection = nn.Linear(self.num_features, self.d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=self.lookback)
        
        # Dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # Causal TransformerEncoder (decoder-style encoder)
        # self_attn and feedforward blocks get: Residual connections, LayerNorm, Dropout
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_ff,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True   # pre-norm is more stable
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.n_layers
        )
        
        # Final norm on past representation
        self.enc_norm = nn.LayerNorm(self.d_model)
        
        # ---- Future decoder (autoregressive over horizons) ----
        # We treat the horizon index as a small sequence: h1, h2, ..., hH
        # GRU input: previous target value (scalar), embedded
        self.future_input_dim = 1          # scalar previous target
        self.future_hidden_dim = self.d_model
        
        self.future_in_proj = nn.Linear(self.future_input_dim, self.future_hidden_dim)
        self.future_decoder = nn.GRU(
            input_size=self.future_hidden_dim,
            hidden_size=self.future_hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Output projection: hidden state -> scalar forecast at each horizon step
        self.future_out_proj = nn.Linear(self.future_hidden_dim, 1)
        
        # Learned start token (what we feed at first horizon step)
        self.start_token = nn.Parameter(torch.zeros(1, self.future_input_dim))
        
        # ---- Causal mask for encoder ----
        self.register_buffer(
            "causal_mask",
            self._generate_square_subsequent_mask(self.lookback)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        Generate causal mask to prevent considering future positions.
        [[ 0,   -inf, -inf, -inf],
         [ 0,    0,   -inf, -inf],
         [ 0,    0,    0,   -inf],
         [ 0,    0,    0,    0  ]]
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def encode_past(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode past window with causal transformer.
        
        Args:
            x: [batch, T, features]
        Returns:
            context: [batch, d_model] representation of last time step
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)  # [B, T, d_model]
        
        # Add positional encoding
        x = self.pos_encoder(x)
        x = self.dropout_layer(x)
        
        # Causal mask for encoder to prevent accessing future positions
        mask = self.causal_mask[:seq_len, :seq_len]  # [T, T]
        
        # Transformer encoder (causal)
        x = self.transformer_encoder(x, mask=mask)   # [B, T, d_model]
        
        # Norm + take last timestep as context
        x = self.enc_norm(x)
        context = x[:, -1, :]  # [B, d_model]
        return context
    
    def decode_future(
        self,
        context: torch.Tensor,
        y_future: Optional[torch.Tensor] = None,
        teacher_forcing: bool = True
    ) -> torch.Tensor:
        """
        Autoregressively decode future horizons given past context.
        
        Args:
            context: [batch, d_model] encoder summary
            y_future: [batch, num_horizons] ground-truth targets (optional)
            teacher_forcing: if True and y_future provided, use them as inputs
        
        Returns:
            preds: [batch, num_horizons]
        """
        batch_size = context.size(0)
        device = context.device
        
        # Initial hidden state of GRU from encoder context
        # GRU expects [num_layers, batch, hidden_dim]
        h0 = context.unsqueeze(0)  # [1, B, d_model]

        # Start token: [B, 1, 1]
        start = self.start_token.expand(batch_size, 1, self.future_input_dim)
        
        if y_future is not None and teacher_forcing:
            # Teacher forcing over horizons:
            # decoder input at step 0: start token
            # decoder input at step t>0: ground-truth y_{t-1}
            # y_future: [B, H]
            y_future = y_future.to(device)
            
            # Build input sequence: [start, y_0, y_1, ..., y_{H-2}]  -> length H
            H = self.num_horizons
            device = y_future.device

            # If we have more than 1 horizon, use ground-truth previous targets
            if H > 1:
                # y_future: [B, H] -> take all but last: [B, H-1] -> [B, H-1, 1]
                prev_targets = y_future[:, :-1].unsqueeze(-1)
                
                # 3) Concatenate along time dimension: [B, 1 + (H-1), 1] = [B, H, 1]
                dec_in = torch.cat([start, prev_targets], dim=1)
            else:
                # Only one horizon: input is just the start token
                dec_in = start  # [B, 1, 1]
            
            # Embed inputs
            dec_in = self.future_in_proj(dec_in)  # [B, H, d_model]
            
            # Run GRU over horizon steps
            dec_out, _ = self.future_decoder(dec_in, h0)  # [B, H, d_model]
            
            # Project to scalar at each horizon
            preds = self.future_out_proj(dec_out).squeeze(-1)  # [B, H]
            return preds
        
        else:
            # Pure autoregressive decoding over horizons (no ground-truth)
            preds = []
            h_t = h0
            prev_input = start
            
            for t in range(self.num_horizons):
                # Embed previous target
                dec_in = self.future_in_proj(prev_input)   # [B,1,d_model]
                
                # GRU step
                dec_out, h_t = self.future_decoder(dec_in, h_t)  # dec_out: [B,1,d_model]
                
                # Predict next horizon target
                y_t = self.future_out_proj(dec_out).squeeze(-1)  # [B,1,1] -> [B,1]
                preds.append(y_t)                                # [B,1]
                
                # Next step's input is current prediction
                prev_input = y_t.unsqueeze(-1)                   # [B,1] -> [B,1,1]
            
            preds = torch.cat(preds, dim=1)  # [B, H]
            return preds
    
    def forward(
        self,
        x: torch.Tensor,
        y_future: Optional[torch.Tensor] = None,
        teacher_forcing: bool = True
    ) -> torch.Tensor:
        """
        Full forward pass: encode past, decode future.
        
        Args:
            x: [batch, lookback, features]
            y_future: [batch, num_horizons] (optional, for teacher forcing)
            teacher_forcing: if True and y_future is provided, use it
        
        Returns:
            predictions: [batch, num_horizons]
        """
        # Encode past window and return last timestamp representation
        context = self.encode_past(x)   # [B, d_model]
        
        # Decode future horizons
        preds = self.decode_future(
            context,
            y_future=y_future,
            teacher_forcing=(y_future is not None and teacher_forcing)
        )
        return preds
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                if 'input_projection' in name:
                    scale = 1.0 / np.sqrt(self.num_features)
                    nn.init.xavier_uniform_(param, gain=scale)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)


# FinCast integration - all import logic is in fincast_extension.py
# That module handles:
#   - Checking if FinCast submodule exists
#   - Importing FFM and dependencies
#   - Providing helpful error messages if missing
#   - Exporting FINCAST_AVAILABLE flag

FINCAST_AVAILABLE = False
DecoderTransformerWithFinCast = None

try:
    # Try relative import first (for package context)
    from .fincast_extension import (
        DecoderTransformerWithFinCast,
        FINCAST_AVAILABLE
    )
except ImportError:
    # Try absolute import (for direct script execution)
    try:
        import importlib.util
        fincast_ext_path = Path(__file__).parent / "fincast_extension.py"
        spec = importlib.util.spec_from_file_location("fincast_extension", fincast_ext_path)
        if spec and spec.loader:
            fincast_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(fincast_module)
            DecoderTransformerWithFinCast = fincast_module.DecoderTransformerWithFinCast
            FINCAST_AVAILABLE = fincast_module.FINCAST_AVAILABLE
    except Exception as e:
        # If both imports fail, FinCast is not available
        pass


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
        if param.grad is None:
            continue
        
        grad_norm = param.grad.data.norm(2).item()
        grad_mean = param.grad.data.mean().item()
        grad_std = param.grad.data.std().item() if param.grad.data.numel() > 1 else 0.0
        grad_max = param.grad.data.abs().max().item()
        
        # Group by rough layer type
        if 'input_projection' in name:
            layer_name = 'Input'
        elif 'transformer_encoder.layers' in name:
            parts = name.split('.')
            layer_idx = parts[2] if len(parts) > 2 else '?'
            layer_name = f'Encoder_L{layer_idx}'
        elif 'future_decoder' in name or 'future_in_proj' in name or 'future_out_proj' in name:
            layer_name = 'FutureDecoder'
        elif 'pos_encoder' in name:
            layer_name = 'PosEnc'
        else:
            layer_name = 'Other'
        
        if layer_name not in layer_stats:
            layer_stats[layer_name] = {'norm': 0.0, 'max': 0.0, 'std': 0.0}
        
        layer_stats[layer_name]['norm'] += grad_norm
        layer_stats[layer_name]['max'] = max(layer_stats[layer_name]['max'], grad_max)
        layer_stats[layer_name]['std'] = max(layer_stats[layer_name]['std'], grad_std)
    
    return layer_stats


def train_epoch(model, train_loader, optimizer, criterion, device, clip_norm=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    unclipped_grad_norms = []
    clipped_grad_norms = []
    
    total_batches = len(train_loader)
    print(f"\n  Training: 0/{total_batches} batches", end='', flush=True)
    
    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        
        # Teacher forcing over horizons during training
        predictions = model(X_batch, y_future=y_batch, teacher_forcing=True)
        loss = criterion(predictions, y_batch)
        
        loss.backward()
        
        # Compute unclipped gradient norm
        unclipped_norm = compute_grad_norm(model)
        unclipped_grad_norms.append(unclipped_norm)
        
        # Apply gradient clipping if specified
        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        
        # Compute clipped gradient norm
        clipped_norm = compute_grad_norm(model)
        clipped_grad_norms.append(clipped_norm)
        
        optimizer.step()
        total_loss += loss.item()
        
        # Progress update every 10 batches or at end
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
            print(f"\r  Training: {batch_idx + 1}/{total_batches} batches (loss: {total_loss / (batch_idx + 1):.4f})", end='', flush=True)
    
    print()  # New line after progress
    avg_loss = total_loss / len(train_loader)
    
    # Unclipped stats
    avg_unclipped = float(np.mean(unclipped_grad_norms))
    max_unclipped = float(np.max(unclipped_grad_norms))
    
    # Clipped stats
    avg_clipped = float(np.mean(clipped_grad_norms))
    max_clipped = float(np.max(clipped_grad_norms))
    
    return avg_loss, avg_unclipped, max_unclipped, avg_clipped, max_clipped


def evaluate(model, val_loader, criterion, device, use_teacher_forcing=True, log_mode=False):
    """Evaluate model on validation set.
    
    Args:
        use_teacher_forcing: If True, use ground truth for next-step inputs (faster, cleaner metrics).
                            If False, use pure autoregressive loop (realistic inference).
        log_mode: If True, print which evaluation mode is being used (for first call)
    """
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    if log_mode:
        mode = "🎯 Teacher Forcing" if use_teacher_forcing else "🔄 Pure Autoregressive"
        print(f"   Eval mode: {mode}")
    
    total_batches = len(val_loader)
    print(f"  Validation: 0/{total_batches} batches", end='', flush=True)
    
    with torch.no_grad():
        for batch_idx, (X_batch, y_batch) in enumerate(val_loader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            if use_teacher_forcing:
                # Teacher forcing: cleaner per-horizon error metrics
                predictions = model(X_batch, y_future=y_batch, teacher_forcing=True)
            else:
                # Pure autoregressive: realistic inference (errors compound)
                predictions = model(X_batch, y_future=None, teacher_forcing=False)
            loss = criterion(predictions, y_batch)
            
            total_loss += loss.item()
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
    
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    avg_loss = total_loss / len(val_loader)
    mae = np.abs(all_predictions - all_targets).mean()
    rmse = np.sqrt(((all_predictions - all_targets) ** 2).mean())
    
    # Directional accuracy on first horizon
    pred_direction = (all_predictions[:, 0] > 0).astype(int)
    true_direction = (all_targets[:, 0] > 0).astype(int)
    dir_acc = (pred_direction == true_direction).mean()
    
    return avg_loss, mae, rmse, dir_acc


def train(
    config_path: str,
    dataloaders: Optional[Dict] = None,
    scalers: Optional[Dict] = None
):
    """
    Core decoder transformer training function.
    
    Args:
        config_path: Path to model config YAML
        dataloaders: Optional pre-loaded DataLoaders (if None, will load from data/processed/)
        scalers: Optional pre-loaded scalers
        
    Note:
        FinCast configuration is now read from the config YAML file under the 'fincast' section.
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data if not provided
    if dataloaders is None:
        print("\n📂 Loading data from data/processed/...")
        from torch.utils.data import TensorDataset, DataLoader
        
        # Load preprocessed data directly from .npy files
        data_dir = Path('data/processed')
        
        train_X_np = np.load(data_dir / 'X_train.npy', allow_pickle=True)
        train_y_np = np.load(data_dir / 'y_train.npy', allow_pickle=True)
        val_X_np = np.load(data_dir / 'X_val.npy', allow_pickle=True)
        val_y_np = np.load(data_dir / 'y_val.npy', allow_pickle=True)
        test_X_np = np.load(data_dir / 'X_test.npy', allow_pickle=True)
        test_y_np = np.load(data_dir / 'y_test.npy', allow_pickle=True)
        
        # Handle object arrays - extract the actual array if wrapped
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
        
        # Create DataLoaders with no workers (Mac compatibility)
        batch_size = config['training']['batch_size']
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        print("✅ Data loaded!")
        scalers = None  # Not needed for inference
    else:
        train_loader = dataloaders['train']
        val_loader = dataloaders['val']
        test_loader = dataloaders['test']
    
    # Setup output directory
    output_dir = Path('models/decoder_transformer')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get num_features from dataset directly (avoid DataLoader iteration on Mac)
    if dataloaders is None:
        # We just created simple datasets, access directly
        num_features = train_X.shape[-1]  # [samples, lookback, features]
        lookback = train_X.shape[1]
        num_horizons = train_y.shape[1]
    else:
        # Using provided dataloaders
        train_dataset = train_loader.dataset
        sample_X, sample_y = train_dataset[0]
        num_features = sample_X.shape[-1]
        lookback = sample_X.shape[0]
        num_horizons = sample_y.shape[0]
    
    # Print detailed configuration
    print("\n" + "="*80)
    print("   Decoder Transformer Configuration")
    print("="*80)
    
    print(f"\n📊 Data Dimensions:")
    print(f"  Lookback window: {lookback} timesteps")
    print(f"  Number of features: {num_features}")
    print(f"  Prediction horizons: {num_horizons}")
    
    # Display date range from config
    if 'data' in config:
        data_cfg = config['data']
        start_date = data_cfg.get('start_date', 'N/A')
        end_date = data_cfg.get('end_date', 'N/A')
        print(f"\n📅 Date Range:")
        print(f"  Start: {start_date}")
        print(f"  End: {end_date}")
        
        # Try to get total samples from metadata
        try:
            metadata_path = Path('data/processed/metadata.yaml')
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = yaml.safe_load(f)
                    train_samples = metadata.get('train_samples', 0)
                    val_samples = metadata.get('val_samples', 0)
                    test_samples = metadata.get('test_samples', 0)
                    total = train_samples + val_samples + test_samples
                    if total > 0:
                        print(f"  Total sequences: {total:,} (train: {train_samples:,}, val: {val_samples:,}, test: {test_samples:,})")
        except Exception:
            pass
    
    # Load feature metadata if available
    try:
        metadata_path = Path('data/processed/metadata.yaml')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = yaml.safe_load(f)
                if 'features' in metadata:
                    actual_features = metadata['features']
                    print(f"\n📋 ACTUAL FEATURES ({len(actual_features)}):")
                    for i, feat in enumerate(actual_features[:10], 1):  # Show first 10
                        print(f"   {i:2d}. {feat}")
                    if len(actual_features) > 10:
                        print(f"   ... and {len(actual_features) - 10} more")
    except Exception as e:
        print(f"\n⚠️  Could not load feature metadata: {e}")
    
    horizons_config = config['data']['prediction_horizons']
    print(f"\n🎯 Output Targets ({len(horizons_config)} horizons):")
    for i, h in enumerate(horizons_config, 1):
        print(f"  {i}. Horizon {h} periods ahead")
    
    # Setup device with detailed info
    device = torch.device(device)
    print(f"\n🖥️  Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Print model architecture
    print("\n" + "="*80)
    print("   Model Architecture")
    print("="*80)
    print(f"\nArchitecture: Decoder-Only Autoregressive Transformer")
    print(f"  d_model: {config['model']['d_model']}")
    print(f"  n_layers: {config['model']['n_layers']}")
    print(f"  n_heads: {config['model']['n_heads']}")
    print(f"  d_ff: {config['model']['d_ff']}")
    print(f"  dropout: {config['model']['dropout']}")
    print(f"\nTraining:")
    print(f"  Epochs: {config['training']['epochs']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    print(f"  Weight decay: {config['training'].get('weight_decay', 0.0)}")
    print(f"  Gradient clip: {config['training'].get('gradient_clip_norm', None)}")
    
    # Read FinCast config from YAML
    fincast_config = config.get('fincast', {})
    use_fincast = fincast_config.get('enabled', False)
    
    # Initialize model (with or without FinCast)
    if use_fincast:
        if not FINCAST_AVAILABLE or DecoderTransformerWithFinCast is None:
            raise ImportError(
                "FinCast is enabled in config but not available.\n"
                "Please ensure the FinCast submodule is properly installed.\n"
                "See scripts/03_training/README.md for setup instructions."
            )
        
        print(f"\n🔧 Initializing model with FinCast integration...")
        
        # Build model config from YAML settings
        model_fincast_config = {
            'd_model': fincast_config.get('d_model', 1280),
            'n_heads': fincast_config.get('n_heads', 16),
            'n_layers': fincast_config.get('n_layers', 50),
            'd_ff': fincast_config.get('d_ff', 5120),
            'dropout': fincast_config.get('dropout', 0.1),
            'freeze': fincast_config.get('freeze_backbone', True),
            'pretrained_path': fincast_config.get('checkpoint_path'),
            'output_dim': fincast_config.get('output_dim', 128)
        }
        
        model = DecoderTransformerWithFinCast(
            config=config,
            num_features=num_features,
            fincast_config=model_fincast_config,
            decoder_transformer_class=DecoderOnlyTransformerAR
        ).to(device)
    else:
        print(f"\n🔧 Initializing standard decoder transformer...")
        model = DecoderOnlyTransformerAR(config, num_features).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🔧 Model Parameters:")
    print(f"   Total: {total_params:,}")
    print(f"   Trainable: {trainable_params:,}")
    
    # Setup TensorBoard (lazy import to avoid Mac ARM64 issues)
    writer = None
    if config.get('logging', {}).get('tensorboard', False):
        try:
            # Lazy import - only load when actually needed
            from torch.utils.tensorboard import SummaryWriter
            
            # Get evaluation mode early for TensorBoard path
            use_teacher_forcing_eval = config['training'].get('eval_teacher_forcing', True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            eval_suffix = "_tf" if use_teacher_forcing_eval else "_ar"
            
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
                base_log_dir = Path(config.get('logging', {}).get('log_dir', 'logs/tensorboard'))
                log_dir = base_log_dir / f"decoder{eval_suffix}" / timestamp
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
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    
    # Adjust learning rate for FinCast runs (larger input dimension: 118 -> 3547 features)
    base_lr = config['training']['learning_rate']
    if use_fincast:
        # Scale LR to handle much larger feature space and prevent gradient explosion
        lr_scale = fincast_config.get('lr_scale', 0.2)
        actual_lr = base_lr * lr_scale
        print(f"\nℹ️  Adjusting learning rate for FinCast: {base_lr:.2e} → {actual_lr:.2e} ({lr_scale}x)")
    else:
        actual_lr = base_lr
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=actual_lr,
        weight_decay=config['training'].get('weight_decay', 0.0)
    )
    
    # Training hyperparams with early stopping
    best_val_loss = float('inf')
    best_val_mae = float('inf')
    patience_counter = 0
    epochs = config['training']['epochs']
    clip_norm = config['training'].get('gradient_clip_norm', None)
    early_stopping_patience = config['training'].get('early_stopping', {}).get('patience', 10)
    use_teacher_forcing_eval = config['training'].get('eval_teacher_forcing', True)
    
    print(f"\n Starting training for {epochs} epochs...")
    print(f"   Device: {device}")
    print(f"   Batch size: {config['training']['batch_size']}")
    print(f"   Learning rate: {config['training']['learning_rate']}")
    print(f"   Gradient clipping: {clip_norm}")
    eval_mode = "Teacher Forcing" if use_teacher_forcing_eval else "Pure Autoregressive"
    print(f"   Evaluation mode: {eval_mode}")
    print("\n" + "="*80)
    
    # Estimate training time
    if use_fincast:
        print("\n⏱️  Estimated time per epoch (with FinCast on CPU): 30-60 minutes")
        print(f"   Total estimated time for {epochs} epochs: {epochs * 0.75:.1f} hours")
    else:
        print(f"\n⏱️  Estimated time per epoch: 5-10 minutes")
        print(f"   Total estimated time for {epochs} epochs: {epochs * 0.125:.1f} hours")
    print(f"   Training on {len(train_loader)} batches per epoch\n")
    
    training_start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        train_loss, avg_unclipped, max_unclipped, avg_clipped, max_clipped = train_epoch(
            model, train_loader, optimizer, criterion, device, clip_norm
        )
        
        val_loss, mae, rmse, dir_acc = evaluate(
            model, val_loader, criterion, device, 
            use_teacher_forcing=use_teacher_forcing_eval,
            log_mode=(epoch == 0)  # Log mode only on first epoch
        )
        
        epoch_time = time.time() - epoch_start_time
        total_elapsed = time.time() - training_start_time
        
        print(f"\nEpoch {epoch+1}/{epochs} - {epoch_time/60:.1f} min (total: {total_elapsed/60:.1f} min)")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val   Loss: {val_loss:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}")
        print(f"  Dir Acc (H1): {dir_acc * 100:.2f}%")
        print(f"  Grad Norm (unclipped): avg={avg_unclipped:.4f}, max={max_unclipped:.4f}")
        print(f"  Grad Norm (clipped):   avg={avg_clipped:.4f}, max={max_clipped:.4f}")
        
        # Log to TensorBoard
        if writer is not None:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Metrics/MAE', mae, epoch)
            writer.add_scalar('Metrics/RMSE', rmse, epoch)
            writer.add_scalar('Metrics/DirectionalAccuracy', dir_acc, epoch)
            writer.add_scalar('Gradients/Unclipped_Avg', avg_unclipped, epoch)
            writer.add_scalar('Gradients/Clipped_Avg', avg_clipped, epoch)
            writer.add_scalar('LR', config['training']['learning_rate'], epoch)
        
        # Every 10 epochs, dump layer-wise grad stats
        if (epoch + 1) % 10 == 0:
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                optimizer.zero_grad()
                preds = model(X_batch, y_future=y_batch, teacher_forcing=True)
                loss = criterion(preds, y_batch)
                loss.backward()
                break
            
            layer_stats = compute_layer_grad_stats(model)
            print("Layer Gradients:")
            for layer_name, stats in sorted(layer_stats.items()):
                print(
                    f"  {layer_name:15s}: "
                    f"norm={stats['norm']:.6f}, "
                    f"max={stats['max']:.6f}, "
                    f"std={stats['std']:.6f}"
                )
        
        print("")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_mae = mae
            patience_counter = 0
            
            # Save checkpoint with all metrics
            # Include eval mode in filename to differentiate models
            eval_suffix = "_tf" if use_teacher_forcing_eval else "_ar"
            checkpoint_path = output_dir / f'decoder_transformer_best{eval_suffix}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mae': mae,
                'val_dir_acc': dir_acc,
                'config': config,
                'eval_teacher_forcing': use_teacher_forcing_eval,  # Track eval mode
                'training_mode': 'teacher_forcing' if use_teacher_forcing_eval else 'autoregressive'
            }, checkpoint_path)
            print(f"  ✅ Saved checkpoint: {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
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
    eval_suffix = "_tf" if use_teacher_forcing_eval else "_ar"
    eval_mode_name = "Teacher Forcing" if use_teacher_forcing_eval else "Autoregressive"
    print(f"  Evaluation mode: {eval_mode_name}")
    print(f"  Model saved: {output_dir / f'decoder_transformer_best{eval_suffix}.pt'}")
    if writer is not None and not os.getenv('CLOUD_ML_JOB_ID'):
        print(f"\n📊 View TensorBoard: tensorboard --logdir logs/tensorboard")
    print("="*80)

    return model, best_val_loss
