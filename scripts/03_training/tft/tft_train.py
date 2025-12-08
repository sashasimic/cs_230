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
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

# Import benchmark utilities
from utils import benchmarks

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
    """Sinusoidal positional encoding for transformer.
    
    Adds positional information to input embeddings to help the model
    understand temporal ordering. Uses sine and cosine functions of
    different frequencies (matching decoder implementation).
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer("pe", pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.
        
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
        
        # Initialize weights properly (CRITICAL for gradient stability)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize GRN weights with scaled initialization."""
        # Scale all linear layers to prevent gradient explosion
        for module in [self.fc1, self.fc2, self.gate]:
            if hasattr(module, 'weight'):
                # Use smaller gain for GRN internal layers
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Skip connection and context projection
        if self.skip is not None:
            nn.init.xavier_uniform_(self.skip.weight, gain=1.0)
            if self.skip.bias is not None:
                nn.init.zeros_(self.skip.bias)
        
        if self.context_dim is not None:
            nn.init.xavier_uniform_(self.context_fc.weight, gain=0.5)
    
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
        
        # Add context if provided (dimension-agnostic)
        if context is not None and self.context_dim is not None:
            context_proj = self.context_fc(context)  # [batch, context_dim] -> [batch, hidden_dim]
            
            # Handle different dimension combinations
            if hidden.dim() == context_proj.dim():
                # Both same dims: direct addition (e.g., both [batch, hidden_dim])
                hidden = hidden + context_proj
            elif hidden.dim() == context_proj.dim() + 1:
                # hidden has extra time dimension: [batch, time, hidden_dim] vs [batch, hidden_dim]
                # Broadcast context across time: [batch, 1, hidden_dim] -> [batch, time, hidden_dim]
                hidden = hidden + context_proj.unsqueeze(1)
            else:
                raise ValueError(
                    f"Unsupported shape combination: hidden {hidden.shape}, context {context_proj.shape}"
                )
        
        # Gated output
        gate = self.sigmoid(self.gate(hidden))
        output = self.fc2(self.dropout(hidden))
        output = gate * output
        
        # Add residual and normalize
        output = self.layer_norm(output + residual)
        
        return output


# TickerGroupAggregator removed - grouping now done at data generation time

class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network (VSN) - learns which features are important."""
    
    def __init__(self, input_dim: int, num_vars: int, hidden_dim: int, dropout: float = 0.0, context_dim: int = None):
        super().__init__()
        self.input_dim = input_dim
        self.num_vars = num_vars
        self.hidden_dim = hidden_dim
        
        # Shared GRN for all variables (more stable than 99 individual GRNs)
        self.shared_grn = GatedResidualNetwork(input_dim, hidden_dim, hidden_dim, dropout, context_dim)
        
        # Variable selection weights - use two-stage reduction for stability
        flattened_dim = num_vars * hidden_dim
        intermediate_dim = max(hidden_dim * 2, num_vars * 2)  # Intermediate bottleneck
        
        # Layer norm before selection to stabilize large concatenated vectors
        self.flatten_norm = nn.LayerNorm(flattened_dim)
        
        # Two-stage dimensional reduction: 12,672 -> 256 -> 99 (more stable than direct)
        self.dimension_reduction = nn.Sequential(
            nn.Linear(flattened_dim, intermediate_dim),
            nn.LayerNorm(intermediate_dim),
            nn.ELU(),
            nn.Dropout(dropout)
        )
        
        self.selection_grn = GatedResidualNetwork(
            intermediate_dim, hidden_dim, num_vars, dropout, context_dim
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
        
        # Process all variables with shared GRN
        processed_vars = []
        for i in range(num_vars):
            processed = self.shared_grn(variables[:, i], context)
            processed_vars.append(processed)
        
        # Stack: [batch, num_vars, hidden_dim]
        processed_vars = torch.stack(processed_vars, dim=1)
        
        # Flatten for selection
        flattened = processed_vars.reshape(variables.shape[0], -1)
        
        # Normalize flattened vector to stabilize gradients
        flattened = self.flatten_norm(flattened)
        
        # Apply two-stage dimensional reduction (12,672 -> 256)
        reduced = self.dimension_reduction(flattened)
        
        # Compute selection weights from reduced representation
        weights = self.selection_grn(reduced, context)
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
        self.use_lstm = config['model'].get('use_lstm', True)
        self.lstm_layers = config['model'].get('lstm_layers', 1) if self.use_lstm else 0
        self.attention_layers = config['model'].get('attention_layers', 1)
        self.attention_heads = config['model']['attention_heads']
        self.dropout = config['model']['dropout']
        self.num_horizons = len(config['data']['prediction_horizons'])
        self.quantiles = config['model'].get('quantiles', [0.5])
        self.num_quantiles = len(self.quantiles)
        self.use_variable_selection = config['model'].get('use_variable_selection', True)
        
        # For simplicity, treat all features as time-varying unknown
        # In production, you'd separate known vs unknown based on config
        self.num_time_varying = self.num_features
        
        print(f"   Initializing TFT with:")
        print(f"   - {self.num_features} input features")
        print(f"   - {self.num_horizons} prediction horizons")
        print(f"   - {self.num_quantiles} quantiles: {self.quantiles}")
        
        # ===== 1. Variable Selection Setup =====
        # Features are already grouped at data generation time
        # We now have ~45-60 compact group-level features instead of 99 raw ticker features
        print(f"\n✅ Using pre-aggregated group features: {self.num_features} features")
        
        # Each feature is already a scalar group-level signal
        vsn_input_dim = 1  # Each feature is scalar
        vsn_num_vars = self.num_features
        
        # ===== 2. Variable Selection Network (VSN) =====
        # Conditionally build VSN based on config
        if self.use_variable_selection:
            # Learns which input features/groups are important
            self.variable_selection = VariableSelectionNetwork(
                input_dim=vsn_input_dim,
                num_vars=vsn_num_vars,
                hidden_dim=self.hidden_size,
                dropout=self.dropout
            )
            self.vsn_norm = nn.LayerNorm(self.hidden_size)  # Normalize VSN output
            print(f"   ✅ Variable Selection Network (VSN) ENABLED")
        else:
            # Simple linear projection instead of VSN (NO NORM - match decoder!)
            self.feature_projection = nn.Linear(self.num_features, self.hidden_size)
            print(f"   ✗ Variable Selection Network (VSN) DISABLED (using linear projection)")
        
        # ===== Positional Encoding & Dropout (Match Decoder) =====
        # Get lookback window from config
        lookback = config['data'].get('lookback_window', config['data'].get('lookback', 192))
        self.pos_encoder = PositionalEncoding(self.hidden_size, max_len=lookback)
        self.dropout_layer = nn.Dropout(self.dropout)
        print(f"   ✅ Added positional encoding (max_len={lookback}) and dropout ({self.dropout})")
        
        # ===== 2. LSTM Encoder (Optional) =====
        # Processes historical sequence
        if self.use_lstm:
            self.lstm_encoder = nn.LSTM(
                input_size=self.hidden_size,
                hidden_size=self.hidden_size,
                num_layers=self.lstm_layers,
                batch_first=True,
                dropout=self.dropout if self.lstm_layers > 1 else 0
            )
            self.lstm_norm = nn.LayerNorm(self.hidden_size)  # Normalize LSTM output
            print(f"   ✅ LSTM Encoder ENABLED ({self.lstm_layers} layer{'s' if self.lstm_layers > 1 else ''}, unidirectional)")
        else:
            print(f"   ✗ LSTM Encoder DISABLED (using direct projection)")
        
        # ===== 3. Static Enrichment (using GRN) =====
        # Embeddings for static categorical features
        self.static_features = static_features
        if static_features:
            # Get cardinalities from config
            augment_groups = config['model'].get('augment_groups', [])
            ticker_groups = config['data'].get('ticker_groups', {})
            
            # Count total tickers across augmentation groups
            total_tickers = 0
            for group_name in augment_groups:
                group_tickers = ticker_groups.get(group_name, {}).get('tickers', [])
                total_tickers += len(group_tickers)
            
            # Ticker embedding: one per ticker in augmentation groups
            ticker_cardinality = total_tickers if total_tickers > 0 else 12
            # Group embedding: one per augmentation group
            group_cardinality = len(augment_groups) if augment_groups else 4
            
            # Use config value if provided, otherwise default to hidden_size // 4
            embedding_dim = config['model'].get('static_embedding_dim', self.hidden_size // 4)
            self.ticker_embedding = nn.Embedding(ticker_cardinality, embedding_dim)
            self.category_embedding = nn.Embedding(group_cardinality, embedding_dim)  # Still named category for backward compat
            
            # Initialize embeddings with smaller values for stability
            nn.init.normal_(self.ticker_embedding.weight, mean=0.0, std=0.01)
            nn.init.normal_(self.category_embedding.weight, mean=0.0, std=0.01)
            
            static_context_dim = embedding_dim * 2  # Concatenate both embeddings
            print(f"   - Static features: {len(static_features)}")
            print(f"     Ticker embedding: {ticker_cardinality} tickers -> {embedding_dim}")
            print(f"     Group embedding: {group_cardinality} groups ({augment_groups}) -> {embedding_dim}")
        else:
            static_context_dim = None
        
        # Static enrichment GRN (optional - can block gradients)
        self.use_static_enrichment = config['model'].get('use_static_enrichment', False)
        if self.use_static_enrichment:
            self.static_enrichment = GatedResidualNetwork(
                input_dim=self.hidden_size,
                hidden_dim=self.hidden_size,
                output_dim=self.hidden_size,
                dropout=self.dropout,
                context_dim=static_context_dim
            )
            self.enrichment_norm = nn.LayerNorm(self.hidden_size)
            print(f"   ✅ Static enrichment GRN ENABLED")
        else:
            print(f"   ✗ Static enrichment GRN DISABLED (better gradient flow)")
        
        # ===== 4. Transformer Encoder (Match Decoder Exactly) =====
        # Use PyTorch's built-in TransformerEncoder for guaranteed correctness
        # This includes: attention, feedforward, dropout, residuals, norms
        dim_feedforward = self.hidden_size * 4  # 384 for hidden_size=96
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.attention_heads,
            dim_feedforward=dim_feedforward,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # pre-norm is more stable
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.attention_layers
        )
        
        # Register causal mask as buffer (won't be trained)
        # This prevents attention from looking at future timesteps
        lookback = config['data'].get('lookback_window', config['data'].get('lookback', 192))
        self.register_buffer(
            'causal_mask',
            self._generate_causal_mask(lookback)
        )
        
        # Final norm on encoder output (matching decoder)
        self.enc_norm = nn.LayerNorm(self.hidden_size)
        
        print(f"   ✅ Using TransformerEncoder ({self.attention_layers} layers, pre-norm, matching decoder)")
        print(f"   ✅ Feedforward dim={dim_feedforward}, GELU activation, dropout={self.dropout}")
        
        # ===== 5. Position-wise Feed-Forward =====
        # Optional GRN (can block gradients)
        self.use_position_wise_grn = config['model'].get('use_position_wise_grn', False)
        if self.use_position_wise_grn:
            self.position_wise_grn = GatedResidualNetwork(
                input_dim=self.hidden_size,
                hidden_dim=self.hidden_size,
                output_dim=self.hidden_size,
                dropout=self.dropout
            )
            print(f"   ✅ Position-wise GRN ENABLED")
        else:
            print(f"   ✗ Position-wise GRN DISABLED (better gradient flow)")
        
        # ===== 6. Future Decoder (Match Decoder Transformer) =====
        # GRU-based autoregressive decoder over horizons (instead of independent heads)
        # This models H1 → H2 → H3 dependencies and distributes gradients better
        self.future_input_dim = 1  # scalar previous target
        self.future_hidden_dim = self.hidden_size
        
        self.future_in_proj = nn.Linear(self.future_input_dim, self.future_hidden_dim)
        self.future_decoder = nn.GRU(
            input_size=self.future_hidden_dim,
            hidden_size=self.future_hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Output projection: hidden state → scalar forecast at each horizon
        self.future_out_proj = nn.Linear(self.future_hidden_dim, 1)
        
        # Learned start token (input at first horizon step)
        self.start_token = nn.Parameter(torch.zeros(1, self.future_input_dim))
        
        print(f"   ✅ Added GRU future decoder (matching decoder transformer)")
        
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
    
    def forward(self, x: torch.Tensor, static_features: torch.Tensor = None, y_future: torch.Tensor = None, teacher_forcing: bool = True, return_attention: bool = False):
        """
        TFT Forward pass.
        
        Args:
            x: [batch, lookback, features]
            static_features: [batch, num_static] - optional static categorical features
            y_future: [batch, num_horizons] - ground truth targets for teacher forcing (optional)
            teacher_forcing: if True and y_future provided, use teacher forcing in future decoder
            return_attention: if True, return attention weights for visualization
        
        Returns:
            predictions: [batch, horizons]
            attention_weights: (optional) [batch, num_heads, seq_len, seq_len] if return_attention=True
        """
        batch_size, lookback, num_features = x.shape
        
        # ===== 1. Variable Selection or Feature Projection =====
        if self.use_variable_selection:
            # Features are already group-level signals from data generation
            # Reshape to [batch, time, num_vars, 1] for VSN
            x_reshaped = x.unsqueeze(-1)  # [batch, lookback, features, 1]
            
            # Apply variable selection to learn feature importance
            selected_features, var_weights = self.variable_selection(x_reshaped)
            # selected_features: [batch, lookback, hidden_size]
            
            selected_features = self.vsn_norm(selected_features)  # Normalize
        else:
            # Simple linear projection without variable selection (no norm - match decoder!)
            selected_features = self.feature_projection(x)  # [batch, lookback, hidden_size]
        
        # ===== Add Positional Encoding & Dropout (Match Decoder) =====
        # Add temporal position information after initial projection
        # This helps the model understand the sequential nature of the data
        selected_features = self.pos_encoder(selected_features)  # [batch, lookback, hidden_size]
        selected_features = self.dropout_layer(selected_features)  # Apply dropout for regularization
        
        # ===== 2. LSTM Encoding (Optional) =====
        if self.use_lstm:
            # Encode the full sequence
            lstm_output, (hidden, cell) = self.lstm_encoder(selected_features)
            # lstm_output: [batch, lookback, hidden_size]
            lstm_output = self.lstm_norm(lstm_output)  # Normalize
            temporal_features = lstm_output
        else:
            # Skip LSTM, use projected features directly
            temporal_features = selected_features
        
        # ===== 3. Static Enrichment =====
        # Create static context from embeddings if available
        static_context = None
        if static_features is not None and self.static_features:
            # static_features: [batch, 2] where column 0=ticker_idx, column 1=category_idx
            ticker_indices = static_features[:, 0]  # [batch]
            category_indices = static_features[:, 1]  # [batch]
            
            # Embed and concatenate
            ticker_emb = self.ticker_embedding(ticker_indices)  # [batch, emb_dim]
            category_emb = self.category_embedding(category_indices)  # [batch, emb_dim]
            static_context = torch.cat([ticker_emb, category_emb], dim=-1)  # [batch, 2*emb_dim]
        
        # Apply GRN to enrich features with static context (if enabled)
        if self.use_static_enrichment:
            enriched = self.static_enrichment(temporal_features, context=static_context)
            enriched = self.enrichment_norm(enriched)  # Normalize to stabilize gradients
        else:
            enriched = temporal_features  # Skip enrichment
        # enriched: [batch, lookback, hidden_size]
        
        # ===== 4. Transformer Encoder =====
        # Use PyTorch built-in encoder (identical to decoder)
        # Causal mask prevents attending to future positions
        batch_size, seq_len, _ = enriched.shape
        mask = self.causal_mask[:seq_len, :seq_len]  # [T, T]
        
        # Use hook to capture attention if needed (doesn't break forward pass)
        attention_weights = None
        if return_attention:
            # Store attention weights using hook
            attn_weights_list = []
            
            def attn_hook(module, input, output):
                # MultiheadAttention returns (output, weights) when need_weights=True
                # But in normal forward, it only returns output
                # We'll capture from the module's internal state instead
                pass
            
            # For now, just run normal forward - attention capture needs deeper integration
            # This prevents breaking the model during logging
            encoded = self.transformer_encoder(enriched, mask=mask)
            # Return None for attention_weights to indicate it's not yet implemented
            attention_weights = None
        else:
            encoded = self.transformer_encoder(enriched, mask=mask)  # [batch, lookback, hidden_size]
        
        # ===== 5. Position-wise Processing =====
        # Apply GRN to each timestep (if enabled)
        if self.use_position_wise_grn:
            processed = self.position_wise_grn(encoded)
        else:
            processed = encoded  # Skip position-wise GRN
        # processed: [batch, lookback, hidden_size]
        
        # ===== 6. Final Norm + Extract Context =====
        # Normalize encoder output (matching decoder)
        processed = self.enc_norm(processed)
        
        # Use last timestep as context for GRU decoder
        context = processed[:, -1, :]  # [batch, hidden_size]
        
        # Decode future horizons autoregressively (matching decoder transformer)
        batch_size = context.size(0)
        device = context.device
        
        # Initial hidden state from encoder context
        h0 = context.unsqueeze(0)  # [1, batch, hidden_size]
        
        # Start token
        start = self.start_token.expand(batch_size, 1, self.future_input_dim)
        
        if y_future is not None and teacher_forcing:
            # Teacher forcing: use ground truth as inputs
            H = self.num_horizons
            
            if H > 1:
                # Build input: [start, y_0, y_1, ..., y_{H-2}]
                prev_targets = y_future[:, :-1].unsqueeze(-1)  # [batch, H-1, 1]
                dec_in = torch.cat([start, prev_targets], dim=1)  # [batch, H, 1]
            else:
                dec_in = start  # [batch, 1, 1]
            
            # Embed inputs
            dec_in = self.future_in_proj(dec_in)  # [batch, H, hidden_size]
            
            # Run GRU over horizons
            dec_out, _ = self.future_decoder(dec_in, h0)  # [batch, H, hidden_size]
            
            # Project to scalar at each horizon
            predictions = self.future_out_proj(dec_out).squeeze(-1)  # [batch, H]
        else:
            # Pure autoregressive (inference mode)
            preds = []
            h_t = h0
            prev_input = start
            
            for t in range(self.num_horizons):
                # Embed previous target
                dec_in = self.future_in_proj(prev_input)  # [batch, 1, hidden_size]
                
                # GRU step
                dec_out, h_t = self.future_decoder(dec_in, h_t)  # [batch, 1, hidden_size]
                
                # Predict next horizon
                y_t = self.future_out_proj(dec_out).squeeze(-1)  # [batch, 1]
                preds.append(y_t)
                
                # Use prediction as next input
                prev_input = y_t.unsqueeze(-1)  # [batch, 1, 1]
            
            predictions = torch.cat(preds, dim=1)  # [batch, H]
        
        if return_attention:
            return predictions, attention_weights
        return predictions
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization (matching decoder exactly)."""
        for name, param in self.named_parameters():
            # Skip GRN layers - they handle their own initialization
            if any(x in name for x in ['static_enrichment', 'position_wise_grn']):
                continue
            
            # Skip TransformerEncoder - it has optimized PyTorch defaults
            if 'transformer_encoder' in name:
                continue
            
            if 'weight' in name and param.dim() >= 2:
                # Scale input projection like decoder does
                if 'feature_projection' in name:
                    scale = 1.0 / np.sqrt(self.num_features)
                    nn.init.xavier_uniform_(param, gain=scale)
                else:
                    # All other layers: use default xavier (matching decoder)
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
            
            # Group by layer type (match decoder's per-layer breakdown)
            if 'variable_selection' in name or 'variable_grns' in name:
                layer_type = 'VSN'
            elif 'feature_projection' in name and 'weight' in name:
                layer_type = 'Input'
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
            elif 'transformer_encoder.layers' in name:
                # TransformerEncoder layers (matching decoder)
                parts = name.split('.')
                layer_idx = parts[2] if len(parts) > 2 else '?'
                # Break out attention vs feedforward within each encoder layer
                if 'self_attn' in name:
                    layer_type = f'Attention_L{layer_idx}'
                elif 'linear1' in name or 'linear2' in name:
                    layer_type = f'Feedforward_L{layer_idx}'
                elif 'norm1' in name or 'norm2' in name:
                    # Norms contribute to the attention/ff layers they're part of
                    if 'norm1' in name:
                        layer_type = f'Attention_L{layer_idx}'
                    else:
                        layer_type = f'Feedforward_L{layer_idx}'
                else:
                    layer_type = f'Encoder_L{layer_idx}_Other'
            elif 'enc_norm' in name:
                layer_type = 'EncoderNorm'
            elif 'future_decoder' in name or 'future_in_proj' in name or 'future_out_proj' in name or 'start_token' in name:
                # Future decoder (GRU-based autoregressive decoder)
                layer_type = 'FutureDecoder'
            elif 'quantile_outputs' in name:
                layer_type = 'Output'
            elif 'pos_encoder' in name:
                layer_type = 'PosEnc'
            else:
                layer_type = 'Other'
            
            if layer_type not in layer_stats:
                layer_stats[layer_type] = {'norm': 0.0, 'max': 0.0, 'std': 0.0}
            
            # Aggregate like decoder does: sum norms, max for max/std
            layer_stats[layer_type]['norm'] += grad_norm
            layer_stats[layer_type]['max'] = max(layer_stats[layer_type]['max'], grad_max)
            layer_stats[layer_type]['std'] = max(layer_stats[layer_type]['std'], grad_std)
    
    return layer_stats


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


def print_attention_console(attention_weights, max_display=36, sample_mode='sample'):
    """Print attention patterns as ASCII art in console.
    
    Args:
        attention_weights: [batch, num_heads, seq_len, seq_len]
        max_display: Maximum sequence length to display (for readability)
        sample_mode: 'full' for first N steps, 'sample' for beginning/middle/end
    """
    if attention_weights is None:
        return
    
    # Get first sample, average across heads
    batch_size, num_heads, seq_len, _ = attention_weights.shape
    attn_avg = attention_weights[0, :, :, :].mean(dim=0)  # [seq, seq]
    attn_avg = attn_avg.detach().cpu().numpy()
    
    if sample_mode == 'sample' and seq_len > max_display:
        # Show beginning, middle, end
        chunk_size = max_display // 3
        indices = list(range(chunk_size)) + \
                  list(range(seq_len//2 - chunk_size//2, seq_len//2 + chunk_size//2)) + \
                  list(range(seq_len - chunk_size, seq_len))
        attn_display = attn_avg[indices][:, indices]
        display_indices = indices
    else:
        # Show first N timesteps
        display_len = min(seq_len, max_display)
        attn_display = attn_avg[:display_len, :display_len]
        display_indices = list(range(display_len))
    
    # Define ASCII characters for different attention levels
    chars = [' ', '·', '░', '▒', '▓', '█']
    
    display_len = len(display_indices)
    if sample_mode == 'sample' and seq_len > max_display:
        print(f"\n  📊 Attention Pattern (Sampled from {seq_len} timesteps: start/mid/end, Avg {num_heads} heads):")
    else:
        print(f"\n  📊 Attention Pattern (First {display_len}/{seq_len} timesteps, Avg {num_heads} heads):")
    
    print(f"     Query → | " + ''.join([f"{display_indices[i]%10}" for i in range(display_len)]))
    print(f"     --------+-" + '-' * display_len)
    
    for i in range(display_len):
        # Convert attention values to ASCII characters
        row = attn_display[i]
        ascii_row = ''
        for val in row:
            # Map [0, 1] to character index
            char_idx = min(int(val * len(chars)), len(chars) - 1)
            ascii_row += chars[char_idx]
        
        t_idx = display_indices[i]
        print(f"     t={t_idx:3d} Key | {ascii_row}")
    
    # Show statistics (use full sequence, not just displayed portion)
    recent_attn = attn_avg[:, -3:].mean()  # Last 3 timesteps (full sequence)
    distant_attn = attn_avg[:, :3].mean()  # First 3 timesteps (full sequence)
    mid_attn = attn_avg[:, seq_len//2-1:seq_len//2+2].mean()  # Middle 3 timesteps
    print(f"\n  💡 Avg attention to recent past (last 3 of {seq_len}): {recent_attn:.3f}")
    print(f"  💡 Avg attention to middle ({seq_len//2-1}-{seq_len//2+1}): {mid_attn:.3f}")
    print(f"  💡 Avg attention to distant past (first 3): {distant_attn:.3f}")


def log_attention_heatmap(writer, attention_weights, epoch, max_samples=4, max_timesteps=64):
    """Log attention heatmap to TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter
        attention_weights: [batch, num_heads, seq_len, seq_len]
        epoch: Current epoch number
        max_samples: Max number of samples to visualize
        max_timesteps: Max sequence length to show (for readability)
    """
    if writer is None or attention_weights is None:
        return
    
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    
    # Get first sample, average across heads
    batch_size, num_heads, seq_len, _ = attention_weights.shape
    num_samples = min(batch_size, max_samples)
    seq_len = min(seq_len, max_timesteps)
    
    # Average across attention heads
    attn_avg = attention_weights[:num_samples, :, :seq_len, :seq_len].mean(dim=1)  # [samples, seq, seq]
    attn_avg = attn_avg.detach().cpu().numpy()
    
    # Create heatmap for each sample
    for sample_idx in range(num_samples):
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(attn_avg[sample_idx], cmap='viridis', aspect='auto')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        ax.set_title(f'Attention Heatmap (Sample {sample_idx+1}, Avg across {num_heads} heads)')
        plt.colorbar(im, ax=ax)
        
        # Log to TensorBoard
        writer.add_figure(f'attention/sample_{sample_idx}', fig, epoch)
        plt.close(fig)


def train_epoch(model: nn.Module, dataloader, criterion, optimizer, device, epoch: int = 0, clip_norm: float = 1.0, has_static_features: bool = False) -> dict:
    """Train for one epoch and return detailed metrics (matches decoder transformer)."""
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
        predictions = model(batch_X, static_features=batch_static, y_future=batch_y, teacher_forcing=True)
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
    
    # Set random seeds for reproducibility (if configured)
    if 'seed' in config:
        seed = config['seed']
        print(f"🎲 Setting random seed: {seed}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    # Initialize static features flag (will be set based on data loading path)
    has_static_features = False
    
    # Load data if not provided
    if dataloaders is None:
        # Always use data/processed/ (unified behavior for local and Vertex AI)
        # Local: Copied from versioned dataset by *_train_local.py
        # Vertex AI: Downloaded from GCS by train_vertex.py
        data_path = Path('data/processed')  # Define here for both branches
        
        if dataset_version:
            print(f"\n📂 Loading data from: {data_path}...")
            
            if not data_path.exists():
                raise FileNotFoundError(
                    f"Data directory not found: {data_path}\n"
                    f"For local training, run:\n"
                    f"  python scripts/03_training/tft/tft_train_local.py --dataset-version {dataset_version}"
                )
            
            # Load preprocessed arrays directly
            from torch.utils.data import TensorDataset, DataLoader
            
            train_X_np = np.load(data_path / 'X_train.npy', allow_pickle=True)
            train_y_np = np.load(data_path / 'y_train.npy', allow_pickle=True)
            val_X_np = np.load(data_path / 'X_val.npy', allow_pickle=True)
            val_y_np = np.load(data_path / 'y_val.npy', allow_pickle=True)
            test_X_np = np.load(data_path / 'X_test.npy', allow_pickle=True)
            test_y_np = np.load(data_path / 'y_test.npy', allow_pickle=True)
            
            # Load static features if available (for augmented datasets)
            static_train_np = None
            static_val_np = None
            static_test_np = None
            has_static_features = False
            
            # Debug: List files in data_path to verify static files were downloaded
            print(f"  🔍 Files in {data_path}:")
            for f in sorted(data_path.glob('*.npy')):
                print(f"     - {f.name}")
            
            if (data_path / 'static_train.npy').exists():
                # Check config to see if static features are enabled
                config_static_features = config['model'].get('static_features', [])
                has_static_features = len(config_static_features) > 0
                
                if has_static_features:
                    # Only load if enabled in config
                    static_train_np = np.load(data_path / 'static_train.npy', allow_pickle=True)
                    static_val_np = np.load(data_path / 'static_val.npy', allow_pickle=True)
                    static_test_np = np.load(data_path / 'static_test.npy', allow_pickle=True)
                    print(f"  ✅ Loaded static features: train{static_train_np.shape}, val{static_val_np.shape}, test{static_test_np.shape}")
                else:
                    # Keep as None when disabled (matching decoder)
                    print(f"  ⚠️  Static feature files exist but DISABLED in config (static_features=[])")
                    print(f"     Will pass None to model (matching decoder)")
            
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
            
            # Convert static features to tensors if present
            if static_train_np is not None:
                # Static features are saved as strings, need to convert to indices
                # Get mappings from config
                augment_groups = config['model'].get('augment_groups', [])
                ticker_groups = config['data'].get('ticker_groups', {})
                
                # Build ticker list and group mapping
                augment_tickers = []
                ticker_to_group = {}
                for group_name in augment_groups:
                    group_tickers = ticker_groups.get(group_name, {}).get('tickers', [])
                    augment_tickers.extend(group_tickers)
                    for ticker in group_tickers:
                        ticker_to_group[ticker] = group_name
                
                # Create ticker -> index mapping
                ticker_to_idx = {ticker: idx for idx, ticker in enumerate(augment_tickers)}
                # Create group -> index mapping (groups are used as categories)
                group_to_idx = {group: idx for idx, group in enumerate(augment_groups)}
                
                # Convert string arrays to index arrays
                def convert_static_to_indices(static_np):
                    result = np.zeros(static_np.shape, dtype=np.int64)
                    for i in range(len(static_np)):
                        ticker_str = static_np[i, 0]
                        group_str = static_np[i, 1]  # Now stores group name instead of old category
                        result[i, 0] = ticker_to_idx.get(ticker_str, 0)
                        result[i, 1] = group_to_idx.get(group_str, 0)
                    return result
                
                train_static_indices = convert_static_to_indices(static_train_np)
                val_static_indices = convert_static_to_indices(static_val_np)
                test_static_indices = convert_static_to_indices(static_test_np)
                
                train_static = torch.tensor(train_static_indices, dtype=torch.long)
                val_static = torch.tensor(val_static_indices, dtype=torch.long)
                test_static = torch.tensor(test_static_indices, dtype=torch.long)
            
            # Create datasets and dataloaders
            # Always use 3-tuple format (X, y, static) for consistency
            # Use zero-filled placeholder tensors when static features not present
            batch_size = config['training']['batch_size']
            if static_train_np is None:
                # Create dummy static tensors (will be ignored by model when None passed)
                train_static = torch.zeros((len(train_X), 2), dtype=torch.long)
                val_static = torch.zeros((len(val_X), 2), dtype=torch.long)
                test_static = torch.zeros((len(test_X), 2), dtype=torch.long)
            
            train_dataset = TensorDataset(train_X, train_y, train_static)
            val_dataset = TensorDataset(val_X, val_y, val_static)
            test_dataset = TensorDataset(test_X, test_y, test_static)
            
            dataloaders = {
                'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
                'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
                'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            }
            scalers = None  # Scalers not available when loading from numpy
            print("✅ Data loaded!")
        else:
            # No preprocessed data found
            raise FileNotFoundError(
                f"No preprocessed data found in {data_path}.\n"
                f"Please generate dataset first using:\n"
                f"  python scripts/05_deployment/generate_dataset.py --model-type tft-augmented --config {config_path}"
            )
    
    # Print detailed feature information
    print("\n" + "="*80)
    print("   Feature Configuration")
    print("="*80)
    
    # Get sample batch to determine actual feature count
    sample_batch = next(iter(dataloaders['train']))
    # Always 3 items: (X, y, static) - static may be empty placeholder
    batch_X, batch_y, _ = sample_batch
    num_features = batch_X.shape[2]
    lookback = batch_X.shape[1]
    num_horizons = batch_y.shape[1]
    
    print(f"\n📊 Data Dimensions:")
    print(f"  Lookback window: {lookback} timesteps")
    print(f"  Number of features: {num_features}")
    print(f"  Prediction horizons: {num_horizons}")
    
    # Update config with actual lookback from data (important for causal mask sizing)
    config['data']['lookback_window'] = lookback
    
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
    
    # Display static features if configured
    static_features = config['model'].get('static_features', [])
    if static_features:
        print(f"\n📌 STATIC Features ({len(static_features)}):")
        for i, feat in enumerate(static_features, 1):
            print(f"  {i:2d}. {feat}")
    
    total_config_features = len(time_varying_known) + len(time_varying_unknown)
    print(f"\n➡️  Total Features (config): {total_config_features}")
    print(f"➡️  Total Features (actual data): {num_features}")
    if static_features:
        print(f"➡️  Static Features: {len(static_features)}")
    
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
    
    # Output targets - MUST come from dataset metadata (single source of truth)
    metadata_path = Path('data/processed/metadata.yaml')
    
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"\n❌ Dataset metadata not found: {metadata_path}\n"
            f"   Prediction horizons MUST come from dataset metadata.\n"
            f"   Please ensure you're using a dataset version or have generated data locally."
        )
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = yaml.safe_load(f)
            # Extract from 'data' section (v11+ metadata structure)
            data_config = metadata.get('data', metadata)
            horizons_config = data_config.get('prediction_horizons', None)
            
            if not horizons_config:
                raise ValueError(
                    f"\n❌ 'prediction_horizons' not found in dataset metadata!\n"
                    f"   Metadata structure: {list(metadata.keys())}\n"
                    f"   Data section keys: {list(data_config.keys()) if data_config else 'None'}\n"
                    f"   Dataset metadata is the single source of truth - config fallback removed."
                )
            
            print(f"\n✅ Using prediction horizons from dataset metadata: {horizons_config}")
    except Exception as e:
        raise RuntimeError(
            f"\n❌ Failed to read prediction horizons from dataset metadata: {e}\n"
            f"   Metadata path: {metadata_path}\n"
            f"   Dataset metadata is required - no config fallback."
        ) from e
    
    # Load feature names from feature_names.txt for benchmark historical returns extraction
    feature_names_path = data_path / 'feature_names.txt'
    if feature_names_path.exists():
        with open(feature_names_path, 'r') as f:
            feature_names = [line.strip() for line in f if line.strip()]
            metadata['feature_names'] = feature_names
            print(f"\n📊 Loaded {len(feature_names)} feature names for benchmark analysis")
    
    print(f"\n🎯 Output Targets ({len(horizons_config)} horizons):")
    for i, h in enumerate(horizons_config, 1):
        print(f"  {i}. Horizon {h} (target_{h}_periods_ahead)")
    
    # ⚠️ CRITICAL: Override config with dataset metadata horizons
    # The model reads num_horizons from config['data']['prediction_horizons']
    config['data']['prediction_horizons'] = horizons_config
    print(f"\n✅ Updated config['data']['prediction_horizons'] = {horizons_config}")
    
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
    use_vsn = model_config.get('use_variable_selection', True)
    if use_vsn:
        print(f"    - Variable Selection Network (VSN) ✓")
    else:
        print(f"    - Variable Selection Network (VSN) ✗ DISABLED")
        print(f"    - Linear Feature Projection (VSN replacement)")
    
    use_lstm = model_config.get('use_lstm', True)
    if use_lstm:
        lstm_layers = model_config.get('lstm_layers', 1)
        print(f"    - LSTM Encoder ({lstm_layers} layer{'s' if lstm_layers > 1 else ''})")
    else:
        print(f"    - LSTM Encoder ✗ DISABLED")
    
    print(f"    - Gated Residual Networks (GRN)")
    
    attention_layers = model_config.get('attention_layers', 1)
    print(f"    - Temporal Self-Attention ({model_config['attention_heads']} heads, {attention_layers} layer{'s' if attention_layers > 1 else ''})")
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
    # Only include enabled components
    additional_model_info = {
        'Model Type': 'Temporal Fusion Transformer',
        'Hidden Size': model_config['hidden_size'],
        'Attention Heads': model_config['attention_heads'],
        'Attention Layers': model_config.get('attention_layers', 1),
        'Dropout': model_config['dropout'],
        'Total Parameters': f"{total_params:,}",
        'Trainable Parameters': f"{trainable_params:,}"
    }
    
    # Conditionally add enabled components
    if model_config.get('use_lstm', False):
        additional_model_info['LSTM Layers'] = model_config.get('lstm_layers', 1)
    
    if model_config.get('use_variable_selection', False):
        additional_model_info['Variable Selection Network'] = 'Enabled'
    
    if model_config.get('use_static_enrichment', False):
        additional_model_info['Static Enrichment'] = 'Enabled'
    
    if model_config.get('use_position_wise_grn', False):
        additional_model_info['Position-wise GRN'] = 'Enabled'
    
    tb_utils.log_experiment_metadata(
        writer, dataset_version, start_date, end_date, horizons_config,
        lookback, num_features, num_horizons,
        train_samples, val_samples, test_samples,
        'Temporal Fusion Transformer', config, total_params, trainable_params,
        additional_model_info
    )
    
    # Log training hyperparameters for HP tuning comparison
    tb_utils.log_training_hyperparameters(writer, training_config, model_config)
    
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
    
    # Create unique output directory per run
    # Get run name from config or generate timestamp-based one
    run_name = config.get('logging', {}).get('run_name')
    if run_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f"run_{timestamp}"
    
    output_dir = Path('models/tft') / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Output directory: {output_dir}")
    print(f"   Run name: {run_name}")
    
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
            epoch=epoch+1, clip_norm=clip_norm, has_static_features=has_static_features
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
            for batch_idx, batch in enumerate(dataloaders['val']):
                # Always 3 items: (X, y, static) - static may be empty placeholder
                batch_X, batch_y, batch_static = batch
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                batch_static = batch_static.to(device) if has_static_features else None
                
                predictions = model(batch_X, static_features=batch_static)
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
                    print(f"    {layer_name:12s}: norm={stats['norm']:.4f}, max={stats['max']:.4f}, std={stats['std']:.4f}")
        
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
                # Get a fresh batch and compute gradients
                model.train()
                for X_batch, y_batch, static_batch in dataloaders['train']:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    static_batch = static_batch.to(device) if static_batch is not None else None
                    
                    optimizer.zero_grad()
                    predictions = model(X_batch, static_features=static_batch, y_future=y_batch, teacher_forcing=True)
                    loss = criterion(predictions, y_batch)
                    loss.backward()
                    break  # Only need one batch for histogram
                
                # Compute detailed layer-wise gradient stats
                layer_stats = compute_layer_grad_stats(model)
                print("  Layer Gradients (detailed):")
                for layer_name, stats in sorted(layer_stats.items()):
                    print(
                        f"    {layer_name:15s}: "
                        f"norm={stats['norm']:.6f}, "
                        f"max={stats['max']:.6f}, "
                        f"std={stats['std']:.6f}"
                    )
                
                # Log histograms to TensorBoard
                tb_utils.log_gradients_and_weights(writer, model, epoch)
                if writer is not None:
                    writer.flush()
                print(f"  ✅ Logged gradient/weight histograms to TensorBoard")
                
                # Simple attention statistics (doesn't break forward pass)
                try:
                    model.eval()
                    with torch.no_grad():
                        # Get one validation batch
                        for val_batch in dataloaders['val']:
                            val_X, val_y, val_static = val_batch
                            val_X = val_X.to(device)
                            val_static = val_static.to(device) if has_static_features else None
                            
                            # Run normal forward pass
                            predictions = model(val_X, static_features=val_static)
                            
                            # Print basic attention layer statistics
                            print(f"\n  📊 Attention Layer Statistics (Epoch {epoch+1}):")
                            
                            # Check attention layer weights/activations
                            for layer_idx, layer in enumerate(model.transformer_encoder.layers):
                                attn_module = layer.self_attn
                                
                                # Get weight norms
                                in_proj_weight = attn_module.in_proj_weight
                                out_proj_weight = attn_module.out_proj.weight
                                
                                in_norm = in_proj_weight.norm().item()
                                out_norm = out_proj_weight.norm().item()
                                in_std = in_proj_weight.std().item()
                                out_std = out_proj_weight.std().item()
                                
                                print(f"     Layer {layer_idx}: in_proj_norm={in_norm:.3f}, out_proj_norm={out_norm:.3f}")
                                print(f"                in_proj_std={in_std:.4f}, out_proj_std={out_std:.4f}")
                            
                            # Check encoder output statistics (proxy for attention effectiveness)
                            with torch.no_grad():
                                # Get intermediate representation
                                if hasattr(model, 'feature_projection'):
                                    features = model.feature_projection(val_X[:1])
                                else:
                                    x_reshaped = val_X[:1].unsqueeze(-1)
                                    features, _ = model.variable_selection(x_reshaped)
                                
                                features = model.pos_encoder(features)
                                features = model.dropout_layer(features)
                                
                                # Pass through transformer
                                mask = model.causal_mask[:features.size(1), :features.size(1)]
                                encoded = model.transformer_encoder(features, mask=mask)
                                
                                # Statistics on encoded output
                                enc_mean = encoded.mean().item()
                                enc_std = encoded.std().item()
                                enc_max = encoded.abs().max().item()
                                
                                print(f"\n     Encoded output: mean={enc_mean:.4f}, std={enc_std:.4f}, max_abs={enc_max:.4f}")
                                
                                # Check if output is collapsing (all near zero)
                                if enc_std < 0.01:
                                    print(f"     ⚠️  WARNING: Low variance - possible attention collapse!")
                                elif enc_std > 0.5:
                                    print(f"     ✅ Good variance - attention is active")
                                
                                # Timestep group analysis - where is attention focusing?
                                seq_len = encoded.size(1)
                                
                                # Split into groups: beginning (0-25%), middle (37.5-62.5%), end (75-100%)
                                begin_end = seq_len // 4
                                mid_start = int(seq_len * 0.375)
                                mid_end = int(seq_len * 0.625)
                                end_start = int(seq_len * 0.75)
                                
                                # Compute statistics for each region
                                begin_region = encoded[:, :begin_end, :]
                                mid_region = encoded[:, mid_start:mid_end, :]
                                end_region = encoded[:, end_start:, :]
                                
                                begin_std = begin_region.std().item()
                                mid_std = mid_region.std().item()
                                end_std = end_region.std().item()
                                
                                begin_mean_abs = begin_region.abs().mean().item()
                                mid_mean_abs = mid_region.abs().mean().item()
                                end_mean_abs = end_region.abs().mean().item()
                                
                                # Normalize to percentages
                                total_activity = begin_mean_abs + mid_mean_abs + end_mean_abs
                                begin_pct = (begin_mean_abs / total_activity) * 100
                                mid_pct = (mid_mean_abs / total_activity) * 100
                                end_pct = (end_mean_abs / total_activity) * 100
                                
                                print(f"\n  📍 Timestep Attention Focus (Activity Distribution):")
                                print(f"     Beginning [t=0-{begin_end-1}]:      {begin_pct:.1f}% (std={begin_std:.3f})")
                                print(f"     Middle [t={mid_start}-{mid_end-1}]:     {mid_pct:.1f}% (std={mid_std:.3f})")
                                print(f"     End [t={end_start}-{seq_len-1}]:        {end_pct:.1f}% (std={end_std:.3f})")
                                
                                # Show bar chart
                                max_pct = max(begin_pct, mid_pct, end_pct)
                                begin_bar = '█' * int((begin_pct / max_pct) * 30)
                                mid_bar = '█' * int((mid_pct / max_pct) * 30)
                                end_bar = '█' * int((end_pct / max_pct) * 30)
                                
                                print(f"\n     Visual:")
                                print(f"     Beginning: {begin_bar} {begin_pct:.1f}%")
                                print(f"     Middle:    {mid_bar} {mid_pct:.1f}%")
                                print(f"     End:       {end_bar} {end_pct:.1f}%")
                                
                                # Interpretation
                                if end_pct > 40:
                                    print(f"     💡 Strong recent focus - model using recent past")
                                elif mid_pct > 40:
                                    print(f"     💡 Balanced temporal focus - looking at history")
                                elif begin_pct > 40:
                                    print(f"     💡 Distant past focus - long-term patterns")
                            
                            break  # Only need one batch
                    model.train()
                except Exception as e:
                    print(f"  ⚠️  ERROR logging attention stats: {e}")
                    import traceback
                    traceback.print_exc()
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
    
    # =========================================================================
    # Validation Set Benchmark Comparison (Best Model)
    # =========================================================================
    print("\n" + "="*80)
    print("   Validation Set Benchmarks (Best Model)")
    print("="*80)
    
    # Re-evaluate on validation set with best model
    model.eval()
    val_preds = []
    val_targets = []
    val_inputs = []  # Collect inputs to extract historical returns
    val_loss = 0.0
    val_batches = 0
    
    with torch.no_grad():
        for batch in dataloaders['val']:
            batch_X, batch_y, batch_static = batch
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            batch_static = batch_static.to(device) if has_static_features else None
            
            predictions = model(batch_X, static_features=batch_static)
            loss = criterion(predictions, batch_y)
            
            val_loss += loss.item()
            val_batches += 1
            val_preds.append(predictions)
            val_targets.append(batch_y)
            val_inputs.append(batch_X.cpu())  # Save inputs for historical returns
    
    val_preds = torch.cat(val_preds, dim=0)
    val_targets = torch.cat(val_targets, dim=0)
    val_inputs = torch.cat(val_inputs, dim=0)  # [batch, lookback, features]
    val_metrics = compute_metrics(val_preds, val_targets, horizons=horizons_config)
    
    print(f"\n📊 Validation Set Results (Best Model):")
    print(f"  Val Loss: {val_loss / val_batches:.6f}")
    print(f"  Val MAE: {val_metrics['mae']:.6f}")
    print(f"  Val RMSE: {val_metrics['rmse']:.6f}")
    print(f"  Val Dir Acc: {val_metrics['dir_acc']:.2f}%")
    
    # Extract historical target returns for MA/EMA benchmarks
    # Find target_basket_close feature index
    target_feature_idx = None
    if 'feature_names' in metadata:
        feature_names = metadata['feature_names']
        if 'target_basket_close' in feature_names:
            target_feature_idx = feature_names.index('target_basket_close')
    
    val_historical_returns = None
    if target_feature_idx is not None:
        # Extract target_basket_close from input sequences: [batch, lookback, features]
        target_prices = val_inputs[:, :, target_feature_idx].numpy()  # [batch, lookback]
        
        # Compute returns from prices (avoid division by zero)
        val_historical_returns = np.zeros_like(target_prices)
        val_historical_returns[:, 1:] = np.diff(target_prices, axis=1) / (target_prices[:, :-1] + 1e-8)
        val_historical_returns[:, 0] = 0  # First return is undefined, set to 0
    
    # Compute validation benchmarks with historical returns
    val_benchmark_results = benchmarks.compute_all_benchmarks(
        targets=val_targets,
        historical_returns=val_historical_returns,
        horizons=horizons_config,
        ma_windows=[5, 10, 21],
        ema_alpha=0.3
    )
    
    # Print validation benchmark comparison
    benchmarks.print_benchmark_comparison(
        model_metrics=val_metrics,
        benchmark_metrics=val_benchmark_results,
        model_name="TFT (Validation)"
    )
    
    # Evaluate on test set
    model.eval()
    test_preds = []
    test_targets = []
    test_inputs = []  # Collect inputs to extract historical returns
    test_loss = 0.0
    test_batches = 0
    
    print(f"  Evaluating on test set...")
    with torch.no_grad():
        for batch in dataloaders['test']:
            # Always 3 items: (X, y, static) - static may be empty placeholder
            batch_X, batch_y, batch_static = batch
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            batch_static = batch_static.to(device) if has_static_features else None
            
            predictions = model(batch_X, static_features=batch_static)
            loss = criterion(predictions, batch_y)
            
            test_loss += loss.item()
            test_batches += 1
            
            test_preds.append(predictions)
            test_targets.append(batch_y)
            test_inputs.append(batch_X.cpu())  # Save inputs for historical returns
    
    test_loss = test_loss / test_batches
    test_preds = torch.cat(test_preds, dim=0)
    test_targets = torch.cat(test_targets, dim=0)
    test_inputs = torch.cat(test_inputs, dim=0)  # [batch, lookback, features]
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
    
    # =========================================================================
    # Test Set Benchmark Comparison
    # =========================================================================
    print("\n" + "="*80)
    print("   Test Set Benchmarks")
    print("="*80)
    
    # Extract historical target returns for MA/EMA benchmarks
    test_historical_returns = None
    if target_feature_idx is not None:
        # Extract target_basket_close from input sequences: [batch, lookback, features]
        target_prices = test_inputs[:, :, target_feature_idx].numpy()  # [batch, lookback]
        
        # Compute returns from prices (avoid division by zero)
        test_historical_returns = np.zeros_like(target_prices)
        test_historical_returns[:, 1:] = np.diff(target_prices, axis=1) / (target_prices[:, :-1] + 1e-8)
        test_historical_returns[:, 0] = 0  # First return is undefined, set to 0
    
    # Compute naïve forecast and other benchmarks with historical returns
    test_benchmark_results = benchmarks.compute_all_benchmarks(
        targets=test_targets,
        historical_returns=test_historical_returns,
        horizons=horizons_config,
        ma_windows=[5, 10, 21],
        ema_alpha=0.3
    )
    
    # Print test benchmark comparison
    benchmarks.print_benchmark_comparison(
        model_metrics=test_metrics,
        benchmark_metrics=test_benchmark_results,
        model_name="TFT (Test)"
    )
    
    # Log hyperparameters to TensorBoard HParams
    hparams = {
        'model': 'TFT',
        'hidden_size': model_config['hidden_size'],
        'lstm_layers': model_config.get('lstm_layers', 1),
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