#!/usr/bin/env python3
"""
Shared Neural Network Modules

Common building blocks used across different model architectures:
- Positional Encoding
- Gated Residual Networks (GRN)
"""

import torch
import torch.nn as nn
import numpy as np


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
    """Gated Residual Network (GRN) - core building block.
    
    Applies non-linear processing with gating and residual connections.
    Used in TFT and can be used in other architectures.
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


def generate_causal_mask(size: int) -> torch.Tensor:
    """Generate causal mask to prevent attention to future positions.
    
    Args:
        size: Sequence length
        
    Returns:
        mask: [size, size] with 0 for allowed, -inf for masked
        [[  0, -inf, -inf],
         [  0,   0, -inf],
         [  0,   0,   0]]
    """
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask