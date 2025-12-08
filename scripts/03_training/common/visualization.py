#!/usr/bin/env python3
"""
Visualization Utilities

Common visualization functions for model analysis:
- Attention heatmaps
- Training curves
- Console visualizations
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import Optional, List, Dict, Any

# Set non-interactive backend to avoid display issues
matplotlib.use('Agg')


def print_attention_console(
    attention_weights: torch.Tensor,
    max_display: int = 36,
    sample_mode: str = 'sample'
) -> None:
    """
    Print attention patterns as ASCII art in console.
    
    Args:
        attention_weights: [batch, num_heads, seq_len, seq_len]
        max_display: Maximum sequence length to display
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
    recent_attn = attn_avg[:, -3:].mean()  # Last 3 timesteps
    distant_attn = attn_avg[:, :3].mean()  # First 3 timesteps
    mid_attn = attn_avg[:, seq_len//2-1:seq_len//2+2].mean()  # Middle 3
    
    print(f"\n  💡 Avg attention to recent past (last 3 of {seq_len}): {recent_attn:.3f}")
    print(f"  💡 Avg attention to middle ({seq_len//2-1}-{seq_len//2+1}): {mid_attn:.3f}")
    print(f"  💡 Avg attention to distant past (first 3): {distant_attn:.3f}")


def log_attention_heatmap(
    writer: Any,  # TensorBoard SummaryWriter
    attention_weights: torch.Tensor,
    epoch: int,
    max_samples: int = 4,
    max_timesteps: int = 64
) -> None:
    """
    Log attention heatmap to TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter
        attention_weights: [batch, num_heads, seq_len, seq_len]
        epoch: Current epoch number
        max_samples: Max number of samples to visualize
        max_timesteps: Max sequence length to show
    """
    if writer is None or attention_weights is None:
        return
    
    # Get first few samples, average across heads
    batch_size, num_heads, seq_len, _ = attention_weights.shape
    num_samples = min(batch_size, max_samples)
    seq_len = min(seq_len, max_timesteps)
    
    # Average across attention heads
    attn_avg = attention_weights[:num_samples, :, :seq_len, :seq_len].mean(dim=1)
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


def print_attention_statistics(
    model: torch.nn.Module,
    val_batch: tuple,
    device: torch.device,
    has_static_features: bool = False
) -> None:
    """
    Print attention layer statistics for analysis.
    
    Args:
        model: The model with attention layers
        val_batch: Validation batch (X, y, static)
        device: Device to run on
        has_static_features: Whether model uses static features
    """
    model.eval()
    with torch.no_grad():
        val_X, val_y, val_static = val_batch
        val_X = val_X.to(device)
        val_static = val_static.to(device) if has_static_features else None
        
        # Run forward pass
        _ = model(val_X, static_features=val_static)
        
        # Check if model has transformer encoder
        if hasattr(model, 'transformer_encoder'):
            print(f"\n  📊 Attention Layer Statistics:")
            
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


def analyze_temporal_focus(
    model: torch.nn.Module,
    val_X: torch.Tensor,
    device: torch.device
) -> None:
    """
    Analyze where the model's attention is focusing temporally.
    
    Args:
        model: Model with attention mechanism
        val_X: Validation input [batch, seq_len, features]
        device: Device to run on
    """
    model.eval()
    with torch.no_grad():
        # Get intermediate representation after transformer
        if hasattr(model, 'feature_projection'):
            features = model.feature_projection(val_X[:1])
        elif hasattr(model, 'variable_selection'):
            x_reshaped = val_X[:1].unsqueeze(-1)
            features, _ = model.variable_selection(x_reshaped)
        else:
            features = val_X[:1]
        
        # Add positional encoding if model has it
        if hasattr(model, 'pos_encoder'):
            features = model.pos_encoder(features)
            features = model.dropout_layer(features)
        
        # Pass through transformer if available
        if hasattr(model, 'transformer_encoder'):
            seq_len = features.size(1)
            if hasattr(model, 'causal_mask'):
                mask = model.causal_mask[:seq_len, :seq_len]
            else:
                mask = None
            encoded = model.transformer_encoder(features, mask=mask)
            
            # Statistics on encoded output
            enc_mean = encoded.mean().item()
            enc_std = encoded.std().item()
            enc_max = encoded.abs().max().item()
            
            print(f"\n     Encoded output: mean={enc_mean:.4f}, std={enc_std:.4f}, max_abs={enc_max:.4f}")
            
            # Check if output is collapsing
            if enc_std < 0.01:
                print(f"     ⚠️  WARNING: Low variance - possible attention collapse!")
            elif enc_std > 0.5:
                print(f"     ✅ Good variance - attention is active")
            
            # Timestep group analysis
            seq_len = encoded.size(1)
            
            # Split into groups
            begin_end = seq_len // 4
            mid_start = int(seq_len * 0.375)
            mid_end = int(seq_len * 0.625)
            end_start = int(seq_len * 0.75)
            
            # Compute statistics for each region
            begin_region = encoded[:, :begin_end, :]
            mid_region = encoded[:, mid_start:mid_end, :]
            end_region = encoded[:, end_start:, :]
            
            begin_mean_abs = begin_region.abs().mean().item()
            mid_mean_abs = mid_region.abs().mean().item()
            end_mean_abs = end_region.abs().mean().item()
            
            # Normalize to percentages
            total_activity = begin_mean_abs + mid_mean_abs + end_mean_abs
            if total_activity > 0:
                begin_pct = (begin_mean_abs / total_activity) * 100
                mid_pct = (mid_mean_abs / total_activity) * 100
                end_pct = (end_mean_abs / total_activity) * 100
                
                print(f"\n  📍 Timestep Attention Focus (Activity Distribution):")
                print(f"     Beginning [t=0-{begin_end-1}]:      {begin_pct:.1f}%")
                print(f"     Middle [t={mid_start}-{mid_end-1}]:     {mid_pct:.1f}%")
                print(f"     End [t={end_start}-{seq_len-1}]:        {end_pct:.1f}%")
                
                # Show bar chart
                max_pct = max(begin_pct, mid_pct, end_pct)
                if max_pct > 0:
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