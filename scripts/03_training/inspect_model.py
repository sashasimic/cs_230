#!/usr/bin/env python3
"""
Generic Model Inspector

Inspects PyTorch model architecture and parameters.
Works with any model type (TFT, LSTM, Transformer, etc.)

Usage:
    # Inspect saved model checkpoint
    python scripts/03_training/inspect_model.py --checkpoint models/tft/tft_best.pt
    
    # Inspect model from config (no checkpoint)
    python scripts/03_training/inspect_model.py --model-type tft --config configs/model_tft_config.yaml
"""

import sys
import os
from pathlib import Path
import argparse
import torch
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def count_parameters(model):
    """Count total, trainable, and non-trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    return total, trainable, non_trainable


def inspect_checkpoint(checkpoint_path):
    """Inspect a saved model checkpoint."""
    print(f"\n{'='*80}")
    print(f"   Inspecting Checkpoint: {checkpoint_path}")
    print(f"{'='*80}\n")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Print checkpoint contents
    print("📦 Checkpoint Contents:")
    for key in checkpoint.keys():
        if key == 'model_state_dict':
            num_params = len(checkpoint[key])
            print(f"   - {key}: {num_params} parameter tensors")
        elif key == 'config':
            print(f"   - {key}: Model configuration")
        elif key == 'scalers':
            num_scalers = len(checkpoint[key]) if isinstance(checkpoint[key], dict) else 1
            print(f"   - {key}: {num_scalers} scalers")
        else:
            print(f"   - {key}: {checkpoint[key]}")
    
    print(f"\n📊 Training Metrics:")
    print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"   Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}" if 'val_loss' in checkpoint else "   Val Loss: N/A")
    print(f"   Val MAE: {checkpoint.get('val_mae', 'N/A'):.4f}" if 'val_mae' in checkpoint else "   Val MAE: N/A")
    print(f"   Dir Accuracy: {checkpoint.get('val_dir_acc', 'N/A'):.2f}%" if 'val_dir_acc' in checkpoint else "   Dir Accuracy: N/A")
    
    # Analyze model state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        
        print(f"\n🏗️  Model Architecture (from state dict):")
        print(f"   Total layers: {len(state_dict)}")
        
        # Count parameters by layer type
        layer_types = {}
        total_params = 0
        
        for name, tensor in state_dict.items():
            layer_type = name.split('.')[0] if '.' in name else name
            params = tensor.numel()
            total_params += params
            
            if layer_type not in layer_types:
                layer_types[layer_type] = {'count': 0, 'params': 0}
            layer_types[layer_type]['count'] += 1
            layer_types[layer_type]['params'] += params
        
        print(f"\n   Layer breakdown:")
        for layer_type, info in sorted(layer_types.items()):
            print(f"      {layer_type:20s}: {info['count']:3d} tensors, {info['params']:>12,} params")
        
        print(f"\n   💾 Total parameters: {total_params:,}")
        print(f"   💾 Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # Show config if available
    if 'config' in checkpoint:
        config = checkpoint['config']
        print(f"\n⚙️  Model Configuration:")
        if 'model' in config:
            for key, value in config['model'].items():
                print(f"      {key:20s}: {value}")
    
    print(f"\n{'='*80}")


def inspect_from_config(model_type, config_path):
    """Inspect model architecture from config (without loading checkpoint)."""
    print(f"\n{'='*80}")
    print(f"   Model Type: {model_type.upper()}")
    print(f"   Config: {config_path}")
    print(f"{'='*80}\n")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("⚠️  Note: This is a placeholder. To fully inspect the model:")
    print("   1. Implement your model class")
    print("   2. Instantiate it with the config")
    print("   3. Use torchinfo.summary() or similar\n")
    
    print("📋 Config Summary:")
    if 'model' in config:
        for key, value in config['model'].items():
            print(f"   {key:20s}: {value}")
    
    print(f"\n{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description='Inspect PyTorch model architecture and parameters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inspect saved checkpoint
  python scripts/03_training/inspect_model.py --checkpoint models/tft/tft_best.pt
  
  # Inspect from config (no checkpoint needed)
  python scripts/03_training/inspect_model.py --model-type tft --config configs/model_tft_config.yaml
        """
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to model checkpoint (.pt file)'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        default='tft',
        help='Model type (tft, lstm, etc.) - used if no checkpoint provided'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/model_tft_config.yaml',
        help='Path to model config YAML'
    )
    
    args = parser.parse_args()
    
    if args.checkpoint:
        # Inspect saved checkpoint
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"❌ Error: Checkpoint not found: {checkpoint_path}")
            sys.exit(1)
        
        inspect_checkpoint(checkpoint_path)
    else:
        # Inspect from config
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"❌ Error: Config file not found: {config_path}")
            sys.exit(1)
        
        inspect_from_config(args.model_type, config_path)


if __name__ == '__main__':
    main()
