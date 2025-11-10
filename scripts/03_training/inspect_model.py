#!/usr/bin/env python
"""Quick model architecture inspection (no forward pass)."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse
from models.model_factory import ModelFactory
from utils.config_loader import load_config

def main():
    parser = argparse.ArgumentParser(description='Inspect model architecture')
    parser.add_argument('--model-type', choices=['lstm', 'gru', 'mlp', 'transformer'], 
                       default='lstm', help='Model type')
    parser.add_argument('--config', default='configs/config.yaml', help='Config file')
    parser.add_argument('--seq-len', type=int, default=30, help='Sequence length')
    parser.add_argument('--n-features', type=int, default=163, help='Number of features')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    config['model']['type'] = args.model_type
    
    # Create model
    input_shape = (args.seq_len, args.n_features)
    model = ModelFactory.create_model(args.model_type, input_shape, config)
    
    print("\n" + "="*70)
    print(f"Model: {args.model_type.upper()}")
    print(f"Config: {args.config}")
    print(f"Input shape: {input_shape}")
    print("="*70)
    model.summary()
    print("="*70)
    print(f"Total parameters: {model.count_params():,}")
    
    # Calculate trainable vs non-trainable
    trainable = sum([w.shape.num_elements() for w in model.trainable_weights])
    non_trainable = model.count_params() - trainable
    print(f"Trainable:        {trainable:,}")
    print(f"Non-trainable:    {non_trainable:,}")
    print("="*70)

if __name__ == '__main__':
    main()
