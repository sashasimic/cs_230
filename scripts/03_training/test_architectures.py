"""Test model architecture with dummy data forward pass."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse
import numpy as np
from utils.config_loader import load_config
from models.model_factory import ModelFactory

def main():
    parser = argparse.ArgumentParser(description='Test model with forward pass')
    parser.add_argument('--model-type', default='mlp', help='Model type')
    parser.add_argument('--config', default='configs/mlp_multi_horizon.yaml', help='Config file')
    parser.add_argument('--seq-len', type=int, default=30, help='Sequence length')
    parser.add_argument('--n-features', type=int, default=10, help='Number of features')
    parser.add_argument('--batch-size', type=int, default=5, help='Test batch size')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    config['model']['type'] = args.model_type

    # Create model
    input_shape = (args.seq_len, args.n_features)
    model = ModelFactory.create_model(args.model_type, input_shape, config)

    print("\n" + "="*60)
    print(f"MODEL ARCHITECTURE: {args.model_type.upper()}")
    print("="*60)
    model.summary()

    print("\n" + "="*60)
    print("FORWARD PASS TEST")
    print("="*60)

    # Test with dummy input
    dummy_input = np.random.randn(args.batch_size, args.seq_len, args.n_features)
    output = model.predict(dummy_input, verbose=0)

    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Check multi-horizon
    n_horizons = config['model'].get('mlp', {}).get('output_horizons', 1)
    print(f"\nExpected: ({args.batch_size}, {n_horizons})")
    print(f"Got:      {output.shape}")

    if n_horizons > 1 and output.shape[1] >= n_horizons:
        print(f"\nFirst sample outputs:")
        horizons = ['1M', '3M', '6M'][:n_horizons]
        for i, horizon in enumerate(horizons):
            print(f"  {horizon}: {output[0, i]:.4f}")
    elif n_horizons == 1:
        print(f"\nFirst sample output: {output[0, 0]:.4f}")

    print("\n" + "="*60)
    success = output.shape[1] == n_horizons
    print("✓ Correct output shape!" if success else "❌ Wrong output shape!")
    print("="*60)

if __name__ == '__main__':
    main()
