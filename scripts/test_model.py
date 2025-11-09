"""Test multi-horizon model architecture."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from utils.config_loader import load_config
from models.model_factory import ModelFactory

# Load config
config = load_config('configs/mlp_multi_horizon.yaml')

# Create model
input_shape = (30, 10)  # 30 timesteps, 10 features
model = ModelFactory.create_model('mlp', input_shape, config)

print("\n" + "="*60)
print("MODEL ARCHITECTURE")
print("="*60)
model.summary()

print("\n" + "="*60)
print("OUTPUT SHAPE TEST")
print("="*60)

# Test with dummy input
dummy_input = np.random.randn(5, 30, 10)  # 5 samples
output = model.predict(dummy_input, verbose=0)

print(f"Input shape:  {dummy_input.shape}")
print(f"Output shape: {output.shape}")
print(f"\nExpected: (5, 3) - 5 samples, 3 horizons")
print(f"Got:      {output.shape}")

print(f"\nFirst sample output (3 single numbers):")
print(f"  1M inflation: {output[0, 0]:.4f}")
print(f"  3M inflation: {output[0, 1]:.4f}")  
print(f"  6M inflation: {output[0, 2]:.4f}")

print("\n" + "="*60)
print("✓ Model outputs 3 single scalars per sample!" if output.shape[1] == 3 else "❌ Wrong output shape!")
print("="*60)
