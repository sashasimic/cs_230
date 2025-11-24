import torch
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path.cwd()))

# Import using importlib to avoid path issues
import importlib.util
spec = importlib.util.spec_from_file_location(
    "fincast_extension",
    "fincast_extension.py"
)
fincast_ext = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fincast_ext)

FinCastBackbone = fincast_ext.FinCastBackbone
FINCAST_AVAILABLE = fincast_ext.FINCAST_AVAILABLE

print('FinCast available:', FINCAST_AVAILABLE)

if FINCAST_AVAILABLE:
    # Create test data
    test_data = torch.randn(2, 60)  # 2 samples, 60 timesteps
    
    print('\nInitializing FinCast (loading 3.97GB checkpoint)...')
    fincast = FinCastBackbone(d_model=128, freeze=True)
    
    print('\nExtracting embeddings...')
    embeddings = fincast(test_data)
    
    print(f'\n✅ Results:')
    print(f'  Input shape: {test_data.shape}')
    print(f'  Output shape: {embeddings.shape}')
    print(f'  Embedding mean: {embeddings.mean().item():.4f}')
    print(f'  Embedding std: {embeddings.std().item():.4f}')
    print(f'  Embedding min: {embeddings.min().item():.4f}')
    print(f'  Embedding max: {embeddings.max().item():.4f}')
    print('\n✅ Extraction successful! Using actual pre-trained FFM hidden states.')
else:
    print('❌ FinCast not available')
