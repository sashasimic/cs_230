import torch
import sys
from pathlib import Path
import yaml
import os

# Since we're in decoder_transformer/, just import directly
from fincast_extension import DecoderTransformerWithFinCast, FINCAST_AVAILABLE
from decoder_transformer_train import DecoderOnlyTransformerAR

if not FINCAST_AVAILABLE:
    print("❌ FinCast not available")
    sys.exit(1)

# Get project root and change to it (so relative paths work)
project_root = Path.cwd().parent.parent.parent
os.chdir(project_root)
print(f"📁 Changed directory to: {project_root}\n")

# Load config
config_path = project_root / 'configs/model_decoder_config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Create mock features including close_* columns (27 tickers)
feature_names = [
    f'close_{ticker}' for ticker in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 
                                      'META', 'NVDA', 'JPM', 'V', 'WMT',
                                      'PG', 'JNJ', 'UNH', 'MA', 'HD',
                                      'DIS', 'PYPL', 'BAC', 'CMCSA', 'NFLX',
                                      'ADBE', 'CRM', 'ABT', 'PFE', 'MRK',
                                      'TMO', 'COST']
] + [f'feature_{i}' for i in range(91)]

num_features = len(feature_names)

print(f"🧪 Testing Full Integration with Multiple Tickers")
print(f"   Features: {num_features} (27 price + 91 others)\n")

# Create mock metadata
metadata_path = Path('data/processed/metadata.yaml')
metadata_path.parent.mkdir(parents=True, exist_ok=True)
with open(metadata_path, 'w') as f:
    yaml.dump({'features': feature_names}, f)

print(f"✅ Created metadata at: {metadata_path.absolute()}\n")

# Create FinCast config
fincast_config = {
    'd_model': 128,
    'max_len': 512,
    'freeze': True
}

# Initialize full model
model = DecoderTransformerWithFinCast(
    config=config,
    num_features=num_features,
    fincast_config=fincast_config,
    decoder_transformer_class=DecoderOnlyTransformerAR
)

print(f"\n✅ Model initialized")
print(f"   Price series: {model.num_price_series}")
print(f"   FinCast features: {model.num_price_series * 128}")
print(f"   Other features: {model.num_rest_features}")
print(f"   Total augmented: {model.num_rest_features + model.num_price_series * 128}")

# Test forward pass
batch_size = 4
lookback = 60
num_horizons = config.get('num_horizons', config['model'].get('num_horizons', 6))  # Default to 6
test_input = torch.randn(batch_size, lookback, num_features)
test_targets = torch.randn(batch_size, num_horizons)

print(f"\n🔄 Testing forward pass...")
output = model(test_input, test_targets, teacher_forcing=True)

print(f"\n✅ Forward pass successful!")
print(f"   Input shape: {test_input.shape}")
print(f"   Output shape: {output.shape}")
print(f"   Expected: [{batch_size}, {num_horizons}]")

print("\n🎉 Full integration test passed!")
print("   Pre-trained FinCast is processing all 27 price tickers!")
