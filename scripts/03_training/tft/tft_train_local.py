#!/usr/bin/env python3
"""
Local TFT Training Script

Train TFT model locally without GCS/Vertex AI dependencies.
Uses data from data/processed/ directory.

Usage:
    # Train with default config
    python scripts/03_training/tft/tft_train_local.py
    
    # Train with custom config
    python scripts/03_training/tft/tft_train_local.py --config configs/model_tft_config.yaml
    
    # Force reload data from BigQuery
    python scripts/03_training/tft/tft_train_local.py --reload
"""

import os
import sys
import argparse
import importlib.util
from pathlib import Path
import yaml
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
env_file = project_root / '.env'
if env_file.exists():
    load_dotenv(env_file)
    print(f"✅ Loaded environment from: {env_file}")
else:
    print(f"⚠️  No .env file found at: {env_file}")


def load_data(config_path: str, force_refresh: bool = False):
    """
    Load training data from local files or BigQuery.
    
    Args:
        config_path: Path to model config YAML
        force_refresh: If True, fetch from BigQuery instead of using cached data
    
    Returns:
        dataloaders: Dict of PyTorch DataLoaders
        scalers: Dict of fitted scalers
    """
    # Import data loader dynamically
    data_loader_path = project_root / 'scripts' / '02_features' / 'tft' / 'tft_data_loader.py'
    spec = importlib.util.spec_from_file_location('tft_data_loader', data_loader_path)
    data_loader_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data_loader_module)
    
    create_data_loaders = data_loader_module.create_data_loaders
    
    print("\n" + "="*80)
    print("   Loading Training Data")
    print("="*80)
    
    if force_refresh:
        print("\n🔄 Force refresh enabled - fetching from BigQuery...")
    else:
        print("\n📂 Checking for cached data in data/processed/...")
    
    dataloaders, scalers = create_data_loaders(
        config_path=config_path,
        export_temp=False,
        force_refresh=force_refresh
    )
    
    print("\n✅ Data loaded successfully!")
    return dataloaders, scalers


def train_model(config_path: str):
    """
    Train TFT model using core training logic.
    
    Args:
        config_path: Path to model config YAML
    """
    # Import training module dynamically
    train_module_path = project_root / 'scripts' / '03_training' / 'tft' / 'tft_train.py'
    
    if not train_module_path.exists():
        raise FileNotFoundError(
            f"Training module not found: {train_module_path}\n"
            "Please ensure tft_train.py exists in scripts/03_training/tft/"
        )
    
    spec = importlib.util.spec_from_file_location('tft_train', train_module_path)
    train_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_module)
    
    train_fn = train_module.train
    
    print("\n" + "="*80)
    print("   Starting TFT Training (Local)")
    print("="*80)
    print(f"\nConfig: {config_path}")
    print(f"Output: models/tft/tft_best.pt")
    print()
    
    # Run training
    train_fn(config_path)
    
    print("\n" + "="*80)
    print("   Training Complete!")
    print("="*80)
    print(f"\n✅ Model saved to: models/tft/tft_best.pt")
    print(f"📊 Logs saved to: logs/")


def main():
    """Main entry point for local training."""
    parser = argparse.ArgumentParser(
        description='Train TFT model locally (no GCS/Vertex AI)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default config
  python scripts/03_training/tft/tft_train_local.py
  
  # Train with custom config
  python scripts/03_training/tft/tft_train_local.py --config configs/my_config.yaml
  
  # Force reload data from BigQuery
  python scripts/03_training/tft/tft_train_local.py --reload
  
  # Combine options
  python scripts/03_training/tft/tft_train_local.py --config configs/my_config.yaml --reload
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/model_tft_config.yaml',
        help='Path to model config YAML (default: configs/model_tft_config.yaml)'
    )
    parser.add_argument(
        '--reload',
        action='store_true',
        help='Force reload data from BigQuery (ignore cached data/processed/)'
    )
    
    args = parser.parse_args()
    
    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Error: Config file not found: {config_path}")
        sys.exit(1)
    
    # Check environment variables
    if not os.getenv('GCP_PROJECT_ID'):
        print("⚠️  Warning: GCP_PROJECT_ID not set")
        print("   Set it with: export GCP_PROJECT_ID=your-project-id")
        print("   Or add to .env file in project root")
        sys.exit(1)
    
    try:
        # Load data
        dataloaders, scalers = load_data(args.config, force_refresh=args.reload)
        
        # Train model
        train_model(args.config)
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()