#!/usr/bin/env python3
"""
Local Training Script for Decoder-Only Transformer (with optional FinCast)

Runs decoder-only AR transformer training locally (not on Vertex AI).
Supports optional FinCast backbone for enhanced price series processing.

Useful for:
- Quick experimentation
- Debugging  
- Small datasets
- Testing FinCast integration

Usage:
    # Standard transformer
    python decoder_transformer_train_local.py --config configs/model_decoder_config.yaml
    
    # With FinCast backbone
    python decoder_transformer_train_local.py --config configs/model_decoder_config.yaml --use-fincast
    
    # With FinCast and custom settings
    python decoder_transformer_train_local.py --config configs/model_decoder_config.yaml \
        --use-fincast --fincast-d-model 64 --fincast-layers 3
"""

import argparse
import os
import sys
import yaml
import torch
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import decoder_transformer training module
import importlib.util
spec = importlib.util.spec_from_file_location(
    "decoder_transformer_train",
    project_root / "scripts/03_training/decoder_transformer/decoder_transformer_train.py"
)
train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_module)
train = train_module.train


def copy_dataset_to_processed(dataset_version: str, config_path: str):
    """
    Copy versioned dataset to data/processed/ directory.
    Mimics Vertex AI's behavior of downloading to data/processed/.
    
    Args:
        dataset_version: Dataset version (e.g., 'v3')
        config_path: Path to config file (to get model type)
    """
    import shutil
    
    # Get model type from config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    model_type = config.get('model', {}).get('type', 'decoder_transformer')
    
    # Source: versioned dataset directory
    source_dir = Path(f'data/datasets/{model_type}/{dataset_version}/processed')
    
    # Destination: data/processed/
    dest_dir = Path('data/processed')
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    if not source_dir.exists():
        raise FileNotFoundError(
            f"Versioned dataset not found: {source_dir}\n"
            f"Please generate it first using:\n"
            f"  python scripts/05_deployment/generate_dataset.py \\\n"
            f"    --model-type {model_type} \\\n"
            f"    --version {dataset_version} \\\n"
            f"    --config {config_path}"
        )
    
    print(f"\n📂 Copying dataset to training location...")
    print(f"   Source: {source_dir}")
    print(f"   Destination: {dest_dir}")
    
    # Copy all files from source to destination
    copied_files = []
    for file in source_dir.glob('*'):
        if file.is_file():
            shutil.copy2(file, dest_dir / file.name)
            copied_files.append(file.name)
    
    print(f"   ✅ Copied {len(copied_files)} files:")
    for fname in sorted(copied_files):
        print(f"      - {fname}")
    print(f"\n✅ Dataset ready at: {dest_dir}/")
    print(f"   (Same as Vertex AI behavior)\n")


def main():
    parser = argparse.ArgumentParser(
        description='Train decoder-only transformer locally',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/model_decoder_config.yaml',
        help='Path to model config YAML'
    )
    parser.add_argument(
        '--dataset-version',
        type=str,
        default='v9',
        help='Dataset version to use'
    )
    parser.add_argument(
        '--reload',
        action='store_true',
        help='Force reload data from BigQuery (ignore cached data/processed/)'
    )
    
    # FinCast arguments (optional overrides - config file is primary source)
    parser.add_argument(
        '--fincast-enabled',
        type=lambda x: x.lower() in ['true', '1', 'yes'],
        default=None,
        help='Override FinCast enabled setting from config (true/false)'
    )
    parser.add_argument(
        '--fincast-checkpoint',
        type=str,
        default=None,
        help='Override FinCast checkpoint path from config'
    )
    
    args = parser.parse_args()
    
    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Error: Config file not found: {config_path}")
        sys.exit(1)
    
    # Check environment variables (optional for local training with cached data)
    if not os.getenv('GCP_PROJECT_ID'):
        print("⚠️  Warning: GCP_PROJECT_ID not set")
        print("   This is OK if using cached data from data/processed/")
        print("   Will fail if --reload is used (requires BigQuery access)")
    
    print("="*80)
    print("🚀 DECODER TRANSFORMER - LOCAL TRAINING")
    print("="*80)
    print(f"\n📋 Configuration:")
    print(f"   Config: {args.config}")
    print(f"   Dataset version: {args.dataset_version}")
    print("="*80)
    
    # Load config to check FinCast settings
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply CLI overrides to FinCast config if provided
    fincast_config = config.get('fincast', {})
    if args.fincast_enabled is not None:
        fincast_config['enabled'] = args.fincast_enabled
    if args.fincast_checkpoint is not None:
        fincast_config['checkpoint_path'] = args.fincast_checkpoint
    
    # Copy versioned dataset to data/processed/ (mimics Vertex AI behavior)
    if args.dataset_version:
        copy_dataset_to_processed(args.dataset_version, args.config)
    
    # Train
    try:
        train(
            config_path=args.config,
            dataset_version=args.dataset_version
        )
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()