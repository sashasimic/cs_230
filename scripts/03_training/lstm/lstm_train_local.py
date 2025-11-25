#!/usr/bin/env python3
"""
Local Training Script for LSTM Multi-Horizon Forecasting

Runs LSTM training locally (not on Vertex AI).
Uses same dataset format as decoder transformer.

Usage:
    python lstm_train_local.py --config configs/model_lstm_config.yaml
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

# Import LSTM training module
import importlib.util
spec = importlib.util.spec_from_file_location(
    "lstm_train",
    project_root / "scripts/03_training/lstm/lstm_train.py"
)
train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_module)
train = train_module.train


def main():
    parser = argparse.ArgumentParser(
        description='Train LSTM multi-horizon model locally',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/model_lstm_config.yaml',
        help='Path to model config YAML'
    )
    parser.add_argument(
        '--dataset-version',
        type=str,
        default='v3',
        help='Dataset version to use'
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
    
    # Check environment variables (optional for local training with cached data)
    if not os.getenv('GCP_PROJECT_ID'):
        print("⚠️  Warning: GCP_PROJECT_ID not set")
        print("   This is OK if using cached data from data/processed/")
        print("   Will fail if --reload is used (requires BigQuery access)")
    
    print("="*80)
    print("🚀 LSTM MULTI-HORIZON - LOCAL TRAINING")
    print("="*80)
    print(f"\n📋 Configuration:")
    print(f"   Config: {args.config}")
    print("="*80)
    
    # Train
    try:
        train(
            config_path=args.config
        )
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()