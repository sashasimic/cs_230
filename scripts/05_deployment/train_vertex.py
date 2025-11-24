#!/usr/bin/env python3
"""
Vertex AI Training Wrapper

Handles:
- Hyperparameter tuning from Vertex AI
- Model checkpointing to GCS
- Metric reporting for Vertex AI
- Parallel training orchestration

Works with any model type (TFT, LSTM, etc.)
"""

import os
import sys
import argparse
import importlib.util
from pathlib import Path
import torch
import yaml
from google.cloud import storage

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def parse_args():
    """Parse command line arguments from Vertex AI."""
    parser = argparse.ArgumentParser(description='Vertex AI Model Training')
    
    # Model hyperparameters (tunable) - None means use config file value
    parser.add_argument('--hidden_size', type=int, default=None)
    parser.add_argument('--lstm_layers', type=int, default=None)
    parser.add_argument('--attention_heads', type=int, default=None)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    
    # Training settings
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--early_stopping_patience', type=int, default=None)
    parser.add_argument('--lookback_window', type=int, default=None)
    
    # Data settings
    parser.add_argument('--start_date', type=str, default='2023-01-01')
    parser.add_argument('--end_date', type=str, default='2024-12-31')
    
    # Dataset version (NEW)
    parser.add_argument('--dataset_version', type=str, default=None,
                       help='Dataset version to use (e.g., v1, v2). If provided, skip BigQuery and load from GCS.')
    
    # GCS paths
    parser.add_argument('--gcs_bucket', type=str, required=True)
    parser.add_argument('--gcs_model_path', type=str, default='models')
    parser.add_argument('--job_name', type=str, default='model-training')
    
    # Config file and model type
    parser.add_argument('--config', type=str, default='configs/model_tft_config.yaml')
    parser.add_argument('--model_type', type=str, default='tft', help='Model type (tft, lstm, etc.)')
    
    return parser.parse_args()


def update_config_with_hyperparameters(config_path: str, args) -> str:
    """Update config with hyperparameters from Vertex AI."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update model architecture (only if explicitly provided and key exists)
    if args.hidden_size is not None and 'hidden_size' in config.get('model', {}):
        config['model']['hidden_size'] = args.hidden_size
    if args.lstm_layers is not None and 'lstm_layers' in config.get('model', {}):
        config['model']['lstm_layers'] = args.lstm_layers
    if args.attention_heads is not None and 'attention_heads' in config.get('model', {}):
        config['model']['attention_heads'] = args.attention_heads
    if args.dropout is not None and 'dropout' in config.get('model', {}):
        config['model']['dropout'] = args.dropout
    
    # Update training settings (only if explicitly provided)
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        config['training']['learning_rate'] = args.learning_rate
    if args.early_stopping_patience is not None and 'early_stopping' in config.get('training', {}):
        config['training']['early_stopping']['patience'] = args.early_stopping_patience
    
    # Update data settings ONLY if not using pre-generated dataset
    # Pre-generated datasets have their own date ranges and lookback windows
    if not args.dataset_version:
        config['data']['start_date'] = args.start_date
        config['data']['end_date'] = args.end_date
        config['data']['lookback_window'] = args.lookback_window
    
    # Save updated config
    temp_config_path = '/tmp/vertex_config.yaml'
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)
    
    print(f"\n" + "="*80)
    print("   Hyperparameters")
    print("="*80)
    # Print model-specific params
    model_config = config.get('model', {})
    if 'hidden_size' in model_config:
        print(f"Hidden size: {model_config['hidden_size']}")
    if 'lstm_layers' in model_config:
        print(f"LSTM layers: {model_config['lstm_layers']}")
    if 'd_model' in model_config:
        print(f"d_model: {model_config['d_model']}")
    if 'n_layers' in model_config:
        print(f"n_layers: {model_config['n_layers']}")
    if 'attention_heads' in model_config:
        print(f"Attention heads: {model_config['attention_heads']}")
    if 'n_heads' in model_config:
        print(f"n_heads: {model_config['n_heads']}")
    if 'dropout' in model_config:
        print(f"Dropout: {model_config['dropout']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Batch size: {config['training']['batch_size']}")
    if 'lookback_window' in config.get('data', {}):
        print(f"Lookback: {config['data']['lookback_window']}")
    print("="*80 + "\n")
    
    return temp_config_path


def upload_data_dir_to_gcs(local_dir: str, gcs_bucket: str, job_name: str, dir_type: str) -> list:
    """Upload data directory (raw or processed) to GCS.
    
    Organizes files differently for single jobs vs HP tuning:
    - Single job:  data/{job_name}/raw/file.parquet
    - HP tuning:   data/{job_name}/trial_{trial_id}/raw/file.parquet
    
    Args:
        local_dir: Local directory path (e.g., 'data/raw')
        gcs_bucket: GCS bucket name
        job_name: Job name for organizing files
        dir_type: 'raw' or 'processed'
    
    Returns:
        List of uploaded GCS paths
    """
    from google.cloud import storage
    from pathlib import Path
    
    local_path = Path(local_dir)
    if not local_path.exists():
        print(f"   ⚠️  {local_dir} not found, skipping...")
        return []
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(gcs_bucket)
    
    # Detect if running in HP tuning (has trial ID)
    trial_id = os.getenv('CLOUD_ML_TRIAL_ID')
    if trial_id:
        # HP Tuning: data/{job_name}/trial_{id}/raw/
        base_path = f"data/{job_name}/trial_{trial_id}"
        job_type = f"HP Tuning Trial {trial_id}"
    else:
        # Single Job: data/{job_name}/raw/
        base_path = f"data/{job_name}"
        job_type = "Single Job"
    
    uploaded = []
    print(f"\n📤 Uploading {dir_type} data to GCS ({job_type})...")
    
    for file_path in local_path.glob('*'):
        if file_path.is_file():
            blob_path = f"{base_path}/{dir_type}/{file_path.name}"
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(str(file_path))
            gcs_uri = f"gs://{gcs_bucket}/{blob_path}"
            uploaded.append(gcs_uri)
            print(f"   ✅ {file_path.name} → {gcs_uri}")
    
    return uploaded


def upload_model_to_gcs(local_path: str, gcs_bucket: str, gcs_model_path: str, job_name: str) -> str:
    """Upload trained model to GCS.
    
    Organizes models differently for single jobs vs HP tuning:
    - Single job:  models/{job_name}/model_best.pt
    - HP tuning:   models/{job_name}/trial_{trial_id}/model_best.pt
    """
    from google.cloud import storage
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(gcs_bucket)
    
    # Detect if running in HP tuning (has trial ID)
    trial_id = os.getenv('CLOUD_ML_TRIAL_ID')
    model_filename = Path(local_path).name  # Use actual filename (e.g., tft_best.pt, lstm_best.pt)
    if trial_id:
        # HP Tuning: models/{job_name}/trial_{id}/model_best.pt
        blob_path = f"{gcs_model_path}/{job_name}/trial_{trial_id}/{model_filename}"
        job_type = f"HP Tuning Trial {trial_id}"
    else:
        # Single Job: models/{job_name}/model_best.pt
        blob_path = f"{gcs_model_path}/{job_name}/{model_filename}"
        job_type = "Single Job"
    
    blob = bucket.blob(blob_path)
    
    print(f"\n📤 Uploading model to GCS ({job_type})...")
    print(f"   Local: {local_path}")
    print(f"   GCS: gs://{gcs_bucket}/{blob_path}")
    
    blob.upload_from_filename(local_path)
    print(f"   ✅ Upload complete!")
    
    return f"gs://{gcs_bucket}/{blob_path}"


def report_metrics_to_vertex(val_loss: float, val_mae: float, dir_acc: float):
    """Report metrics to Vertex AI hyperparameter tuning."""
    try:
        import hypertune
        hpt = hypertune.HyperTune()
        
        # Primary metric for optimization
        hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag='val_loss',
            metric_value=val_loss,
            global_step=1
        )
        
        # Additional metrics
        hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag='val_mae',
            metric_value=val_mae,
            global_step=1
        )
        
        hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag='directional_accuracy',
            metric_value=dir_acc,
            global_step=1
        )
        
        print(f"\n📊 Reported metrics to Vertex AI:")
        print(f"   Val Loss: {val_loss:.4f}")
        print(f"   Val MAE: {val_mae:.4f}")
        print(f"   Dir Acc: {dir_acc:.2f}%")
    except ImportError:
        print("\n⚠️  Hypertune not available, skipping metric reporting")


def download_fincast_checkpoint(gcs_bucket: str, checkpoint_gcs_path: str = 'models/fincast/v1.pth'):
    """
    Download FinCast pre-trained checkpoint from GCS if needed.
    
    Args:
        gcs_bucket: GCS bucket name
        checkpoint_gcs_path: Path to checkpoint in GCS
    
    Returns:
        Local path to checkpoint
    """
    local_path = Path('external/fincast/checkpoints/v1.pth')
    
    # Check if checkpoint already exists
    if local_path.exists():
        size_mb = local_path.stat().st_size / (1024 * 1024)
        print(f"✅ FinCast checkpoint already exists ({size_mb:.0f} MB)")
        return str(local_path)
    
    print(f"\n📥 Downloading FinCast checkpoint from GCS...")
    print(f"   gs://{gcs_bucket}/{checkpoint_gcs_path}")
    print(f"   → {local_path}")
    print(f"   Size: ~3.97 GB (may take 2-5 minutes)\n")
    
    # Create directory
    local_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Download
    storage_client = storage.Client()
    bucket = storage_client.bucket(gcs_bucket)
    blob = bucket.blob(checkpoint_gcs_path)
    
    if not blob.exists():
        raise FileNotFoundError(
            f"\n❌ FinCast checkpoint not found in GCS!\n"
            f"   gs://{gcs_bucket}/{checkpoint_gcs_path}\n\n"
            f"📤 To upload the checkpoint to GCS, run:\n"
            f"   gsutil cp external/fincast/checkpoints/v1.pth "
            f"gs://{gcs_bucket}/{checkpoint_gcs_path}\n"
        )
    
    blob.download_to_filename(str(local_path))
    
    size_mb = local_path.stat().st_size / (1024 * 1024)
    print(f"✅ FinCast checkpoint downloaded ({size_mb:.0f} MB)\n")
    
    return str(local_path)


def load_from_dataset_version(version, gcs_bucket=None):
    """
    Load a pre-generated dataset version from GCS to local data/ directory.
    This allows training jobs to use versioned datasets instead of re-querying BigQuery.
    
    Args:
        version: Dataset version (e.g., 'v1', 'v2')
        gcs_bucket: GCS bucket name
    """
    from google.cloud import storage
    
    # Use environment variable if bucket not provided
    if gcs_bucket is None:
        project_id = os.getenv('GCP_PROJECT_ID', 'your-project')
        gcs_bucket = f"{project_id}-models"
    
    print(f"\n" + "="*80)
    print(f"   📦 Loading Dataset Version: {version}")
    print("="*80)
    
    # Special case: 'local' means use mounted local data (for Docker testing)
    if version.lower() == 'local':
        print(f"\n💻 Using local data (skipping GCS download)")
        print(f"   Verifying local data exists...")
        
        if not Path('data/processed').exists() or not any(Path('data/processed').iterdir()):
            raise FileNotFoundError(
                "Local data not found in data/processed/\n"
                "Please ensure data is mounted or run data generation first."
            )
        
        print(f"   ✅ Local data found in data/processed/")
        print(f"\n✅ Dataset 'local' ready!")
        print("="*80)
        return
    
    print(f"   GCS Bucket: {gcs_bucket}")
    
    # Initialize GCS client
    storage_client = storage.Client()
    bucket = storage_client.bucket(gcs_bucket)
    
    # Create local directories
    Path('data/raw').mkdir(parents=True, exist_ok=True)
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    
    def download_directory(gcs_prefix, local_dir):
        """Download all blobs with a given prefix to local directory."""
        blobs = list(bucket.list_blobs(prefix=gcs_prefix))
        if not blobs:
            return False
        
        for blob in blobs:
            if blob.name.endswith('/'):
                continue  # Skip directory markers
            
            # Get relative path within the prefix
            relative_path = blob.name[len(gcs_prefix):]
            local_path = Path(local_dir) / relative_path
            
            # Create parent directories
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download file
            blob.download_to_filename(str(local_path))
        
        return True
    
    # Download processed data (needed for training)
    print(f"\n📥 Downloading processed data...")
    gcs_processed_prefix = f"datasets/{version}/processed/"
    if download_directory(gcs_processed_prefix, 'data/processed'):
        print(f"   ✅ Processed data loaded to: data/processed/")
    else:
        raise Exception(f"Failed to download processed data from gs://{gcs_bucket}/{gcs_processed_prefix}")
    
    # Download raw data (for reference/debugging)
    print(f"\n📥 Downloading raw data...")
    gcs_raw_prefix = f"datasets/{version}/raw/"
    if download_directory(gcs_raw_prefix, 'data/raw'):
        print(f"   ✅ Raw data loaded to: data/raw/")
    else:
        print(f"   ⚠️  Raw data download failed (non-critical)")
    
    # Download manifest
    print(f"\n📥 Downloading manifest...")
    try:
        manifest_blob = bucket.blob(f"datasets/{version}/manifest.yaml")
        manifest_blob.download_to_filename('data/manifest.yaml')
        print(f"   ✅ Manifest loaded")
    except Exception as e:
        print(f"   ⚠️  Manifest download failed (non-critical): {e}")
    
    print(f"\n✅ Dataset version '{version}' loaded successfully!")
    print("="*80 + "\n")


def main():
    """Main training loop for Vertex AI."""
    args = parse_args()
    
    # Auto-select config based on model_type if using default TFT config
    if args.config == 'configs/model_tft_config.yaml' and args.model_type != 'tft':
        config_map = {
            'decoder_transformer': 'configs/model_decoder_config.yaml',
            'lstm': 'configs/model_lstm_config.yaml',
            'transformer': 'configs/model_transformer_config.yaml',
        }
        if args.model_type in config_map:
            args.config = config_map[args.model_type]
            print(f"📋 Auto-selected config: {args.config}")
    
    # Set environment variables (needed for BigQuery access and TensorBoard logging)
    # Extract project ID from GCS bucket name (format: {project_id}-*-models)
    # Handle various bucket naming patterns
    if '-models' in args.gcs_bucket:
        project_id = args.gcs_bucket.split('-models')[0].rsplit('-', 1)[0]
    else:
        # Fallback: use environment variable or bucket name as-is
        project_id = os.getenv('GCP_PROJECT_ID', args.gcs_bucket.split('/')[0])
    os.environ['GCP_PROJECT_ID'] = project_id
    os.environ['GCP_REGION'] = os.getenv('GCP_REGION', 'us-central1')
    os.environ['GCS_BUCKET'] = args.gcs_bucket  # For TensorBoard GCS logging
    os.environ['JOB_NAME'] = args.job_name       # For TensorBoard log organization
    
    print("\n" + "="*80)
    print(f"   Vertex AI Training ({args.model_type.upper()})")
    print("="*80)
    print(f"Job: {args.job_name}")
    print(f"GCS Bucket: {args.gcs_bucket}")
    print(f"Project ID: {project_id}")
    if args.dataset_version:
        print(f"Dataset Version: {args.dataset_version} (pre-generated)")
    else:
        print(f"Dataset: Will generate from BigQuery")
    print("="*80 + "\n")
    
    # Load data from dataset version OR generate from BigQuery
    if args.dataset_version:
        # Load from pre-generated dataset version
        load_from_dataset_version(args.dataset_version, args.gcs_bucket)
        # Data is now in data/processed/, skip data generation in training
    else:
        # Will generate data from BigQuery during training (current behavior)
        print("\n⚠️  No dataset version provided, will generate from BigQuery...")
    
    # Download FinCast checkpoint if needed (check config first)
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if config.get('fincast', {}).get('enabled', False):
        print("\n🔧 FinCast enabled in config, downloading checkpoint...")
        download_fincast_checkpoint(
            gcs_bucket=args.gcs_bucket,
            checkpoint_gcs_path='models/fincast/v1.pth'
        )
    
    # Update config with hyperparameters
    temp_config_path = update_config_with_hyperparameters(args.config, args)
    
    # Train model
    print("\n🚀 Starting training...\n")
    
    try:
        # Import training module dynamically (can't use standard import with numeric prefix)
        # Look for model-specific training script (e.g., tft/tft_train.py, lstm/lstm_train.py)
        train_module_path = Path(__file__).parent.parent / '03_training' / args.model_type / f'{args.model_type}_train.py'
        if not train_module_path.exists():
            # Fallback to generic location
            train_module_path = Path(__file__).parent.parent / '03_training' / f'{args.model_type}_train.py'
        
        if not train_module_path.exists():
            raise FileNotFoundError(f"Training module not found: {train_module_path}")
        
        spec = importlib.util.spec_from_file_location(f'{args.model_type}_train', train_module_path)
        train_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(train_module)
        train = train_module.train
        
        # Run training
        train(temp_config_path)
        
        # Load best checkpoint to get metrics
        # Look for model in model_type subdirectory
        # For decoder_transformer, check for _ar or _tf variants first
        local_model_path = None
        if args.model_type == 'decoder_transformer':
            # Try autoregressive variant first, then teacher forcing
            for suffix in ['_ar', '_tf']:
                candidate = f'models/{args.model_type}/{args.model_type}_best{suffix}.pt'
                if os.path.exists(candidate):
                    local_model_path = candidate
                    break
        
        # Fallback to standard naming for other models or if variants not found
        if local_model_path is None:
            local_model_path = f'models/{args.model_type}/{args.model_type}_best.pt'
        
        if os.path.exists(local_model_path):
            # Note: weights_only=False is safe here since we're loading our own checkpoint
            # PyTorch 2.6+ requires this for checkpoints containing numpy objects
            checkpoint = torch.load(local_model_path, map_location='cpu', weights_only=False)
            val_loss = checkpoint.get('val_loss', 0.0)
            val_mae = checkpoint.get('val_mae', 0.0)
            dir_acc = checkpoint.get('val_dir_acc', 0.0)  # Match key from tft_train.py
            
            print(f"\n✅ Training complete!")
            print(f"   Best Val Loss: {val_loss:.4f}")
            print(f"   Best Val MAE: {val_mae:.4f}")
            print(f"   Dir Accuracy: {dir_acc:.2f}%")
            
            # Upload data files to GCS only if NOT using pre-generated dataset
            if not args.dataset_version:
                print(f"\n📤 Uploading data to GCS (freshly generated from BigQuery)...")
                raw_files = upload_data_dir_to_gcs('data/raw', args.gcs_bucket, args.job_name, 'raw')
                processed_files = upload_data_dir_to_gcs('data/processed', args.gcs_bucket, args.job_name, 'processed')
            else:
                print(f"\n⏭️  Skipping data upload (using pre-generated dataset: {args.dataset_version})")
            
            # Upload model to GCS
            gcs_model_uri = upload_model_to_gcs(
                local_model_path,
                args.gcs_bucket,
                args.gcs_model_path,
                args.job_name
            )
            
            # TensorBoard logs are automatically synced by Vertex AI when using AIP_TENSORBOARD_LOG_DIR
            # No manual upload needed!
            
            # Report metrics to Vertex AI (for hyperparameter tuning)
            report_metrics_to_vertex(val_loss, val_mae, dir_acc)
        else:
            print(f"\n⚠️  Model file not found: {local_model_path}")
            sys.exit(1)
        
        print("\n" + "="*80)
        print("   Training Complete!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()