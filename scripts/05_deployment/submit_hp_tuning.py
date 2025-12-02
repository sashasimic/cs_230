#!/usr/bin/env python3
"""
Submit Vertex AI Hyperparameter Tuning Job
Uses Google's built-in intelligent search instead of grid search
"""

import os
import yaml
from pathlib import Path
from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
project_root = Path(__file__).parent.parent.parent
env_file = project_root / '.env'

if not env_file.exists():
    raise FileNotFoundError(f".env file not found at {env_file}")

load_dotenv(env_file)

# Configuration
PROJECT_ID = os.getenv('GCP_PROJECT_ID')
REGION = os.getenv('GCP_REGION', 'us-central1')

if not PROJECT_ID:
    raise ValueError("GCP_PROJECT_ID not set in .env file")

GCS_BUCKET = f"{PROJECT_ID}-models"
IMAGE_URI = f"gcr.io/{PROJECT_ID}/model-trainer:latest"


def submit_hyperparameter_tuning_job(
    job_name=None,
    machine_type='n2-standard-8',  # N2 with modern CPUs (Cascade Lake, 8 vCPUs)
    accelerator_type=None,  # CPU-only training (no GPU)
    accelerator_count=0,  # No GPU
    max_trial_count=20,  # Total trials to run
    parallel_trial_count=4,  # How many to run simultaneously
    dataset_version=None,  # Dataset version (e.g., 'v1', 'v2')
    model_type='tft',  # Model type (e.g., 'tft', 'lstm', 'transformer')
    phase=1,  # HP tuning phase: 1=architecture search, 2=regularization tuning
):
    """
    Submit a hyperparameter tuning job using Vertex AI's native service.
    Uses Bayesian optimization by default.
    
    Args:
        job_name: Name for the tuning job
        machine_type: GCE machine type
        max_trial_count: Maximum number of trials to run
        parallel_trial_count: Number of trials to run in parallel
        dataset_version: Dataset version (e.g., 'v1', 'v2'). If not provided, each trial generates from BigQuery.
        model_type: Model type (e.g., 'tft', 'lstm', 'transformer')
    
    Returns:
        HyperparameterTuningJob object
    """
    
    if job_name is None:
        job_name = f"model-hp-tuning-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # Initialize Vertex AI
    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
        staging_bucket=f'gs://{GCS_BUCKET}'
    )
    
    print(f"\n{'='*80}")
    print(f"   Vertex AI Hyperparameter Tuning Job")
    print(f"{'='*80}")
    print(f"Job Name: {job_name}")
    print(f"Model Type: {model_type}")
    if phase == 2:
        print(f"   Phase 2: Regularization Tuning (FinCast Prep)")
    else:
        print(f"   Phase 1: Architecture Search")
    print(f"Algorithm: Bayesian Optimization (default)")
    print(f"Max Trials: {max_trial_count}")
    print(f"Parallel Trials: {parallel_trial_count}")
    if dataset_version:
        full_dataset_version = f"{model_type}/{dataset_version}"
        print(f"Dataset Version: {full_dataset_version} (shared across all trials)")
    else:
        print(f"Dataset: Each trial will generate from BigQuery")
    
    # Phase-specific param info
    if phase == 2:
        print(f"\n🔒 LOCKED (from Phase 1 winners):")
        print(f"   - hidden_size: 128")
        print(f"   - lstm_layers: 2")
        print(f"   - attention_layers: 2")
        print(f"   - attention_heads: 8")
        print(f"   - batch_size: 64")
        print(f"   - learning_rate: 5e-5")
        print(f"\n🎯 TUNING (Phase 2):")
        print(f"   - dropout: [0.35, 0.40, 0.45, 0.50]")
        print(f"   - weight_decay: [0.0, 0.01, 0.02, 0.03]")
    
    print(f"{'='*80}\n")
    
    # Define hyperparameter search space based on phase
    if phase == 2:
        # Phase 2: Lock architecture params from Phase 1, tune regularization
        hyperparameter_specs = {
            # LOCKED architecture (from Phase 1 winners)
            'hidden_size': hpt.DiscreteParameterSpec(values=[128], scale='linear'),
            'lstm_layers': hpt.DiscreteParameterSpec(values=[2], scale='linear'),
            'attention_layers': hpt.DiscreteParameterSpec(values=[2], scale='linear'),
            'attention_heads': hpt.DiscreteParameterSpec(values=[8], scale='linear'),
            'learning_rate': hpt.DiscreteParameterSpec(values=[0.00005], scale='linear'),
            'batch_size': hpt.DiscreteParameterSpec(values=[64], scale='linear'),
            
            # TUNING (Phase 2 focus)
            'dropout': hpt.DiscreteParameterSpec(
                values=[0.35, 0.40, 0.45, 0.50],  # Tune around Phase 1 value
                scale='linear'
            ),
            'weight_decay': hpt.DiscreteParameterSpec(
                values=[0.0, 0.01, 0.02, 0.03],  # L2 regularization
                scale='linear'
            ),
        }
    else:
        # Phase 1: Original architecture search
        hyperparameter_specs = {
            'hidden_size': hpt.DiscreteParameterSpec(
                values=[64, 128],  # TFT benefits from larger hidden sizes
                scale='linear'
            ),
            'lstm_layers': hpt.DiscreteParameterSpec(
                values=[1, 2],  # LSTM layers (1=baseline, 2=deeper)
                scale='linear'
            ),
            'attention_layers': hpt.DiscreteParameterSpec(
                values=[1, 2, 3],  # Transformer encoder layers (1=shallow, 2=baseline, 3=deep)
                scale='linear'
            ),
            'attention_heads': hpt.DiscreteParameterSpec(
                values=[4, 8],  # Multi-head attention (must divide hidden_size)
                scale='linear'
            ),
            'learning_rate': hpt.DiscreteParameterSpec(
                values=[0.00005, 0.0001, 0.0005],  # TFT prefers lower LR
                scale='linear'
            ),
            'dropout': hpt.DiscreteParameterSpec(
                values=[0.3, 0.4, 0.5],  # Higher dropout for regularization
                scale='linear'
            ),
            'batch_size': hpt.DiscreteParameterSpec(
                values=[32, 64],  # TFT is memory-intensive
                scale='linear'
            ),
        }
    
    # Define metrics to track
    # Primary metric (optimized): val_loss
    # Secondary metrics (monitored): val_mae, directional_accuracy
    metric_spec = {
        'val_loss': 'minimize',  # Primary optimization target
        # Uncomment to also optimize for directional accuracy:
        # 'directional_accuracy': 'maximize',
    }
    
    # Create worker pool spec
    # Build container args
    container_args = [
        f'--gcs_bucket={GCS_BUCKET}',
        f'--job_name={job_name}',
        f'--model_type={model_type}',  # Pass model type to training wrapper
    ]
    
    # Add dataset version if provided
    # All trials will share the same dataset!
    # Construct full dataset path: model_type/version (e.g., 'tft/v1')
    if dataset_version:
        full_dataset_version = f"{model_type}/{dataset_version}"
        container_args.append(f'--dataset_version={full_dataset_version}')
    
    worker_pool_specs = [{
        'machine_spec': {
            'machine_type': machine_type,
        },
        'replica_count': 1,
        'container_spec': {
            'image_uri': IMAGE_URI,
            'args': container_args,
        },
    }]
    
    # Add GPU accelerator if specified
    if accelerator_type and accelerator_count > 0:
        worker_pool_specs[0]['machine_spec']['accelerator_type'] = accelerator_type
        worker_pool_specs[0]['machine_spec']['accelerator_count'] = accelerator_count
    
    # Create labels for easy identification in Experiments UI
    # Labels must be lowercase, alphanumeric, hyphens, underscores
    labels = {
        'model_type': model_type.lower().replace('_', '-'),
        'job_name': job_name.lower().replace('_', '-'),
        'job_type': f'hp-phase{phase}',
        'phase': 'regularization' if phase == 2 else 'architecture',
    }
    if dataset_version:
        labels['dataset_version'] = dataset_version.lower().replace('/', '-').replace('_', '-')
    
    # Try to extract date range and hyperparams from dataset metadata
    try:
        # Parse dataset version to find metadata file
        # dataset_version can be "v1" or "model_type/v1"
        if dataset_version:
            if '/' in dataset_version:
                # Already has model_type prefix
                dataset_path = f"data/datasets/{dataset_version}/processed/metadata.yaml"
            else:
                # Add model_type prefix
                dataset_path = f"data/datasets/{model_type}/{dataset_version}/processed/metadata.yaml"
            
            if Path(dataset_path).exists():
                with open(dataset_path, 'r') as f:
                    dataset_metadata = yaml.safe_load(f)
                
                # Extract data section (metadata has nested 'data' section in v11+)
                data_config = dataset_metadata.get('data', dataset_metadata)
                
                # Date range
                start_date = data_config.get('start_date', '')
                end_date = data_config.get('end_date', '')
                
                if start_date:
                    # GCP labels: lowercase, alphanumeric, hyphens, underscores only
                    labels['start_date'] = start_date.replace('/', '-')
                if end_date:
                    labels['end_date'] = end_date.replace('/', '-')
                
                # Data configuration (useful for filtering)
                if 'lookback_window' in data_config:
                    labels['lookback'] = str(data_config['lookback_window'])
                if 'prediction_horizons' in data_config:
                    horizons = data_config['prediction_horizons']
                    labels['num_horizons'] = str(len(horizons))
                    # Format horizons as underscore-separated string (e.g., "7_14_28")
                    labels['horizons'] = '_'.join(map(str, horizons))
            else:
                print(f"   ⚠️  Dataset metadata not found: {dataset_path}")
    except Exception as e:
        # Non-critical, continue without metadata labels
        print(f"   ⚠️  Could not extract metadata: {e}")
    
    # Add FinCast enabled label from model config
    try:
        config_path = f"configs/model_{model_type}_config.yaml"
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                model_config = yaml.safe_load(f)
                fincast_enabled = model_config.get('fincast', {}).get('enabled', False)
                labels['fincast'] = 'enabled' if fincast_enabled else 'disabled'
    except Exception as e:
        # Non-critical, default to disabled
        labels['fincast'] = 'disabled'
        print(f"   ⚠️  Could not read FinCast status from config: {e}")
    
    # Add phase-specific labels
    if phase == 2:
        # Phase 2: Add locked architecture params for easy filtering
        labels['hidden_size'] = '128'
        labels['lstm_layers'] = '2'
        labels['attention_layers'] = '2'
        labels['attention_heads'] = '8'
        labels['batch_size'] = '64'
        labels['lr'] = '5e-05'  # 'learning_rate' is too long
        labels['tuning'] = 'dropout-weight_decay'
    else:
        # Phase 1: Indicate architecture search
        labels['tuning'] = 'architecture'
    
    print(f"\n🏷️  Adding labels for identification:")
    for key, value in labels.items():
        print(f"   {key}: {value}")
    
    # Create custom job for HP tuning
    custom_job = aiplatform.CustomJob(
        display_name=f"{job_name}-base",
        worker_pool_specs=worker_pool_specs,
        labels=labels,
    )
    
    # Get or create TensorBoard instance (matching submit_job.py pattern)
    tensorboard_resource_name = None
    try:
        print(f"\n📊 Setting up TensorBoard...")
        tensorboards = aiplatform.Tensorboard.list(filter=f'display_name="tensorboard-{PROJECT_ID}"')
        
        if tensorboards:
            tensorboard = tensorboards[0]
            tensorboard_resource_name = tensorboard.resource_name
            print(f"   ✅ Using existing TensorBoard: {tensorboard.display_name}")
            print(f"   Resource: {tensorboard_resource_name}")
        else:
            print(f"   Creating new TensorBoard instance...")
            tensorboard = aiplatform.Tensorboard.create(
                display_name=f"tensorboard-{PROJECT_ID}",
                project=PROJECT_ID,
                location=REGION,
            )
            tensorboard_resource_name = tensorboard.resource_name
            print(f"   ✅ Created: {tensorboard.display_name}")
            print(f"   Resource: {tensorboard_resource_name}")
    except Exception as e:
        print(f"\n⚠️  TensorBoard setup failed: {e}")
        print(f"   HP tuning will continue without TensorBoard")
        tensorboard_resource_name = None
    
    # Create hyperparameter tuning job with TensorBoard
    # Note: search_algorithm defaults to Bayesian optimization when not specified
    hp_job = aiplatform.HyperparameterTuningJob(
        display_name=job_name,
        custom_job=custom_job,
        metric_spec=metric_spec,
        parameter_spec=hyperparameter_specs,
        max_trial_count=max_trial_count,
        parallel_trial_count=parallel_trial_count,
        labels=labels,  # Add labels to HP tuning job for visibility in console
    )
    
    print(f"\n🚀 Submitting hyperparameter tuning job...")
    
    # Prepare run parameters
    run_params = {}
    
    # Add TensorBoard and service account if TensorBoard is configured
    if tensorboard_resource_name:
        # Get project number for default compute service account
        # This SA already has necessary GCS and Vertex AI permissions
        import subprocess
        result = subprocess.run(
            ['gcloud', 'projects', 'describe', PROJECT_ID, '--format=value(projectNumber)'],
            capture_output=True, text=True, check=True
        )
        project_number = result.stdout.strip()
        service_account = f"{project_number}-compute@developer.gserviceaccount.com"
        
        run_params['tensorboard'] = tensorboard_resource_name
        run_params['service_account'] = service_account
        print(f"   📊 TensorBoard integration: ENABLED")
        print(f"   Service account: {service_account}")
    else:
        print(f"   📊 TensorBoard integration: DISABLED")
    
    hp_job.run(**run_params)
    
    print(f"\n✅ Job submitted!")
    print(f"📊 Monitor at:")
    print(f"https://console.cloud.google.com/vertex-ai/training/training-pipelines?project={PROJECT_ID}")
    
    if tensorboard_resource_name:
        tb_id = tensorboard_resource_name.split('/')[-1]
        print(f"\n📈 TensorBoard:")
        print(f"https://console.cloud.google.com/vertex-ai/experiments/tensorboard-instances/{tb_id}/experiments?project={PROJECT_ID}")
    
    print(f"\n💡 Best trial will be automatically identified!\n")
    
    return hp_job


if __name__ == '__main__':
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Submit Vertex AI hyperparameter tuning job')
    parser.add_argument('--dataset-version', type=str, default=None,
                       help='Dataset version to use (e.g., v1, v2). Combined with model-type to form full path.')
    parser.add_argument('--model-type', type=str, default='tft',
                       help='Model type (e.g., tft, lstm, transformer). Default: tft')
    parser.add_argument('--job-name', type=str, default=None,
                       help='Job name (auto-generated if not provided)')
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2],
                       help='HP tuning phase: 1=architecture search, 2=regularization tuning (default: 1)')
    parser.add_argument('--max-trials', type=int, default=None,
                       help='Max trials (default: 20 for phase 1, 16 for phase 2)')
    parser.add_argument('--parallel-trials', type=int, default=None,
                       help='Parallel trials (default: 3 for phase 1, 4 for phase 2)')
    args = parser.parse_args()
    
    # Set defaults based on phase
    max_trials = args.max_trials or (16 if args.phase == 2 else 20)
    parallel_trials = args.parallel_trials or (4 if args.phase == 2 else 3)
    
    # CPU-only (slower but cheaper)
    submit_hyperparameter_tuning_job(
        job_name=args.job_name,
        dataset_version=args.dataset_version,
        model_type=args.model_type,
        phase=args.phase,
        machine_type='e2-highmem-4',         # CPU-only: 32GB RAM (~$0.21/hr)
        accelerator_type=None,
        accelerator_count=0,
        max_trial_count=max_trials,
        parallel_trial_count=parallel_trials,
    )
    
    # CPU-only option (uncomment to use - cheaper but slower)
    # job = submit_hyperparameter_tuning_job(
    #     job_name=args.job_name or 'model-hp-tuning-cpu',
    #     machine_type='e2-standard-4',  # E2 = CPU only (~$0.13/hr per trial)
    #     accelerator_type=None,
    #     accelerator_count=0,
    #     max_trial_count=20,
    #     parallel_trial_count=4,
    #     dataset_version=args.dataset_version,  # NEW
    # )