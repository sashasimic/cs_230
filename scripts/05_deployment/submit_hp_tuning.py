#!/usr/bin/env python3
"""
Submit Vertex AI Hyperparameter Tuning Job
Uses Google's built-in intelligent search instead of grid search
"""

import os
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
    machine_type='n1-standard-4',  # N1 supports GPUs
    accelerator_type='NVIDIA_TESLA_T4',  # T4 GPU enabled by default
    accelerator_count=1,  # 1 GPU per trial
    max_trial_count=20,  # Total trials to run
    parallel_trial_count=4,  # How many to run simultaneously
    dataset_version=None,  # NEW: Dataset version to use
):
    """
    Submit a hyperparameter tuning job using Vertex AI's native service.
    Uses Bayesian optimization by default.
    
    Args:
        job_name: Name for the tuning job
        machine_type: GCE machine type
        max_trial_count: Maximum number of trials to run
        parallel_trial_count: Number of trials to run in parallel
        dataset_version: Dataset version to use (e.g., 'v1', 'v2'). If not provided, each trial generates from BigQuery.
    
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
    print(f"Algorithm: Bayesian Optimization (default)")
    print(f"Max Trials: {max_trial_count}")
    print(f"Parallel Trials: {parallel_trial_count}")
    if dataset_version:
        print(f"Dataset Version: {dataset_version} (shared across all trials)")
    else:
        print(f"Dataset: Each trial will generate from BigQuery")
    print(f"{'='*80}\n")
    
    # Define hyperparameter search space
    # Matches HYPERPARAMETER_GRID from submit_parallel.py
    hyperparameter_specs = {
        'hidden_size': hpt.DiscreteParameterSpec(
            values=[32, 64, 128],
            scale='linear'
        ),
        'lstm_layers': hpt.DiscreteParameterSpec(
            values=[1, 2],
            scale='linear'
        ),
        'learning_rate': hpt.DiscreteParameterSpec(
            values=[0.0001, 0.001, 0.01],
            scale='linear'
        ),
        'dropout': hpt.DiscreteParameterSpec(
            values=[0.1, 0.2],
            scale='linear'
        ),
        'batch_size': hpt.DiscreteParameterSpec(
            values=[64, 128],
            scale='linear'
        ),
        'lookback_window': hpt.DiscreteParameterSpec(
            values=[96, 192],
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
    ]
    
    # Add dataset version if provided (NEW)
    # All trials will share the same dataset!
    if dataset_version:
        container_args.append(f'--dataset_version={dataset_version}')
    
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
    
    # Create custom job for HP tuning
    custom_job = aiplatform.CustomJob(
        display_name=f"{job_name}-base",
        worker_pool_specs=worker_pool_specs,
    )
    
    # Create hyperparameter tuning job
    # Note: search_algorithm defaults to Bayesian optimization when not specified
    hp_job = aiplatform.HyperparameterTuningJob(
        display_name=job_name,
        custom_job=custom_job,
        metric_spec=metric_spec,
        parameter_spec=hyperparameter_specs,
        max_trial_count=max_trial_count,
        parallel_trial_count=parallel_trial_count,
    )
    
    print(f"\n🚀 Submitting hyperparameter tuning job...")
    hp_job.run()
    
    print(f"\n✅ Job submitted!")
    print(f"📊 Monitor at:")
    print(f"https://console.cloud.google.com/vertex-ai/training/training-pipelines?project={PROJECT_ID}")
    print(f"\n💡 Best trial will be automatically identified!\n")
    
    return hp_job


if __name__ == '__main__':
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Submit Vertex AI hyperparameter tuning job')
    parser.add_argument('--dataset-version', type=str, default=None,
                       help='Dataset version to use (e.g., v1, v2). If not provided, each trial generates from BigQuery.')
    parser.add_argument('--job-name', type=str, default=None,
                       help='Job name (auto-generated if not provided)')
    args = parser.parse_args()
    
    # GPU-enabled (default) - 2-3x faster per trial!
    job = submit_hyperparameter_tuning_job(
        job_name=args.job_name or 'model-hp-tuning-gpu',
        machine_type='n1-standard-4',        # N1 supports GPUs (~$0.19/hr)
        accelerator_type='NVIDIA_TESLA_T4',   # T4 GPU (~$0.35/hr)
        accelerator_count=1,                  # 1 GPU per trial
        max_trial_count=20,
        parallel_trial_count=4,  # 4 parallel trials = 4 GPUs running simultaneously!
        dataset_version=args.dataset_version,  # NEW
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