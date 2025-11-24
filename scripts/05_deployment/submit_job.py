#!/usr/bin/env python3
"""
Submit single training job to Vertex AI
"""

import os
from pathlib import Path
from google.cloud import aiplatform
from datetime import datetime

# Load environment variables from .env file
from dotenv import load_dotenv
project_root = Path(__file__).parent.parent.parent
env_file = project_root / '.env'

if not env_file.exists():
    raise FileNotFoundError(f".env file not found at {env_file}")

load_dotenv(env_file)

# Configuration from environment
PROJECT_ID = os.getenv('GCP_PROJECT_ID')
REGION = os.getenv('GCP_REGION', 'us-central1')

if not PROJECT_ID:
    raise ValueError(
        f"GCP_PROJECT_ID not set in .env file ({env_file}). "
        "Please set it in .env or use: export GCP_PROJECT_ID=your-project-id"
    )

GCS_BUCKET = f"{PROJECT_ID}-models"
IMAGE_URI = f"gcr.io/{PROJECT_ID}/model-trainer:latest"


def submit_training_job(
    job_name=None,
    machine_type='n1-standard-4',  # N1 supports GPUs
    accelerator_type='NVIDIA_TESLA_T4',  # T4 GPU enabled by default
    accelerator_count=1,  # 1 GPU
    use_spot=True,  # Use spot instances for faster provisioning & lower cost
    dataset_version=None,  # Dataset version (e.g., 'v1', 'v2')
    model_type='tft',  # Model type (e.g., 'tft', 'lstm', 'transformer')
    **hyperparameters
):
    """
    Submit a custom training job to Vertex AI.
    
    Args:
        job_name: Name for the training job
        machine_type: GCE machine type
        accelerator_type: GPU type (optional)
        accelerator_count: Number of GPUs
        use_spot: Use spot (preemptible) instances
        dataset_version: Dataset version (e.g., 'v1', 'v2'). If not provided, generates from BigQuery.
        model_type: Model type (e.g., 'tft', 'lstm', 'transformer')
        **hyperparameters: Model hyperparameters to pass
    """
    
    if job_name is None:
        job_name = f"model-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # Initialize Vertex AI with staging bucket
    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
        staging_bucket=f'gs://{GCS_BUCKET}'
    )
    
    # Build args list
    args = [
        f'--gcs_bucket={GCS_BUCKET}',
        f'--job_name={job_name}',
        f'--model_type={model_type}',  # Pass model type to training wrapper
    ]
    
    # Add dataset version if provided
    # Construct full dataset path: model_type/version (e.g., 'tft/v1')
    if dataset_version:
        full_dataset_version = f"{model_type}/{dataset_version}"
        args.append(f'--dataset_version={full_dataset_version}')
    
    for key, value in hyperparameters.items():
        args.append(f'--{key}={value}')
    
    print(f"\n{'='*80}")
    print(f"   Submitting Vertex AI Training Job")
    print(f"{'='*80}")
    print(f"Job Name: {job_name}")
    print(f"Model Type: {model_type}")
    print(f"Machine: {machine_type}")
    print(f"Image: {IMAGE_URI}")
    if dataset_version:
        full_dataset_version = f"{model_type}/{dataset_version}"
        print(f"Dataset Version: {full_dataset_version} (pre-generated)")
    else:
        print(f"Dataset: Will generate from BigQuery")
    print(f"Args: {args}")
    print(f"{'='*80}\n")
    
    # Create custom job
    machine_spec = {
        'machine_type': machine_type,
    }
    
    if accelerator_type and accelerator_count > 0:
        machine_spec['accelerator_type'] = accelerator_type
        machine_spec['accelerator_count'] = accelerator_count
    
    # Build worker pool spec
    worker_pool_spec = {
        'machine_spec': machine_spec,
        'replica_count': 1,
        'container_spec': {
            'image_uri': IMAGE_URI,
            'command': ['python', 'scripts/05_deployment/train_vertex.py'],
            'args': args,
        },
    }
    
    job = aiplatform.CustomJob(
        display_name=job_name,
        worker_pool_specs=[worker_pool_spec],
    )
    
    # Get or create TensorBoard instance
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
        print(f"   Training will continue without TensorBoard")
        tensorboard_resource_name = None
    
    # Submit job
    print(f"\n🚀 Submitting job to Vertex AI...")
    if use_spot:
        print(f"   ⚠️  Warning: Spot instances requested but not supported via Python SDK")
        print(f"   Using regular instances (spot requires gcloud CLI or REST API)")
    
    # Prepare job run parameters
    run_params = {'sync': False}
    
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
    
    job.run(**run_params)
    
    # Wait a moment for job to be created
    import time
    time.sleep(2)
    
    print(f"\n✅ Job submitted successfully!")
    
    # Job properties may not be available immediately after async submission
    try:
        print(f"   Job Name: {job.display_name}")
    except (RuntimeError, AttributeError):
        print(f"   Job Name: {job_name}")
    
    try:
        print(f"   Resource: {job.resource_name}")
    except (RuntimeError, AttributeError):
        print(f"   Resource: (being created...)")
    
    print(f"\n📊 Monitor at:")
    print(f"https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT_ID}")
    
    return job


if __name__ == '__main__':
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Submit Vertex AI training job')
    parser.add_argument('--dataset-version', type=str, default=None,
                       help='Dataset version to use (e.g., v1, v2). Combined with model-type to form full path.')
    parser.add_argument('--model-type', type=str, default='tft',
                       help='Model type (e.g., tft, lstm, transformer). Default: tft')
    parser.add_argument('--job-name', type=str, default='model-test-run-cpu',
                       help='Job name')
    parser.add_argument('--wait', action='store_true',
                       help='Wait and monitor job status (default: exit immediately)')
    args = parser.parse_args()
    
    # GPU option (commented - quota exceeded, request increase at console.cloud.google.com/iam-admin/quotas)
    # Training time: ~5-10 hours (vs ~47 hours CPU), Cost: ~$5.40/run
    # Hyperparameters from best HP tuning trial (trial_1, val_loss=0.670)
    # job = submit_training_job(
    #     job_name=args.job_name,
    #     machine_type='n1-standard-4',        # N1 supports GPUs (~$0.19/hr)
    #     accelerator_type='NVIDIA_TESLA_T4',   # T4 GPU (~$0.35/hr)
    #     accelerator_count=1,                  # 1 GPU
    #     dataset_version=args.dataset_version,
    #     hidden_size=64,           # Best from HP tuning
    #     lstm_layers=2,            # Best from HP tuning
    #     learning_rate=0.001,      # Best from HP tuning
    #     dropout=0.2,              # Best from HP tuning (was 0.1)
    #     batch_size=128,           # Best from HP tuning
    #     lookback_window=192,      # Best from HP tuning (was 100)
    # )
    
    # CPU-only option - uses values from config file
    # Training time: ~50-70 hours (3x data vs original), Cost: ~$12-18/run
    job = submit_training_job(
        job_name=args.job_name,
        machine_type='e2-highmem-4',  # E2 high-mem = 32GB RAM (~$0.24/hr)
        accelerator_type=None,
        accelerator_count=0,
        dataset_version=args.dataset_version,
        model_type=args.model_type,
        # No hyperparameters - use config file defaults
    )
    
    # Monitor job if --wait flag is set
    if args.wait:
        print(f"\n🔍 Monitoring job status...")
        print(f"   (Press Ctrl+C to stop monitoring, job will continue running)\n")
        try:
            job.wait()
            print(f"\n✅ Job completed successfully!")
        except KeyboardInterrupt:
            print(f"\n\n⏸️  Stopped monitoring (job still running in background)")
    else:
        print(f"\n🚀 Job submitted in background")
        print(f"   Use --wait flag to monitor status, or check console")
    
    # Print final status (handle case where job resource isn't available yet)
    try:
        print(f"\n📊 Job: {job.display_name}")
    except (RuntimeError, AttributeError) as e:
        # Check if it's a quota error
        if "quota" in str(e).lower():
            print(f"\n❌ Job creation failed: GPU quota exceeded")
            print(f"   Error: {str(e)}")
            print(f"\n💡 Solutions:")
            print(f"   1. Use CPU-only training (see commented code above)")
            print(f"   2. Request GPU quota increase: https://console.cloud.google.com/iam-admin/quotas")
            print(f"   3. Check for running jobs consuming quota")
        else:
            print(f"\n🚀 Job submitted (resource being created...)")