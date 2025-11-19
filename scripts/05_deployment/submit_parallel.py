#!/usr/bin/env python3
"""
Submit parallel training jobs with different hyperparameters
"""

import os
import itertools
from pathlib import Path
from google.cloud import aiplatform
from datetime import datetime
import time

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

# Define hyperparameter grid
HYPERPARAMETER_GRID = {
    'hidden_size': [32, 64, 128],
    'lstm_layers': [1, 2],
    'learning_rate': [0.0001, 0.001, 0.01],
    'dropout': [0.1, 0.2],
    'batch_size': [64, 128],
    'lookback_window': [96, 192],
}


def generate_configurations(grid, max_trials=20):
    """
    Generate hyperparameter configurations from grid.
    
    Args:
        grid: Dict of parameter names to lists of values
        max_trials: Maximum number of configurations to generate
    
    Returns:
        List of dicts, each containing one configuration
    """
    keys = grid.keys()
    values = grid.values()
    
    # Generate all combinations
    all_combos = list(itertools.product(*values))
    
    # Limit to max_trials
    combos = all_combos[:max_trials]
    
    # Convert to list of dicts
    configs = [
        dict(zip(keys, combo))
        for combo in combos
    ]
    
    return configs


def submit_parallel_jobs(configs, machine_type='e2-standard-4', delay_between_jobs=5):
    """
    Submit multiple training jobs in parallel.
    
    Args:
        configs: List of hyperparameter configurations
        machine_type: GCE machine type for all jobs
        delay_between_jobs: Seconds to wait between submissions
    
    Returns:
        List of submitted job objects
    """
    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
        staging_bucket=f'gs://{GCS_BUCKET}'
    )
    
    jobs = []
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    
    print(f"\n{'='*80}")
    print(f"   Submitting {len(configs)} Parallel Training Jobs")
    print(f"{'='*80}\n")
    
    for i, config in enumerate(configs, 1):
        job_name = f"model-parallel-{timestamp}-{i:02d}"
        
        # Build args
        args = [
            f'--gcs_bucket={GCS_BUCKET}',
            f'--job_name={job_name}',
        ]
        
        for key, value in config.items():
            args.append(f'--{key}={value}')
        
        print(f"[{i}/{len(configs)}] Submitting: {job_name}")
        print(f"  Config: {config}")
        
        # Create and submit job
        job = aiplatform.CustomJob(
            display_name=job_name,
            worker_pool_specs=[{
                'machine_spec': {'machine_type': machine_type},
                'replica_count': 1,
                'container_spec': {
                    'image_uri': IMAGE_URI,
                    'args': args,
                },
            }],
        )
        
        job.run(sync=False)  # Async submission
        jobs.append(job)
        
        print(f"  ✅ Submitted: {job.resource_name}\n")
        
        # Delay to avoid API rate limits
        if i < len(configs):
            time.sleep(delay_between_jobs)
    
    print(f"{'='*80}")
    print(f"✅ All {len(jobs)} jobs submitted!")
    print(f"\n📊 Monitor at:")
    print(f"https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT_ID}")
    print(f"{'='*80}\n")
    
    return jobs


if __name__ == '__main__':
    # Generate configurations
    configs = generate_configurations(HYPERPARAMETER_GRID, max_trials=12)
    
    print(f"\nGenerated {len(configs)} configurations:")
    for i, cfg in enumerate(configs[:5], 1):
        print(f"  {i}. {cfg}")
    if len(configs) > 5:
        print(f"  ... and {len(configs) - 5} more\n")
    
    # Submit jobs
    jobs = submit_parallel_jobs(configs, machine_type='n1-highmem-4')
    
    print(f"\n🚀 {len(jobs)} jobs running in parallel!")