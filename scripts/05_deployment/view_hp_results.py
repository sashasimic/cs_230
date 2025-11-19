#!/usr/bin/env python3
"""
View Hyperparameter Tuning Results
Displays all trials with metrics and hyperparameters in a formatted table
"""

import os
import sys
import argparse
from pathlib import Path
from google.cloud import aiplatform
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
project_root = Path(__file__).parent.parent.parent
env_file = project_root / '.env'

if env_file.exists():
    load_dotenv(env_file)

PROJECT_ID = os.getenv('GCP_PROJECT_ID')
REGION = os.getenv('GCP_REGION', 'us-central1')

if not PROJECT_ID:
    print("⚠️  GCP_PROJECT_ID not set in .env file")
    PROJECT_ID = input("Enter project ID: ").strip()


def list_hp_tuning_jobs(limit=10):
    """List recent HP tuning jobs."""
    aiplatform.init(project=PROJECT_ID, location=REGION)
    
    jobs = aiplatform.HyperparameterTuningJob.list(
        filter=None,
        order_by='create_time desc',
    )[:limit]
    
    if not jobs:
        print("❌ No hyperparameter tuning jobs found.")
        return
    
    print(f"\n{'='*80}")
    print(f"   Recent Hyperparameter Tuning Jobs")
    print(f"{'='*80}\n")
    
    for i, job in enumerate(jobs, 1):
        status = job.state.name
        status_emoji = "✅" if status == "JOB_STATE_SUCCEEDED" else "🔄" if "RUNNING" in status else "❌"
        
        print(f"{i}. {status_emoji} {job.display_name}")
        print(f"   ID: {job.name.split('/')[-1]}")
        print(f"   Status: {status}")
        print(f"   Created: {job.create_time}")
        if hasattr(job, 'trial_job_spec'):
            trials = len(job.trials) if hasattr(job, 'trials') and job.trials else 0
            print(f"   Trials: {trials}")
        print()
    
    return jobs


def get_hp_tuning_results(job_id_or_name, export_csv=None, sort_by='val_loss'):
    """
    Get detailed results from a hyperparameter tuning job.
    
    Args:
        job_id_or_name: Job ID or full resource name
        export_csv: Optional path to export results as CSV
        sort_by: Metric to sort by (default: val_loss)
    """
    aiplatform.init(project=PROJECT_ID, location=REGION)
    
    # Construct full resource name if only ID provided
    if not job_id_or_name.startswith('projects/'):
        resource_name = f"projects/{PROJECT_ID}/locations/{REGION}/hyperparameterTuningJobs/{job_id_or_name}"
    else:
        resource_name = job_id_or_name
    
    print(f"\n🔍 Fetching results for: {resource_name.split('/')[-1]}...\n")
    
    try:
        hp_job = aiplatform.HyperparameterTuningJob.get(resource_name)
    except Exception as e:
        print(f"❌ Error fetching job: {e}")
        return None
    
    print(f"Job: {hp_job.display_name}")
    print(f"Status: {hp_job.state.name}")
    print(f"Created: {hp_job.create_time}")
    
    if not hasattr(hp_job, 'trials') or not hp_job.trials:
        print("\n⚠️  No trials found yet. Job may still be starting.")
        return None
    
    print(f"Total Trials: {len(hp_job.trials)}")
    print(f"{'='*80}\n")
    
    # Extract trial data
    trials_data = []
    for trial in hp_job.trials:
        row = {
            'trial_id': trial.id,
            'state': trial.state.name,
        }
        
        # Add metrics
        if hasattr(trial, 'final_measurement') and trial.final_measurement:
            for metric in trial.final_measurement.metrics:
                row[metric.metric_id] = metric.value
        
        # Add hyperparameters
        if hasattr(trial, 'parameters'):
            for param in trial.parameters:
                row[param.parameter_id] = param.value
        
        trials_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(trials_data)
    
    # Sort by specified metric
    if sort_by in df.columns:
        ascending = 'accuracy' not in sort_by.lower()  # Descending for accuracy metrics
        df = df.sort_values(sort_by, ascending=ascending)
    
    # Format display
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 20)
    
    # Print summary statistics
    print("\n📊 Metrics Summary:")
    print("="*80)
    metrics_cols = [col for col in df.columns if col not in ['trial_id', 'state']]
    if metrics_cols:
        summary = df[metrics_cols].describe().loc[['mean', 'min', 'max']]
        print(summary.to_string())
    
    # Print full table
    print(f"\n\n📋 All Trials (sorted by {sort_by}):")
    print("="*80)
    print(df.to_string(index=False))
    
    # Highlight best trial
    if sort_by in df.columns:
        best_idx = df[sort_by].idxmax() if 'accuracy' in sort_by.lower() else df[sort_by].idxmin()
        best_trial = df.loc[best_idx]
        
        print(f"\n\n🏆 Best Trial (by {sort_by}):")
        print("="*80)
        print(f"Trial ID: {best_trial['trial_id']}")
        for col in df.columns:
            if col not in ['trial_id', 'state']:
                print(f"{col}: {best_trial[col]}")
    
    # Export to CSV if requested
    if export_csv:
        csv_path = Path(export_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"\n💾 Results exported to: {csv_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='View Vertex AI Hyperparameter Tuning Results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List recent HP tuning jobs
  python view_hp_results.py --list
  
  # View detailed results for a specific job
  python view_hp_results.py --job-id 1234567890
  
  # Export results to CSV
  python view_hp_results.py --job-id 1234567890 --export results.csv
  
  # Sort by directional accuracy
  python view_hp_results.py --job-id 1234567890 --sort-by directional_accuracy
        """
    )
    
    parser.add_argument('--list', action='store_true', 
                       help='List recent HP tuning jobs')
    parser.add_argument('--job-id', type=str,
                       help='HP tuning job ID to view details')
    parser.add_argument('--export', type=str,
                       help='Export results to CSV file')
    parser.add_argument('--sort-by', type=str, default='val_loss',
                       help='Metric to sort by (default: val_loss)')
    
    args = parser.parse_args()
    
    if args.list:
        list_hp_tuning_jobs()
    elif args.job_id:
        get_hp_tuning_results(args.job_id, export_csv=args.export, sort_by=args.sort_by)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()