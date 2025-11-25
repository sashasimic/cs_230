#!/usr/bin/env python3
"""
Dataset Generation and GCS Upload Script

This script generates versioned datasets for model training and uploads them to GCS.
It supports multiple model types with automatic data loader selection.

 MODEL-TYPE-SPECIFIC DATA LOADERS:
   The script dynamically loads the appropriate data loader based on --model-type:
   
   - 'tft' → scripts/02_features/tft/tft_data_loader.py (MultiTickerDataLoader)
   - 'lstm' → scripts/02_features/lstm/lstm_data_loader.py (LSTMDataLoader)
   - 'transformer' → scripts/02_features/transformer/transformer_data_loader.py (TransformerDataLoader)
   
   Each data loader implements its own feature engineering pipeline and exports
   model-specific features to data/raw/ and data/processed/.

 VERSIONING STRUCTURE:
   Datasets are stored with model-type-specific paths:
   
   Local:  data/datasets/{model_type}/{version}/
   GCS:    gs://bucket/datasets/{model_type}/{version}/
   
   Example:
   - data/datasets/tft/v1/raw/tft_features.csv
   - data/datasets/tft/v1/processed/X_train.npy
   - gs://my-bucket/datasets/tft/v1/

 WORKFLOW:
   1. Load model-specific data loader (based on --model-type)
   2. Generate features from BigQuery using that loader
   3. Export to data/datasets/{model_type}/{version}/
   4. Upload to gs://bucket/datasets/{model_type}/{version}/
   5. Register in datasets_registry.yaml
   6. Create Vertex AI Managed Datasets

Usage:
    # Generate TFT dataset version 1
    python generate_dataset.py --version v1 --model-type tft
    
    # Generate LSTM dataset version 1
    python generate_dataset.py --version v1 --model-type lstm
    
    # Use existing local data (skip generation)
    python generate_dataset.py --version v1 --model-type tft --use-existing
    
    # Delete old versions
    python generate_dataset.py --delete-versions v1 v2 v3
"""

import argparse
import os
import sys
import subprocess
import yaml
import importlib.util
from pathlib import Path
from datetime import datetime


def load_data_loader_for_model(model_type: str):
    """
    Dynamically load the appropriate data loader based on model type.
    
    Args:
        model_type: Model type ('tft', 'lstm', 'transformer', etc.)
    
    Returns:
        Data loader class for the specified model type
    
    Raises:
        ValueError: If model type is not supported
    """
    # Map model types to their data loader paths and class names
    model_loaders = {
        'tft': {
            'path': Path(__file__).parent.parent / '02_features' / 'tft' / 'tft_data_loader.py',
            'module_name': 'tft_data_loader',
            'class_name': 'MultiTickerDataLoader'
        },
        'decoder_transformer': {
            # Uses same data format as TFT
            'path': Path(__file__).parent.parent / '02_features' / 'tft' / 'tft_data_loader.py',
            'module_name': 'tft_data_loader',
            'class_name': 'MultiTickerDataLoader'
        },
        'lstm': {
            # Uses same data format as TFT and decoder_transformer
            'path': Path(__file__).parent.parent / '02_features' / 'tft' / 'tft_data_loader.py',
            'module_name': 'tft_data_loader',
            'class_name': 'MultiTickerDataLoader'
        },
        'transformer': {
            'path': Path(__file__).parent.parent / '02_features' / 'transformer' / 'transformer_data_loader.py',
            'module_name': 'transformer_data_loader',
            'class_name': 'TransformerDataLoader'
        }
    }
    
    if model_type not in model_loaders:
        available = ', '.join(model_loaders.keys())
        raise ValueError(f"Unsupported model type '{model_type}'. Available: {available}")
    
    loader_config = model_loaders[model_type]
    loader_path = loader_config['path']
    
    if not loader_path.exists():
        raise FileNotFoundError(
            f"Data loader not found for model type '{model_type}' at: {loader_path}\n"
            f"Please ensure the data loader exists or use a supported model type."
        )
    
    # Dynamically import the data loader module
    spec = importlib.util.spec_from_file_location(loader_config['module_name'], loader_path)
    data_loader_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data_loader_module)
    
    # Get the data loader class
    DataLoaderClass = getattr(data_loader_module, loader_config['class_name'])
    
    print(f"\n📦 Loaded data loader: {loader_config['class_name']} for model type '{model_type}'")
    print(f"   Path: {loader_path}")
    
    return DataLoaderClass


def generate_data(config_path: str, version: str, model_type: str = 'tft', base_dir: str = 'data/datasets') -> dict:
    """
    Generate raw and processed data using unified logic.
    
    Args:
        config_path: Path to model config YAML
        version: Dataset version (e.g., 'v1', 'v2', 'nov2024')
        model_type: Model type for versioning (e.g., 'tft', 'lstm')
        base_dir: Base directory for dataset storage
    
    Returns:
        Dictionary with paths to generated data
    """
    print("\n" + "="*80)
    print(f"   GENERATING DATASET: {model_type}/{version}")
    print("="*80)
    
    # Create model-type-specific version directory
    version_dir = Path(base_dir) / model_type / version
    raw_dir = version_dir / 'raw'
    processed_dir = version_dir / 'processed'
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📂 Local directories:")
    print(f"   Raw: {raw_dir}")
    print(f"   Processed: {processed_dir}")
    
    # Step 1: Load appropriate data loader for model type
    print("\n" + "-"*80)
    print(f"STEP 1: Preparing data using {model_type.upper()} data loader")
    print("-"*80)
    
    try:
        DataLoaderClass = load_data_loader_for_model(model_type)
    except (ValueError, FileNotFoundError) as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    
    # Initialize and run data loader
    loader = DataLoaderClass(config_path=config_path)
    splits = loader.prepare_data()
    
    # prepare_data() automatically exports to:
    #   - data/raw/{model_type}_features.{csv,parquet}
    #   - data/processed/X_*.npy, y_*.npy, ts_*.npy (9 files for 3 splits)
    #   - data/processed/scalers.pkl
    #   - data/processed/metadata.yaml
    
    print("\n✅ Data generated by data loader:")
    print(f"   Raw: data/raw/ (model-specific features)")
    print(f"   Processed: data/processed/X_*.npy, y_*.npy, ts_*.npy")
    print(f"   Metadata: data/processed/{{scalers.pkl,metadata.yaml,feature_names.txt}}")
    
    # Step 2: Copy raw data to versioned directory
    print("\n" + "-"*80)
    print("STEP 2: Copying raw data to versioned directory")
    print("-"*80)
    
    import shutil
    from pathlib import Path as P
    
    # Copy from data/raw to versioned raw directory
    source_raw = P('data/raw')
    if source_raw.exists():
        for file in source_raw.glob('*'):
            if file.is_file():
                shutil.copy2(file, raw_dir / file.name)
                print(f"   ✅ {file.name} → {raw_dir / file.name}")
    
    # Step 3: Copy processed data to versioned directory
    print("\n" + "-"*80)
    print("STEP 3: Copying processed data to versioned directory")
    print("-"*80)
    
    # Copy from data/processed to versioned processed directory
    source_processed = P('data/processed')
    if source_processed.exists():
        for file in source_processed.glob('*'):
            if file.is_file():
                shutil.copy2(file, processed_dir / file.name)
                print(f"   ✅ {file.name} → {processed_dir / file.name}")
    
    print("\n💡 Note: Using numpy arrays (.npy) for training data")
    print("   Vertex AI Managed Datasets will use raw CSV for schema viewing")
    
    # Step 4: Create manifest
    manifest = {
        'version': version,
        'created': datetime.now().isoformat(),
        'config': config_path,
        'raw_dir': str(raw_dir),
        'processed_dir': str(processed_dir),
        'files': {
            'raw': [f.name for f in raw_dir.glob('*') if f.is_file()],
            'processed': [f.name for f in processed_dir.glob('*') if f.is_file()]
        },
        'data_stats': {
            'train_samples': splits['train'][0].shape[0],
            'val_samples': splits['val'][0].shape[0],
            'test_samples': splits['test'][0].shape[0],
            'num_features': splits['train'][0].shape[2],  # (samples, lookback, features)
            'lookback_window': splits['train'][0].shape[1],
            'prediction_horizons': splits['train'][1].shape[1]
        }
    }
    
    manifest_file = version_dir / 'manifest.yaml'
    with open(manifest_file, 'w') as f:
        yaml.dump(manifest, f, default_flow_style=False)
    
    print(f"\n📄 Manifest saved: {manifest_file}")
    
    print("\n" + "="*80)
    print(f"✅ Dataset {version} generated successfully!")
    print("="*80)
    print(f"\n📊 Summary:")
    print(f"   Train: {manifest['data_stats']['train_samples']:,} sequences")
    print(f"   Val: {manifest['data_stats']['val_samples']:,} sequences")
    print(f"   Test: {manifest['data_stats']['test_samples']:,} sequences")
    print(f"   Features: {manifest['data_stats']['num_features']} (lookback: {manifest['data_stats']['lookback_window']})")
    print(f"   Horizons: {manifest['data_stats']['prediction_horizons']}")
    
    return manifest


def upload_to_gcs(version: str, gcs_bucket: str, base_dir: str = 'data/datasets') -> str:
    """
    Upload versioned dataset to GCS.
    
    Args:
        version: Dataset version
        gcs_bucket: GCS bucket name
        base_dir: Local base directory
    
    Returns:
        GCS base path for the version
    """
    print("\n" + "="*80)
    print(f"   UPLOADING TO GCS")
    print("="*80)
    
    # version already includes model_type (e.g., 'tft/v1')
    version_dir = Path(base_dir) / version
    gcs_base_path = f"gs://{gcs_bucket}/datasets/{version}/"
    
    print(f"\n☁️  Destination: {gcs_base_path}")
    
    # Upload entire version directory (raw + processed + manifest)
    # Use rsync to exclude .gitkeep files (cp doesn't support -x)
    cmd = [
        'gsutil', '-m', 'rsync', '-r',
        '-x', r'.*/\.gitkeep$',  # Exclude .gitkeep files
        str(version_dir),
        gcs_base_path.rstrip('/')
    ]
    
    print(f"\n🚀 Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Upload failed: {result.stderr}")
        raise Exception(f"GCS upload failed: {result.stderr}")
    
    print(f"\n✅ Upload complete!")
    print(f"   Raw data: {gcs_base_path}raw/")
    print(f"   Processed data: {gcs_base_path}processed/")
    print(f"   Manifest: {gcs_base_path}manifest.yaml")
    
    # Verify upload by listing files on GCS
    print(f"\n🔍 Verifying GCS upload...")
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(gcs_bucket)
        
        prefix = f"datasets/{version}/"
        all_blobs = list(bucket.list_blobs(prefix=prefix))
        
        if all_blobs:
            print(f"   ✅ Found {len(all_blobs)} files on GCS")
            
            # Define expected files (model-agnostic check)
            expected_files = {
                'raw': [],  # Skip specific file checks, model-dependent
                'processed': [
                    'X_train.npy', 'y_train.npy', 'ts_train.npy',
                    'X_val.npy', 'y_val.npy', 'ts_val.npy',
                    'X_test.npy', 'y_test.npy', 'ts_test.npy',
                    'scalers.pkl', 'metadata.yaml', 'feature_names.txt'
                ],
                'root': ['manifest.yaml']
            }
            
            # Group by directory
            raw_files = [b for b in all_blobs if '/raw/' in b.name]
            processed_files = [b for b in all_blobs if '/processed/' in b.name]
            root_files = [b for b in all_blobs if b.name.count('/') == 2]
            
            validation_errors = []
            
            # Validate raw files
            if raw_files:
                print(f"\n   📂 raw/ ({len(raw_files)} files):")
                raw_names = set()
                for blob in sorted(raw_files, key=lambda b: b.name):
                    filename = blob.name.split('/')[-1]
                    raw_names.add(filename)
                    size_mb = blob.size / (1024 * 1024)
                    # .gitkeep files are expected to be empty
                    is_gitkeep = filename == '.gitkeep'
                    status = "✅" if (blob.size > 0 or is_gitkeep) else "⚠️"
                    print(f"      {status} {filename:<35} {size_mb:>8.2f} MB")
                    if blob.size == 0 and not is_gitkeep:
                        validation_errors.append(f"raw/{filename} is empty (0 bytes)")
                
                # Check for missing expected files
                for expected in expected_files['raw']:
                    if expected not in raw_names:
                        validation_errors.append(f"Missing expected file: raw/{expected}")
            
            # Validate processed files
            if processed_files:
                print(f"\n   📂 processed/ ({len(processed_files)} files):")
                processed_names = set()
                for blob in sorted(processed_files, key=lambda b: b.name):
                    filename = blob.name.split('/')[-1]
                    processed_names.add(filename)
                    size_mb = blob.size / (1024 * 1024)
                    # .gitkeep files are expected to be empty
                    is_gitkeep = filename == '.gitkeep'
                    status = "✅" if (blob.size > 0 or is_gitkeep) else "⚠️"
                    print(f"      {status} {filename:<35} {size_mb:>8.2f} MB")
                    if blob.size == 0 and not is_gitkeep:
                        validation_errors.append(f"processed/{filename} is empty (0 bytes)")
                
                # Check for missing expected files
                for expected in expected_files['processed']:
                    if expected not in processed_names:
                        validation_errors.append(f"Missing expected file: processed/{expected}")
            
            # Validate root files
            if root_files:
                print(f"\n   📂 root/ ({len(root_files)} files):")
                root_names = set()
                for blob in sorted(root_files, key=lambda b: b.name):
                    filename = blob.name.split('/')[-1]
                    root_names.add(filename)
                    size_kb = blob.size / 1024
                    status = "✅" if blob.size > 0 else "⚠️"
                    print(f"      {status} {filename:<35} {size_kb:>8.2f} KB")
                    if blob.size == 0:
                        validation_errors.append(f"{filename} is empty (0 bytes)")
                
                # Check for missing expected files
                for expected in expected_files['root']:
                    if expected not in root_names:
                        validation_errors.append(f"Missing expected file: {expected}")
            
            total_size_mb = sum(b.size for b in all_blobs) / (1024 * 1024)
            print(f"\n   📊 Total size: {total_size_mb:.2f} MB")
            
            # Report validation results
            if validation_errors:
                print(f"\n   ⚠️  Validation warnings ({len(validation_errors)}):")
                for error in validation_errors:
                    print(f"      • {error}")
                print(f"\n   ⚠️  Upload may be incomplete or corrupted!")
            else:
                print(f"\n   ✅ All expected files present and non-empty!")
        else:
            print(f"   ⚠️  No files found on GCS at {gcs_base_path}")
            print(f"   This may indicate an upload failure.")
    except Exception as e:
        print(f"   ⚠️  Could not verify GCS upload: {e}")
    
    return gcs_base_path


def create_managed_dataset(version: str, gcs_bucket: str, project: str, region: str, model_type: str = 'tft') -> dict:
    """
    Create Vertex AI Managed Datasets for raw and processed data.
    Uses TabularDataset with CSV files to enable schema viewing in Vertex AI Console.
    
    Args:
        version: Dataset version (e.g., 'tft/v2' or just 'v2')
        gcs_bucket: GCS bucket name
        project: GCP project ID
        region: GCP region
        model_type: Model type (e.g., 'tft', 'lstm')
    
    Returns:
        Dictionary with dataset IDs
    """
    print(f"\n" + "="*80)
    print(f"   CREATING VERTEX AI MANAGED DATASETS")
    print("="*80)
    
    try:
        from google.cloud import aiplatform
    except ImportError:
        print("⚠️  google-cloud-aiplatform not installed, skipping Managed Dataset creation")
        return {}
    
    aiplatform.init(project=project, location=region)
    
    gcs_base_path = f"gs://{gcs_bucket}/datasets/{version}/"
    
    # Helper function to find existing dataset by display name
    def find_existing_dataset(display_name: str):
        """Find an existing TabularDataset by display name."""
        try:
            datasets = aiplatform.TabularDataset.list(
                filter=f'display_name="{display_name}"',
                order_by='create_time desc'
            )
            if datasets:
                return datasets[0]  # Return most recent
        except Exception as e:
            print(f"   ⚠️  Could not check for existing dataset: {e}")
        return None
    
    # Create Managed Dataset for CSV schema viewing
    # Note: This is ONLY for viewing the feature schema in Vertex AI Console.
    # Actual training loads .npy files directly from GCS (much faster).
    # Convert version path to valid display name (replace / with -)
    # e.g., 'tft/v2' -> 'tft-v2-features-csv'
    version_safe = version.replace('/', '-')
    raw_display_name = f"{version_safe}-features-csv"
    print(f"\n🔍 Checking for existing schema dataset '{raw_display_name}'...")
    raw_dataset = find_existing_dataset(raw_display_name)
    
    if raw_dataset:
        print(f"   ♻️  Reusing existing dataset: {raw_dataset.resource_name}")
    else:
        print(f"\n🏗️  Creating Managed Dataset (CSV for schema viewing only)...")
        try:
            # Find CSV file in raw directory (model-agnostic)
            from google.cloud import storage
            client = storage.Client()
            bucket = client.bucket(gcs_bucket)
            prefix = f"datasets/{version}/raw/"
            csv_files = [b for b in bucket.list_blobs(prefix=prefix) if b.name.endswith('.csv')]
            
            if not csv_files:
                print(f"   ⚠️  No CSV files found in {gcs_base_path}raw/")
                print(f"   Skipping Managed Dataset creation (CSV required for schema viewing)")
                raw_dataset = None
            else:
                # Use the first CSV file found
                csv_blob = csv_files[0]
                csv_gcs_path = f"gs://{gcs_bucket}/{csv_blob.name}"
                print(f"   📄 Using CSV: {csv_blob.name}")
                
                raw_dataset = aiplatform.TabularDataset.create(
                    display_name=raw_display_name,
                    gcs_source=csv_gcs_path,
                    labels={
                        'model_type': model_type,
                        'version': version.replace('/', '-').replace('.', '_'),  # Labels can't have dots or slashes
                        'type': 'schema',
                        'purpose': 'inspection',
                        'note': 'training_uses_npy_files'
                    },
                    sync=True
                )
                print(f"   ✅ Managed Dataset: {raw_dataset.resource_name}")
                print(f"   📊 Purpose: View feature schema in Vertex AI Console")
                print(f"   ⚠️  Note: Training loads .npy files directly, not this CSV")
        except Exception as e:
            print(f"   ⚠️  Warning: Could not create raw dataset: {e}")
            raw_dataset = None
    
    # Note: No Managed Dataset for processed data (numpy arrays)
    # Training jobs load .npy files directly from GCS - much faster than CSV!
    print(f"\n💡 Training Data Location:")
    print(f"   Training jobs load .npy files from: {gcs_base_path}processed/X_*.npy")
    print(f"   The Managed Dataset above is ONLY for viewing schema, not for training")
    processed_dataset = None
    
    datasets = {
        'version': version,
        'raw_dataset_id': raw_dataset.resource_name if raw_dataset else 'N/A',
        'processed_dataset_id': processed_dataset.resource_name if processed_dataset else 'N/A',
        'gcs_base_path': gcs_base_path,
        'created': datetime.now().isoformat()
    }
    
    print(f"\n💡 Note: Managed Dataset '{raw_display_name}' is for schema viewing only.")
    print(f"   Training jobs load NumPy arrays (.npy) directly from GCS - fastest format!")
    
    return datasets


def delete_dataset_version(version: str, gcs_bucket: str, project: str, region: str, 
                          delete_local: bool = True, registry_file: str = 'datasets_registry.yaml'):
    """
    Delete a dataset version from GCS, Vertex AI, local storage, and registry.
    
    Args:
        version: Dataset version to delete
        gcs_bucket: GCS bucket name
        project: GCP project ID
        region: GCP region
        delete_local: Whether to delete local files
        registry_file: Path to registry file
    """
    print(f"\n" + "="*80)
    print(f"   DELETING DATASET VERSION: {version}")
    print("="*80)
    
    # Load registry
    registry_path = Path(registry_file)
    registry = {}
    if registry_path.exists():
        with open(registry_path, 'r') as f:
            registry = yaml.safe_load(f) or {}
    
    if version not in registry:
        print(f"\n⚠️  Version '{version}' not found in registry.")
        print(f"   Available versions: {', '.join(registry.keys()) if registry else 'None'}")
        return False
    
    version_info = registry[version]
    errors = []
    
    # Step 1: Delete Managed Datasets
    print(f"\n🗑️  Step 1: Deleting Vertex AI Managed Datasets...")
    try:
        from google.cloud import aiplatform
        aiplatform.init(project=project, location=region)
        
        for ds_type in ['raw_dataset_id', 'processed_dataset_id']:
            if ds_type in version_info and version_info[ds_type] != 'N/A':
                try:
                    dataset = aiplatform.TabularDataset(version_info[ds_type])
                    dataset.delete()
                    print(f"   ✅ Deleted {ds_type}: {version_info[ds_type]}")
                except Exception as e:
                    error_msg = f"Could not delete {ds_type}: {e}"
                    print(f"   ⚠️  {error_msg}")
                    errors.append(error_msg)
    except ImportError:
        print("   ⚠️  google-cloud-aiplatform not installed, skipping Managed Dataset deletion")
    except Exception as e:
        error_msg = f"Error deleting Managed Datasets: {e}"
        print(f"   ⚠️  {error_msg}")
        errors.append(error_msg)
    
    # Step 2: Delete GCS files
    print(f"\n🗑️  Step 2: Deleting GCS files...")
    try:
        from google.cloud import storage
        client = storage.Client(project=project)
        bucket = client.bucket(gcs_bucket)
        
        prefix = f"datasets/{version}/"
        blobs = list(bucket.list_blobs(prefix=prefix))
        
        if blobs:
            for blob in blobs:
                blob.delete()
            print(f"   ✅ Deleted {len(blobs)} files from gs://{gcs_bucket}/{prefix}")
        else:
            print(f"   ℹ️  No files found at gs://{gcs_bucket}/{prefix}")
    except Exception as e:
        error_msg = f"Error deleting GCS files: {e}"
        print(f"   ⚠️  {error_msg}")
        errors.append(error_msg)
    
    # Step 3: Delete local files
    if delete_local:
        print(f"\n🗑️  Step 3: Deleting local files...")
        local_path = Path(version_info.get('local_path', f'data/datasets/{version}'))
        if local_path.exists():
            import shutil
            shutil.rmtree(local_path)
            print(f"   ✅ Deleted local directory: {local_path}")
        else:
            print(f"   ℹ️  Local directory not found: {local_path}")
    else:
        print(f"\n⏭️  Step 3: Skipping local file deletion (--keep-local)")
    
    # Step 4: Remove from registry
    print(f"\n🗑️  Step 4: Removing from registry...")
    del registry[version]
    with open(registry_path, 'w') as f:
        yaml.dump(registry, f, default_flow_style=False, sort_keys=False)
    print(f"   ✅ Removed '{version}' from {registry_path}")
    
    # Summary
    print(f"\n" + "="*80)
    if errors:
        print(f"⚠️  Version '{version}' deleted WITH WARNINGS")
        print(f"\n⚠️  Errors encountered:")
        for error in errors:
            print(f"   - {error}")
    else:
        print(f"✅ Version '{version}' completely deleted!")
    print("="*80)
    
    remaining = list(registry.keys())
    if remaining:
        print(f"\n📊 Remaining versions: {', '.join(remaining)}")
    else:
        print(f"\n📊 No dataset versions remaining.")
    
    return len(errors) == 0


def save_dataset_registry(datasets: dict, registry_file: str = 'datasets_registry.yaml'):
    """
    Save or update dataset registry file.
    
    Args:
        datasets: Dataset information to save
        registry_file: Path to registry file
    """
    registry_path = Path(registry_file)
    
    # Load existing registry
    if registry_path.exists():
        with open(registry_path, 'r') as f:
            registry = yaml.safe_load(f) or {}
    else:
        registry = {}
    
    # Add or update version
    version = datasets['version']
    registry[version] = datasets
    
    # Save registry
    with open(registry_path, 'w') as f:
        yaml.dump(registry, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n💾 Dataset registry updated: {registry_path}")
    print(f"   Versions available: {', '.join(registry.keys())}")


def main():
    # Load environment variables from .env file
    from dotenv import load_dotenv
    project_root = Path(__file__).parent.parent.parent
    env_file = project_root / '.env'
    if env_file.exists():
        load_dotenv(env_file)
        print(f"✅ Loaded environment from: {env_file}")
    else:
        print(f"⚠️  Warning: .env file not found at {env_file}")
    
    parser = argparse.ArgumentParser(
        description='Generate versioned datasets for Vertex AI cloud training (supports multiple model types)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate v1 dataset and upload to GCS
  python scripts/05_deployment/generate_dataset.py --version v1
  
  # Use existing local data, just upload to GCS
  python scripts/05_deployment/generate_dataset.py --version v1 --use-existing
  
  # Specify custom GCS bucket
  python scripts/05_deployment/generate_dataset.py --version v2 --gcs-bucket my-bucket
  
  # Delete a single version
  python scripts/05_deployment/generate_dataset.py --delete-versions v1
  
  # Delete multiple versions
  python scripts/05_deployment/generate_dataset.py --delete-versions v1 v2 v3
        """
    )
    
    parser.add_argument('--version', type=str, required=False,
                       help='Dataset version (e.g., v1, v2, nov2024)')
    parser.add_argument('--model-type', type=str, default='tft',
                       help='Model type for dataset versioning (default: tft)')
    parser.add_argument('--delete-versions', type=str, nargs='+',
                       help='Delete specified dataset versions (e.g., --delete-versions v1 v2)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to model config YAML (default: auto-select based on --model-type)')
    parser.add_argument('--gcs-bucket', type=str, 
                       default=os.getenv('GCP_PROJECT_ID', 'your-project') + '-models',
                       help='GCS bucket name (default: {GCP_PROJECT_ID}-models)')
    parser.add_argument('--project', type=str, 
                       default=os.getenv('GCP_PROJECT_ID', 'your-project'),
                       help='GCP project ID (default: from GCP_PROJECT_ID env var)')
    parser.add_argument('--region', type=str, default='us-central1',
                       help='GCP region')
    parser.add_argument('--use-existing', action='store_true',
                       help='Skip data generation, use existing local data')
    parser.add_argument('--base-dir', type=str, default='data/datasets',
                       help='Base directory for dataset storage')
    
    args = parser.parse_args()
    
    # Auto-select config file based on model type if not specified
    if args.config is None:
        config_map = {
            'tft': 'configs/model_tft_config.yaml',
            'decoder_transformer': 'configs/model_decoder_config.yaml',
            'lstm': 'configs/model_lstm_config.yaml',
            'transformer': 'configs/model_transformer_config.yaml'
        }
        args.config = config_map.get(args.model_type, 'configs/model_tft_config.yaml')
        print(f"\n📋 Auto-selected config: {args.config} (for model type: {args.model_type})")
    
    # Handle deletion mode
    if args.delete_versions:
        print(f"\n🔴 DELETION MODE: Deleting {len(args.delete_versions)} version(s)")
        print(f"   Versions to delete: {', '.join(args.delete_versions)}")
        
        confirm = input(f"\n⚠️  This will permanently delete data from GCS, Vertex AI, and local storage.\n   Continue? [y/N]: ")
        if confirm.lower() != 'y':
            print("\n❌ Deletion cancelled.")
            return
        
        all_success = True
        for version in args.delete_versions:
            success = delete_dataset_version(
                version=version,
                gcs_bucket=args.gcs_bucket,
                project=args.project,
                region=args.region,
                delete_local=True
            )
            if not success:
                all_success = False
        
        if all_success:
            print(f"\n✅ All versions deleted successfully!")
        else:
            print(f"\n⚠️  Some versions had errors during deletion.")
        return
    
    # Generation mode requires --version
    if not args.version:
        parser.error("--version is required for dataset generation (or use --delete-versions to delete)")
    
    # Validate version naming
    poor_names = ['test', 'temp', 'tmp', 'debug', 'dev', 'latest', 'current']
    if args.version.lower() in poor_names:
        print(f"\n⚠️  Warning: '{args.version}' is not a good version name.")
        print(f"   Recommended: Use semantic versions like 'v1', 'v2', or dates like '2024-11'")
        print(f"   Your current version: {args.version}")
        confirm = input("\n   Continue anyway? [y/N]: ")
        if confirm.lower() != 'y':
            print("\n❌ Cancelled.")
            return
    
    # Calculate paths with model type
    version_dir = Path(args.base_dir) / args.model_type / args.version
    registry_file = Path('datasets_registry.yaml')
    
    # Full version key includes model type (e.g., 'tft/v1')
    full_version_key = f"{args.model_type}/{args.version}"
    
    # Check local directory
    local_exists = version_dir.exists()
    
    # Check registry
    registry_exists = False
    registry = {}
    if registry_file.exists():
        with open(registry_file, 'r') as f:
            registry = yaml.safe_load(f) or {}
        registry_exists = full_version_key in registry
    
    # Check GCS (via registry)
    gcs_exists = registry_exists  # If in registry, it's in GCS
    
    # Handle conflicts
    if (local_exists or registry_exists) and not args.use_existing:
        print(f"\n⚠️  Dataset version '{full_version_key}' already exists:")
        if local_exists:
            print(f"   📂 Local: {version_dir}")
        if registry_exists:
            print(f"   📊 Registry: datasets_registry.yaml")
        if gcs_exists:
            gcs_path = registry.get(full_version_key, {}).get('gcs_base_path', 'Unknown')
            print(f"   ☁️  GCS: {gcs_path}")
        
        print(f"\n💡 Options:")
        print(f"   1. Use existing data (recommended): --use-existing")
        print(f"   2. Create new version: --version v{int(args.version[1:])+1 if args.version.startswith('v') and args.version[1:].isdigit() else '2'}")
        print(f"   3. Overwrite {full_version_key} (destructive): Continue below")
        
        response = input(f"\n⚠️  Overwrite version '{full_version_key}'? [y/N]: ")
        if response.lower() != 'y':
            print("\n❌ Aborted. No changes made.")
            return
        else:
            print(f"\n⚠️  Proceeding with overwrite of version '{full_version_key}'...")
    
    try:
        # Step 1: Generate data locally
        if args.use_existing:
            print(f"\n📂 Using existing data at {version_dir}")
            manifest_file = version_dir / 'manifest.yaml'
            if not manifest_file.exists():
                raise FileNotFoundError(f"Manifest not found: {manifest_file}")
            with open(manifest_file, 'r') as f:
                manifest = yaml.safe_load(f)
        else:
            manifest = generate_data(args.config, args.version, args.model_type, args.base_dir)
        
        datasets_info = {
            'version': f"{args.model_type}/{args.version}",
            'model_type': args.model_type,
            'version_number': args.version,
            'local_path': str(version_dir),
            'created': manifest.get('created', datetime.now().isoformat())
        }
        
        # Step 2: Upload to GCS
        gcs_base_path = upload_to_gcs(f"{args.model_type}/{args.version}", args.gcs_bucket, args.base_dir)
        datasets_info['gcs_base_path'] = gcs_base_path
        
        # Step 3: Create Managed Datasets
        managed_datasets = create_managed_dataset(
            f"{args.model_type}/{args.version}",  # Full version path (e.g., 'tft/v1')
            args.gcs_bucket,
            args.project,
            args.region,
            args.model_type
        )
        
        if managed_datasets:
            datasets_info.update(managed_datasets)
        
        # Step 4: Save to registry
        save_dataset_registry(datasets_info)
        
        # Final summary
        print("\n" + "="*80)
        print("   ✨ DATASET READY FOR CLOUD TRAINING!")
        print("="*80)
        print(f"\n📂 Local path: {version_dir}")
        print(f"☁️  GCS path: {datasets_info.get('gcs_base_path', 'N/A')}")
        print(f"\n📊 Managed Datasets:")
        print(f"   Raw: {datasets_info.get('raw_dataset_id', 'N/A')}")
        print(f"   Processed: {datasets_info.get('processed_dataset_id', 'N/A')}")
        
        print(f"\n💡 Next steps:")
        print(f"   # Single training job:")
        print(f"   python scripts/05_deployment/submit_job.py --dataset-version {args.version} --model-type {args.model_type}")
        print(f"   ")
        print(f"   # Hyperparameter tuning (all trials share this dataset):")
        print(f"   python scripts/05_deployment/submit_hp_tuning.py --dataset-version {args.version} --model-type {args.model_type}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()