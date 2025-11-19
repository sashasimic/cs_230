# Cloud Deployment Scripts

Scripts for training models on Google Cloud Vertex AI at scale.

## Overview

Deploy and train models on Vertex AI with:
- **GPU/TPU acceleration** - 10-50x faster than CPU
- **Hyperparameter tuning** - Automated Bayesian optimization
- **Parallel execution** - Run multiple experiments simultaneously
- **Versioned datasets** - Reproducible training with dataset snapshots
- **Cost optimization** - Spot instances, auto-scaling

## Prerequisites

1. **GCP Account** with billing enabled
2. **Environment variables** in `.env`:
   ```bash
   GCP_PROJECT_ID=your-project-id
   GCP_REGION=us-central1
   ```
3. **gcloud CLI** installed and authenticated
4. **Docker** installed (for building images)

## Setup

### Step 1: Configure GCP

```bash
# Run setup script (creates bucket, service account, IAM roles)
bash scripts/05_deployment/setup_gcp.sh
```

**This creates:**
- GCS bucket: `gs://{project-id}-models`
- Service account: `vertex-model-trainer@{project}.iam.gserviceaccount.com`
- IAM roles: AI Platform, Storage, BigQuery, Logging
- Docker configuration for GCR

### Step 2: Build and Test Docker Image

**Recommended Workflow (Hybrid Approach):**

1. **Develop locally with venv** (fast iteration)
2. **Test in Docker** (validate before deploying)
3. **Deploy to GCS** (confident it will work)

#### Option A: Automated Test Script (Recommended)

```bash
# Run automated test (builds, tests, and shows deployment commands)
bash scripts/05_deployment/test_docker_local.sh
```

This script will:
- ✅ Build Docker image for linux/amd64
- ✅ Check for required data
- ✅ Run training in Docker with mounted volumes
- ✅ Show deployment commands if successful

#### Option B: Manual Testing

```bash
# Build for linux/amd64 (required for GCP)
docker build --platform linux/amd64 \
  -f scripts/05_deployment/Dockerfile.vertex \
  -t gcr.io/{project-id}/model-trainer .

# Test locally before pushing
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  --env-file .env \
  gcr.io/{project-id}/model-trainer \
  --config configs/model_tft_config.yaml

# If successful, push to GCR
docker push gcr.io/{project-id}/model-trainer
```

**Note:** Building for `linux/amd64` is critical for GCP compatibility, especially on Apple Silicon Macs.

## Scripts

### Dataset Management

#### `generate_dataset.py` - Create Versioned Datasets

Create reproducible dataset snapshots for training.

```bash
# Generate v1 dataset
python scripts/05_deployment/generate_dataset.py --version v1

# Use existing local data (skip regeneration)
python scripts/05_deployment/generate_dataset.py --version v1 --use-existing

# Delete old versions
python scripts/05_deployment/generate_dataset.py --delete-versions v1 v2
```

**Output:**
- **Local:** `data/datasets/v1/` (raw + processed + manifest)
- **GCS:** `gs://{bucket}/datasets/v1/`
- **Registry:** `datasets_registry.yaml` (tracks all versions)

**Why version datasets?**
- Reproducibility: Same data for all experiments
- Speed: No need to regenerate for each job
- Sharing: Multiple jobs can use same dataset
- Rollback: Easy to revert to previous data

### Job Submission

#### `submit_job.py` - Single Training Job

Submit a single training job to Vertex AI.

```bash
# Submit job with dataset v1
python scripts/05_deployment/submit_job.py --dataset-version v1

# Custom job name
python scripts/05_deployment/submit_job.py \
  --dataset-version v1 \
  --job-name my-experiment
```

**Default configuration:**
- Machine: `e2-highmem-4` (4 vCPUs, 32GB RAM)
- Accelerator: None (CPU-only)
- Cost: ~$0.24/hr
- Training time: ~50-70 hours

**GPU configuration (commented in script):**
- Machine: `n1-standard-4` (4 vCPUs, 15GB RAM)
- Accelerator: `NVIDIA_TESLA_T4` (1 GPU)
- Cost: ~$0.54/hr
- Training time: ~5-10 hours (10x faster!)

#### `submit_parallel.py` - Parallel Grid Search

Submit multiple jobs with different hyperparameters.

```bash
# Run grid search (20 configurations)
python scripts/05_deployment/submit_parallel.py
```

**Hyperparameter grid:**
```python
HYPERPARAMETER_GRID = {
    'hidden_size': [32, 64, 128],
    'lstm_layers': [1, 2],
    'learning_rate': [0.0001, 0.001, 0.01],
    'dropout': [0.1, 0.2],
    'batch_size': [64, 128],
    'lookback_window': [96, 192],
}
```

**Output:**
- Jobs: `model-parallel-{timestamp}-01`, `model-parallel-{timestamp}-02`, ...
- Models: `gs://{bucket}/models/model-parallel-{timestamp}-{i}/`

#### `submit_hp_tuning.py` - Bayesian Optimization

Vertex AI native hyperparameter tuning with intelligent search.

```bash
# Run HP tuning (20 trials, 4 parallel)
python scripts/05_deployment/submit_hp_tuning.py
```

**Features:**
- **Bayesian optimization**: Smarter than grid search
- **Parallel trials**: Run 4 trials simultaneously
- **Early stopping**: Stop unpromising trials early
- **Metric optimization**: Minimize validation loss

**Tunable parameters:**
- `hidden_size`: [32, 64, 128, 256]
- `lstm_layers`: [1, 2, 3]
- `learning_rate`: [0.0001, 0.001, 0.01]
- `dropout`: [0.1, 0.2, 0.3]
- `batch_size`: [64, 128, 256]

#### `view_hp_results.py` - View Tuning Results

Analyze hyperparameter tuning results.

```bash
# List recent HP tuning jobs
python scripts/05_deployment/view_hp_results.py

# View specific job results
python scripts/05_deployment/view_hp_results.py --job-id {job-id}
```

**Output:**
- Trial rankings by validation loss
- Best hyperparameters
- Convergence plots
- Parameter importance

### Training Wrapper

#### `train_vertex.py` - Cloud Training Wrapper

Generic wrapper called by Vertex AI (works with any model type).

**Features:**
- Downloads dataset from GCS
- Calls model-specific training script (e.g., `tft/tft_train.py`)
- Uploads results to GCS
- Reports metrics for HP tuning
- Supports multiple model types via `--model_type` flag

**Model type support:**
```bash
# TFT model (default)
--model_type tft  # Looks for scripts/03_training/tft/tft_train.py

# LSTM model (future)
--model_type lstm  # Looks for scripts/03_training/lstm/lstm_train.py
```

## Workflow

### Complete Training Pipeline

```
1. Generate dataset
   python scripts/05_deployment/generate_dataset.py --version v1
   ↓
2. Upload to GCS
   (automatic in step 1)
   ↓
3. Submit training job(s)
   python scripts/05_deployment/submit_job.py --dataset-version v1
   ↓
4. Monitor in Vertex AI Console
   https://console.cloud.google.com/vertex-ai/training/custom-jobs
   ↓
5. Download trained models from GCS
   gsutil cp gs://{bucket}/models/{job-name}/tft_best.pt models/
```

### Hyperparameter Tuning Workflow

```
1. Generate dataset (once)
   python scripts/05_deployment/generate_dataset.py --version v1
   ↓
2. Submit HP tuning job
   python scripts/05_deployment/submit_hp_tuning.py
   ↓
3. Wait for completion (~4-8 hours)
   ↓
4. View results
   python scripts/05_deployment/view_hp_results.py
   ↓
5. Use best hyperparameters in config
   Update configs/model_tft_config.yaml
   ↓
6. Train final model
   python scripts/05_deployment/submit_job.py --dataset-version v1
```

## Monitoring

### Vertex AI Console

View jobs at:
```
https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={project-id}
```

**Monitor:**
- Job status (running, succeeded, failed)
- Resource usage (CPU, memory, GPU)
- Logs (stdout, stderr)
- Metrics (loss, accuracy)
- Duration and cost

### GCS Bucket

View outputs at:
```
https://console.cloud.google.com/storage/browser/{project-id}-models
```

**Structure:**
```
gs://{bucket}/
├── datasets/
│   ├── v1/
│   │   ├── raw/
│   │   ├── processed/
│   │   └── manifest.yaml
│   └── v2/
├── models/
│   ├── {job-name}/
│   │   └── tft_best.pt
│   └── {hp-job-name}/
│       ├── trial_1/tft_best.pt
│       ├── trial_2/tft_best.pt
│       └── ...
└── logs/
```

## Cost Optimization

### Machine Types

| Type | vCPUs | RAM | GPU | Cost/hr | Use Case |
|------|-------|-----|-----|---------|----------|
| `e2-standard-4` | 4 | 16GB | None | $0.13 | Small experiments |
| `e2-highmem-4` | 4 | 32GB | None | $0.24 | CPU training |
| `n1-standard-4` | 4 | 15GB | T4 | $0.54 | GPU training |
| `n1-standard-8` | 8 | 30GB | T4 | $0.73 | Large models |

### GPU Types

| GPU | Memory | Cost/hr | Speed | Use Case |
|-----|--------|---------|-------|----------|
| T4 | 16GB | $0.35 | 1x | Standard |
| V100 | 16GB | $2.48 | 3x | Large models |
| A100 | 40GB | $3.67 | 5x | Huge models |

### Spot Instances

- **Discount**: 60-91% off regular price
- **Risk**: May be preempted (interrupted)
- **Best for**: Experiments, HP tuning, non-critical jobs

**Note:** Spot instances not yet supported via Python SDK (use gcloud CLI)

### Cost Estimation

**CPU-only training:**
- Machine: `e2-highmem-4` ($0.24/hr)
- Duration: ~60 hours
- **Total: ~$14.40**

**GPU training:**
- Machine: `n1-standard-4` ($0.19/hr)
- GPU: T4 ($0.35/hr)
- Duration: ~6 hours
- **Total: ~$3.24** (4.5x cheaper + 10x faster!)

**HP tuning (20 trials, 4 parallel):**
- Per trial: ~$3.24
- Total trials: 20
- **Total: ~$64.80**

## Configuration

### Environment Variables (`.env`)
```bash
GCP_PROJECT_ID=your-project-id
GCP_REGION=us-central1
```

### Model Config (`configs/model_tft_config.yaml`)
```yaml
model:
  hidden_size: 64
  lstm_layers: 2
  attention_heads: 4
  dropout: 0.2

training:
  epochs: 100
  batch_size: 128
  learning_rate: 0.001
```

## Troubleshooting

**Issue: Docker build fails**
- Ensure `--platform linux/amd64` is specified
- Check Dockerfile.vertex exists
- Verify Docker daemon is running

**Issue: Image push fails**
- Run `gcloud auth configure-docker`
- Check GCR API is enabled
- Verify project ID is correct

**Issue: Job submission fails**
- Check service account has required permissions
- Verify GCS bucket exists
- Ensure dataset version exists (if using `--dataset-version`)

**Issue: Job fails during training**
- Check logs in Vertex AI Console
- Verify dataset is complete
- Check for out-of-memory errors (reduce batch size)

**Issue: Quota exceeded**
- Request quota increase: https://console.cloud.google.com/iam-admin/quotas
- Use CPU instead of GPU
- Reduce parallel trials

## Best Practices

1. **Version datasets**: Always use versioned datasets for reproducibility
2. **Start small**: Test with CPU before scaling to GPU
3. **Monitor costs**: Set up billing alerts in GCP Console
4. **Use HP tuning**: Don't manually grid search - use Bayesian optimization
5. **Save checkpoints**: Enable checkpointing for long-running jobs
6. **Clean up**: Delete old jobs and datasets to save storage costs

## Next Steps

1. **Run setup**: `bash scripts/05_deployment/setup_gcp.sh`
2. **Build image**: Build and push Docker image
3. **Generate dataset**: Create v1 dataset
4. **Submit job**: Start your first training job
5. **Monitor**: Watch progress in Vertex AI Console
6. **Iterate**: Use HP tuning to find best hyperparameters