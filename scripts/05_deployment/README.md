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
- **TensorBoard permissions:** Configures default compute service account for TensorBoard integration
- Docker configuration for GCR

**Service Accounts Configured:**
1. **`vertex-model-trainer@{project}.iam.gserviceaccount.com`** - Custom service account (legacy, not used)
2. **`{project-number}-compute@developer.gserviceaccount.com`** - Default compute SA with:
   - `roles/storage.objectAdmin` on GCS bucket (for dataset access)
   - `roles/aiplatform.user` (for TensorBoard logging)

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

Create reproducible dataset snapshots for training with model-type-specific data loaders.

```bash
# Generate TFT dataset v1
python scripts/05_deployment/generate_dataset.py --version v1 --model-type tft

# Generate decoder_transformer dataset v1
python scripts/05_deployment/generate_dataset.py --version v1 --model-type decoder_transformer

# Generate LSTM dataset v1 (when LSTM loader exists)
python scripts/05_deployment/generate_dataset.py --version v1 --model-type lstm

# Use existing local data (skip regeneration)
python scripts/05_deployment/generate_dataset.py --version v1 --model-type tft --use-existing

# Delete old versions
python scripts/05_deployment/generate_dataset.py --delete-versions v1 v2
```

**Model-Type-Specific Data Loaders:**
The script automatically selects the correct data loader based on `--model-type`:
- `tft` → `scripts/02_features/tft/tft_data_loader.py` (MultiTickerDataLoader)
- `decoder_transformer` → `scripts/02_features/decoder_transformer/decoder_transformer_data_loader.py`
- `lstm` → `scripts/02_features/lstm/lstm_data_loader.py` (LSTMDataLoader)
- `transformer` → `scripts/02_features/transformer/transformer_data_loader.py` (TransformerDataLoader)

**Output:**
- **Local:** `data/datasets/{model_type}/{version}/` (raw + processed + manifest)
- **GCS:** `gs://{bucket}/datasets/{model_type}/{version}/`
- **Registry:** `datasets_registry.yaml` (tracks all versions)

**Example Structure:**
```
data/datasets/
├── tft/
│   ├── v1/
│   │   ├── raw/tft_features.csv
│   │   ├── processed/X_*.npy
│   │   └── manifest.yaml
│   └── v2/
├── lstm/
│   └── v1/
└── transformer/
    └── v1/
```

**Why version datasets?**
- Reproducibility: Same data for all experiments
- Speed: No need to regenerate for each job
- Sharing: Multiple jobs can use same dataset
- Rollback: Easy to revert to previous data
- Model isolation: Different models can have different feature engineering

#### `delete_dataset.py` - Delete Dataset Versions

Delete dataset versions from all locations (local, GCS, Vertex AI Managed Datasets, and registry).

```bash
# Delete specific version (with confirmation prompt)
python scripts/05_deployment/delete_dataset.py --version v3 --model-type tft

# Delete without confirmation
python scripts/05_deployment/delete_dataset.py --version v3 --model-type tft --yes

# Delete LSTM dataset
python scripts/05_deployment/delete_dataset.py --version v1 --model-type lstm -y
```

**What gets deleted:**
1. **Local disk:** `data/datasets/{model_type}/{version}/`
2. **GCS:** `gs://{bucket}/datasets/{model_type}/{version}/`
3. **Managed Dataset:** Vertex AI TabularDataset (e.g., `tft-v3-features-csv`)
4. **Registry:** Entry in `datasets_registry.yaml`

**Options:**
- `--version`: Dataset version to delete (e.g., `v1`, `v2`, `v3`)
- `--model-type`: Model type (default: `tft`)
- `--yes`, `-y`: Skip confirmation prompt

**Safety:**
- Prompts for confirmation by default
- Shows what will be deleted before proceeding
- Gracefully handles missing files/datasets

**Example output:**
```
🗑️  Deleting dataset: tft/v3
================================================================================
✅ Deleted local: data/datasets/tft/v3
☁️  Deleting from GCS: gs://bucket/datasets/tft/v3
✅ Deleted from GCS
🗂️  Deleting Managed Dataset: tft-v3-features-csv
✅ Deleted Managed Dataset
✅ Removed from registry: tft/v3
================================================================================
✅ Deletion complete!
```

### Job Submission

#### `submit_job.py` - Single Training Job

Submit a single training job to Vertex AI.

```bash
# Submit job with TFT dataset v1
python scripts/05_deployment/submit_job.py --dataset-version v1 --model-type tft

# Submit decoder_transformer job
python scripts/05_deployment/submit_job.py --dataset-version v1 --model-type decoder_transformer

# Custom job name
python scripts/05_deployment/submit_job.py \
  --dataset-version v1 \
  --model-type decoder_transformer \
  --job-name my-experiment

# Submit LSTM model (when LSTM implementation exists)
python scripts/05_deployment/submit_job.py --dataset-version v1 --model-type lstm
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
# Run HP tuning for TFT model (20 trials, 4 parallel)
python scripts/05_deployment/submit_hp_tuning.py --dataset-version v1 --model-type tft

# With custom job name
python scripts/05_deployment/submit_hp_tuning.py \
  --dataset-version v1 \
  --model-type tft \
  --job-name tft-hp-tuning-experiment
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

## FinCast Deployment

### Overview

Deploy decoder transformers with FinCast Foundation Model (FFM) on Vertex AI. FinCast adds 991M pre-trained parameters for enhanced price series processing.

**Key Features:**
- ✅ **Auto CPU/GPU detection**: Works on both CPU and GPU instances
- ✅ **Checkpoint management**: Downloads 3.97 GB checkpoint from GCS automatically
- ✅ **Memory efficient**: Checkpoint excluded from Docker image
- ✅ **Transfer learning**: Only trains 900K projection parameters, FFM stays frozen

### Prerequisites

1. **FinCast checkpoint uploaded to GCS**:
   ```bash
   # Upload checkpoint (one-time, ~3.97 GB)
   gsutil cp external/fincast/checkpoints/v1.pth \
       gs://YOUR-PROJECT-ID-models/models/fincast/v1.pth
   ```

2. **FinCast enabled in config** (`configs/model_decoder_config.yaml`):
   ```yaml
   fincast:
     enabled: true
     checkpoint_path: 'external/fincast/checkpoints/v1.pth'
     output_dim: 128
     freeze_backbone: true
     lr_scale: 0.2
   ```

3. **Dataset version created**:
   ```bash
   # Create dataset for decoder transformer
   python scripts/05_deployment/generate_dataset.py \
       --version v3 \
       --model-type decoder_transformer
   ```

### Deployment

#### **CPU Training (Testing)**

```bash
# Submit CPU job (slower, cheaper, good for testing)
python scripts/05_deployment/submit_job.py \
    --dataset-version v3 \
    --model-type decoder_transformer \
    --machine-type e2-highmem-8 \
    --job-name decoder-fincast-cpu-test
```

**Performance**:
- **Time per epoch**: 30-60 minutes
- **Cost**: ~$0.38/hr
- **Use case**: Quick validation, debugging

#### **GPU Training (Production)**

```bash
# Submit GPU job (faster, better for production)
python scripts/05_deployment/submit_job.py \
    --dataset-version v3 \
    --model-type decoder_transformer \
    --machine-type n1-standard-4 \
    --accelerator NVIDIA_TESLA_T4 \
    --accelerator-count 1 \
    --job-name decoder-fincast-gpu
```

**Performance**:
- **Time per epoch**: 5-10 minutes (5-10x faster than CPU)
- **Cost**: ~$0.54/hr (GPU T4)
- **Use case**: Production training, hyperparameter tuning

### What Happens During Training

1. **Container starts**
   - FinCast submodule copied to image (code only, no checkpoint)
   - Dependencies installed from `external/fincast/requirement_v2.txt`

2. **Checkpoint download** (automatic)
   - Checks if checkpoint exists locally
   - Downloads from `gs://YOUR-BUCKET/models/fincast/v1.pth` if missing
   - Takes 2-5 minutes on first run
   - Cached for subsequent epochs

3. **Model initialization**
   - Loads 3.97 GB checkpoint from local path
   - Auto-detects CPU vs GPU:
     ```python
     backend="cpu" if not torch.cuda.is_available() else "gpu"
     ```
   - Creates projection layer (1280 → 128 dims)
   - Freezes 991M FFM parameters
   - Adds LayerNorm for feature stability

4. **Training**
   - Trains only 900K projection + decoder parameters
   - Learning rate auto-adjusted (5x lower for FinCast)
   - Gradient norms stabilized with LayerNorm

### Logs

Expected output:
```
🔧 FinCast enabled in config, downloading checkpoint...
📥 Downloading FinCast checkpoint from GCS...
   gs://your-bucket/models/fincast/v1.pth
   → external/fincast/checkpoints/v1.pth
   Size: ~3.97 GB (may take 2-5 minutes)

✅ FinCast checkpoint downloaded (3979 MB)

🔧 Initializing model with FinCast integration...
✅ Initializing pre-trained FinCast (FFM) model
   Architecture: 50 layers, 1280 dims, 16 heads
   Context length: 512
   Checkpoint: external/fincast/checkpoints/v1.pth
   Loading pre-trained weights (3.97 GB, may take 30-60 seconds on CPU)...
   ✅ Pre-trained FinCast loaded successfully (4.1s)
   Adding projection: 1280 -> 128 dims
   🔒 FinCast backbone frozen (no gradients)

🔧 FinCast Integration:
   Price series: 27
   FinCast output dim: 128 (projected from 1280)
   FinCast features: 3456
   Rest features: 91
   Total augmented: 3547
   ⚖️  Added LayerNorm(3547) for feature normalization

🔧 Model Parameters:
   Total: 992,340,504
   Trainable: 903,544  ← Only 900K trainable!

ℹ️  Adjusting learning rate for FinCast: 5.00e-05 → 1.00e-05 (0.2x)
```

### Cost Comparison

| Configuration | Machine | GPU | Time/Epoch | Cost/Epoch | 100 Epochs |
|--------------|---------|-----|------------|------------|------------|
| **CPU** | e2-highmem-8 | None | 45 min | $0.28 | $28 |
| **GPU T4** | n1-standard-4 | T4 | 7 min | $0.06 | $6 |
| **GPU V100** | n1-standard-4 | V100 | 3 min | $0.12 | $12 |

**Recommendation**: Use GPU T4 for FinCast training (5-10x faster, lower total cost).

### Hyperparameter Tuning with FinCast

```bash
# Tune FinCast-specific parameters
python scripts/05_deployment/submit_hp_tuning.py \
    --dataset-version v3 \
    --model-type decoder_transformer \
    --config configs/hp_decoder_fincast.yaml \
    --max-trials 20 \
    --parallel-trials 5 \
    --machine-type n1-standard-4 \
    --accelerator NVIDIA_TESLA_T4 \
    --job-name decoder-fincast-hptune
```

**Tunable parameters** (in config):
- `fincast.output_dim`: [64, 128, 256]
- `fincast.lr_scale`: [0.1, 0.2, 0.5]
- `model.d_model`: [64, 96, 128]
- `model.n_layers`: [2, 3, 4]
- `training.learning_rate`: [1e-5, 5e-5, 1e-4]

### Troubleshooting

**Issue: Checkpoint not found**
```
❌ FinCast checkpoint not found in GCS!
   gs://your-bucket/models/fincast/v1.pth
```

**Fix**: Upload checkpoint to GCS:
```bash
gsutil cp external/fincast/checkpoints/v1.pth \
    gs://YOUR-BUCKET/models/fincast/v1.pth
```

**Issue: Out of memory on CPU**
```
RuntimeError: [enforce fail at alloc_cpu.cpp:...] . DefaultCPUAllocator: can't allocate memory
```

**Fix**: Use higher-memory machine:
```bash
--machine-type e2-highmem-16  # 128 GB RAM
```

**Issue: CUDA out of memory**
```
RuntimeError: CUDA out of memory. Tried to allocate 3.97 GB...
```

**Fix**: Reduce batch size in config:
```yaml
training:
  batch_size: 32  # or 16
```

**Issue: Gradients exploding**
```
Grad Norm (unclipped): avg=158.6, max=764.4
```

**Fix**: Already handled by:
1. LayerNorm after feature augmentation ✅
2. Lower learning rate (`lr_scale: 0.2`) ✅
3. Gradient clipping (`gradient_clip_norm: 1.0`) ✅

If still seeing high gradients, reduce `lr_scale` further:
```yaml
fincast:
  lr_scale: 0.1  # Even lower
```

### Best Practices

1. **Always test locally first** before deploying to Vertex AI
2. **Use GPU for production** - Total cost is lower despite higher $/hr
3. **Monitor first epoch carefully** - Should stabilize within 2-3 epochs
4. **Check gradient norms** - Should be avg < 10, max < 50
5. **Version your datasets** - FinCast requires specific feature engineering
6. **Cache checkpoint** - First epoch downloads checkpoint, subsequent runs reuse it

## Workflow

### Complete Training Pipeline

```
1. Generate dataset (with model type)
   python scripts/05_deployment/generate_dataset.py --version v1 --model-type tft
   ↓
2. Upload to GCS
   (automatic in step 1)
   ↓
3. Submit training job(s)
   python scripts/05_deployment/submit_job.py --dataset-version v1 --model-type tft
   ↓
4. Monitor in Vertex AI Console
   https://console.cloud.google.com/vertex-ai/training/custom-jobs
   ↓
5. Download trained models from GCS
   gsutil cp gs://{bucket}/models/{job-name}/tft_best.pt models/
```

### Hyperparameter Tuning Workflow

```
1. Generate dataset (once, with model type)
   python scripts/05_deployment/generate_dataset.py --version v1 --model-type tft
   ↓
2. Submit HP tuning job
   python scripts/05_deployment/submit_hp_tuning.py --dataset-version v1
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
   python scripts/05_deployment/submit_job.py --dataset-version v1 --model-type tft
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

### TensorBoard on Vertex AI

**View TensorBoard experiments:**

1. **Navigate to Vertex AI TensorBoard:**
   ```
   https://console.cloud.google.com/vertex-ai/experiments/tensorboard?project={project-id}
   ```

2. **Find your experiment:**
   - Logs are organized by job name and timestamp
   - For decoder_transformer: Look for runs with `_ar` or `_tf` suffix
   - Path: `gs://{bucket}/tensorboard_logs/{job-name}/{timestamp}`

3. **Compare experiments:**
   - Select multiple runs to compare
   - View loss curves, metrics, gradients side-by-side
   - Filter by tags (e.g., `decoder_ar` vs `decoder_tf`)

**Metrics available:**
- Loss/train, Loss/val
- Metrics/MAE, Metrics/RMSE, Metrics/DirectionalAccuracy
- Gradients/Unclipped_Avg, Gradients/Clipped_Avg
- LR (learning rate)

**How It Works:**
- Vertex AI sets `AIP_TENSORBOARD_LOG_DIR` environment variable automatically
- Training scripts detect this and write TensorBoard logs to the specified directory
- Logs are auto-synced to the Vertex AI TensorBoard instance in real-time
- No manual GCS upload needed!

**Permissions Required:**
- Default compute service account (`{project-number}-compute@developer.gserviceaccount.com`) needs:
  - `roles/aiplatform.user` - To write to TensorBoard experiments
  - `roles/storage.objectAdmin` - To access training data from GCS
- These are automatically configured by `setup_gcp.sh`

**Troubleshooting:**
- If you see "⚠️ TensorBoard not available" in logs, check service account permissions
- Training continues without TensorBoard if permissions are missing
- Look for "📊 TensorBoard (Vertex AI):" message in logs to confirm it's working

### GCS Bucket

View outputs at:
```
https://console.cloud.google.com/storage/browser/{project-id}-models
```

**Structure:**
```
gs://{bucket}/
├── datasets/
│   ├── tft/                    # Model-type-specific datasets
│   │   ├── v1/
│   │   │   ├── raw/
│   │   │   ├── processed/
│   │   │   └── manifest.yaml
│   │   └── v2/
│   ├── decoder_transformer/
│   │   └── v1/
│   ├── lstm/
│   │   └── v1/
│   └── transformer/
│       └── v1/
├── models/
│   ├── {job-name}/
│   │   ├── tft_best.pt
│   │   ├── decoder_transformer_best_ar.pt   # Autoregressive eval
│   │   └── decoder_transformer_best_tf.pt   # Teacher forcing eval
│   └── {hp-job-name}/
│       ├── trial_1/decoder_transformer_best_ar.pt
│       ├── trial_2/decoder_transformer_best_ar.pt
│       └── ...
├── tensorboard_logs/
│   ├── {job-name}/
│   │   └── {timestamp}/   # TensorBoard events
│   └── {hp-job-name}/
│       ├── trial_1/{timestamp}/
│       └── trial_2/{timestamp}/
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
3. **Generate dataset**: `python scripts/05_deployment/generate_dataset.py --version v1 --model-type tft`
4. **Submit job**: `python scripts/05_deployment/submit_job.py --dataset-version v1 --model-type tft`
5. **Monitor**: Watch progress in Vertex AI Console
6. **Iterate**: Use HP tuning to find best hyperparameters