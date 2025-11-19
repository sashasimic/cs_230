# Model Training Scripts

Scripts for training machine learning models locally and in the cloud.

## Development Workflow (Hybrid Approach)

We use a **hybrid approach** for development and deployment:

### **Local Development (Fast)** ⚡

Use Python venv for rapid iteration:

```bash
# Activate virtual environment
source venv/bin/activate

# Train locally (fast, no Docker overhead)
python scripts/03_training/tft/tft_train_local.py
```

**Pros:**
- ⚡ Fast iteration (no Docker rebuild)
- 💻 Native performance
- 🛠️ Easy debugging

**Use for:**
- Daily development
- Quick experiments
- Debugging

### **Docker Testing (Before Deployment)** 🐳

Test in the same environment as GCS before deploying:

```bash
# Test in Docker (validates GCS compatibility)
bash scripts/05_deployment/test_docker_local.sh
```

**Pros:**
- ✅ Same environment as GCS
- ✅ Catches issues before deployment
- ✅ No wasted GCS costs on broken code

**Use for:**
- Before deploying to GCS
- Final validation
- Debugging cloud-specific issues

### **Cloud Training (Production)** ☁️

Deploy to Vertex AI for production training:

```bash
# Deploy to GCS (after Docker test passes)
python scripts/05_deployment/submit_job.py --dataset-version v1
```

**Pros:**
- 🚀 GPU/TPU acceleration
- 📊 Hyperparameter tuning
- 💾 Large-scale training

**Use for:**
- Final model training
- Hyperparameter tuning
- Large datasets

---

## Structure

```
03_training/
├── tft/
│   ├── tft_train.py         # Core TFT training logic
│   └── tft_train_local.py   # Local training wrapper
├── inspect_model.py          # Model inspection utilities
└── test_architectures.py    # Architecture testing
```

## TFT Training

### Local Training

#### `tft/tft_train_local.py` - Local Training Wrapper

Train TFT model locally without any cloud dependencies.

```bash
# Train with default config
python scripts/03_training/tft/tft_train_local.py

# Train with custom config
python scripts/03_training/tft/tft_train_local.py --config configs/my_config.yaml

# Force reload data from BigQuery
python scripts/03_training/tft/tft_train_local.py --reload
```

**Features:**
- No GCS/Vertex AI dependencies
- Loads data from `data/processed/` (or BigQuery with `--reload`)
- Saves model to `models/tft/tft_best.pt`
- Logs to `logs/` directory

**Requirements:**
- Environment variable: `GCP_PROJECT_ID` (for BigQuery access)
- Config file: `configs/model_tft_config.yaml`
- Data: Run `scripts/02_features/tft/tft_data_loader.py` first

### Core Training Logic

#### `tft/tft_train.py` - Shared Training Function

Core training logic used by both local and cloud training.

**Features:**
- Loads data from `data/processed/`
- Trains model with early stopping
- Saves checkpoints to `models/tft/tft_best.pt`
- Can be called by `tft_train_local.py` or cloud wrapper

**Note:** Currently a placeholder - you need to add your TFT model class and training loop.

**TODO:**
- Implement TFT model architecture
- Add forward pass and loss computation
- Add metrics calculation (MAE, directional accuracy)
- Add TensorBoard logging

## Training Pipeline

```
1. Load data
   ├─ From cache: data/processed/*.npy (fast)
   └─ From BigQuery: fetch + process (slow)
   ↓
2. Initialize model
   ├─ TFT architecture
   └─ Optimizer, scheduler
   ↓
3. Training loop
   ├─ Forward pass
   ├─ Loss computation
   ├─ Backpropagation
   └─ Validation
   ↓
4. Early stopping
   └─ Save best checkpoint
   ↓
5. Output
   ├─ Model: models/tft/tft_best.pt
   └─ Logs: logs/
```

## Configuration

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
  early_stopping:
    patience: 10
    min_delta: 0.0001

data:
  lookback_window: 192
  prediction_horizons: [4, 8, 16]
  train_ratio: 0.70
  val_ratio: 0.15
```

## Output

### Model Checkpoint

**Location:** `models/tft/tft_best.pt`

**Contents:**
```python
{
    'epoch': 42,
    'model_state_dict': {...},
    'optimizer_state_dict': {...},
    'val_loss': 0.0234,
    'val_mae': 0.0156,
    'val_dir_acc': 58.3,
    'config': {...},
    'scalers': {...}
}
```

### Logs

**Location:** `logs/`

**Contents:**
- TensorBoard logs (loss curves, metrics)
- Training metrics (CSV)
- Model architecture summary
- Hyperparameters used

## Loading Trained Model

```python
import torch
import yaml

# Load checkpoint
checkpoint = torch.load('models/tft/tft_best.pt', map_location='cpu')

# Get model state and metrics
model_state = checkpoint['model_state_dict']
val_loss = checkpoint['val_loss']
config = checkpoint['config']
scalers = checkpoint['scalers']

print(f"Best validation loss: {val_loss:.4f}")
print(f"Trained for {checkpoint['epoch']} epochs")
```

## Common Workflows

### Initial Training
```bash
# 1. Prepare data
python scripts/02_features/tft/tft_data_loader.py

# 2. Train model
python scripts/03_training/tft/tft_train_local.py

# 3. Check output
ls -lh models/tft/
```

### Resume Training
```bash
# Modify config to load from checkpoint
# Then run training again
python scripts/03_training/tft/tft_train_local.py
```

### Train with Different Config
```bash
# Create custom config
cp configs/model_tft_config.yaml configs/my_experiment.yaml
# Edit my_experiment.yaml

# Train with custom config
python scripts/03_training/tft/tft_train_local.py --config configs/my_experiment.yaml
```

### Reload Data and Train
```bash
# Force reload data from BigQuery and train
python scripts/03_training/tft/tft_train_local.py --reload
```

## Monitoring Training

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir logs/

# Open browser to http://localhost:6006
```

**Metrics to monitor:**
- Training loss (should decrease)
- Validation loss (should decrease, watch for overfitting)
- MAE (Mean Absolute Error)
- Directional accuracy (% of correct up/down predictions)

### Console Output

Training progress is printed to console:
```
Epoch 1/100
  Train batches: 18, Val batches: 4
  ✅ Saved checkpoint: models/tft/tft_best.pt

Epoch 2/100
  Train batches: 18, Val batches: 4
  ...
```

## Utilities

### `inspect_model.py` - Model Inspection

Inspect PyTorch model checkpoints and analyze architecture.

```bash
# Inspect saved TFT model
python scripts/03_training/inspect_model.py --checkpoint models/tft/tft_best.pt

# Inspect from config (no checkpoint needed)
python scripts/03_training/inspect_model.py --model-type tft --config configs/model_tft_config.yaml
```

**Parameters:**
- `--checkpoint`: Path to model checkpoint file (.pt)
- `--model-type`: Model type (tft, lstm, etc.) - used if no checkpoint
- `--config`: Path to config YAML file

**Features:**
- Checkpoint contents analysis
- Training metrics (loss, MAE, accuracy)
- Parameter count by layer type
- Model size estimation
- Configuration display

**Output Example:**
```
📦 Checkpoint Contents:
   - model_state_dict: 42 parameter tensors
   - config: Model configuration
   - scalers: 5 scalers
   - epoch: 42
   - val_loss: 0.0234

🏗️  Model Architecture:
   Total layers: 42
   
   Layer breakdown:
      encoder      :  12 tensors,      245,760 params
      decoder      :  18 tensors,      589,824 params
      attention    :   8 tensors,       98,304 params
   
   💾 Total parameters: 933,888
   💾 Model size: 3.56 MB (float32)
```

### `test_architectures.py` - Architecture Testing

Test model forward pass and compare multiple checkpoints.

```bash
# Test single model with dummy data
python scripts/03_training/test_architectures.py --checkpoint models/tft/tft_best.pt

# Test with custom dimensions
python scripts/03_training/test_architectures.py \
    --checkpoint models/tft/tft_best.pt \
    --batch-size 8 --seq-len 192 --n-features 7

# Compare multiple models
python scripts/03_training/test_architectures.py \
    --compare models/tft/tft_best.pt models/tft/tft_v2.pt models/tft/tft_v3.pt
```

**Parameters:**
- `--checkpoint`: Path to single model checkpoint to test
- `--compare`: Paths to multiple checkpoints to compare
- `--batch-size`: Test batch size (default: 4)
- `--seq-len`: Sequence length (default: 192)
- `--n-features`: Number of features (default: 7)

**Features:**
- Forward pass testing with dummy data
- Model comparison (parameters, metrics)
- Expected vs actual output shape validation
- Best model identification

**Comparison Output Example:**
```
Model                                    Params   Val Loss    Val MAE  Epoch
--------------------------------------------------------------------------------
tft_best                                933,888     0.0234     0.0156     42
tft_v2                                1,245,632     0.0198     0.0142     38
tft_v3                                  756,432     0.0256     0.0168     35

🏆 Best model: tft_v2
   Val Loss: 0.0198
   Val MAE: 0.0142
   Parameters: 1,245,632
```

**Note:** These scripts work with any PyTorch model checkpoint that follows the standard format:
```python
{
    'model_state_dict': {...},
    'optimizer_state_dict': {...},
    'epoch': int,
    'val_loss': float,
    'val_mae': float,
    'config': {...}
}
```

## Performance Tips

- **Use GPU**: Training is 10-50x faster on GPU
- **Batch size**: Larger batches = faster training (if memory allows)
- **Early stopping**: Prevents overfitting and saves time
- **Cached data**: Use `data/processed/*.npy` instead of reloading from BigQuery
- **Mixed precision**: Use `torch.cuda.amp` for faster training on modern GPUs

## Troubleshooting

**Issue: Out of memory**
- Reduce `batch_size` in config
- Reduce `hidden_size` or `lstm_layers`
- Reduce `lookback_window`
- Use gradient accumulation

**Issue: Training too slow**
- Use GPU instead of CPU
- Increase `batch_size` (if memory allows)
- Reduce `lookback_window`
- Use cached data (don't use `--reload`)

**Issue: Model not converging**
- Adjust `learning_rate` (try 0.0001 or 0.01)
- Check data normalization
- Verify no NaN values in data
- Try different optimizer (Adam, AdamW, SGD)

**Issue: Overfitting**
- Increase `dropout`
- Add weight decay
- Reduce model size (`hidden_size`, `lstm_layers`)
- Get more training data
- Use data augmentation

## Next Steps

1. **Implement TFT model** in `tft_train.py`
2. **Add training loop** with proper loss and metrics
3. **Test locally** with `tft_train_local.py`
4. **Deploy to cloud** using `scripts/05_deployment/`

## Integration with Cloud Training

The same `tft_train.py` is used by cloud training:

```python
# In train_vertex.py (cloud wrapper)
from scripts.03_training.tft.tft_train import train

# Call shared training function
train(config_path)
```

This ensures consistency between local and cloud training!