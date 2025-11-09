# CS230 Deep Learning Project - Time Series Regression

**A modular TensorFlow/Keras framework for tabular time-series regression with support for multiple model architectures, BigQuery integration, and Google Cloud deployment.**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
  - [Local Setup](#local-setup)
  - [Docker Setup](#docker-setup)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Data Generation](#data-generation)
  - [Training Models](#training-models)
  - [Model Architectures](#model-architectures)
  - [Visualization](#visualization)
- [Configuration](#configuration)
- [Workflows](#workflows)
  - [Local Development](#local-development-workflow)
  - [Docker Development](#docker-development-workflow)
  - [GCP Deployment](#gcp-deployment-workflow)
- [Google Cloud Integration](#google-cloud-integration)
- [Troubleshooting](#troubleshooting)
- [Project Features](#project-features)

---

## 🎯 Overview

This project provides a complete deep learning pipeline for time series regression tasks, designed for the CS230 Deep Learning course. It includes:

- **3 Model Architectures**: MLP, LSTM, Transformer
- **Flexible Data Pipeline**: Local CSV/Parquet, BigQuery, or dummy data generation
- **Production-Ready**: Docker containerization, GCP integration, TensorBoard logging
- **Modular Design**: Easy to extend with new models, data sources, or features
- **Comprehensive Logging**: File logs, TensorBoard metrics, training plots

---

## 📁 Project Structure

```
inflation_predictor/
│
├── configs/
│   └── config.yaml              # Main configuration file
│
├── data/
│   ├── __init__.py
│   ├── data_loader.py           # Data loading & preprocessing
│   ├── dummy_generator.py       # Synthetic data generation
│   ├── raw/                     # Raw data files
│   ├── processed/               # Preprocessed data
│   └── dummy/                   # Generated dummy data
│
├── models/
│   ├── __init__.py
│   ├── model_factory.py         # MLP, LSTM, Transformer builders
│   └── *.h5                     # Saved models
│
├── training/
│   ├── __init__.py
│   ├── trainer.py               # Training orchestration
│   └── callbacks.py             # Keras callbacks
│
├── utils/
│   ├── __init__.py
│   ├── config_loader.py         # YAML config management
│   ├── logger.py                # Logging setup
│   └── visualization.py         # Plotting utilities
│
├── logs/
│   ├── tensorboard/             # TensorBoard logs
│   ├── metrics/                 # CSV metric logs
│   ├── plots/                   # Training history plots
│   └── *.log                    # Text logs
│
├── checkpoints/                 # Model checkpoints during training
│
├── notebooks/                   # Jupyter notebooks for analysis
│
├── scripts/                     # Utility scripts
│   ├── __init__.py
│   └── generate_data.py         # Data generation script
│
├── train.py                     # Main training script
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker image definition
├── docker-compose.yml           # Docker Compose configuration
└── README.md                    # This file
```

---

## 🚀 Installation

### Local Setup

**Prerequisites:**
- Python 3.9+ (3.10 recommended)
- pip

**Steps:**

```bash
# Clone the repository
git clone <your-repo-url>
cd inflation_predictor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**macOS with Apple Silicon (M1/M2):**
```bash
# Use TensorFlow for macOS
pip install tensorflow-macos==2.13.0 tensorflow-metal==1.0.0
```

### Docker Setup

**Prerequisites:**
- Docker Desktop
- docker-compose

**Steps:**

```bash
# Build the Docker image
docker-compose build

# Verify installation
docker-compose run --rm trainer python --version
```

### When to Rebuild Docker Images

**You MUST rebuild the Docker image when:**

1. **Dockerfile changes** - Modified base image, system packages, or build steps
   ```bash
   docker-compose build --no-cache
   ```

2. **requirements.txt changes** - Added or updated Python dependencies
   ```bash
   docker-compose build
   ```

3. **New Python packages needed** - Installing additional libraries
   ```bash
   # Update requirements.txt first, then:
   docker-compose build
   ```

**You DON'T need to rebuild when:**

- ✅ Changing Python code (`.py` files) - mounted as volume, live updates
- ✅ Modifying config files (`config.yaml`) - mounted as volume
- ✅ Updating data files - mounted as volume
- ✅ Changing logs or outputs - mounted as volume

### Docker Build Commands

```bash
# Standard rebuild (uses cache when possible)
docker-compose build

# Force complete rebuild (no cache, slower but clean)
docker-compose build --no-cache

# Rebuild specific service
docker-compose build trainer

# Pull latest base images and rebuild
docker-compose build --pull

# View image size and details
docker images | grep inflation_predictor

# Remove old images to save space
docker image prune -f
```

### Troubleshooting Docker Builds

**Problem: Build fails with "no space left"**
```bash
# Clean up Docker system
docker system prune -a --volumes
```

**Problem: Changes not reflected in container**
```bash
# Rebuild without cache
docker-compose build --no-cache

# Or restart container
docker-compose down
docker-compose up -d
```

**Problem: Container uses old code**
```bash
# Check if volumes are mounted correctly
docker-compose config

# Code should be mounted at: .:/app
```

---

## ⚡ Quick Start

### Local (macOS/Linux)

```bash
# Generate dummy data and train an LSTM model
python train.py --generate-dummy --model-type lstm --epochs 30

# View results
tensorboard --logdir=logs/tensorboard
open logs/plots/*_history.png
```

### Docker

```bash
# Start interactive container
docker-compose run --rm trainer bash

# Inside container:
python train.py --generate-dummy --model-type lstm --epochs 30

# View TensorBoard (in separate terminal)
docker-compose up tensorboard
# Open: http://localhost:6006
```

---

## 📖 Usage

### Data Generation

**Generate synthetic time series data:**

```bash
# Default: 10,000 samples, 10 features
python scripts/generate_data.py

# Custom configuration
python scripts/generate_data.py --n-samples 50000 --n-features 20 --output-dir data/dummy
```

**Output files:**
- `data/dummy/train.csv` (70% of data)
- `data/dummy/val.csv` (15% of data)
- `data/dummy/test.csv` (15% of data)

### Training Models

**Basic training:**

```bash
# Train with default config (LSTM, 100 epochs)
python train.py

# Train specific model
python train.py --model-type mlp --epochs 30 --batch-size 64

# Override multiple parameters
python train.py \
  --model-type transformer \
  --epochs 100 \
  --learning-rate 0.0001 \
  --batch-size 32
```

**Using existing data:**

```bash
# Use data in data/dummy/ (configured in config.yaml)
python train.py --model-type lstm --epochs 50
```

**Generate fresh data each run:**

```bash
python train.py --generate-dummy --model-type lstm --epochs 30
```

### Model Architectures

#### 1. MLP (Multi-Layer Perceptron)

**Best for:** Simple patterns, baseline model

```bash
python train.py --model-type mlp --epochs 30
```

**Architecture:**
- Flatten layer
- Dense layers: [256, 128, 64]
- Dropout: 0.2
- Output layer: 1 neuron

#### 2. LSTM (Long Short-Term Memory)

**Best for:** Sequential patterns, time dependencies

```bash
python train.py --model-type lstm --epochs 50
```

**Architecture:**
- LSTM layers: 2 × 128 units
- Dropout: 0.2
- Optional bidirectional
- Dense output layer

#### 3. Transformer

**Best for:** Complex patterns, long sequences

```bash
python train.py --model-type transformer --epochs 100
```

**Architecture:**
- Multi-head attention: 4 heads
- Feed-forward dim: 256
- Transformer blocks: 2
- Layer normalization
- Dropout: 0.1

### Visualization

**TensorBoard:**

```bash
# Start TensorBoard server
tensorboard --logdir=logs/tensorboard

# Open browser: http://localhost:6006
```

**Training Plots:**

```bash
# View generated plots
open logs/plots/*_history.png

# Or on Linux:
xdg-open logs/plots/*_history.png
```

**Metrics CSV:**

```bash
# View training metrics
cat logs/metrics/*_metrics.csv
```

---

## ⚙️ Configuration

### Config File: `configs/config.yaml`

The YAML config controls all aspects of the pipeline:

```yaml
# Data source: 'local' or 'bigquery'
data:
  source: 'local'
  
  local:
    train_path: 'data/dummy/train.csv'
    val_path: 'data/dummy/val.csv'
    test_path: 'data/dummy/test.csv'
  
  features:
    sequence_length: 30  # Time steps to look back
    target_column: 'target'

# Model architecture
model:
  type: 'lstm'  # 'mlp', 'lstm', or 'transformer'
  hidden_dim: 128
  num_layers: 2
  dropout: 0.2

# Training hyperparameters
training:
  batch_size: 32
  epochs: 100
  learning_rate: 0.001
  optimizer: 'adam'
  
  early_stopping:
    enabled: true
    patience: 15
    monitor: 'val_loss'
```

### Command-Line Overrides

```bash
# Override any config parameter
python train.py \
  --model-type transformer \
  --batch-size 64 \
  --epochs 50 \
  --learning-rate 0.0005
```

---

## 🔄 Workflows

### Local Development Workflow

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Generate or prepare data
python scripts/generate_data.py --n-samples 10000

# 3. Train model (quick test)
python train.py --model-type mlp --epochs 5

# 4. Train production model
python train.py --model-type lstm --epochs 100

# 5. Monitor with TensorBoard
tensorboard --logdir=logs/tensorboard

# 6. Evaluate results
open logs/plots/*_history.png
cat logs/metrics/*_metrics.csv
```

### Docker Development Workflow

```bash
# 1. Build image (first time or after Dockerfile changes)
docker-compose build

# 2. Start interactive session
docker-compose run --rm trainer bash

# Inside container:
# 3. Generate data
python scripts/generate_data.py --n-samples 10000

# 4. Train models
python train.py --model-type mlp --epochs 30
python train.py --model-type lstm --epochs 50
python train.py --model-type transformer --epochs 100

# 5. Exit container
exit

# 6. View results (on local machine)
open logs/plots/*_history.png

# 7. Start TensorBoard
docker-compose up tensorboard
# Open: http://localhost:6006
```

**Docker one-liner training:**

```bash
# Train without interactive shell
docker-compose run --rm trainer \
  python train.py --generate-dummy --model-type lstm --epochs 30
```

### GCP Deployment Workflow

```bash
# 1. Set GCP project
export PROJECT_ID=your-gcp-project-id
export REGION=us-central1

# 2. Build and push Docker image
docker build -t gcr.io/$PROJECT_ID/cs230-trainer:latest .
docker push gcr.io/$PROJECT_ID/cs230-trainer:latest

# 3. Run on Vertex AI Custom Job
gcloud ai custom-jobs create \
  --region=$REGION \
  --display-name=cs230-lstm-training \
  --worker-pool-spec=machine-type=n1-standard-4,replica-count=1,container-image-uri=gcr.io/$PROJECT_ID/cs230-trainer:latest \
  --args="--model-type=lstm,--epochs=100"

# 4. Monitor job
gcloud ai custom-jobs list --region=$REGION

# 5. Run on Cloud Run (for API serving)
gcloud run deploy cs230-predictor \
  --image gcr.io/$PROJECT_ID/cs230-trainer:latest \
  --region $REGION \
  --platform managed
```

---

## ☁️ Google Cloud Integration

### BigQuery Data Source

**Update `configs/config.yaml`:**

```yaml
data:
  source: 'bigquery'
  
  bigquery:
    project_id: 'your-gcp-project'
    dataset_id: 'your-dataset'
    train_table: 'train_data'
    val_table: 'val_data'
    test_table: 'test_data'
  
  gcs:
    bucket_name: 'your-gcs-bucket'
    export_path: 'exports/'
    model_path: 'models/'
```

**Train with BigQuery data:**

```bash
python train.py --model-type lstm --epochs 50
# Data will be exported from BigQuery → GCS → loaded into tf.data
```

### GCS Model Storage

Models are automatically saved to GCS when using BigQuery source:

```
gs://your-bucket/models/lstm_20241108_221349_final.h5
gs://your-bucket/models/lstm_20241108_221349_best.h5
```

---

## 🐛 Troubleshooting

### TensorFlow Installation Issues (macOS)

**Problem:** TensorFlow hangs or "Illegal instruction" error

**Solution:**
```bash
pip uninstall tensorflow -y
pip install tensorflow-macos==2.13.0 tensorflow-metal==1.0.0
```

### Docker "Illegal instruction" on Apple Silicon

**Problem:** Container crashes with "Illegal instruction"

**Solution:** Already fixed in `Dockerfile` using `python:3.10-slim` base image

### Data Not Found Error

**Problem:** `FileNotFoundError: data/raw/train.csv`

**Solution:**
```bash
# Either generate data:
python train.py --generate-dummy

# Or update config.yaml to point to correct path:
data:
  local:
    train_path: 'data/dummy/train.csv'
```

### Permission Issues (Docker)

**Problem:** Files created in Docker are owned by root

**Solution:**
```bash
# Run container with your user ID
docker-compose run --rm --user $(id -u):$(id -g) trainer bash

# Or fix permissions after:
sudo chown -R $USER:$USER data/ logs/ checkpoints/ models/
```

### Out of Memory

**Problem:** Training crashes with OOM error

**Solution:**
```bash
# Reduce batch size
python train.py --batch-size 16

# Or reduce model size in config.yaml:
model:
  hidden_dim: 64  # Instead of 128
  num_layers: 1   # Instead of 2
```

---

## ✨ Project Features

### ✅ Completed Features

- [x] Modular project structure
- [x] 3 model architectures (MLP, LSTM, Transformer)
- [x] Flexible data pipeline (local, BigQuery, dummy)
- [x] Configurable training (YAML + CLI)
- [x] Early stopping & checkpointing
- [x] TensorBoard integration
- [x] Training visualization plots
- [x] CSV metric logging
- [x] Docker containerization
- [x] GCP deployment support
- [x] Comprehensive logging
- [x] Reproducible experiments (seeding)

### 🚧 Future Enhancements

- [ ] Hyperparameter tuning (Optuna/Keras Tuner)
- [ ] Model ensembling
- [ ] Real-time prediction API
- [ ] MLflow experiment tracking
- [ ] Automated testing suite
- [ ] Data augmentation strategies
- [ ] Multi-step forecasting
- [ ] Attention visualization

---

## 📚 Additional Resources

### Documentation

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Keras Guide](https://keras.io/guides/)
- [Google Cloud AI Platform](https://cloud.google.com/ai-platform/docs)

### Notebooks

Explore the `notebooks/` directory for:
- Data exploration
- Model comparison
- Error analysis
- Hyperparameter tuning experiments

---

## 📄 License

This project is for educational purposes (CS230 Deep Learning).

---

## 🤝 Contributing

For teammates:

1. Pull latest changes: `git pull`
2. Create feature branch: `git checkout -b feature/your-feature`
3. Make changes and test locally
4. Commit: `git commit -m "Add: your feature"`
5. Push: `git push origin feature/your-feature`
6. Create Pull Request

---

## 📧 Contact

For questions or issues, please contact the team or open an issue in the repository
