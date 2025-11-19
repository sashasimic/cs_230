# CS230 Deep Learning Project - Inflation Prediction with TFT

**A PyTorch-based Temporal Fusion Transformer (TFT) for multi-horizon inflation forecasting with BigQuery integration and Google Cloud Vertex AI deployment.**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
  - [Local Setup](#local-setup)
  - [Docker Setup](#docker-setup)
  - [GCP Setup](#google-cloud-platform-gcp-setup)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Phase 1: Extract from BigQuery](#phase-1-extract-data-from-bigquery)
  - [Data Generation (Synthetic)](#data-generation-synthetic)
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

This project implements a Temporal Fusion Transformer (TFT) for multi-horizon inflation prediction, designed for the CS230 Deep Learning course. It includes:

- **PyTorch TFT Architecture**: LSTM encoder, multi-head attention, variable selection network
- **Multi-Horizon Forecasting**: Predict multiple time steps ahead simultaneously
- **Flexible Data Pipeline**: BigQuery integration for market data (SPY, QQQ, IWM, RSP)
- **Production-Ready**: Docker containerization, GCP Vertex AI deployment, TensorBoard logging
- **Hybrid Development**: Fast local iteration with venv, validated Docker testing before cloud deployment
- **Comprehensive Logging**: Detailed feature tracking, gradient monitoring, TensorBoard metrics

---

## 📁 Project Structure

```
inflation_predictor/
│
├── configs/
│   ├── config.yaml              # Main configuration file
│   ├── mlp_multi_horizon.yaml   # Multi-horizon MLP config
│   └── tickers.yaml             # Ticker list for BigQuery extraction
│
├── data/
│   ├── __init__.py
│   ├── data_loader.py           # Data loading & preprocessing
│   ├── dummy_generator.py       # Synthetic data generation
│   ├── raw/                     # Raw data from BigQuery
│   │   └── stocks_raw.parquet   # Phase 1 output
│   ├── processed/               # Preprocessed data for training
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
├── credentials/                 # GCP service account keys (gitignored)
│   └── gcp-key.json             # Your BigQuery credentials
│
├── scripts/                     # Utility scripts
│   ├── __init__.py
│   ├── extract_tickers_from_bigquery.py  # Phase 1: BigQuery extraction
│   ├── generate_data.py         # Generate synthetic data
│   └── test_model.py            # Test model architectures
│
├── train.py                     # Main training script
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker image definition
├── docker-compose.yml           # Docker Compose configuration
└── README.md                    # This file
```

---

## ⚡ Quick Start (Local Training)

Get started with local TFT training in 3 steps:

### **Step 1: Setup Environment**

```bash
# Clone repository
git clone <your-repo-url>
cd inflation_predictor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and set: GCP_PROJECT_ID=your-project-id
```

### **Step 2: Generate Training Data**

```bash
# Load data from BigQuery and create TFT features
python scripts/02_features/tft/tft_data_loader.py

# This will:
# - Query market data (SPY, QQQ, IWM, RSP) from BigQuery
# - Generate technical indicators (SMA, volume features)
# - Create time-varying features (month_sin/cos, is_weekend)
# - Save to data/processed/ (X_train.npy, y_train.npy, etc.)
# - Save scalers for inference (scalers.pkl)
```

**Output:**
```
data/processed/
├── X_train.npy          # Training sequences (2198, 192, 7)
├── y_train.npy          # Training targets (2198, 3)
├── X_val.npy            # Validation sequences
├── y_val.npy            # Validation targets
├── X_test.npy           # Test sequences
├── y_test.npy           # Test targets
├── timestamps_train.npy # Timestamps for each sample
├── timestamps_val.npy
├── timestamps_test.npy
└── scalers.pkl          # StandardScaler objects
```

### **Step 3: Train TFT Model**

```bash
# Train locally with default config
python scripts/03_training/tft/tft_train_local.py

# Or with custom config
python scripts/03_training/tft/tft_train_local.py --config configs/my_config.yaml

# Force reload data from BigQuery
python scripts/03_training/tft/tft_train_local.py --reload
```

**Training Output:**
- Model checkpoint: `models/tft/tft_best.pt`
- TensorBoard logs: `logs/tensorboard/`
- Training checkpoints: `checkpoints/tft/`

**View Training Progress:**
```bash
# Open TensorBoard
tensorboard --logdir logs/tensorboard

# Open browser to http://localhost:6006
```

### **What You'll See:**

```
================================================================================
   Feature Configuration
================================================================================

📊 Data Dimensions:
  Lookback window: 192 timesteps
  Number of features: 7
  Prediction horizons: 3

🔑 TIME-VARYING KNOWN Features (3):
   1. month_sin
   2. month_cos
   3. is_weekend

📊 TIME-VARYING UNKNOWN Features (4):
   1. close
   2. volume
   3. sma_50
   4. sma_200

🎯 Output Targets (3 horizons):
  1. Horizon 4 (target_4_periods_ahead)
  2. Horizon 8 (target_8_periods_ahead)
  3. Horizon 16 (target_16_periods_ahead)

================================================================================
Epoch 1/100
  Train Loss: 0.0234
  Val Loss: 0.0256, MAE: 0.0123, RMSE: 0.0178
  Dir Acc: 52.34%
  Grad Norm: avg=0.1234, max=0.5678, min=0.0123
  Layer Gradients (batch 1):
    Input       : norm=0.1234, max=0.5678, std=0.0234
    LSTM_L0     : norm=0.2345, max=0.6789, std=0.0345
    ...
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

---

## ☁️ Quick Start (GCP Cloud Training)

Deploy and train on Google Cloud Vertex AI for production workloads:

### **Step 1: GCP Setup**

```bash
# Install gcloud CLI (if not already installed)
# https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login
gcloud auth application-default login

# Set project
gcloud config set project YOUR_PROJECT_ID

# Run setup script (creates bucket, service account, IAM roles)
bash scripts/05_deployment/setup_gcp.sh
```

**This creates:**
- GCS bucket: `gs://YOUR_PROJECT_ID-models`
- Service account: `vertex-model-trainer@YOUR_PROJECT.iam.gserviceaccount.com`
- IAM roles: AI Platform Admin, Storage Admin, BigQuery User

### **Step 2: Create Dataset Version**

```bash
# Generate and upload dataset to GCS
python scripts/05_deployment/generate_dataset.py --version v1

# This will:
# - Run tft_data_loader.py to generate features
# - Package data/processed/ and data/raw/
# - Upload to gs://YOUR_PROJECT_ID-models/datasets/v1/
# - Create manifest with metadata
```

**Output in GCS:**
```
gs://YOUR_PROJECT_ID-models/datasets/v1/
├── processed/
│   ├── X_train.npy
│   ├── y_train.npy
│   ├── X_val.npy
│   ├── y_val.npy
│   ├── X_test.npy
│   ├── y_test.npy
│   ├── timestamps_*.npy
│   └── scalers.pkl
├── raw/
│   └── (raw data files)
└── manifest.json
```

### **Step 3: Build and Push Docker Image**

```bash
# Build Docker image for Vertex AI
docker build --platform linux/amd64 \
  -f scripts/05_deployment/Dockerfile.vertex \
  -t gcr.io/YOUR_PROJECT_ID/model-trainer:latest \
  .

# Push to Google Container Registry
docker push gcr.io/YOUR_PROJECT_ID/model-trainer:latest
```

**Note:** Building for `linux/amd64` is required for GCP, especially on Apple Silicon Macs.

### **Step 4: Test Docker Locally (Optional but Recommended)**

```bash
# Test in same environment as GCS before deploying
bash scripts/05_deployment/test_docker_local.sh

# This will:
# - Build Docker image
# - Run training with mounted local data
# - Validate everything works before spending GCS credits
```

### **Step 5: Submit Training Job to Vertex AI**

```bash
# Submit job with dataset version v1
python scripts/05_deployment/submit_job.py --dataset-version v1

# Or with custom machine type
python scripts/05_deployment/submit_job.py \
  --dataset-version v1 \
  --machine-type n1-highmem-8
```

**Monitor Training:**
```bash
# View in GCP Console
https://console.cloud.google.com/vertex-ai/training/custom-jobs

# Or check logs
gcloud ai custom-jobs stream-logs JOB_ID --region=us-central1
```

### **Step 6: Retrieve Trained Model**

```bash
# Download model from GCS
gsutil cp gs://YOUR_PROJECT_ID-models/models/tft/tft_best.pt models/tft/

# Or download TensorBoard logs
gsutil -m cp -r gs://YOUR_PROJECT_ID-models/tensorboard_logs/ logs/
```

### **What You'll See in Vertex AI Logs:**

```
================================================================================
   Vertex AI Training (TFT)
================================================================================
Job: model-training-20251119-170000
GCS Bucket: YOUR_PROJECT_ID-models
Dataset Version: v1

📦 Loading Dataset Version: v1
  ✅ Processed data loaded to: data/processed/
  ✅ Raw data loaded to: data/raw/

================================================================================
   Feature Configuration
================================================================================

📊 Data Dimensions:
  Lookback window: 192 timesteps
  Number of features: 7
  Prediction horizons: 3

🔑 TIME-VARYING KNOWN Features (3):
   1. month_sin
   2. month_cos
   3. is_weekend

📊 TIME-VARYING UNKNOWN Features (4):
   1. close
   2. volume
   3. sma_50
   4. sma_200

================================================================================
Epoch 1/100
  Train Loss: 0.0234
  Val Loss: 0.0256, MAE: 0.0123, RMSE: 0.0178
  Dir Acc: 52.34%
  Grad Norm: avg=0.1234, max=0.5678, min=0.0123
  Layer Gradients (batch 1):
    Input       : norm=0.1234, max=0.5678, std=0.0234
    LSTM_L0     : norm=0.2345, max=0.6789, std=0.0345
    LSTM_L1     : norm=0.3456, max=0.7890, std=0.0456
    ...
```

---

## 🚀 Installation

**All installation and setup instructions are in the Quick Start sections above:**

- **For local development:** See [⚡ Quick Start (Local Training)](#-quick-start-local-training)
- **For cloud deployment:** See [☁️ Quick Start (GCP Cloud Training)](#%EF%B8%8F-quick-start-gcp-cloud-training)

**Additional Resources:**
- Detailed training documentation: `scripts/03_training/README.md`
- Deployment documentation: `scripts/05_deployment/README.md`
- Data pipeline documentation: `scripts/02_features/README.md`

---

## 📚 Project Structure

**1. Create a GCP Service Account:**

```bash
# In Google Cloud Console:
# 1. Navigate to: IAM & Admin > Service Accounts
# 2. Click "Create Service Account"
# 3. Name: "cs230-bigquery-reader"
# 4. Grant roles:
#    - BigQuery Data Viewer
#    - BigQuery Job User
# 5. Create and download JSON key
```

**2. Store Service Account Key:**

```bash
# Create credentials directory (gitignored)
mkdir -p credentials

# Move your downloaded key
mv ~/Downloads/your-service-account-key.json credentials/gcp-key.json

# Verify .gitignore includes credentials/
grep -q "credentials/" .gitignore || echo "credentials/" >> .gitignore
```

**3. Set Environment Variable:**

```bash
# For local development (add to ~/.bashrc or ~/.zshrc)
export GOOGLE_APPLICATION_CREDENTIALS="/full/path/to/credentials/gcp-key.json"

# Or set per-session
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/credentials/gcp-key.json"
```

**4. Configure BigQuery Details:**

Edit `configs/tickers.yaml`:

```yaml
bigquery:
  project_id: 'your-gcp-project-id'      # e.g., 'my-ml-project'
  dataset_id: 'your-bigquery-dataset'    # e.g., 'stock_data'
  table_name: 'stock_ohlcv'              # Your table name
```

**5. Verify Connection:**

**Install Google Cloud SDK (if needed):**

```bash
# macOS
brew install google-cloud-sdk

# Linux
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Or download from: https://cloud.google.com/sdk/docs/install
```

**Test Authentication:**

```bash
# Method 1: Using gcloud CLI
gcloud auth application-default login

# Method 2: Using service account key (recommended for automation)
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/credentials/gcp-key.json"

# Verify authentication works
gcloud auth list
```

**Test BigQuery Access:**

```bash
# 1. Test project access
gcloud config set project your-project-id
gcloud projects describe your-project-id

# 2. List datasets (verify you can see your dataset)
bq ls

# 3. List tables in your dataset
bq ls your-dataset-id

# 4. Test query on your table
bq query --use_legacy_sql=false \
  'SELECT COUNT(*) as row_count FROM `your-project-id.your-dataset-id.stock_ohlcv` LIMIT 1'

# 5. Preview table data
bq head -n 5 your-project-id:your-dataset-id.stock_ohlcv

# 6. Show table schema
bq show --schema your-project-id:your-dataset-id.stock_ohlcv
```

**Test with Python:**

```bash
# Test connection from Python
python -c "
from google.cloud import bigquery
import os

print('Testing BigQuery connection...')
client = bigquery.Client()
print(f'✓ Connected to project: {client.project}')

# List datasets
datasets = list(client.list_datasets())
if datasets:
    print(f'✓ Found {len(datasets)} datasets:')
    for dataset in datasets:
        print(f'  - {dataset.dataset_id}')
else:
    print('⚠️ No datasets found')

print('✓ Connection successful!')
"

# Test extraction script
python scripts/extract_tickers_from_bigquery.py --tickers AAPL --start-date 2024-01-01 --end-date 2024-01-31
```

**Troubleshoot Connection Issues:**

```bash
# Check environment variable is set
echo $GOOGLE_APPLICATION_CREDENTIALS

# Verify key file exists and is readable
ls -l $GOOGLE_APPLICATION_CREDENTIALS
cat $GOOGLE_APPLICATION_CREDENTIALS | python -m json.tool | head -20

# Check service account permissions
gcloud projects get-iam-policy your-project-id \
  --flatten="bindings[].members" \
  --filter="bindings.members:serviceAccount:YOUR_SERVICE_ACCOUNT_EMAIL"

# Test with verbose output
bq --apilog query --use_legacy_sql=false 'SELECT 1 as test'
```

**Expected Output:**

```
✓ Connected to project: your-project-id
✓ Found 3 datasets:
  - stock_data
  - analytics
  - ml_models
✓ Connection successful!
```


**Security Best Practices:**

- ✅ **Never commit** service account keys to git
- ✅ Store keys in `credentials/` directory (gitignored)
- ✅ Use read-only permissions (BigQuery Data Viewer)
- ✅ Rotate keys periodically
- ✅ Use different keys for dev/prod
- ❌ Don't hardcode credentials in code
- ❌ Don't share keys via email/Slack

#### Recommended: User Account Authentication (No Keys Needed)

**If service account key creation is disabled by your organization**, use personal account authentication:

```bash
# 1. Install Google Cloud SDK
brew install google-cloud-sdk  # macOS

# 2. Set your GCP project
gcloud config set project sp-indicators  # Replace with your project ID

# 3. Authenticate with your Google account
gcloud auth application-default login
# This opens a browser - sign in with your Google account

# 4. Set environment variable (permanent)
echo 'export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcloud/application_default_credentials.json"' >> ~/.zshrc
source ~/.zshrc

# 5. Verify connection
python -c "from google.cloud import bigquery; c = bigquery.Client(); print(f'✓ Connected to {c.project}')"
```

**Benefits:**
- ✅ No keys to manage or rotate
- ✅ More secure (no files to leak)
- ✅ Better audit trail (actions tracked per user)
- ✅ Works immediately
- ✅ Bypasses organization policy restrictions

**Team Setup:** Each team member runs the same authentication steps with their own Google account.

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

---

## 📖 Usage

### Quick Start Workflow

#### Prerequisites: GCP Authentication

**First-time setup** - Authenticate with Google Cloud for BigQuery access:

```bash
# Using your Google account
gcloud auth application-default login

# Option 2: Using your Google account (no key needed)
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

**📚 See [GCP Setup](#google-cloud-platform-gcp-setup) for detailed authentication instructions.**

---


#### Complete Pipeline

```bash
# ============================================================
# Phase 1a: Extract Ticker Data from BigQuery
# ============================================================

# Check data availability for configured tickers
python scripts/01_extract/extract_tickers.py --check-availability

# Extract all tickers to parquet
python scripts/01_extract/extract_tickers.py

# Verify ticker extraction
python scripts/01_extract/verify_ticker_extraction.py
python scripts/01_extract/verify_ticker_extraction.py --ticker SPY  # Detailed analysis

# ============================================================
# Phase 1b: Extract Google Trends Data
# ============================================================

# Test with last 30 days (quick test)
python scripts/01_extract/extract_google_trends.py --test

# Full extraction with aggressive retry settings (if hitting rate limits)
python scripts/01_extract/extract_google_trends.py

# Verify Google Trends extraction
python scripts/01_extract/verify_google_trends.py

# ============================================================
# Phase 2: Build Features
# ============================================================

# Build features with default config
python scripts/02_features/build_features.py

# Verify processed features
python scripts/02_features/verify_features.py

# ============================================================
# Phase 3: Train Multi-Horizon Models
# ============================================================

# Train multi-horizon MLP model
python train.py --model-type mlp --config configs/mlp_multi_horizon.yaml --epochs 30

# ============================================================
# Monitor Training (in separate terminal)
# ============================================================

# View TensorBoard
tensorboard --logdir logs/tensorboard
# Open: http://localhost:6006
```

**📚 For detailed usage of each script, see [scripts/README.md](scripts/README.md)**

---

### Phase 1: Data Extraction

**Phase 1 extracts two types of data:**
1. **Ticker/Stock Data** from BigQuery (OHLCV price data)
2. **Google Trends Data** (search interest for inflation-related keywords)

#### Phase 1a: Extract Ticker Data from BigQuery

**Extract real stock/index data from your BigQuery table.**

**Configuration:** `configs/tickers.yaml` - Edit to specify tickers, date range, and BigQuery details.

```bash
# Check data availability first
python scripts/01_extract/extract_tickers.py --check-availability

# Extract all configured tickers (uses date range from config or default 2015-10-28 to today)
python scripts/01_extract/extract_tickers.py

# Extract with custom date range
python scripts/01_extract/extract_tickers.py --start-date 2015-10-28 --end-date 2025-10-24

# Verify extraction
python scripts/01_extract/verify_ticker_extraction.py

# Verify specific ticker with date range
python scripts/01_extract/verify_ticker_extraction.py --ticker SPY
```

**Output:** `data/raw/stocks_raw.parquet` - Wide-format parquet with one row per date, columns for each ticker's OHLCV data.

#### Phase 1b: Extract Google Trends Data

**Extract search interest data for inflation indicators.**

**Configuration:** `configs/google_trends.yaml` - Edit to specify keywords, date range, and region.

```bash
# Quick test (last 30 days)
python scripts/01_extract/extract_google_trends.py --test

# Full extraction with retry settings for rate limiting
python scripts/01_extract/extract_google_trends.py \
  --start-date 2015-10-28 \
  --end-date 2025-10-24 \
  --throttle 10 \
  --max-retries 8 \
  --force

# Verify Google Trends extraction
python scripts/01_extract/verify_google_trends.py
```

**Output:** `data/raw/google_trends.parquet` - Daily search interest (0-100 scale) for keywords like "inflation", "food prices".

**📚 For detailed options (ticker management, custom dates, verification), see [scripts/README.md](scripts/README.md#phase-1-data-extraction)**

---

### Phase 2: Feature Engineering

**Transform raw OHLCV data into ML-ready features with technical indicators and multi-horizon targets.**

```bash
# Build features with default config
python scripts/02_features/build_features.py

# Verify processed data
python scripts/02_features/verify_features.py
```

**Output:**
- `data/processed/train.parquet` - Training set
- `data/processed/val.parquet` - Validation set  
- `data/processed/test.parquet` - Test set
- `data/processed/scaler.pkl` - Fitted scalers
- `data/processed/feature_names.txt` - Feature list

**Features Created:**
- Technical indicators: Returns, Volatility, Moving Averages, RSI, Momentum
- Multi-horizon targets: 1M, 3M, 6M forward returns
- Weighted target: Configurable ticker weights (default: 60% SPY + 40% QQQ)
- ~300+ features → ~160 after correlation removal

**Configuration:** Edit `configs/features.yaml` to customize input tickers, technical indicators, target horizons, and feature engineering options.

**📚 For detailed configuration options, see [scripts/README.md](scripts/README.md#phase-2-feature-engineering)**

---

### Dummy Data Generation

**Generate synthetic time series data for testing without BigQuery access:**

```bash
# Generate dummy dataset
python scripts/dummy/generate_dummy_data.py

# Custom configuration
python scripts/dummy/generate_dummy_data.py --n-samples 50000 --n-features 20
```

**📚 For detailed options, see [scripts/README.md](scripts/README.md#dummy-data-generation)**

**Output files:**
- `data/dummy/train.csv` - Training set (default: 70%)
- `data/dummy/val.csv` - Validation/Dev set (default: 15%)
- `data/dummy/test.csv` - Test set (default: 15%)

**Note:** Test set percentage is automatically calculated as `1 - train_ratio - val_ratio`. For example:
- `--train-ratio 0.8 --val-ratio 0.1` → Test gets 10%
- `--train-ratio 0.6 --val-ratio 0.2` → Test gets 20%

**Data Split Options:**

| Dataset Size | Recommended Split | Command |
|--------------|-------------------|----------|
| Small (<10K) | 70/15/15 | `--train-ratio 0.7 --val-ratio 0.15` (default) |
| Medium (10K-100K) | 80/10/10 | `--train-ratio 0.8 --val-ratio 0.1` |
| Large (>100K) | 90/5/5 | `--train-ratio 0.9 --val-ratio 0.05` |

### Phase 3: Training Multi-Horizon Models

**Train models to predict multiple future time horizons (1M, 3M, 6M forward returns).**

#### Multi-Horizon Training (Recommended)

```bash
# Train LSTM with multi-horizon config
python train.py --model-type lstm --config configs/mlp_multi_horizon.yaml --epochs 50

# Train MLP with multi-horizon config
python train.py --model-type mlp --config configs/mlp_multi_horizon.yaml --epochs 30

# Train Transformer with multi-horizon config
python train.py --model-type transformer --config configs/mlp_multi_horizon.yaml --epochs 100
```

**Multi-horizon models predict:**
- **1M horizon**: 1-month forward returns
- **3M horizon**: 3-month forward returns  
- **6M horizon**: 6-month forward returns
- **Weighted target**: Configurable blend (default: 60% SPY + 40% QQQ)

#### Basic Training (Single Horizon)

```bash
# Train with default config (LSTM, single target)
python train.py --model-type lstm --epochs 50

# Override parameters
python train.py \
  --model-type transformer \
  --epochs 100 \
  --learning-rate 0.0001 \
  --batch-size 32
```

#### Quick Testing with Dummy Data

```bash
# Generate synthetic data and train (no BigQuery needed)
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

##  Troubleshooting

### TensorFlow Installation Issues (macOS)

**Problem:** TensorFlow hangs or "Illegal instruction" error

**Solution:**
```bash
pip uninstall tensorflow -y
pip install tensorflow-macos==2.13.0 tensorflow-metal==1.0.0
```


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

### GCP Deployment Workflow

```bash
# 1. Set GCP project
export PROJECT_ID=your-gcp-project-id
export REGION=us-central1

# 2. Run on Vertex AI Custom Job
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
