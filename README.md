# CS230 Deep Learning Project - Inflation Prediction with TFT

**A PyTorch-based Temporal Fusion Transformer (TFT) for multi-horizon inflation forecasting with BigQuery integration and Google Cloud Vertex AI deployment.**

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Quick Start: Setting Up Your Data](#-quick-start-setting-up-your-data)
- [Quick Start: Local Training](#-quick-start-local-training)
- [Quick Start: GCP Cloud Training](#%EF%B8%8F-quick-start-gcp-cloud-training)
- [Troubleshooting](#-troubleshooting)
- [Project Features](#-project-features)
- [Additional Resources](#-additional-resources)
- [Contributing](#-contributing)

---

## 📍 CS230 Project Navigation

**Important:** This project spans multiple branches and repositories. Here's where to find each component:

### **This Repository (`inflation_predictor`)**

| Component | Branch | Description |
|-----------|--------|-------------|
| **Data Pipeline & Feature Engineering** | `main` | BigQuery integration, feature generation, data preprocessing |
| **LSTM Baseline** | `main` | LSTM model for time series forecasting |
| **Decoder Transformer** | `main` | Decoder-only transformer architecture |
| **TFT (Temporal Fusion Transformer)** | `main` | Full TFT implementation with VSN, attention, and GRU decoder |
| **TFT + FinCast Integration** | `sasha-v4` | TFT with FinCast price encoder for enhanced financial features |

### **External Repositories**

| Model | Repository | Description |
|-------|------------|-------------|
| **PAN-NAN** | [feiyangk/230proj](https://github.com/feiyangk/230proj) | PAN-NAN architecture (separate team implementation) |

**Quick Navigation:**
```bash
# Work with main models (LSTM, Decoder, TFT)
git checkout main

# Work with TFT + FinCast integration
git checkout sasha-v4

# View PAN-NAN implementation
# Visit: https://github.com/feiyangk/230proj
```

---

## 🎯 Overview

This project implements a Temporal Fusion Transformer (TFT) for multi-horizon inflation prediction, designed for the CS230 Deep Learning course. It includes:

- **PyTorch TFT Architecture**: LSTM encoder, multi-head attention, variable selection network, GRU future decoder
- **Multi-Horizon Forecasting**: Predict multiple time steps ahead simultaneously  
- **Flexible Data Pipeline**: Model-agnostic BigQuery integration with ticker grouping and augmentation
- **FinCast Integration**: Optional price encoder integration for enhanced financial features
- **Production-Ready**: Docker containerization, GCP Vertex AI deployment, TensorBoard logging
- **Modular Codebase**: Shared training utilities, clean separation between models and common components
- **Optimized Logging**: Clean console output with detailed TensorBoard metrics

---

## 📁 Project Structure

```
inflation_predictor/
│
├── configs/
│   ├── model_tft_config.yaml    # TFT model configuration
│   └── google_trends.yaml       # Google Trends config (legacy)
│
├── data/
│   ├── datasets/                # Versioned datasets for reproducibility
│   │   ├── tft/                 # TFT model datasets
│   │   │   ├── v1/              # Dataset version 1
│   │   │   ├── v2/              # Dataset version 2
│   │   │   └── v{N}/            # Each version contains:
│   │   │       ├── raw/         #   - tft_features.csv (pivoted, one row per date)
│   │   │       ├── processed/   #   - X_*.npy, y_*.npy (train/val/test)
│   │   │       └── manifest.yaml#   - Metadata and feature list
│   │   ├── decoder_transformer/ # Decoder transformer datasets
│   │   └── lstm/                # LSTM baseline datasets
│   ├── raw/                     # Temporary raw data (latest generation)
│   └── processed/               # Temporary processed data (latest generation)
│
├── models/
│   └── tft/                     # Saved TFT models
│       └── tft_best.pt          # Best model checkpoint
│
├── scripts/
│   ├── 01_extract/              # Data extraction from BigQuery
│   │   └── extract_tickers.py
│   ├── 02_features/             # Feature engineering (model-agnostic modules)
│   │   ├── data_loader.py       # Base multi-ticker data loader
│   │   ├── data_grouping.py     # Ticker group feature aggregation
│   │   ├── data_augmentation.py # Data augmentation utilities
│   │   └── tft_pipeline.py      # TFT-specific pipeline (uses model-agnostic modules)
│   ├── 03_training/             # Model training
│   │   ├── common/              # Shared training utilities
│   │   │   ├── nn_modules.py    #   - Neural network components (GRN, PositionalEncoding)
│   │   │   ├── training_utils.py#   - Training loops, optimizer/scheduler creation
│   │   │   ├── metrics.py       #   - Metric computation (MAE, RMSE, directional accuracy)
│   │   │   └── visualization.py #   - TensorBoard and plotting utilities
│   │   ├── tft/                 # Temporal Fusion Transformer
│   │   │   ├── tft_train.py     #   - Core training logic (shared by local & cloud)
│   │   │   └── tft_train_local.py  #   - Local training wrapper
│   │   ├── decoder_transformer/ # Decoder-only Transformer
│   │   │   ├── decoder_transformer_train.py       # Core training logic
│   │   │   ├── decoder_transformer_train_local.py # Local training wrapper
│   │   │   └── fincast_extension.py               # FinCast integration
│   │   ├── lstm/                # LSTM Baseline
│   │   │   ├── lstm_train.py    #   - Core training logic
│   │   │   └── lstm_train_local.py #   - Local training wrapper
│   │   ├── inspect_model.py     # Model architecture inspection tool
│   │   └── test_architectures.py # Architecture comparison tests
│   ├── 04_inference/            # Model inference
│   ├── 05_deployment/           # GCP deployment utilities
│   │   ├── generate_dataset.py  # Create versioned datasets
│   │   ├── submit_job.py        # Submit Vertex AI jobs
│   │   ├── Dockerfile.vertex    # Docker for cloud training
│   │   └── setup_gcp.sh         # GCP infrastructure setup
│   └── dummy/                   # Legacy dummy data generation
│
├── utils/
│   ├── __init__.py              # Package initialization
│   ├── config_loader.py         # YAML config utilities
│   ├── logger.py                # Logging setup
│   ├── benchmarks.py            # Model benchmarking utilities
│   └── visualization.py         # Plotting utilities
│
├── external/
│   └── fincast/                 # FinCast price encoder (git submodule)
│
├── logs/                        # Training logs and metrics
├── checkpoints/                 # Training checkpoints
├── notebooks/                   # Jupyter notebooks
├── temp/                        # Temporary files (gitignored)
├── .env                         # Environment variables (gitignored)
├── .env.example                 # Example environment file
├── .gitignore                   # Git ignore rules
├── .dockerignore                # Docker ignore rules
├── datasets_registry.yaml       # Dataset version registry
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 🚀 Installation

### Prerequisites
- Python 3.12+ (3.12 recommended for Apple Silicon compatibility)
- pip package manager
- Git (with submodules support for FinCast)
- Google Cloud SDK (for cloud deployment)
- Access to BigQuery with ticker data and GDELT sentiment data

### Local Setup

**1. Clone the Repository**

```bash
# Clone the project
git clone https://github.com/your-org/inflation_predictor.git
cd inflation_predictor
```

**2. Create Virtual Environment**

```bash
# Create virtual environment
python -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

**3. Install Dependencies**

```bash
# Install all required packages
pip install -r requirements.txt
```

**4. Configure Environment Variables**

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and set your values:
# - GCP_PROJECT_ID: Your Google Cloud project ID
# - BIGQUERY_DATASET: Your BigQuery dataset name
# - Other configuration as needed
```

**5. Verify Installation**

```bash
# Test imports
python -c "import torch; import pandas; import google.cloud.bigquery; print('✅ All dependencies installed')"
```

---

## ⚡ Quick Start: Setting Up Your Data

**Assumptions:**
- ✅ Ticker OHLCV data already loaded in BigQuery
- ✅ GDELT sentiment data already loaded in BigQuery  
- ✅ (Optional) Synthetic/agriculture basket data available

### **Step 1: Configure Data Sources**

Edit `configs/model_tft_config.yaml` to point to your BigQuery tables:

```yaml
data:
  # BigQuery configuration
  bigquery:
    project_id: 'your-project-id'
    
    # Ticker/market data table
    ticker_dataset: 'your_dataset'
    ticker_table: 'ticker_ohlcv'  # Should have: ticker, date, open, high, low, close, volume
    
    # GDELT sentiment data table  
    gdelt_dataset: 'your_dataset'
    gdelt_table: 'gdelt_daily'    # Should have: date, weighted_avg_tone, num_articles, etc.
    
    # Agriculture basket (optional)
    agriculture_table: 'agriculture_basket'  # Optional: WEAT, SOYB, RJA prices
  
  # Date range
  start_date: '2020-01-01'
  end_date: '2025-05-05'
  
  # Tickers to query
  tickers: ['SPY', 'QQQ', 'IWM', 'RSP']  # Market index ETFs
```

### **Step 2: Test BigQuery Connection**

```bash
# Authenticate with Google Cloud
gcloud auth application-default login

# Set project
gcloud config set project your-project-id

# Test query (should return row count)
bq query --use_legacy_sql=false \
  'SELECT COUNT(*) FROM `your-project.your-dataset.ticker_ohlcv`'
```

### **Step 3: Generate Local Training Data**

```bash
# Run TFT data loader to create local training data
python scripts/02_features/tft_pipeline.py

# This will:
# 1. Query ticker data from BigQuery (SPY, QQQ, IWM, RSP)
# 2. Query GDELT sentiment data
# 3. Join data on date
# 4. Pivot tickers (one row per date, ticker-specific columns: close_SPY, close_QQQ, etc.)
# 5. Generate technical indicators (SMA, volume features)
# 6. Create time features (month_sin/cos, is_weekend)
# 7. Split into train/val/test (70/15/15)
# 8. Normalize and save sequences
# 9. Export to data/raw/ and data/processed/
```

**Output Structure:**
```
data/
├── raw/
│   ├── tft_features.csv         # Pivoted: one row per date, all ticker columns
│   ├── gdelt_raw_weighted_daily.parquet
│   └── google_trends.parquet
└── processed/
    ├── X_train.npy              # Training sequences [N, 192, 26]
    ├── y_train.npy              # Training targets [N, 3]
    ├── X_val.npy, y_val.npy
    ├── X_test.npy, y_test.npy
    ├── timestamps_*.npy
    ├── scalers.pkl
    ├── metadata.yaml            # Dataset metadata
    └── feature_names.txt        # List of 26 features
```

**Verify Dataset:**
```bash
# Check metadata
cat data/processed/metadata.yaml

# Check features (should show 26 features: 16 ticker-specific + 7 GDELT + 3 time)
cat data/processed/feature_names.txt

# Check raw data format (one row per date)
head -5 data/raw/tft_features.csv
```

**💡 For Cloud Training:**
If you want to create a versioned dataset for GCS upload and cloud training:
```bash
# Generate versioned dataset for cloud deployment
python scripts/05_deployment/generate_dataset.py --version v1 --model-type tft

# This creates data/datasets/tft/v1/ and uploads to GCS
# See "Quick Start: GCP Cloud Training" section below
```

---

## ⚡ Quick Start: Local Training

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

### **Step 2: Train Model**

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

**View Training Progress with TensorBoard:**
```bash
# Start TensorBoard server
tensorboard --logdir logs/tensorboard

# Open browser to http://localhost:6006

# View specific model logs
tensorboard --logdir logs/tensorboard/tft  # TFT model only
tensorboard --logdir logs/tensorboard/decoder_transformer  # Decoder only
```

### **What You'll See:**

```
🏗️  Initializing TFT model...
   Parameters: 1,234,567 trainable

🏋️  Starting training...

Epoch 1/200 (2.3min) - Train: 0.0234, Val: 0.0256, MAE: 0.0123, Dir Acc: 52.3%

Epoch 2/200 (2.1min) - Train: 0.0198, Val: 0.0215, MAE: 0.0107, Dir Acc: 54.1%
  ⭐ New best model! Saving to models/tft/tft_best.pt

Epoch 3/200 (2.2min) - Train: 0.0176, Val: 0.0189, MAE: 0.0095, Dir Acc: 56.8%
  ⭐ New best model! Saving to models/tft/tft_best.pt

...

Epoch 50/200 (2.0min) - Train: 0.0045, Val: 0.0052, MAE: 0.0026, Dir Acc: 67.2%
  ⭐ New best model! Saving to models/tft/tft_best.pt

✅ Training complete!

📊 Validation Set Results (Best Model):
  Val Loss: 0.005234
  MAE: 0.002567, RMSE: 0.003421
  Directional Accuracy: 67.2%

📊 Test Set Results:
  Test Loss: 0.005489
  MAE: 0.002634, RMSE: 0.003512
  Directional Accuracy: 66.8%
```

**Notes:**
- Clean, concise output per epoch
- Detailed metrics logged to TensorBoard
- Best model automatically saved
- Final evaluation on validation and test sets


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

### **Step 2: Create and Upload Dataset Version**

```bash
# Generate dataset v1 and upload to GCS
python scripts/05_deployment/generate_dataset.py --version v1 --model-type tft

# This will:
# - Run tft_pipeline.py to generate features from BigQuery
# - Package data to data/datasets/tft/v1/
# - Upload to gs://YOUR_PROJECT_ID-models/datasets/tft/v1/
# - Register in datasets_registry.yaml
# - Create manifest with metadata
```

**Output in GCS:**
```
gs://YOUR_PROJECT_ID-models/datasets/tft/v1/
├── processed/
│   ├── X_train.npy              # [N, 192, 26] training sequences  
│   ├── y_train.npy              # [N, 3] targets (3 horizons)
│   ├── X_val.npy, y_val.npy
│   ├── X_test.npy, y_test.npy
│   ├── timestamps_*.npy
│   ├── scalers.pkl
│   ├── metadata.yaml
│   └── feature_names.txt
├── raw/
│   ├── tft_features.csv         # Pivoted: one row per date
│   ├── gdelt_raw_weighted_daily.parquet
│   └── google_trends.parquet
└── manifest.yaml
```

### **Step 3: Build and Push Docker Image**

```bash
# Build Docker image for Vertex AI
docker build --platform linux/amd64 \
  -f scripts/05_deployment/Dockerfile.vertex \
  -t gcr.io/inflation-prediction-478715/model-trainer:latest \
  .

# Push to Google Container Registry
docker push gcr.io/inflation-prediction-478715/model-trainer:latest
```

**Note:** Building for `linux/amd64` is required for GCP, especially on Apple Silicon Macs.

**Note:** If you need fill rebuild
```bash
docker build --no-cache --platform linux/amd64 \
-f scripts/05_deployment/Dockerfile.vertex \
-t gcr.io/inflation-prediction-478715/model-trainer:latest \
.
```

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

# Or submit HP tuning job
python scripts/05_deployment/submit_hp_tuning.py \                 
  --phase 1 \
  --dataset-version v22 \
  --model-type tft
```

**Monitor Training:**
```bash
# View in GCP Console
https://console.cloud.google.com/vertex-ai/training/custom-jobs

# Or check logs
gcloud ai custom-jobs stream-logs JOB_ID --region=us-central1
```

### **Step 6: Monitor Training with TensorBoard**

#### **View TensorBoard on GCP Console**

```bash
# Access TensorBoard directly in GCP Console:
https://console.cloud.google.com/vertex-ai/experiments/tensorboard?project=YOUR_PROJECT_ID

# Or use the link from job submission output:
# View Tensorboard:
# https://us-central1.tensorboard.googleusercontent.com/experiment/projects+...
```

**In the GCP Console:**
1. Navigate to **Vertex AI > Experiments > TensorBoard**
2. Find your TensorBoard instance: `tensorboard-YOUR_PROJECT_ID`
3. Click on the experiment matching your job name (e.g., `model-training-20251119-170000`)
4. View metrics in real-time:
   - **Scalars**: Loss, MAE, RMSE, directional accuracy
   - **Training/Validation**: Separate tabs for train vs. val metrics
   - **Autoregressive vs. Teacher Forcing**: Compare evaluation modes (decoder_transformer only)

**TensorBoard automatically syncs** from `/tmp/tensorboard/` in the training VM to the managed TensorBoard service.

#### **Download and View TensorBoard Locally**

To view historical TensorBoard logs locally:

```bash
# Option 1: Download from TensorBoard's managed storage
# (Logs are stored in a Google-managed GCS bucket)
# This is automatic - you can access via console URL above

# Option 2: If you manually saved logs to your GCS bucket
gsutil -m cp -r gs://YOUR_PROJECT_ID-models/tensorboard_logs/ logs/gcp_tensorboard/

# View locally
tensorboard --logdir logs/gcp_tensorboard/

# Open browser to http://localhost:6006
```

**Note:** Vertex AI TensorBoard stores logs in a Google-managed storage location. To persist logs in your own GCS bucket, you would need to explicitly copy them in your training script.

### **Step 7: Retrieve Trained Model**

```bash
# Download model from GCS
gsutil cp gs://YOUR_PROJECT_ID-models/models/tft/tft_best.pt models/tft/

# Download checkpoints
gsutil -m cp -r gs://YOUR_PROJECT_ID-models/checkpoints/ checkpoints/
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

🏗️  Initializing TFT model...
   Parameters: 1,234,567 trainable

🏋️  Starting training...

Epoch 1/200 (2.3min) - Train: 0.0234, Val: 0.0256, MAE: 0.0123, Dir Acc: 52.3%

Epoch 2/200 (2.1min) - Train: 0.0198, Val: 0.0215, MAE: 0.0107, Dir Acc: 54.1%
  ⭐ New best model! Saving checkpoint...

Epoch 3/200 (2.2min) - Train: 0.0176, Val: 0.0189, MAE: 0.0095, Dir Acc: 56.8%
  ⭐ New best model! Saving checkpoint...

...

✅ Training complete!

📊 Final Results:
  Best Val Loss: 0.005234
  Test MAE: 0.002634, RMSE: 0.003512
  Directional Accuracy: 66.8%

📤 Uploading model to GCS...
✅ Model saved to: gs://YOUR_PROJECT_ID-models/models/tft/tft_best.pt
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

- [x] Modular project structure with shared utilities
- [x] TFT architecture with LSTM, attention, VSN, and GRU decoder
- [x] Model-agnostic data pipeline (BigQuery integration)
- [x] Ticker grouping and data augmentation
- [x] FinCast price encoder integration (optional)
- [x] Configurable training (YAML + CLI)
- [x] Early stopping & checkpointing
- [x] TensorBoard integration with clean console output
- [x] GCP Vertex AI deployment with Docker
- [x] Versioned datasets with GCS sync
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

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch nn.Transformer](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html)
- [Google Cloud Vertex AI](https://cloud.google.com/vertex-ai/docs)
- [BigQuery Python Client](https://cloud.google.com/python/docs/reference/bigquery/latest)

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
