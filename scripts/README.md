# Utility Scripts

This directory contains standalone utility scripts for the CS230 project.

## Available Scripts

### [generate_data.py](cci:7://file:///Users/sasasimic/CascadeProjects/cs230/inflation_predictor/generate_data.py:0:0-0:0)

Generate synthetic time series data for testing and development.

**Usage:**
```bash
# Generate default dataset (10,000 samples, 10 features)
python scripts/generate_data.py

# Custom configuration
python scripts/generate_data.py \
  --n-samples 50000 \
  --n-features 20 \
  --output-dir data/dummy