# Model Inference Scripts

Scripts for making predictions with trained models.

## Overview

This folder will contain scripts for:
- Loading trained models
- Making predictions on new data
- Batch inference
- Real-time inference
- Model serving

## Coming Soon

Inference scripts will be added after model training is complete.

## Planned Scripts

### `predict.py` - Make Predictions

Load a trained model and make predictions on new data.

```bash
# Predict on new data
python scripts/04_inference/predict.py \
    --checkpoint models/tft/tft_best.pt \
    --input data/new_data.parquet \
    --output predictions.csv
```

### `batch_inference.py` - Batch Predictions

Run predictions on large datasets in batches.

```bash
# Batch inference
python scripts/04_inference/batch_inference.py \
    --checkpoint models/tft/tft_best.pt \
    --input-dir data/inference/ \
    --output-dir predictions/
```

### `serve_model.py` - Model Serving

Serve model as REST API for real-time predictions.

```bash
# Start model server
python scripts/04_inference/serve_model.py \
    --checkpoint models/tft/tft_best.pt \
    --port 8000
```

## Typical Workflow

```
1. Train model
   python scripts/03_training/tft/tft_train_local.py
   ↓
2. Make predictions
   python scripts/04_inference/predict.py --checkpoint models/tft/tft_best.pt
   ↓
3. Analyze results
   python scripts/04_inference/analyze_predictions.py
```

## Integration with Training

Inference scripts will work with any trained model checkpoint:

- **Local models**: `models/tft/tft_best.pt`
- **Cloud models**: Downloaded from GCS
- **Versioned models**: From model registry

## Requirements

- Trained model checkpoint
- Input data in same format as training
- Scalers from training (for inverse transform)

---

**Status**: 🚧 Under development

**Next steps**: Implement basic prediction script after TFT training is complete.