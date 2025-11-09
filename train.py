"""Main training script."""
# CRITICAL: Set TensorFlow environment variables BEFORE any imports
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTHONUNBUFFERED'] = '1'

import argparse
import sys
from datetime import datetime

# Print immediate feedback (flush to avoid buffering)
print("[1/4] Starting CS230 Training Script...", flush=True)
print("[2/4] Loading utilities...", flush=True)

from utils.config_loader import load_config, update_config
from utils.logger import setup_logger

logger = setup_logger()
logger.info("Logger initialized")

print("[3/4] Loading TensorFlow in CPU-only mode (30-60 seconds)...", flush=True)

from data import DataLoader, generate_dummy_data
from training import Trainer
from utils.visualization import plot_predictions

print("[4/4] All modules loaded successfully!", flush=True)
logger.info("All imports completed")

def parse_args():
    parser = argparse.ArgumentParser(description='Train time series model')
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--model-type', choices=['mlp', 'lstm', 'transformer'])
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--generate-dummy', action='store_true')
    parser.add_argument('--model-name', type=str)
    return parser.parse_args()

def main():
    args = parse_args()
    print("\n" + "="*80, flush=True)
    print("CS230 Deep Learning Project - Time Series Regression", flush=True)
    print("="*80 + "\n", flush=True)
    
    logger.info("="*80)
    logger.info("CS230 Deep Learning Project")
    logger.info("="*80)
    
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    logger.info("✓ Configuration loaded")
    
    # Override config
    overrides = {}
    if args.model_type: overrides['model.type'] = args.model_type
    if args.batch_size: overrides['training.batch_size'] = args.batch_size
    if args.epochs: overrides['training.epochs'] = args.epochs
    if args.learning_rate: overrides['training.learning_rate'] = args.learning_rate
    if overrides:
        logger.info(f"Overriding config: {overrides}")
        config = update_config(config, **overrides)
    
    if args.generate_dummy:
        logger.info("Generating dummy data...")
        print("Generating dummy data (10,000 samples)...", flush=True)
        generate_dummy_data(output_dir='data/dummy', seed=config['seed'])
        config['data']['local']['train_path'] = 'data/dummy/train.csv'
        config['data']['local']['val_path'] = 'data/dummy/val.csv'
        config['data']['local']['test_path'] = 'data/dummy/test.csv'
        logger.info("✓ Dummy data generated")
    
    # Load data
    logger.info("Initializing data loader...")
    print("\nLoading data...", flush=True)
    data_loader = DataLoader(config)
    X_train, y_train = data_loader.load_data('train')
    X_val, y_val = data_loader.load_data('val')
    X_test, y_test = data_loader.load_data('test')
    logger.info("✓ Data loaded")
    
    # Preprocess
    logger.info("Preprocessing data...")
    print("Preprocessing and normalizing...", flush=True)
    X_train, y_train = data_loader.preprocess(X_train, y_train, fit=True)
    X_val, y_val = data_loader.preprocess(X_val, y_val)
    X_test, y_test = data_loader.preprocess(X_test, y_test)
    logger.info("✓ Data preprocessed")
    
    # Create sequences
    logger.info("Creating sequences...")
    print("Creating time series sequences...", flush=True)
    X_train_seq, y_train_seq = data_loader.create_sequences(X_train, y_train)
    X_val_seq, y_val_seq = data_loader.create_sequences(X_val, y_val)
    X_test_seq, y_test_seq = data_loader.create_sequences(X_test, y_test)
    logger.info(f"✓ Sequences created: train={X_train_seq.shape}, val={X_val_seq.shape}, test={X_test_seq.shape}")
    print(f"  Train: {X_train_seq.shape}, Val: {X_val_seq.shape}, Test: {X_test_seq.shape}", flush=True)
    
    # Create datasets
    logger.info("Creating TensorFlow datasets...")
    print("Creating TensorFlow datasets...", flush=True)
    batch_size = config['training']['batch_size']
    train_ds = data_loader.create_tf_dataset(X_train_seq, y_train_seq, batch_size, True)
    val_ds = data_loader.create_tf_dataset(X_val_seq, y_val_seq, batch_size, False)
    test_ds = data_loader.create_tf_dataset(X_test_seq, y_test_seq, batch_size, False)
    logger.info("✓ Datasets created")
    
    # Train
    logger.info("Initializing trainer...")
    print(f"\nBuilding {config['model']['type'].upper()} model...", flush=True)
    trainer = Trainer(config)
    input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
    trainer.build_model(input_shape)
    
    model_name = args.model_name or f"{config['model']['type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger.info(f"Starting training: {model_name}")
    print(f"\nStarting training for {config['training']['epochs']} epochs...\n", flush=True)
    trainer.train(train_ds, val_ds, model_name)
    
    # Evaluate
    logger.info("Evaluating on test set...")
    print("\nEvaluating on test set...", flush=True)
    results = trainer.evaluate(test_ds)
    
    logger.info("Saving model...")
    trainer.save_model(f"models/{model_name}_final.keras")  # Changed from .h5 to .keras for Keras 3.x
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nTest Results: {results}")
    print(f"Model saved: {model_name}")
    print(f"\nCheckpoints: checkpoints/")
    print(f"Logs: logs/")
    print(f"Plots: logs/plots/")
    print("\nTo view TensorBoard: tensorboard --logdir=logs/tensorboard")
    print("="*80 + "\n")
    
    logger.info(f"Test Results: {results}")
    logger.info(f"Model: {model_name}")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.exception(f"Error: {e}")
        sys.exit(1)
