"""Training callbacks."""

import os
import tensorflow as tf
from tensorflow import keras
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def get_callbacks(config: Dict[str, Any], model_name: str) -> List[keras.callbacks.Callback]:
    """
    Create training callbacks based on config.
    
    Args:
        config: Configuration dictionary
        model_name: Name for saving checkpoints
        
    Returns:
        List of callbacks
    """
    callbacks = []
    training_config = config['training']
    logging_config = config['logging']
    
    # Early stopping
    if training_config['early_stopping']['enabled']:
        es_config = training_config['early_stopping']
        early_stopping = keras.callbacks.EarlyStopping(
            monitor=es_config['monitor'],
            patience=es_config['patience'],
            restore_best_weights=es_config['restore_best_weights'],
            min_delta=es_config['min_delta'],
            verbose=1
        )
        callbacks.append(early_stopping)
        logger.info(f"Added early stopping: monitor={es_config['monitor']}, patience={es_config['patience']}")
    
    # Model checkpointing
    if training_config['checkpointing']['enabled']:
        checkpoint_dir = logging_config['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}_best.h5')
        cp_config = training_config['checkpointing']
        
        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=cp_config['monitor'],
            save_best_only=cp_config['save_best_only'],
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(checkpoint)
        logger.info(f"Added checkpointing: {checkpoint_path}")
    
    # Learning rate scheduling
    if training_config['lr_schedule']['enabled']:
        lr_config = training_config['lr_schedule']
        
        if lr_config['type'] == 'reduce_on_plateau':
            lr_scheduler = keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=lr_config['factor'],
                patience=lr_config['patience'],
                min_lr=lr_config['min_lr'],
                verbose=1
            )
            callbacks.append(lr_scheduler)
            logger.info("Added ReduceLROnPlateau scheduler")
        
        elif lr_config['type'] == 'exponential':
            def exponential_decay(epoch, lr):
                return lr * lr_config['factor']
            
            lr_scheduler = keras.callbacks.LearningRateScheduler(exponential_decay)
            callbacks.append(lr_scheduler)
            logger.info("Added exponential decay scheduler")
        
        elif lr_config['type'] == 'cosine':
            # Cosine decay will be handled by optimizer
            pass
    
    # TensorBoard
    if logging_config['tensorboard']['enabled']:
        tb_config = logging_config['tensorboard']
        tensorboard_dir = os.path.join(tb_config['log_dir'], model_name)
        os.makedirs(tensorboard_dir, exist_ok=True)
        
        tensorboard = keras.callbacks.TensorBoard(
            log_dir=tensorboard_dir,
            update_freq=tb_config['update_freq'],
            profile_batch=tb_config['profile_batch']
        )
        callbacks.append(tensorboard)
        logger.info(f"Added TensorBoard: {tensorboard_dir}")
    
    # CSV logger
    metrics_dir = logging_config['metrics_dir']
    os.makedirs(metrics_dir, exist_ok=True)
    csv_path = os.path.join(metrics_dir, f'{model_name}_metrics.csv')
    csv_logger = keras.callbacks.CSVLogger(csv_path)
    callbacks.append(csv_logger)
    logger.info(f"Added CSV logger: {csv_path}")
    
    return callbacks
