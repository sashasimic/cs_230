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
        checkpoint_dir = logging_config.get('checkpoint_dir', 'checkpoints')
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
    lr_schedule_config = training_config.get('learning_rate_schedule', {})
    if lr_schedule_config.get('enabled', False):
        lr_config = lr_schedule_config
        
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
    tensorboard_config = logging_config.get('tensorboard', False)
    if isinstance(tensorboard_config, bool):
        tensorboard_enabled = tensorboard_config
        tb_log_dir = logging_config.get('log_dir', 'logs')
    else:
        tensorboard_enabled = tensorboard_config.get('enabled', False)
        tb_log_dir = tensorboard_config.get('log_dir', 'logs/tensorboard')
    
    if tensorboard_enabled:
        tensorboard_dir = os.path.join(tb_log_dir, model_name)
        os.makedirs(tensorboard_dir, exist_ok=True)
        
        tensorboard = keras.callbacks.TensorBoard(
            log_dir=tensorboard_dir,
            update_freq='epoch',
            profile_batch=0
        )
        callbacks.append(tensorboard)
        logger.info(f"Added TensorBoard: {tensorboard_dir}")
    
    # CSV logger
    metrics_dir = logging_config.get('metrics_dir', 'logs/metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    csv_path = os.path.join(metrics_dir, f'{model_name}_metrics.csv')
    csv_logger = keras.callbacks.CSVLogger(csv_path)
    callbacks.append(csv_logger)
    logger.info(f"Added CSV logger: {csv_path}")
    
    return callbacks
