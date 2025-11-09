"""Training orchestration."""

import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
from datetime import datetime

from models import create_model
from data import DataLoader
from training.callbacks import get_callbacks
from utils.visualization import plot_training_history, plot_predictions

logger = logging.getLogger(__name__)


class Trainer:
    """Training orchestrator."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model = None
        self.history = None
        
        # Set random seeds
        self._set_seeds(config['seed'])
        
        # Enable mixed precision if configured
        if config['training'].get('mixed_precision', False):
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            logger.info("Mixed precision training enabled")
    
    def _set_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        tf.random.set_seed(seed)
        logger.info(f"Random seed set to {seed}")
    
    def build_model(self, input_shape: tuple):
        """
        Build model architecture.
        
        Args:
            input_shape: Input shape (sequence_length, n_features)
        """
        model_type = self.config['model']['type']
        self.model = create_model(model_type, input_shape, self.config)
        
        # Compile model
        self._compile_model()
        
        # Print model summary
        logger.info("\nModel Summary:")
        self.model.summary(print_fn=logger.info)
    
    def _compile_model(self):
        """Compile model with optimizer and loss."""
        training_config = self.config['training']
        
        # Get optimizer
        optimizer = self._get_optimizer(training_config)
        
        # Get loss
        loss = self._get_loss(training_config['loss'])
        
        # Get metrics
        metrics = self._get_metrics(training_config['metrics'])
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        logger.info(f"Model compiled with optimizer={training_config['optimizer']}, loss={training_config['loss']}")
    
    def _get_optimizer(self, training_config: Dict[str, Any]) -> keras.optimizers.Optimizer:
        """Get optimizer from config."""
        optimizer_name = training_config['optimizer']
        lr = training_config['learning_rate']
        
        if optimizer_name == 'adam':
            params = training_config['optimizer_params']['adam']
            return keras.optimizers.Adam(
                learning_rate=lr,
                beta_1=params['beta_1'],
                beta_2=params['beta_2'],
                epsilon=params['epsilon']
            )
        elif optimizer_name == 'sgd':
            params = training_config['optimizer_params']['sgd']
            return keras.optimizers.SGD(
                learning_rate=lr,
                momentum=params['momentum'],
                nesterov=params['nesterov']
            )
        elif optimizer_name == 'rmsprop':
            return keras.optimizers.RMSprop(learning_rate=lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _get_loss(self, loss_name: str) -> keras.losses.Loss:
        """Get loss function from config."""
        if loss_name == 'mse':
            return keras.losses.MeanSquaredError()
        elif loss_name == 'mae':
            return keras.losses.MeanAbsoluteError()
        elif loss_name == 'huber':
            return keras.losses.Huber()
        else:
            raise ValueError(f"Unknown loss: {loss_name}")
    
    def _get_metrics(self, metric_names: list) -> list:
        """Get metrics from config."""
        metrics = []
        for name in metric_names:
            if name == 'mae':
                metrics.append(keras.metrics.MeanAbsoluteError(name='mae'))
            elif name == 'mse':
                metrics.append(keras.metrics.MeanSquaredError(name='mse'))
            elif name == 'rmse':
                metrics.append(keras.metrics.RootMeanSquaredError(name='rmse'))
        return metrics
    
    def train(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        model_name: Optional[str] = None
    ):
        """
        Train model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            model_name: Optional model name for checkpoints
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        if model_name is None:
            model_name = f"{self.config['model']['type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        training_config = self.config['training']
        
        # Get callbacks
        callbacks = get_callbacks(self.config, model_name)
        
        # Train
        logger.info(f"\nStarting training for {training_config['epochs']} epochs...")
        
        self.history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=training_config['epochs'],
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Training completed")
        
        # Plot training history
        visualization_config = self.config.get('logging', {}).get('visualization', {})
        if visualization_config.get('enabled', True):  # Default to True
            plot_dir = visualization_config.get('plot_dir', 'logs/plots')
            os.makedirs(plot_dir, exist_ok=True)
            plot_path = os.path.join(plot_dir, f'{model_name}_history.png')
            plot_training_history(self.history.history, plot_path)
            logger.info(f"Training history plot saved to {plot_path}")
    
    def evaluate(
        self,
        test_dataset: tf.data.Dataset
    ) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Args:
            test_dataset: Test dataset
            
        Returns:
            Dictionary of metrics
        """
        if self.model is None:
            raise ValueError("Model not built")
        
        logger.info("\nEvaluating model on test set...")
        
        results = self.model.evaluate(test_dataset, verbose=1)
        
        # Create results dictionary
        metric_names = ['loss'] + [m.name for m in self.model.metrics]
        results_dict = dict(zip(metric_names, results))
        
        logger.info("\nTest Results:")
        for name, value in results_dict.items():
            logger.info(f"  {name}: {value:.6f}")
        
        return results_dict
    
    def predict(
        self,
        X: np.ndarray,
        denormalize: bool = False,
        data_loader: Optional[DataLoader] = None
    ) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input data
            denormalize: Whether to denormalize predictions
            data_loader: DataLoader for denormalization
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not built")
        
        predictions = self.model.predict(X, verbose=0)
        
        if denormalize and data_loader is not None:
            # Denormalize predictions
            if hasattr(data_loader, 'mean_y'):
                predictions = predictions * data_loader.std_y + data_loader.mean_y
            elif hasattr(data_loader, 'min_y'):
                predictions = predictions * data_loader.range_y + data_loader.min_y
        
        return predictions
    
    def save_model(self, path: str):
        """Save model to file."""
        if self.model is None:
            raise ValueError("Model not built")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model from file."""
        self.model = keras.models.load_model(path)
        logger.info(f"Model loaded from {path}")
