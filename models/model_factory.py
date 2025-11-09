"""Model factory for creating different architectures."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory for creating time series regression models."""
    
    @staticmethod
    def create_model(
        model_type: str,
        input_shape: tuple,
        config: Dict[str, Any]
    ) -> keras.Model:
        """
        Create model based on type.
        
        Args:
            model_type: Model type ('mlp', 'lstm', 'transformer')
            input_shape: Input shape (sequence_length, n_features)
            config: Model configuration
            
        Returns:
            Compiled Keras model
        """
        logger.info(f"Creating {model_type.upper()} model with input shape {input_shape}")
        
        if model_type == 'mlp':
            return ModelFactory._create_mlp(input_shape, config)
        elif model_type == 'lstm':
            return ModelFactory._create_lstm(input_shape, config)
        elif model_type == 'transformer':
            return ModelFactory._create_transformer(input_shape, config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def _create_mlp(input_shape: tuple, config: Dict[str, Any]) -> keras.Model:
        """Create MLP model."""
        model_config = config['model']
        mlp_config = model_config['mlp']
        
        inputs = layers.Input(shape=input_shape)
        
        # Flatten time series
        x = layers.Flatten()(inputs)
        
        # Hidden layers
        for dim in mlp_config['layer_dims']:
            x = layers.Dense(
                dim,
                activation=model_config['activation']
            )(x)
            x = layers.Dropout(model_config['dropout'])(x)
        
        # Output layer
        outputs = layers.Dense(1)(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='MLP')
        
        logger.info(f"MLP model created with {model.count_params()} parameters")
        
        return model
    
    @staticmethod
    def _create_lstm(input_shape: tuple, config: Dict[str, Any]) -> keras.Model:
        """Create LSTM model."""
        model_config = config['model']
        lstm_config = model_config['lstm']
        
        inputs = layers.Input(shape=input_shape)
        x = inputs
        
        # LSTM layers
        for i in range(model_config['num_layers']):
            return_sequences = (i < model_config['num_layers'] - 1) or lstm_config['return_sequences']
            
            if lstm_config['bidirectional']:
                x = layers.Bidirectional(
                    layers.LSTM(
                        model_config['hidden_dim'],
                        return_sequences=return_sequences,
                        dropout=model_config['dropout']
                    )
                )(x)
            else:
                x = layers.LSTM(
                    model_config['hidden_dim'],
                    return_sequences=return_sequences,
                    dropout=model_config['dropout']
                )(x)
        
        # Dense layers
        x = layers.Dense(
            model_config['hidden_dim'] // 2,
            activation=model_config['activation']
        )(x)
        x = layers.Dropout(model_config['dropout'])(x)
        
        # Output layer
        outputs = layers.Dense(1)(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='LSTM')
        
        logger.info(f"LSTM model created with {model.count_params()} parameters")
        
        return model
    
    @staticmethod
    def _create_transformer(input_shape: tuple, config: Dict[str, Any]) -> keras.Model:
        """Create Transformer model."""
        model_config = config['model']
        transformer_config = model_config['transformer']
        
        inputs = layers.Input(shape=input_shape)
        
        # Positional encoding
        x = ModelFactory._positional_encoding(inputs, model_config['hidden_dim'])
        
        # Transformer blocks
        for _ in range(model_config['num_layers']):
            x = ModelFactory._transformer_block(
                x,
                model_config['hidden_dim'],
                transformer_config['num_heads'],
                transformer_config['ff_dim'],
                transformer_config['attention_dropout']
            )
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = layers.Dense(
            model_config['hidden_dim'],
            activation=model_config['activation']
        )(x)
        x = layers.Dropout(model_config['dropout'])(x)
        
        # Output layer
        outputs = layers.Dense(1)(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='Transformer')
        
        logger.info(f"Transformer model created with {model.count_params()} parameters")
        
        return model
    
    @staticmethod
    def _positional_encoding(x: tf.Tensor, d_model: int) -> tf.Tensor:
        """Add positional encoding to input."""
        # Project to model dimension
        x = layers.Dense(d_model)(x)
        
        # Simple learned positional encoding
        seq_len = tf.shape(x)[1]
        position = tf.range(start=0, limit=seq_len, delta=1)
        position_embedding = layers.Embedding(
            input_dim=1000,  # Max sequence length
            output_dim=d_model
        )(position)
        
        return x + position_embedding
    
    @staticmethod
    def _transformer_block(
        x: tf.Tensor,
        d_model: int,
        num_heads: int,
        ff_dim: int,
        dropout: float
    ) -> tf.Tensor:
        """Transformer encoder block."""
        # Multi-head attention
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout
        )(x, x)
        attn_output = layers.Dropout(dropout)(attn_output)
        x1 = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
        
        # Feed-forward network
        ff_output = layers.Dense(ff_dim, activation='relu')(x1)
        ff_output = layers.Dropout(dropout)(ff_output)
        ff_output = layers.Dense(d_model)(ff_output)
        ff_output = layers.Dropout(dropout)(ff_output)
        x2 = layers.LayerNormalization(epsilon=1e-6)(x1 + ff_output)
        
        return x2


def create_model(
    model_type: str,
    input_shape: tuple,
    config: Dict[str, Any]
) -> keras.Model:
    """
    Convenience function to create model.
    
    Args:
        model_type: Model type
        input_shape: Input shape
        config: Configuration
        
    Returns:
        Keras model
    """
    return ModelFactory.create_model(model_type, input_shape, config)
