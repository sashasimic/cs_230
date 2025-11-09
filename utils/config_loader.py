"""Configuration loader utility."""

import yaml
import os
from typing import Dict, Any


def load_config(config_path: str = 'configs/config.yaml') -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config YAML file
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with environment variables if present
    if 'GCP_PROJECT_ID' in os.environ:
        config['data']['bigquery']['project_id'] = os.environ['GCP_PROJECT_ID']
    
    if 'GCS_BUCKET' in os.environ:
        config['data']['gcs']['bucket_name'] = os.environ['GCS_BUCKET']
    
    if 'MODEL_TYPE' in os.environ:
        config['model']['type'] = os.environ['MODEL_TYPE']
    
    return config


def update_config(config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Update config with command-line arguments.
    
    Args:
        config: Base configuration
        **kwargs: Override parameters
        
    Returns:
        Updated configuration
    """
    for key, value in kwargs.items():
        if value is not None:
            # Handle nested keys like 'training.batch_size'
            keys = key.split('.')
            current = config
            for k in keys[:-1]:
                current = current.setdefault(k, {})
            current[keys[-1]] = value
    
    return config
