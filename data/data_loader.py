"""Data loading and preprocessing pipeline."""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Tuple, Optional, List, Dict, Any
from google.cloud import bigquery
from google.cloud import storage
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Data loader supporting local files and BigQuery."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data loader.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_config = config['data']
        self.source = self.data_config['source']
        
        # Set up GCP clients if needed
        if self.source == 'bigquery':
            service_account = config['gcp'].get('service_account')
            if service_account and os.path.exists(service_account):
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account
            
            self.bq_client = bigquery.Client(
                project=self.data_config['bigquery']['project_id']
            )
            self.gcs_client = storage.Client(
                project=self.data_config['bigquery']['project_id']
            )
        
        self.feature_columns = self.data_config['features']['feature_columns']
        self.sequence_length = self.data_config['features']['sequence_length']
        
        # Handle multi-horizon vs single target
        horizons_config = self.data_config.get('horizons', {})
        self.multi_horizon = horizons_config.get('enabled', False)
        
        if self.multi_horizon:
            self.target_columns = horizons_config.get('targets', [])
            if not self.target_columns:
                raise ValueError("Multi-horizon enabled but no target columns specified")
            logger.info(f"Multi-horizon mode: {len(self.target_columns)} targets - {self.target_columns}")
        else:
            self.target_column = self.data_config['features'].get('target_column', 'target')
            self.target_columns = [self.target_column]
            logger.info(f"Single target mode: {self.target_column}")
    
    def load_data(
        self,
        split: str = 'train'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data from configured source.
        
        Args:
            split: Data split ('train', 'val', or 'test')
            
        Returns:
            Tuple of (features, targets)
        """
        if self.source == 'local':
            return self._load_local_data(split)
        elif self.source == 'bigquery':
            return self._load_bigquery_data(split)
        else:
            raise ValueError(f"Unknown data source: {self.source}")
    
    def _load_local_data(self, split: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from local CSV or Parquet files."""
        local_config = self.data_config['local']
        
        # Get file path
        if split == 'train':
            file_path = local_config['train_path']
        elif split == 'val':
            file_path = local_config['val_path']
        elif split == 'test':
            file_path = local_config['test_path']
        else:
            raise ValueError(f"Unknown split: {split}")
        
        logger.info(f"Loading {split} data from {file_path}")
        
        # Load file
        file_format = local_config['format']
        if file_format == 'csv':
            df = pd.read_csv(file_path)
        elif file_format == 'parquet':
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unknown file format: {file_format}")
        
        # Auto-detect feature columns if not specified
        if not self.feature_columns:
            # Exclude targets and timestamp from features
            timestamp_col = self.data_config['features'].get('timestamp_column', 'timestamp')
            exclude_cols = self.target_columns + [timestamp_col, 'date', 'timestamp']  # Cover all date column names
            # Only select numeric columns that aren't in exclude list
            self.feature_columns = [
                col for col in df.columns 
                if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
            ]
        
        # Extract features and targets
        X = df[self.feature_columns].values
        
        # Handle multi-horizon or single target
        if self.multi_horizon:
            y = df[self.target_columns].values  # Shape: (n_samples, n_horizons)
            logger.info(f"Loaded {len(df)} samples with {len(self.feature_columns)} features, "
                       f"{len(self.target_columns)} target horizons")
        else:
            y = df[self.target_columns[0]].values  # Shape: (n_samples,)
            logger.info(f"Loaded {len(df)} samples with {len(self.feature_columns)} features")
        
        return X, y
    
    def _load_bigquery_data(self, split: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from BigQuery via GCS export."""
        bq_config = self.data_config['bigquery']
        gcs_config = self.data_config['gcs']
        
        # Construct query or table reference
        if bq_config.get('query_template'):
            query = bq_config['query_template'].format(
                project_id=bq_config['project_id'],
                dataset_id=bq_config['dataset_id'],
                table=bq_config[f'{split}_table']
            )
        else:
            table_id = f"{bq_config['project_id']}.{bq_config['dataset_id']}.{bq_config[f'{split}_table']}"
            query = f"SELECT * FROM `{table_id}`"
        
        logger.info(f"Querying BigQuery for {split} data")
        
        # Execute query
        df = self.bq_client.query(query).to_dataframe()
        
        # Export to GCS for reproducibility
        bucket_name = gcs_config['bucket_name']
        export_path = f"{gcs_config['export_path']}{split}.parquet"
        
        logger.info(f"Exporting to GCS: gs://{bucket_name}/{export_path}")
        
        # Save locally first, then upload
        local_path = f"data/processed/{split}.parquet"
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        df.to_parquet(local_path)
        
        bucket = self.gcs_client.bucket(bucket_name)
        blob = bucket.blob(export_path)
        blob.upload_from_filename(local_path)
        
        # Auto-detect feature columns if not specified
        if not self.feature_columns:
            # Exclude targets and timestamp from features
            timestamp_col = self.data_config['features'].get('timestamp_column', 'timestamp')
            exclude_cols = self.target_columns + [timestamp_col, 'date', 'timestamp']  # Cover all date column names
            # Only select numeric columns that aren't in exclude list
            self.feature_columns = [
                col for col in df.columns 
                if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
            ]
        
        # Extract features and targets
        X = df[self.feature_columns].values
        
        # Handle multi-horizon or single target
        if self.multi_horizon:
            y = df[self.target_columns].values  # Shape: (n_samples, n_horizons)
            logger.info(f"Loaded {len(df)} samples with {len(self.feature_columns)} features, "
                       f"{len(self.target_columns)} target horizons from BigQuery")
        else:
            y = df[self.target_columns[0]].values  # Shape: (n_samples,)
            logger.info(f"Loaded {len(df)} samples with {len(self.feature_columns)} features from BigQuery")
        
        return X, y
    
    def create_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series modeling.
        
        Args:
            X: Feature array
            y: Target array
            
        Returns:
            Tuple of (sequence_features, sequence_targets)
        """
        sequences_X = []
        sequences_y = []
        
        for i in range(len(X) - self.sequence_length):
            sequences_X.append(X[i:i + self.sequence_length])
            sequences_y.append(y[i + self.sequence_length])
        
        return np.array(sequences_X), np.array(sequences_y)
    
    def preprocess(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess features and targets.
        
        Args:
            X: Feature array
            y: Target array
            fit: Whether to fit normalization parameters
            
        Returns:
            Tuple of (preprocessed_X, preprocessed_y)
        """
        preprocessing_config = self.data_config['preprocessing']
        normalization = preprocessing_config['normalization']
        
        # Handle NaN values by filling with column means (only for features)
        if np.isnan(X).any():
            if fit:
                # Store column means for filling NaNs
                self.fill_values_X = np.nanmean(X, axis=0)
            # Fill NaNs with stored means
            mask = np.isnan(X)
            X = X.copy()
            for col_idx in range(X.shape[1]):
                if mask[:, col_idx].any():
                    X[mask[:, col_idx], col_idx] = self.fill_values_X[col_idx]
        
        if normalization == 'standard':
            if fit or not hasattr(self, 'mean_X'):
                self.mean_X = np.mean(X, axis=0)
                self.std_X = np.std(X, axis=0) + 1e-8
                self.mean_y = np.mean(y)
                self.std_y = np.std(y) + 1e-8
            
            X = (X - self.mean_X) / self.std_X
            y = (y - self.mean_y) / self.std_y
        
        elif normalization == 'minmax':
            if fit or not hasattr(self, 'min_X'):
                self.min_X = np.min(X, axis=0)
                self.max_X = np.max(X, axis=0)
                self.range_X = self.max_X - self.min_X + 1e-8
                self.min_y = np.min(y)
                self.max_y = np.max(y)
                self.range_y = self.max_y - self.min_y + 1e-8
            
            X = (X - self.min_X) / self.range_X
            y = (y - self.min_y) / self.range_y
        
        return X, y
    
    def create_tf_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
        cache: bool = True
    ) -> tf.data.Dataset:
        """
        Create TensorFlow dataset.
        
        Args:
            X: Feature array
            y: Target array
            batch_size: Batch size
            shuffle: Whether to shuffle
            cache: Whether to cache dataset
            
        Returns:
            TensorFlow dataset
        """
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        
        if cache:
            dataset = dataset.cache()
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(X))
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def get_num_features(self) -> int:
        """Get number of features."""
        return len(self.feature_columns) if self.feature_columns else 0
