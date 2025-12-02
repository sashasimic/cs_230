#!/usr/bin/env python3
"""
FinCast Extension for Temporal Fusion Transformer (TFT)

Integrates the official FinCast foundation model (FFM) as a frozen backbone
for processing individual price series in TFT.

Requires:
- FinCast submodule: https://github.com/vincent05r/FinCast-fts
- Pre-trained weights: external/fincast/checkpoints/v1.pth (3.97GB)
"""

import sys
import torch
import torch.nn as nn
import yaml
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Try to import FinCast from submodule
FINCAST_AVAILABLE = False
FinCastModel = None
FFmHparams = None

try:
    # Add submodule to path
    fincast_path = project_root / 'external' / 'fincast' / 'src'
    if fincast_path.exists():
        sys.path.insert(0, str(fincast_path))
        # Import FFM (FinCast Foundation Model)
        from ffm import FFM as FinCastModel
        from ffm import FFmHparams
        FINCAST_AVAILABLE = True
        print("✅ FinCast (FFM) imported successfully")
    else:
        raise ImportError(f"FinCast submodule not found at: {fincast_path}")
except (ImportError, ModuleNotFoundError) as e:
    FINCAST_AVAILABLE = False
    FinCastModel = None
    FFmHparams = None
    print(f"❌ FinCast import failed: {e}")
    print("")
    print("ℹ️  To use FinCast, follow setup instructions in scripts/03_training/README.md")
    print("   Or set fincast.enabled: false in your config to disable FinCast.")
    print("")


class FinCastBackbone(nn.Module):
    """
    FinCast: Financial Forecasting Transformer Backbone
    
    Wrapper for the official pre-trained FinCast (FFM) foundation model.
    Uses the full pre-trained model (50 layers, 1280 dims) and extracts 
    hidden embeddings for downstream TFT processing.
    
    Pre-trained on 20B+ financial time points across multiple domains.
    """
    
    def __init__(
        self,
        d_model: int = 1280,  # Fixed by pre-trained model
        n_heads: int = 16,     # Fixed by pre-trained model  
        n_layers: int = 50,    # Fixed by pre-trained model
        d_ff: int = 128,       # Not used (FFM architecture)
        dropout: float = 0.1,
        max_len: int = 512,
        freeze: bool = True,
        pretrained_path: Optional[str] = None
    ):
        super().__init__()
        
        if not FINCAST_AVAILABLE:
            raise ImportError("FinCast model not available. Install the submodule first.")
        
        # FFM outputs 1280-dimensional embeddings (fixed architecture)
        self.d_model = 1280
        self.freeze = freeze
        self.max_len = max_len
        
        # Set default checkpoint path if not provided
        if pretrained_path is None:
            pretrained_path = str(project_root / 'external' / 'fincast' / 'checkpoints' / 'v1.pth')
        
        if not Path(pretrained_path).exists():
            raise FileNotFoundError(
                f"Pre-trained FinCast checkpoint not found: {pretrained_path}\n"
                f"Please download it using:\n"
                f"  python -c \"from huggingface_hub import snapshot_download; "
                f"snapshot_download(repo_id='Vincent05R/FinCast', local_dir='external/fincast/checkpoints')\""
            )
        
        # Initialize FFM with pre-trained hyperparameters
        print(f"✅ Initializing pre-trained FinCast (FFM) model")
        print(f"   Architecture: {n_layers} layers, {self.d_model} dims, {n_heads} heads")
        print(f"   Context length: {max_len}")
        print(f"   Checkpoint: {pretrained_path}")
        
        # Create FFM hyperparameters matching the checkpoint
        self.hparams = FFmHparams(
            context_len=max_len,
            horizon_len=128,        # Not used for embedding extraction
            input_patch_len=32,     # From pre-trained model
            output_patch_len=128,   # From pre-trained model
            num_layers=50,          # From pre-trained model
            num_heads=16,           # From pre-trained model
            model_dims=1280,        # From pre-trained model
            per_core_batch_size=32,
            backend="cpu" if not torch.cuda.is_available() else "gpu",
            use_positional_embedding=False,
            num_experts=4,          # MoE configuration (must match checkpoint!)
            gating_top_n=2
        )
        
        # Load the pre-trained FFM model
        print(f"   Loading pre-trained weights (3.97 GB, may take 30-60 seconds on CPU)...")
        import time
        start_time = time.time()
        self.ffm_wrapper = FinCastModel(
            hparams=self.hparams,
            checkpoint=pretrained_path,
            loading_mode=0  # Load without compilation
        )
        elapsed = time.time() - start_time
        print(f"   ✅ Pre-trained FinCast loaded successfully ({elapsed:.1f}s)")
        
        # Register the actual PyTorch model (FFmTorch._model) as a submodule
        # so PyTorch tracks its parameters
        self.ffm_model = self.ffm_wrapper._model
        
        # Create a projection layer to reduce dimensions if needed
        # This allows flexibility in output dimension
        self.output_projection = None
        if d_model != self.d_model:
            self.output_projection = nn.Linear(self.d_model, d_model)
            print(f"   Adding projection: {self.d_model} -> {d_model} dims")
        
        # Freeze if specified
        if freeze:
            self.freeze_backbone()
            print(f"   🔒 FinCast backbone frozen (991M params, no gradients)")
    
    def freeze_backbone(self):
        """Freeze all parameters in the FFM backbone."""
        # Freeze the FFM model parameters
        for param in self.ffm_model.parameters():
            param.requires_grad = False
        # Keep projection layer trainable if it exists
        if self.output_projection is not None:
            for param in self.output_projection.parameters():
                param.requires_grad = True
    
    def unfreeze_top_layers(self, n_layers: int = 1):
        """Unfreeze top n transformer layers for fine-tuning."""
        if hasattr(self.ffm_model, 'stacked_transformer') and hasattr(self.ffm_model.stacked_transformer, 'layers'):
            total_layers = len(self.ffm_model.stacked_transformer.layers)
            for i in range(max(0, total_layers - n_layers), total_layers):
                for param in self.ffm_model.stacked_transformer.layers[i].parameters():
                    param.requires_grad = True
            print(f"🔓 Unfroze top {n_layers} FinCast transformer layers for fine-tuning")
        else:
            print(f"⚠️  Could not unfreeze layers - FFM structure not as expected")
    
    def _extract_embeddings(self, x: np.ndarray) -> torch.Tensor:
        """
        Extract hidden embeddings from FFM model's stacked transformer.
        
        FFM uses patch-based processing:
        1. Splits input into patches
        2. Processes through transformer
        3. We extract the transformer hidden states
        
        Args:
            x: [batch_size, seq_len] numpy array
            
        Returns:
            embeddings: [batch_size, seq_len, d_model] tensor (interpolated from patches)
        """
        batch_size, seq_len = x.shape
        # Get device from the FFM model
        device = next(self.ffm_model.parameters()).device
        
        # Pad/truncate to match context length
        original_len = seq_len
        if seq_len < self.max_len:
            # Pad with zeros at the beginning
            padding = np.zeros((batch_size, self.max_len - seq_len))
            x = np.concatenate([padding, x], axis=1)
            seq_len = self.max_len
        elif seq_len > self.max_len:
            # Take last max_len points
            x = x[:, -self.max_len:]
            seq_len = self.max_len
        
        # Convert to tensor
        input_ts = torch.tensor(x, dtype=torch.float32, device=device)
        
        # Create padding mask (all valid data) - must be float32 for FFM
        input_padding = torch.zeros_like(input_ts, dtype=torch.float32, device=device)
        
        # Create frequency embedding (assume daily = 0)
        freq = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        
        # Extract hidden states from FFM's internal layers
        with torch.no_grad() if self.freeze else torch.enable_grad():
            # Step 1: Preprocess (patching and normalization)
            model_input, patched_padding, stats, _ = self.ffm_model._preprocess_input(
                input_ts=input_ts,
                input_padding=input_padding
            )
            # model_input shape: [batch_size, num_patches, hidden_size]
            
            # Step 2: Add frequency embedding
            f_emb = self.ffm_model.freq_emb(freq)  # [B, 1, hidden_size]
            model_input = model_input + f_emb
            
            # Step 3: Pass through stacked transformer
            hidden_states, _ = self.ffm_model.stacked_transformer(model_input, patched_padding)
            # hidden_states shape: [batch_size, num_patches, hidden_size=1280]
            
            # Step 4: Interpolate patches back to original sequence length
            # FFM uses patches of size 32, so we need to expand back to timesteps
            num_patches = hidden_states.shape[1]
            patch_len = self.hparams.input_patch_len
            
            # Repeat each patch embedding across its patch length
            # [B, N, H] -> [B, N, P, H] -> [B, N*P, H]
            expanded = hidden_states.unsqueeze(2).repeat(1, 1, patch_len, 1)
            expanded = expanded.view(batch_size, num_patches * patch_len, self.d_model)
            
            # Trim/pad to exact sequence length needed
            if expanded.shape[1] > original_len:
                # Take last original_len timesteps
                embeddings = expanded[:, -original_len:, :]
            elif expanded.shape[1] < original_len:
                # Pad at beginning
                pad_len = original_len - expanded.shape[1]
                padding = torch.zeros(batch_size, pad_len, self.d_model, device=device)
                embeddings = torch.cat([padding, expanded], dim=1)
            else:
                embeddings = expanded
        
        return embeddings
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process univariate price series through FinCast and extract embeddings.
        
        Args:
            x: [batch_size, seq_len] - scalar price series
            
        Returns:
            hidden_states: [batch_size, seq_len, d_model (1280)] or projected dims
        """
        # Convert to numpy for FFM processing
        x_np = x.detach().cpu().numpy()
        
        # Extract embeddings from FFM
        embeddings = self._extract_embeddings(x_np)
        
        # Apply output projection if specified
        if self.output_projection is not None:
            embeddings = self.output_projection(embeddings)
        
        return embeddings


class TemporalFusionTransformerWithFinCast(nn.Module):
    """
    Temporal Fusion Transformer with FinCast Backbone Integration.
    
    Uses FinCast to process price series individually, then combines
    representations with other features for the main TFT.
    """
    
    def __init__(
        self,
        config: dict,
        num_features: int,
        static_features: List[str],
        fincast_config: Optional[Dict] = None,
        tft_class = None
    ):
        super().__init__()
        
        if not FINCAST_AVAILABLE:
            raise ImportError("FinCast not available. Cannot use TemporalFusionTransformerWithFinCast.")
        
        self.config = config
        self.num_features = num_features
        self.static_features = static_features
        
        # Extract price and rest indices from config
        self.price_indices = self._get_price_indices(config, fincast_config)
        self.num_price_series = len(self.price_indices)
        self.rest_indices = [i for i in range(num_features) if i not in self.price_indices]
        self.num_rest_features = len(self.rest_indices)
        
        if self.num_price_series == 0:
            raise ValueError("No price series found in fincast.price_tickers. Check your config.")
        
        # Initialize FinCast backbone
        fincast_config = fincast_config or {}
        
        # Determine output dimension for FinCast projection
        # FFM outputs 1280 dims, we project to output_dim for efficiency
        fincast_output_dim = fincast_config.get('output_dim', 128)
        
        self.fincast = FinCastBackbone(
            d_model=fincast_output_dim,  # Will create projection from 1280 -> this size
            n_heads=16,                   # Fixed by pre-trained model
            n_layers=50,                  # Fixed by pre-trained model
            d_ff=128,                     # Not used
            dropout=fincast_config.get('dropout', 0.1),
            max_len=fincast_config.get('max_context_len', 512),
            freeze=fincast_config.get('freeze_backbone', True),
            pretrained_path=fincast_config.get('checkpoint_path')
        )
        
        # Augmented feature dimension
        # Each price series -> fincast_output_dim dimensional embedding
        fincast_feat_dim = self.num_price_series * fincast_output_dim
        augmented_features = self.num_rest_features + fincast_feat_dim
        
        print(f"\n🔧 FinCast Integration:")
        print(f"   Price series: {self.num_price_series}")
        print(f"   FinCast output dim: {fincast_output_dim} (projected from 1280)")
        print(f"   FinCast features: {fincast_feat_dim}")
        print(f"   Rest features: {self.num_rest_features}")
        print(f"   Total augmented: {augmented_features}")
        
        # Add LayerNorm to normalize mixed features (FinCast embeddings + rest features)
        # This is critical to prevent gradient explosion from feature scale mismatch
        self.feature_norm = nn.LayerNorm(augmented_features)
        print(f"   ⚖️  Added LayerNorm({augmented_features}) for feature normalization")
        
        # Initialize main TFT with augmented features
        if tft_class is None:
            raise ValueError("tft_class must be provided")
        
        self.tft = tft_class(
            config=config,
            num_features=augmented_features
        )
        
        # Store feature indices as buffers (for device handling)
        self.register_buffer('price_idx', torch.tensor(self.price_indices, dtype=torch.long))
        self.register_buffer('rest_idx', torch.tensor(self.rest_indices, dtype=torch.long))
    
    def _get_price_indices(self, config: dict, fincast_config: dict) -> List[int]:
        """
        Extract indices of price features based on fincast.price_tickers config.
        """
        try:
            # Get tickers to process from config
            price_tickers = fincast_config.get('price_tickers', [])
            if not price_tickers:
                raise ValueError("fincast.price_tickers is empty. Specify which tickers to process.")
            
            # Try to load feature names from feature_names.txt (TFT format)
            feature_names_path = Path('data/processed/feature_names.txt')
            if feature_names_path.exists():
                with open(feature_names_path, 'r') as f:
                    features = [line.strip() for line in f.readlines()]
                
                # Find close_<ticker> features for specified tickers
                price_indices = []
                for ticker in price_tickers:
                    # Look for close_{ticker} feature
                    feature_name = f'close_{ticker}'
                    try:
                        idx = features.index(feature_name)
                        price_indices.append(idx)
                    except ValueError:
                        print(f"⚠️  Warning: {feature_name} not found in features")
                
                if price_indices:
                    print(f"\n📈 Found {len(price_indices)} price series from config:")
                    for idx in price_indices:
                        print(f"   {idx:3d}: {features[idx]}")
                    return price_indices
                else:
                    raise ValueError(f"None of the specified tickers found in features: {price_tickers}")
            
            # Fallback: Try loading from metadata.yaml (decoder format)
            metadata_path = Path('data/processed/metadata.yaml')
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = yaml.safe_load(f)
                    features = metadata.get('features', [])
                    
                    # Check if features is a list (decoder format)
                    if isinstance(features, list):
                        price_indices = []
                        for ticker in price_tickers:
                            feature_name = f'close_{ticker}'
                            try:
                                idx = features.index(feature_name)
                                price_indices.append(idx)
                            except ValueError:
                                print(f"⚠️  Warning: {feature_name} not found in features")
                        
                        if price_indices:
                            print(f"\n📈 Found {len(price_indices)} price series from config:")
                            for idx in price_indices:
                                print(f"   {idx:3d}: {features[idx]}")
                            return price_indices
        except Exception as e:
            print(f"⚠️  Error loading price indices: {e}")
            print("   Falling back to all close_* features")
            
            # Fallback: use all close_* features from feature_names.txt
            try:
                feature_names_path = Path('data/processed/feature_names.txt')
                if feature_names_path.exists():
                    with open(feature_names_path, 'r') as f:
                        features = [line.strip() for line in f.readlines()]
                    price_indices = [
                        i for i, feat in enumerate(features)
                        if feat.startswith('close_') and not feat.endswith('_returns')
                    ]
                    if price_indices:
                        print(f"\n📈 Using all {len(price_indices)} close_* features as fallback")
                        return price_indices
            except Exception:
                pass
        
        raise ValueError("Could not determine price feature indices. Check config and feature_names.txt.")
    
    def process_with_fincast(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process input through FinCast for price series and combine with other features.
        
        Args:
            x: [batch_size, seq_len, num_features]
            
        Returns:
            x_augmented: [batch_size, seq_len, augmented_features]
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Extract price and rest features
        x_prices = x[:, :, self.price_idx]  # [B, T, N_prices]
        x_rest = x[:, :, self.rest_idx]     # [B, T, F_rest]
        
        # Reshape prices for FinCast processing
        # [B, T, N] -> [B*N, T]
        x_prices_flat = x_prices.permute(0, 2, 1).contiguous()  # [B, N, T]
        x_prices_flat = x_prices_flat.view(-1, seq_len)  # [B*N, T]
        
        # Process through FinCast
        # FinCast may pad/truncate internally, so output might be different length
        fincast_hidden = self.fincast(x_prices_flat)  # [B*N, T', H_f] where T' might != T
        
        # Get actual output sequence length
        _, output_seq_len, hidden_dim = fincast_hidden.shape
        
        # If FinCast changed sequence length, we need to handle it
        if output_seq_len != seq_len:
            if output_seq_len > seq_len:
                # FinCast padded - take last seq_len steps
                fincast_hidden = fincast_hidden[:, -seq_len:, :]
            else:
                # FinCast truncated - pad at beginning
                padding = torch.zeros(
                    fincast_hidden.shape[0], 
                    seq_len - output_seq_len,
                    hidden_dim,
                    device=device
                )
                fincast_hidden = torch.cat([padding, fincast_hidden], dim=1)
        
        # Reshape back and flatten ticker dimension
        # [B*N, T, H_f] -> [B, N, T, H_f]
        fincast_hidden = fincast_hidden.view(
            batch_size, self.num_price_series, seq_len, hidden_dim
        )
        # [B, N, T, H_f] -> [B, T, N, H_f]
        fincast_hidden = fincast_hidden.permute(0, 2, 1, 3).contiguous()
        # [B, T, N, H_f] -> [B, T, N*H_f]
        fincast_hidden = fincast_hidden.view(batch_size, seq_len, -1)
        
        # Concatenate with rest features
        x_augmented = torch.cat([x_rest, fincast_hidden], dim=-1)
        
        # Normalize mixed features to prevent gradient explosion
        x_augmented = self.feature_norm(x_augmented)
        
        return x_augmented
    
    def forward(
        self,
        x: torch.Tensor,
        static_features: Optional[torch.Tensor] = None,
        y_future: Optional[torch.Tensor] = None,
        teacher_forcing: bool = True,
        return_attention: bool = False
    ):
        """
        Forward pass with FinCast preprocessing.
        
        Args:
            x: [batch_size, lookback, num_features]
            static_features: [batch_size, num_static] (optional)
            y_future: [batch_size, num_horizons] (optional, for teacher forcing)
            teacher_forcing: whether to use teacher forcing
            return_attention: whether to return attention weights
            
        Returns:
            predictions: [batch_size, num_horizons]
            attention_weights: (optional) if return_attention=True
        """
        # Process through FinCast
        x = self.process_with_fincast(x)
        
        # Forward through main TFT
        return self.tft(x, static_features, y_future, teacher_forcing, return_attention)
    
    def unfreeze_fincast_top(self, n_layers: int = 1):
        """Unfreeze top layers of FinCast for fine-tuning."""
        if self.fincast is not None:
            self.fincast.unfreeze_top_layers(n_layers)


def create_fincast_tft(config: dict, num_features: int, static_features: List[str], 
                       fincast_config: Dict, tft_class):
    """
    Factory function to create a TFT with FinCast backbone.
    
    Args:
        config: Model configuration
        num_features: Number of input features
        static_features: List of static feature names
        fincast_config: FinCast configuration
        tft_class: The base TFT class
        
    Returns:
        TemporalFusionTransformerWithFinCast instance
    """
    if not FINCAST_AVAILABLE:
        raise ImportError(
            "FinCast not available. Either:\n"
            "1. Set fincast.enabled: false in config, or\n"
            "2. Install FinCast submodule (see scripts/03_training/README.md)"
        )
    
    return TemporalFusionTransformerWithFinCast(
        config=config,
        num_features=num_features,
        static_features=static_features,
        fincast_config=fincast_config,
        tft_class=tft_class
    )