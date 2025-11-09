"""Generate dummy time series data for training."""
import argparse
import sys
from pathlib import Path

# Add parent directory to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import generate_dummy_data
from utils.logger import setup_logger

logger = setup_logger()

def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic time series data for inflation forecasting'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=10000,
        help='Number of time steps to generate (default: 10000)'
    )
    parser.add_argument(
        '--n-features',
        type=int,
        default=10,
        help='Number of features (default: 10)'
    )
    parser.add_argument(
        '--output-dir',
        default='data/dummy',
        help='Output directory (default: data/dummy)'
    )
    parser.add_argument(
        '--multi-horizon',
        action='store_true',
        default=True,
        help='Generate multi-horizon targets (1m, 3m, 6m inflation)'
    )
    parser.add_argument(
        '--single-target',
        action='store_true',
        help='Generate single target instead of multi-horizon'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Training set ratio (default: 0.7 for 70%%)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Validation set ratio (default: 0.15 for 15%%)'
    )
    parser.add_argument(
        '--shuffle',
        action='store_true',
        help='Shuffle data before splitting (breaks temporal order)'
    )
    
    args = parser.parse_args()
    
    # Determine if multi-horizon
    multi_horizon = not args.single_target
    
    logger.info("Generating dummy data...")
    logger.info(f"  Samples: {args.n_samples}")
    logger.info(f"  Features: {args.n_features}")
    logger.info(f"  Multi-horizon: {multi_horizon}")
    logger.info(f"  Output: {args.output_dir}")
    
    generate_dummy_data(
        n_samples=args.n_samples,
        n_features=args.n_features,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        output_dir=args.output_dir,
        multi_horizon=multi_horizon,
        shuffle=args.shuffle
    )
    
    logger.info("✓ Data generation complete!")
    logger.info(f"\nFiles created:")
    logger.info(f"  {args.output_dir}/train.csv")
    logger.info(f"  {args.output_dir}/val.csv")
    logger.info(f"  {args.output_dir}/test.csv")
    
    if multi_horizon:
        logger.info(f"\nTarget columns: target_1m, target_3m, target_6m")
    else:
        logger.info(f"\nTarget column: target")

if __name__ == '__main__':
    main()
