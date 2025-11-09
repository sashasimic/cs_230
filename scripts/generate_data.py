"""Generate dummy data."""
import argparse
from data import generate_dummy_data
from utils.logger import setup_logger

logger = setup_logger()

parser = argparse.ArgumentParser()
parser.add_argument('--n-samples', type=int, default=10000)
parser.add_argument('--n-features', type=int, default=10)
parser.add_argument('--output-dir', default='data/dummy')
args = parser.parse_args()

generate_dummy_data(n_samples=args.n_samples, n_features=args.n_features, 
                   output_dir=args.output_dir)
logger.info("✓ Done")
