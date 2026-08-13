import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import ID_COL, SUBMISSION_DIR, SAMPLE_SUBMISSION_PATH
from src.utils import get_logger, format_and_save_submission
from src.data import load_raw_data
from src.features import engineer_features

logger = get_logger("predict_pipeline")

def parse_args():
    parser = argparse.ArgumentParser(description="Zindi Financial Stress Prediction Inference")
    parser.add_argument("--test-file", type=str, default=None, help="Optional path to custom test CSV file")
    return parser.parse_args()

def main():
    args = parse_args()
    logger.info("Running standalone prediction pipeline...")
    
    train_df, test_df, sample_sub_df = load_raw_data()
    if args.test_file:
        logger.info(f"Overriding test dataset with custom file: {args.test_file}")
        test_df = pd.read_csv(args.test_file)
        
    logger.info(f"Test samples to predict: {len(test_df)}")
    # Features can be extracted and used for prediction as needed
    logger.info("Prediction pipeline ready.")

if __name__ == "__main__":
    main()
