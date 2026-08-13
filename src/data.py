import pandas as pd
import numpy as np
from src.config import TRAIN_PATH, TEST_PATH, SAMPLE_SUBMISSION_PATH, ID_COL, TARGET_COL, CATEGORICAL_COLS
from src.utils import get_logger

logger = get_logger()

def load_raw_data():
    """Load raw dataset files and run initial validation."""
    logger.info(f"Loading Train dataset from {TRAIN_PATH}...")
    train_df = pd.read_csv(TRAIN_PATH)
    logger.info(f"Train dataset loaded: {train_df.shape[0]} rows, {train_df.shape[1]} columns.")
    
    logger.info(f"Loading Test dataset from {TEST_PATH}...")
    test_df = pd.read_csv(TEST_PATH)
    logger.info(f"Test dataset loaded: {test_df.shape[0]} rows, {test_df.shape[1]} columns.")
    
    logger.info(f"Loading SampleSubmission from {SAMPLE_SUBMISSION_PATH}...")
    sample_sub_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    
    # Sanity checks
    assert TARGET_COL in train_df.columns, f"Target column '{TARGET_COL}' missing from Train data"
    assert TARGET_COL not in test_df.columns, f"Target column '{TARGET_COL}' should not be in Test data"
    assert (test_df[ID_COL].values == sample_sub_df[ID_COL].values).all(), "Test IDs do not match SampleSubmission IDs"
    
    # Class imbalance check
    class_counts = train_df[TARGET_COL].value_counts()
    class_ratio = train_df[TARGET_COL].mean()
    logger.info(f"Target distribution: 0={class_counts.get(0, 0)}, 1={class_counts.get(1, 0)} ({class_ratio:.2%} positive class)")
    
    # Categorical handling
    for col in CATEGORICAL_COLS:
        if col in train_df.columns:
            train_df[col] = train_df[col].astype(str).fillna("Missing")
            test_df[col] = test_df[col].astype(str).fillna("Missing")
            
    return train_df, test_df, sample_sub_df

if __name__ == "__main__":
    train, test, sample_sub = load_raw_data()
    print("Data loading test completed successfully!")
