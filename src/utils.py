import os
import random
import logging
import zipfile
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

def set_seed(seed=42):
    """Set seeds for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass

def get_logger(name="financial_stress_ml"):
    """Set up structured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s', datefmt='%H:%M:%S')
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

def evaluate_predictions(y_true, y_prob):
    """Calculate comprehensive evaluation metrics."""
    # Clip probabilities to avoid infinite log_loss
    y_prob_clipped = np.clip(y_prob, 1e-15, 1 - 1e-15)
    loss = log_loss(y_true, y_prob_clipped)
    auc = roc_auc_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob_clipped)
    return {
        "log_loss": loss,
        "roc_auc": auc,
        "brier_score": brier
    }

def format_and_save_submission(sub_df, sample_sub_df, output_dir, prefix="sub"):
    """Validate and save submission file and zip archive."""
    logger = get_logger()
    
    # Check shape & columns
    assert len(sub_df) == len(sample_sub_df), f"Submission length ({len(sub_df)}) does not match expected ({len(sample_sub_df)})"
    assert "ID" in sub_df.columns and "Target" in sub_df.columns, "Submission missing required columns ID and Target"
    
    # Verify ID alignment
    assert (sub_df["ID"].values == sample_sub_df["ID"].values).all(), "ID alignment mismatch with SampleSubmission"
    
    # Verify probabilities
    assert not sub_df["Target"].isna().any(), "Submission contains NaN values!"
    assert (sub_df["Target"] >= 0).all() and (sub_df["Target"] <= 1).all(), "Predictions outside valid probability range [0, 1]"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"{prefix}_{timestamp}.csv"
    csv_path = output_dir / csv_filename
    zip_filename = f"{prefix}_{timestamp}.zip"
    zip_path = output_dir / zip_filename
    
    # Save CSV
    sub_df.to_csv(csv_path, index=False)
    logger.info(f"Saved submission CSV to {csv_path}")
    
    # Save latest submission copy
    latest_csv_path = output_dir / "submission.csv"
    sub_df.to_csv(latest_csv_path, index=False)
    logger.info(f"Saved latest submission to {latest_csv_path}")
    
    # Create ZIP archive
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(csv_path, arcname="submission.csv")
    logger.info(f"Created submission ZIP archive at {zip_path}")
    
    return csv_path, zip_path
