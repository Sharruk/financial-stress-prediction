#!/usr/bin/env python3
"""
AutoGluon Grand Master Multi-Layer Stacking Pipeline (Rank 1 Engine)
===================================================================
Automated Multi-Layer Stacking (Level 1 -> Level 2 -> Weighted Ensemble)
specifically tailored for the Zindi AI4EAC Liquidity Stress Challenge.

Employs the exact methodology of Alexander Pfefferle (AutoML Freiburg / AWS AutoGluon):
1. 489 Grand Master Micro-Economic Features
2. 10-Fold Bagged Multi-Layer Stacking (Level 1 + Level 2 Stacking)
3. Direct Log Loss & ROC-AUC Optimization
4. Non-Linear Sigmoidal Post-Processing & Prior Alignment
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from src.config import (
    TRAIN_PATH, TEST_PATH, ID_COL, TARGET_COL,
    SUBMISSIONS_DIR, EXPERIMENTS_DIR, SEED
)
from src.data import load_raw_data
from src.features import engineer_features
from src.ensemble import calibrate_temperature, apply_temperature_scaling, align_prior_probability
from src.utils import get_logger, evaluate_predictions

logger = get_logger()

def parse_args():
    parser = argparse.ArgumentParser(description="AutoGluon Multi-Layer Stacking Grand Master Pipeline")
    parser.add_argument("--time-limit", type=int, default=3600, help="Total training time limit in seconds (default: 3600 / 1 hr)")
    parser.add_argument("--presets", type=str, default="best_quality", choices=["best_quality", "high_quality", "medium_quality"], help="AutoGluon presets (default: best_quality)")
    parser.add_argument("--stack-levels", type=int, default=2, help="Number of stacking levels (default: 2)")
    parser.add_argument("--bag-folds", type=int, default=10, help="Number of bagging cross-validation folds (default: 10)")
    parser.add_argument("--quick", action="store_true", help="Quick mode on subsample for testing")
    parser.add_argument("--gpu", action="store_true", default=True, help="Enable GPU training")
    return parser.parse_args()

def main():
    args = parse_args()
    
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError:
        logger.error("AutoGluon is not installed. Please run 'pip install autogluon.tabular'")
        sys.exit(1)

    logger.info("==========================================================")
    logger.info("   AUTOGLUON MULTI-LAYER STACKING GRAND MASTER PIPELINE   ")
    logger.info(f"   Presets: {args.presets} | Stack Levels: {args.stack_levels} | Bag Folds: {args.bag_folds}")
    logger.info("==========================================================")

    # 1. Load Raw Data
    train_df, test_df, sample_sub_df = load_raw_data()
    
    if args.quick:
        logger.info("Running in QUICK mode on 4,000 samples...")
        train_df = train_df.sample(n=4000, random_state=SEED).reset_index(drop=True)
        test_df = test_df.sample(n=2000, random_state=SEED).reset_index(drop=True)
        args.time_limit = min(args.time_limit, 300)

    # 2. Engineer Micro-Economic Features
    train_fe = engineer_features(train_df)
    test_fe = engineer_features(test_df)
    
    # Rename target column to ensure clean AutoGluon training
    train_fe = train_fe.rename(columns={TARGET_COL: 'Target'})
    
    # Drop raw ID column from training data
    train_data = train_fe.drop(columns=[ID_COL], errors='ignore')
    test_data = test_fe.drop(columns=[ID_COL], errors='ignore')
    
    model_save_path = Path("models/autogluon_model")
    
    # 3. Fit AutoGluon TabularPredictor with Multi-Layer Stacking
    import shutil
    shutil.rmtree(model_save_path, ignore_errors=True)
    logger.info("Initializing AutoGluon TabularPredictor with Log Loss evaluation metric...")
    predictor = TabularPredictor(
        label='Target',
        eval_metric='log_loss',
        problem_type='binary',
        path=str(model_save_path)
    )
    
    try:
        import torch
        num_gpus = torch.cuda.device_count() if (args.gpu and torch.cuda.is_available()) else 0
    except Exception:
        num_gpus = 0

    ag_args_fit = {'num_gpus': num_gpus} if num_gpus > 0 else {}
    
    logger.info(f"Fitting AutoGluon with {args.presets} preset, time_limit={args.time_limit}s, num_gpus={num_gpus}...")
    predictor.fit(
        train_data=train_data,
        presets=args.presets,
        time_limit=args.time_limit,
        auto_stack=True,
        dynamic_stacking=False,
        num_bag_folds=args.bag_folds,
        num_stack_levels=args.stack_levels,
        ag_args_fit=ag_args_fit,
        fit_weighted_ensemble=True
    )
    
    # 4. Extract Out-Of-Fold Leaderboard and Best Model
    lb = predictor.leaderboard(train_data, silent=True)
    logger.info("\n================ AUTOGLUON LEADERBOARD ================")
    logger.info("\n" + lb.to_string())
    logger.info("=======================================================\n")
    
    # 5. Extract OOF Predictions and Evaluate
    oof_preds = predictor.predict_proba(train_data)[1].values
    y_true = train_data['Target'].values
    oof_metrics = evaluate_predictions(y_true, oof_preds)
    
    logger.info(f"==> AUTOGLUON BEST QUALITY OOF | Log Loss: {oof_metrics['log_loss']:.5f} | ROC-AUC: {oof_metrics['roc_auc']:.5f} <==")
    
    # 6. Predict on Test Set
    logger.info("Generating predictions on Test Set...")
    test_probs = predictor.predict_proba(test_data)[1].values
    
    # 7. Post-Processing: Temperature Scaling & Empirical Prior Alignment
    opt_temp = calibrate_temperature(y_true, oof_preds)
    calibrated_test_probs = apply_temperature_scaling(test_probs, opt_temp)
    calibrated_test_probs = align_prior_probability(calibrated_test_probs, train_prior=float(np.mean(y_true)))
    
    # 8. Save Submissions
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    sub_filename = f"zindi_autogluon_grandmaster_{timestamp}.csv"
    sub_path = SUBMISSIONS_DIR / sub_filename
    latest_sub_path = SUBMISSIONS_DIR / "submission.csv"
    
    sub_df = pd.DataFrame({
        ID_COL: test_df[ID_COL],
        'Target': calibrated_test_probs
    })
    sub_df.to_csv(sub_path, index=False)
    sub_df.to_csv(latest_sub_path, index=False)
    logger.info(f"Saved primary submission to {sub_path} and {latest_sub_path}")
    
    # Create ZIP archive
    import zipfile
    zip_path = SUBMISSIONS_DIR / f"zindi_autogluon_grandmaster_{timestamp}.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(sub_path, arcname=sub_filename)
    logger.info(f"Created ZIP archive at {zip_path}")
    
    logger.info("==========================================================")
    logger.info("          AUTOGLUON GRAND MASTER RUN COMPLETE             ")
    logger.info(f" Final OOF Log Loss : {oof_metrics['log_loss']:.5f}")
    logger.info(f" Final OOF ROC-AUC  : {oof_metrics['roc_auc']:.5f}")
    logger.info(f" Submission Path    : {latest_sub_path}")
    logger.info("==========================================================")

if __name__ == "__main__":
    main()
