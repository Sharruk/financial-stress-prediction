import argparse
import json
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import TARGET_COL, ID_COL, SUBMISSION_DIR, MODELS_DIR, SEED, N_SPLITS
from src.utils import set_seed, get_logger, evaluate_predictions, format_and_save_submission
from src.data import load_raw_data
from src.features import engineer_features
from src.validation import train_cv_model
from src.ensemble import (
    optimize_ensemble_weights,
    compute_blend_predictions,
    compute_logit_blend,
    compute_rank_average,
    train_stacking_meta_learner
)
from src.persistence import (
    prepare_full_features,
    refit_and_save_model,
    save_ensemble_config,
    save_te_artifacts,
)

logger = get_logger("train_pipeline")

def parse_args():
    parser = argparse.ArgumentParser(description="Zindi Financial Stress Prediction Training Pipeline")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lightgbm", "lightgbm_dart", "xgboost", "hist_gbm", "random_forest", "pytorch_mlp"],
        help="List of models to train (lightgbm, lightgbm_dart, xgboost, hist_gbm, random_forest, extra_trees, pytorch_mlp)"
    )
    parser.add_argument("--quick", action="store_true", help="Run quick baseline mode with fewer iterations on subsample")
    parser.add_argument("--folds", type=int, default=None, help="Number of cross-validation folds (default: 10 full, 5 quick)")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    n_splits = args.folds if args.folds is not None else (5 if args.quick else N_SPLITS)

    logger.info("==========================================================")
    logger.info("   ZINDI FINANCIAL STRESS PREDICTION - GRAND MASTER V3   ")
    logger.info(f"   Mode: {'QUICK (10k sample)' if args.quick else 'FULL DATA (40,000 samples)'} | Folds: {n_splits}")
    logger.info("==========================================================")

    # 1. Load Data
    train_df, test_df, sample_sub_df = load_raw_data()

    # Quick mode adjustments if requested
    if args.quick:
        logger.info("Quick mode enabled: Subsampling 10,000 train rows for fast baseline verification.")
        train_df = train_df.sample(n=10000, random_state=args.seed).reset_index(drop=True)

    # 2. Engineer Features
    logger.info("Running Grand Master Feature Engineering Pipeline on Train and Test...")
    train_fe = engineer_features(train_df)
    test_fe = engineer_features(test_df)

    # 3. Train Models via Stratified CV
    oof_dict = {}
    test_dict = {}
    model_summaries = []

    selected_models = args.models
    if args.quick:
        # Reduced iterations for fast verification
        lgb_params = {'n_estimators': 150, 'learning_rate': 0.05}
        lgb_dart_params = {'n_estimators': 120, 'learning_rate': 0.05}
        xgb_params = {'n_estimators': 150, 'learning_rate': 0.05}
        hgb_params = {'max_iter': 100, 'learning_rate': 0.05}
        rf_params = {'n_estimators': 100}
        mlp_params = {'epochs': 10}
    else:
        lgb_params = {'n_estimators': 1500, 'learning_rate': 0.018}
        lgb_dart_params = {'n_estimators': 1200, 'learning_rate': 0.025}
        xgb_params = {'n_estimators': 1200, 'learning_rate': 0.018}
        hgb_params = {'max_iter': 900, 'learning_rate': 0.02}
        rf_params = {'n_estimators': 500}
        mlp_params = {'epochs': 25}

    param_map = {
        'lightgbm': lgb_params,
        'lightgbm_dart': lgb_dart_params,
        'xgboost': xgb_params,
        'hist_gbm': hgb_params,
        'random_forest': rf_params,
        'pytorch_mlp': mlp_params
    }

    for m_name in selected_models:
        m_params = param_map.get(m_name, {})
        try:
            results = train_cv_model(m_name, train_fe, test_fe, model_params=m_params, n_splits=n_splits)
            oof_dict[m_name] = results['oof_probs']
            test_dict[m_name] = results['test_probs']

            m_metrics = results['oof_metrics']
            train_metrics = results.get('train_metrics', {})
            from src.utils import calculate_generalization_gap
            gap = calculate_generalization_gap(train_metrics, m_metrics)

            model_summaries.append({
                'Model': m_name.upper(),
                'Log Loss': m_metrics['log_loss'],
                'ROC-AUC': m_metrics['roc_auc'],
                'Brier Score': m_metrics['brier_score'],
                '_metrics_full': m_metrics,
                '_train_metrics': train_metrics,
                '_gap': gap,
                '_params': m_params
            })

            if results['feature_importances'] is not None:
                top10 = results['feature_importances'].head(10)
                logger.info(f"Top 5 Features for {m_name.upper()}:\n" + "\n".join([f"  - {feat}: {imp:.4f}" for feat, imp in top10.head(5).items()]))

        except Exception as e:
            logger.error(f"Failed to train model '{m_name}': {e}", exc_info=True)

    if not oof_dict:
        logger.error("No models trained successfully. Exiting.")
        sys.exit(1)

    # -------------------------------------------------------------------
    # 3b. Full-data refit for persistence
    # -------------------------------------------------------------------
    logger.info("Preparing full-data features for model persistence refits...")
    X_train_full, X_test_full, te_maps, feature_cols = prepare_full_features(train_fe, test_fe)
    y_train_full = train_fe[TARGET_COL].values

    save_te_artifacts(te_maps, feature_cols, MODELS_DIR)

    logger.info("Refitting models on full training data for persistence...")
    for m_name in selected_models:
        if m_name not in oof_dict:
            logger.warning(f"Skipping full-data refit for '{m_name}' (CV failed).")
            continue
        m_params = param_map.get(m_name, {})
        try:
            refit_and_save_model(m_name, m_params, X_train_full, y_train_full, MODELS_DIR)
        except Exception as e:
            logger.error(f"Full-data refit failed for '{m_name}': {e}", exc_info=True)

    # Print Single Models Benchmark Table
    summary_df = pd.DataFrame(model_summaries).sort_values(by="Log Loss")
    logger.info("\n================ SINGLE MODELS CV BENCHMARK ================")
    logger.info("\n" + summary_df.to_string(index=False))
    logger.info("============================================================\n")

    # 4. Multi-Model Ensembling & Blending
    y_true = train_fe[TARGET_COL].values
    best_test_preds = None

    if len(oof_dict) > 1:
        logger.info("Computing Optimal Ensemble Weights...")
        best_weights, weight_dict = optimize_ensemble_weights(oof_dict, y_true)
        
        # Standard Weighted Blend
        blend_oof, blend_test = compute_blend_predictions(oof_dict, test_dict, best_weights)
        blend_metrics = evaluate_predictions(y_true, blend_oof)
        logger.info(f"==> OPTIMAL PROBABILITY BLEND OOF | Log Loss: {blend_metrics['log_loss']:.5f} | ROC-AUC: {blend_metrics['roc_auc']:.5f} <==")

        # Logit Space Blend
        logit_oof, logit_test = compute_logit_blend(oof_dict, test_dict, best_weights)
        logit_metrics = evaluate_predictions(y_true, logit_oof)
        logger.info(f"==> LOGIT-SPACE BLEND OOF         | Log Loss: {logit_metrics['log_loss']:.5f} | ROC-AUC: {logit_metrics['roc_auc']:.5f} <==")

        # Rank Averaging
        rank_oof, rank_test = compute_rank_average(oof_dict, test_dict, best_weights)
        rank_metrics = evaluate_predictions(y_true, rank_oof)
        logger.info(f"==> RANK-AVERAGED BLEND OOF        | Log Loss: {rank_metrics['log_loss']:.5f} | ROC-AUC: {rank_metrics['roc_auc']:.5f} <==")

        # Stacking Meta Learner
        meta_oof, meta_test, meta_metrics = train_stacking_meta_learner(oof_dict, test_dict, y_true)

        # Select best calibrated strategy for primary submission
        if logit_metrics['log_loss'] <= blend_metrics['log_loss']:
            logger.info("Selected LOGIT-SPACE BLEND for final primary submission.")
            best_test_preds = logit_test
            final_oof_loss = logit_metrics['log_loss']
            final_oof_auc = logit_metrics['roc_auc']
            selected_strategy = "logit_blend"
            strategy_metrics = logit_metrics
        else:
            logger.info("Selected PROBABILITY BLEND for final primary submission.")
            best_test_preds = blend_test
            final_oof_loss = blend_metrics['log_loss']
            final_oof_auc = blend_metrics['roc_auc']
            selected_strategy = "weighted_blend"
            strategy_metrics = blend_metrics

        # Persist ensemble config
        save_ensemble_config(
            weight_dict=weight_dict,
            model_names=list(oof_dict.keys()),
            strategy=selected_strategy,
            models_dir=MODELS_DIR,
        )

        # Also save rank-averaged submission as alternative candidate
        rank_sub_df = pd.DataFrame({ID_COL: test_df[ID_COL], 'Target': rank_test})
        format_and_save_submission(rank_sub_df, sample_sub_df, SUBMISSION_DIR, prefix="zindi_stress_sub_rank")
    else:
        m_single = list(test_dict.keys())[0]
        best_test_preds = test_dict[m_single]
        final_oof_loss = summary_df.iloc[0]['Log Loss']
        final_oof_auc = summary_df.iloc[0]['ROC-AUC']
        selected_strategy = "single_model"
        strategy_metrics = {"log_loss": final_oof_loss, "roc_auc": final_oof_auc}
        single_weight = {m_single: 1.0}
        save_ensemble_config(
            weight_dict=single_weight,
            model_names=[m_single],
            strategy=selected_strategy,
            models_dir=MODELS_DIR,
        )

    # 5. Build and Save Primary Zindi Submission File
    sub_df = pd.DataFrame({
        ID_COL: test_df[ID_COL],
        'Target': best_test_preds
    })

    csv_path, zip_path = format_and_save_submission(sub_df, sample_sub_df, SUBMISSION_DIR, prefix="zindi_stress_sub")

    # -------------------------------------------------------------------
    # 6. Write experiment metadata JSON
    # -------------------------------------------------------------------
    experiments_dir = Path(__file__).resolve().parent / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)

    import subprocess
    def get_git_commit():
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
        except Exception:
            return None

    run_ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    run_id = f"EXP-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    experiment_record = {
        "experiment_id": run_id,
        "timestamp": run_ts,
        "git_commit": get_git_commit(),
        "model": "ensemble" if selected_strategy != "single_model" else list(test_dict.keys())[0],
        "model_version": "v3_grandmaster",
        "feature_version": "v3",
        "feature_count": len(train_fe.columns) - 1,
        "cv_folds": n_splits,
        "hyperparameters": {s['Model']: s['_params'] for s in model_summaries},
        "training_metrics": None,
        "validation_metrics": None,
        "oof_metrics": strategy_metrics,
        "generalization_gap": None,
        "ensemble_information": {
            "strategy": selected_strategy,
            "weights": weight_dict if selected_strategy in ["weighted_blend", "logit_blend"] else None
        },
        "training_time": None,
        "status": "completed",
        "notes": f"Grand Master training run with {n_splits}-fold CV"
    }

    experiment_record["_base_models"] = [
        {
            "name": s["Model"],
            "params": s["_params"],
            "training_metrics": s["_train_metrics"],
            "oof_metrics": s["_metrics_full"],
            "generalization_gap": s["_gap"]
        }
        for s in model_summaries
    ]

    run_json_path = experiments_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(run_json_path, "w") as f:
        json.dump(experiment_record, f, indent=2)
    logger.info(f"Experiment record saved to {run_json_path}")

    logger.info("==========================================================")
    logger.info("                  TRAINING PIPELINE COMPLETE               ")
    logger.info(f" Final OOF Log Loss : {final_oof_loss:.5f}")
    logger.info(f" Final OOF ROC-AUC  : {final_oof_auc:.5f}")
    logger.info(f" Primary CSV Path   : {csv_path}")
    logger.info(f" Latest CSV Path    : {SUBMISSION_DIR / 'submission.csv'}")
    logger.info(f" Experiment record  : {run_json_path}")
    logger.info("==========================================================")

if __name__ == "__main__":
    main()
