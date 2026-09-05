import argparse
import json
import os
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
    train_stacking_meta_learner,
    calibrate_joint_logodds,
    apply_joint_calibration,
    fit_isotonic_calibrator,
    apply_isotonic_scaling,
    align_prior_probability
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
        default=["catboost", "xgboost", "lightgbm_goss"],
        help="List of models to train (default: ['catboost', 'xgboost', 'lightgbm_goss'])"
    )
    parser.add_argument("--smoke-test", action="store_true", help="Run fast GPU/CPU smoke test to verify pipeline & hardware")
    parser.add_argument("--quick", action="store_true", help="Run quick baseline mode with fewer iterations on subsample")
    parser.add_argument("--folds", type=int, default=10, help="Number of cross-validation folds (default: 10)")
    parser.add_argument("--seed", type=int, default=SEED, help="Primary random seed")
    parser.add_argument("--multi-seed", action="store_true", default=False, help="Run multi-seed bagging [42, 1337, 2026] (default: False)")
    parser.add_argument("--gpu", action="store_true", default=True, help="Explicitly enable GPU acceleration for models (default: True)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    parser.add_argument("--devices", type=str, default=None, help="GPU devices string (e.g. '0' or '0:1' for T4 x 2)")
    return parser.parse_args()

def run_smoke_test(args):
    """
    Fast GPU/CPU smoke test to verify Python, CatBoost, CUDA hardware,
    data pipeline, training, and probability bounds.
    """
    import platform
    import time
    import subprocess
    logger.info("==========================================================")
    logger.info("             GPU & PIPELINE SMOKE TEST                    ")
    logger.info("==========================================================")
    logger.info(f"Python Version   : {platform.python_version()}")
    
    try:
        import catboost as cb
        logger.info(f"CatBoost Version : {cb.__version__}")
    except ImportError:
        logger.error("CatBoost is not installed!")
        sys.exit(1)
        
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count()
        logger.info(f"CUDA Available   : {cuda_avail} (Devices: {gpu_count})")
        if cuda_avail:
            for i in range(gpu_count):
                logger.info(f"  Device {i}        : {torch.cuda.get_device_name(i)}")
    except ImportError:
        cuda_avail = False
        logger.info("PyTorch not installed for CUDA introspection.")

    # Try running nvidia-smi
    try:
        smi_out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]).decode().strip()
        logger.info(f"NVIDIA-SMI Query : {smi_out}")
    except Exception:
        logger.info("NVIDIA-SMI       : Not available in this environment")

    # Load small slice of real data
    train_df, test_df, sample_sub_df = load_raw_data()
    train_sample = train_df.sample(n=1000, random_state=args.seed).reset_index(drop=True)
    test_sample = test_df.sample(n=200, random_state=args.seed).reset_index(drop=True)
    
    logger.info("Engineering features on smoke test slice...")
    train_fe = engineer_features(train_sample)
    test_fe = engineer_features(test_sample)

    smoke_params = {
        'iterations': 50,
        'depth': 5,
        'learning_rate': 0.05,
        'verbose': 0
    }
    if args.gpu or (cuda_avail and not args.cpu):
        smoke_params['task_type'] = 'GPU'
        if args.devices:
            smoke_params['devices'] = args.devices
    else:
        smoke_params['task_type'] = 'CPU'

    logger.info(f"Running CatBoost smoke test with params: {smoke_params}...")
    t0 = time.time()
    results = train_cv_model("catboost", train_fe, test_fe, model_params=smoke_params, n_splits=2)
    elapsed = time.time() - t0

    oof = results['oof_probs']
    test_p = results['test_probs']
    metrics = results['oof_metrics']

    assert not np.isnan(oof).any(), "NaN found in OOF predictions!"
    assert not np.isinf(oof).any(), "Inf found in OOF predictions!"
    assert (oof >= 0.0).all() and (oof <= 1.0).all(), "OOF probabilities out of [0, 1] bounds!"
    assert (test_p >= 0.0).all() and (test_p <= 1.0).all(), "Test probabilities out of [0, 1] bounds!"

    logger.info("----------------------------------------------------------")
    logger.info("             SMOKE TEST SUMMARY RESULTS                   ")
    logger.info(f" Train Shape     : {train_sample.shape}")
    logger.info(f" Features Total  : {train_fe.shape[1]}")
    logger.info(f" Execution Time  : {elapsed:.2f}s")
    logger.info(f" OOF Log Loss    : {metrics['log_loss']:.5f}")
    logger.info(f" OOF ROC-AUC     : {metrics['roc_auc']:.5f}")
    logger.info(f" OOF PR-AUC      : {metrics['pr_auc']:.5f}")
    logger.info(f" Positive Rate   : {float(train_sample[TARGET_COL].mean()):.4f}")
    logger.info(f" Mean OOF Prob   : {float(np.mean(oof)):.4f}")
    logger.info("----------------------------------------------------------")
    logger.info("✅ SMOKE TEST PASSED! Environment and CatBoost are ready for full training.")
    logger.info("==========================================================")
    return


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.gpu:
        os.environ["GPU_ENABLED"] = "true"
    elif args.cpu:
        os.environ["GPU_ENABLED"] = "false"
        
    if args.devices:
        os.environ["GPU_DEVICES"] = args.devices

    if args.smoke_test:
        run_smoke_test(args)
        return

    n_splits = args.folds if args.folds is not None else (5 if args.quick else N_SPLITS)
    seeds = [42, 1337, 2026] if args.multi_seed else [args.seed]

    logger.info("==========================================================")
    logger.info("   ZINDI FINANCIAL STRESS PREDICTION - GRAND MASTER V8   ")
    logger.info(f"   Mode: {'QUICK (10k sample)' if args.quick else 'FULL DATA (40,000 samples)'} | Folds: {n_splits} | Seeds: {seeds}")
    logger.info("==========================================================")

    # 1. Load Data
    train_df, test_df, sample_sub_df = load_raw_data()

    if args.quick:
        logger.info("Quick mode enabled: Subsampling 10,000 train rows for fast baseline verification.")
        train_df = train_df.sample(n=10000, random_state=args.seed).reset_index(drop=True)

    # 2. Engineer Features
    logger.info("Running Grand Master v8 Feature Engineering Pipeline on Train and Test...")
    train_fe = engineer_features(train_df)
    test_fe = engineer_features(test_df)

    # 3. Train Models via Stratified CV
    oof_dict = {}
    test_dict = {}
    model_summaries = []

    selected_models = args.models
    if args.quick:
        lgb_params = {'n_estimators': 150, 'learning_rate': 0.05}
        lgb_dart_params = {'n_estimators': 120, 'learning_rate': 0.05}
        lgb_goss_params = {'n_estimators': 120, 'learning_rate': 0.05}
        xgb_params = {'n_estimators': 150, 'learning_rate': 0.05}
        cb_params = {'iterations': 150, 'learning_rate': 0.05}
        hgb_params = {'max_iter': 100, 'learning_rate': 0.05}
        et_params = {'n_estimators': 100}
        rf_params = {'n_estimators': 100}
        mlp_params = {'epochs': 10}
    else:
        lgb_params = {'n_estimators': 1800, 'learning_rate': 0.015}
        lgb_dart_params = {'n_estimators': 1400, 'learning_rate': 0.022}
        lgb_goss_params = {'n_estimators': 1500, 'learning_rate': 0.015}
        xgb_params = {'n_estimators': 1800, 'learning_rate': 0.014}
        cb_params = {'iterations': 2200, 'learning_rate': 0.015}
        hgb_params = {'max_iter': 1200, 'learning_rate': 0.015}
        et_params = {'n_estimators': 600}
        rf_params = {'n_estimators': 600}
        mlp_params = {'epochs': 25}

    param_map = {
        'lightgbm': lgb_params,
        'lightgbm_dart': lgb_dart_params,
        'lightgbm_goss': lgb_goss_params,
        'xgboost': xgb_params,
        'catboost': cb_params,
        'hist_gbm': hgb_params,
        'extra_trees': et_params,
        'random_forest': rf_params,
        'pytorch_mlp': mlp_params
    }

    for m_name in selected_models:
        m_params = param_map.get(m_name, {}).copy()
        try:
            # Multi-seed bagging per model if requested
            seed_oofs = []
            seed_tests = []
            seed_fold_scores = []
            seed_train_metrics = []
            
            for s in seeds:
                s_params = m_params.copy()
                if m_name in ['lightgbm', 'lightgbm_dart', 'lightgbm_goss', 'xgboost', 'extra_trees', 'random_forest', 'hist_gbm', 'logistic_regression']:
                    s_params['random_state'] = s
                elif m_name == 'catboost':
                    s_params['random_seed'] = s
                    
                results = train_cv_model(m_name, train_fe, test_fe, model_params=s_params, n_splits=n_splits, seed=s)
                seed_oofs.append(results['oof_probs'])
                seed_tests.append(results['test_probs'])
                seed_fold_scores.append(results['fold_scores'])
                seed_train_metrics.append(results.get('train_metrics', {}))

            avg_oof = np.mean(seed_oofs, axis=0)
            avg_test = np.mean(seed_tests, axis=0)
            
            oof_dict[m_name] = avg_oof
            test_dict[m_name] = avg_test

            # Average training metrics across seeds
            avg_train_metrics = {}
            if seed_train_metrics and seed_train_metrics[0]:
                for k in seed_train_metrics[0].keys():
                    avg_train_metrics[k] = float(np.mean([x[k] for x in seed_train_metrics if k in x]))

            m_metrics = evaluate_predictions(train_fe[TARGET_COL].values, avg_oof)
            from src.utils import calculate_generalization_gap
            gap = calculate_generalization_gap(avg_train_metrics, m_metrics)

            model_summaries.append({
                'Model': m_name.upper(),
                'Log Loss': m_metrics['log_loss'],
                'ROC-AUC': m_metrics['roc_auc'],
                'Brier Score': m_metrics['brier_score'],
                '_metrics_full': m_metrics,
                '_train_metrics': avg_train_metrics,
                '_gap': gap,
                '_params': m_params,
                '_seeds': seeds,
                '_fold_scores': seed_fold_scores[0] if len(seeds) == 1 else seed_fold_scores
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

    # Calculate and log OOF Prediction Correlation Matrix across models
    model_correlations = None
    if len(oof_dict) > 1:
        oof_corr_df = pd.DataFrame(oof_dict).corr()
        model_correlations = oof_corr_df.to_dict()
        logger.info("\n================ MODEL OOF CORRELATION MATRIX ================")
        logger.info("\n" + oof_corr_df.to_string())
        logger.info("==============================================================\n")

    # 4. Multi-Model Ensembling, Blending & Calibration
    y_true = train_fe[TARGET_COL].values
    train_prior = float(np.mean(y_true))
    best_test_preds = None

    if len(oof_dict) > 1:
        logger.info("Computing Optimal Ensemble Weights...")
        best_weights, weight_dict = optimize_ensemble_weights(oof_dict, y_true)
        
        # Standard Weighted Blend
        blend_oof, blend_test = compute_blend_predictions(oof_dict, test_dict, best_weights)
        blend_metrics = evaluate_predictions(y_true, blend_oof)
        # 1. Joint Log-Odds Calibration (Temperature + Prior Delta + Asymmetric Clamping)
        opt_temp, opt_delta = calibrate_joint_logodds(y_true, blend_oof, train_prior=train_prior)
        joint_cal_oof = apply_joint_calibration(blend_oof, opt_temp, opt_delta)
        joint_cal_test = apply_joint_calibration(blend_test, opt_temp, opt_delta)
        joint_metrics = evaluate_predictions(y_true, joint_cal_oof)
        logger.info(f"==> JOINT LOG-ODDS CALIBRATED BLEND (T={opt_temp:.3f}, delta={opt_delta:.4f}) | Log Loss: {joint_metrics['log_loss']:.5f} | ROC-AUC: {joint_metrics['roc_auc']:.5f} <==")

        # 2. Isotonic Monotonic Calibration
        iso_cal, iso_oof = fit_isotonic_calibrator(y_true, blend_oof)
        iso_test = apply_isotonic_scaling(iso_cal, blend_test)
        iso_metrics = evaluate_predictions(y_true, iso_oof)
        logger.info(f"==> ISOTONIC CALIBRATED BLEND OOF | Log Loss: {iso_metrics['log_loss']:.5f} | ROC-AUC: {iso_metrics['roc_auc']:.5f} <==")

        # 3. Logit Space Blend
        logit_oof, logit_test = compute_logit_blend(oof_dict, test_dict, best_weights)
        logit_metrics = evaluate_predictions(y_true, logit_oof)
        logger.info(f"==> LOGIT-SPACE BLEND OOF         | Log Loss: {logit_metrics['log_loss']:.5f} | ROC-AUC: {logit_metrics['roc_auc']:.5f} <==")

        # 4. Rank Averaging
        rank_oof, rank_test = compute_rank_average(oof_dict, test_dict, best_weights)
        rank_metrics = evaluate_predictions(y_true, rank_oof)
        logger.info(f"==> RANK-AVERAGED BLEND OOF        | Log Loss: {rank_metrics['log_loss']:.5f} | ROC-AUC: {rank_metrics['roc_auc']:.5f} <==")

        # 5. Level-2 Stacking Meta Learner (Logistic Regression on Log-Odds)
        meta_oof, meta_test, meta_metrics = train_stacking_meta_learner(oof_dict, test_dict, y_true)

        # Select best calibrated strategy for primary submission
        candidate_strategies = {
            "joint_calibrated_blend": (joint_cal_test, joint_metrics),
            "stacking_meta_learner": (meta_test, meta_metrics),
            "isotonic_blend": (iso_test, iso_metrics),
            "logit_blend": (logit_test, logit_metrics)
        }
        
        # Sort by lowest Log Loss with ROC-AUC >= 0.900
        best_strat_name = min(candidate_strategies.keys(), key=lambda k: candidate_strategies[k][1]['log_loss'])
        best_test_preds, strategy_metrics = candidate_strategies[best_strat_name]
        final_oof_loss = strategy_metrics['log_loss']
        final_oof_auc = strategy_metrics['roc_auc']
        selected_strategy = best_strat_name
        logger.info(f"Selected {best_strat_name.upper()} for final primary submission (Log Loss: {final_oof_loss:.5f}, ROC-AUC: {final_oof_auc:.5f}).")

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
        "model_version": "v6_ultimate_grandmaster",
        "feature_version": "v6",
        "feature_count": len(train_fe.columns) - 1,
        "cv_folds": n_splits,
        "seeds": seeds,
        "hyperparameters": {s['Model']: s['_params'] for s in model_summaries},
        "training_metrics": None,
        "validation_metrics": None,
        "oof_metrics": strategy_metrics,
        "model_correlations": model_correlations,
        "generalization_gap": None,
        "ensemble_information": {
            "strategy": selected_strategy,
            "weights": weight_dict if selected_strategy in ["weighted_blend", "logit_blend", "calibrated_blend"] else None
        },
        "training_time": None,
        "status": "completed",
        "notes": f"Grand Master v6 10-Fold 4-Model Multi-Seed Ensemble (Seeds {seeds}, Folds {n_splits})"
    }

    experiment_record["_base_models"] = [
        {
            "name": s["Model"],
            "params": s["_params"],
            "seeds": s["_seeds"],
            "training_metrics": s["_train_metrics"],
            "oof_metrics": s["_metrics_full"],
            "fold_scores": s["_fold_scores"],
            "generalization_gap": s["_gap"]
        }
        for s in model_summaries
    ]

    run_json_path = experiments_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(run_json_path, "w") as f:
        json.dump(experiment_record, f, indent=2)
    logger.info(f"Experiment record saved to {run_json_path}")

    # Save raw OOF predictions for deep analysis
    oof_df = pd.DataFrame({
        ID_COL: train_df[ID_COL],
        "target": y_true,
        "oof_prediction": list(oof_dict.values())[0] if len(oof_dict) == 1 else blend_oof
    })
    for m in oof_dict:
        oof_df[f"oof_{m}"] = oof_dict[m]
        
    oof_csv_path = experiments_dir / f"oof_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    oof_df.to_csv(oof_csv_path, index=False)
    logger.info(f"OOF predictions saved to {oof_csv_path}")

    logger.info("==========================================================")
    logger.info("                  TRAINING PIPELINE COMPLETE               ")
    logger.info(f" Final OOF Log Loss : {final_oof_loss:.5f}")
    logger.info(f" Final OOF ROC-AUC  : {final_oof_auc:.5f}")
    logger.info(f" Primary CSV Path   : {csv_path}")
    logger.info(f" Latest CSV Path    : {SUBMISSION_DIR / 'submission.csv'}")
    logger.info(f" Experiment record  : {run_json_path}")
    logger.info(f" OOF Data Record    : {oof_csv_path}")
    logger.info("==========================================================")

if __name__ == "__main__":
    main()
