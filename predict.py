import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import ID_COL, SUBMISSION_DIR, MODELS_DIR
from src.utils import get_logger, format_and_save_submission
from src.data import load_raw_data
from src.features import engineer_features
from src.persistence import (
    load_model,
    load_ensemble_config,
    load_te_artifacts,
    apply_te_maps,
)

logger = get_logger("predict_pipeline")


def parse_args():
    parser = argparse.ArgumentParser(description="Zindi Financial Stress Prediction Inference")
    parser.add_argument("--test-file", type=str, default=None, help="Optional path to custom test CSV file")
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("Running standalone prediction pipeline...")

    # ------------------------------------------------------------------
    # 1. Guard: ensure training artifacts exist before doing anything else
    # ------------------------------------------------------------------
    ensemble_config_path = MODELS_DIR / "ensemble_config.json"
    te_maps_path = MODELS_DIR / "te_maps.json"
    feature_cols_path = MODELS_DIR / "feature_cols.json"

    missing_artifacts = []
    for p in [ensemble_config_path, te_maps_path, feature_cols_path]:
        if not p.exists():
            missing_artifacts.append(str(p))

    if missing_artifacts:
        logger.error(
            "\n"
            "========================================================\n"
            "  ERROR: Required training artifacts not found.\n"
            "  Missing files:\n" +
            "\n".join(f"    - {p}" for p in missing_artifacts) + "\n"
            "\n"
            "  Please run training first:\n"
            "      python train.py --quick     # fast verification\n"
            "      python train.py             # full training run\n"
            "========================================================\n"
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Load ensemble config and TE artifacts
    # ------------------------------------------------------------------
    try:
        ensemble_config = load_ensemble_config(MODELS_DIR)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    try:
        te_maps, feature_cols = load_te_artifacts(MODELS_DIR)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    strategy = ensemble_config["strategy"]
    model_names = ensemble_config["model_names"]
    weights_dict = ensemble_config["weights"]

    logger.info(f"Ensemble strategy: {strategy}")
    logger.info(f"Models to load: {model_names}")

    # ------------------------------------------------------------------
    # 3. Load persisted models
    # ------------------------------------------------------------------
    loaded_models = {}
    for m_name in model_names:
        try:
            loaded_models[m_name] = load_model(m_name, MODELS_DIR)
        except FileNotFoundError as e:
            logger.error(
                f"Model artifact missing for '{m_name}': {e}\n"
                "Please run 'python train.py' to regenerate model artifacts."
            )
            sys.exit(1)

    # ------------------------------------------------------------------
    # 4. Load and engineer test data
    # ------------------------------------------------------------------
    train_df, test_df, sample_sub_df = load_raw_data()
    if args.test_file:
        logger.info(f"Overriding test dataset with custom file: {args.test_file}")
        test_df = pd.read_csv(args.test_file)
        # Re-load sample_sub_df aligned to custom test IDs
        sample_sub_df = pd.DataFrame({ID_COL: test_df[ID_COL], "Target": 0.0})

    logger.info(f"Test samples to predict: {len(test_df)}")
    logger.info("Running feature engineering on test data...")
    test_fe = engineer_features(test_df)

    # ------------------------------------------------------------------
    # 5. Apply saved TE maps to produce inference-ready feature matrix
    # ------------------------------------------------------------------
    logger.info("Applying target-encoding maps to test features...")
    X_test = apply_te_maps(test_fe, te_maps, feature_cols)
    logger.info(f"Inference feature matrix shape: {X_test.shape}")

    # ------------------------------------------------------------------
    # 6. Generate per-model predictions and combine with ensemble weights
    # ------------------------------------------------------------------
    X_test_arr = X_test.values

    if strategy == "stacking":
        # For stacking: we need the stacking meta-learner.
        # The stacking meta-model is a LogisticRegression trained on OOF stacks —
        # it is NOT currently persisted separately. Fall back to weighted blend.
        logger.warning(
            "Stacking meta-learner is not persisted as a separate artifact. "
            "Falling back to weighted blend using optimized weights from ensemble_config."
        )
        strategy_for_inference = "weighted_blend"
    else:
        strategy_for_inference = strategy

    # Compute per-model test probabilities
    model_preds = {}
    for m_name, model in loaded_models.items():
        logger.info(f"Generating predictions with {m_name.upper()}...")
        preds = model.predict_proba(X_test_arr)[:, 1]
        model_preds[m_name] = preds

    # Combine predictions
    if len(model_preds) == 1:
        final_preds = list(model_preds.values())[0]
    else:
        # Weighted blend using persisted optimized weights
        weights_arr = np.array([weights_dict.get(m, 1.0 / len(model_names)) for m in model_names])
        weights_arr = weights_arr / weights_arr.sum()  # normalise to sum to 1
        pred_matrix = np.column_stack([model_preds[m] for m in model_names])
        final_preds = np.dot(pred_matrix, weights_arr)

    final_preds = np.clip(final_preds, 0.0, 1.0)
    logger.info(f"Final predictions — mean: {final_preds.mean():.4f}, std: {final_preds.std():.4f}")

    # ------------------------------------------------------------------
    # 7. Save submission
    # ------------------------------------------------------------------
    sub_df = pd.DataFrame({
        ID_COL: test_df[ID_COL],
        "Target": final_preds,
    })

    csv_path, zip_path = format_and_save_submission(
        sub_df, sample_sub_df, SUBMISSION_DIR, prefix="predict_only_sub"
    )

    logger.info("=========================================================")
    logger.info("             PREDICTION PIPELINE COMPLETE                ")
    logger.info(f" Ensemble strategy  : {strategy_for_inference}")
    logger.info(f" Models used        : {list(loaded_models.keys())}")
    logger.info(f" Primary CSV Path   : {csv_path}")
    logger.info(f" Latest CSV Path    : {SUBMISSION_DIR / 'submission.csv'}")
    logger.info("=========================================================")


if __name__ == "__main__":
    main()
