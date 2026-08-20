"""
src/persistence.py
------------------
Model and artifact persistence utilities for the financial stress prediction pipeline.

Responsibilities:
  - prepare_full_features(): replicate fold TE logic on full training data, save TE maps.
  - save_model() / load_model(): dispatch between joblib (sklearn-API) and torch.save/load.
  - save_ensemble_config() / load_ensemble_config(): persist ensemble strategy JSON.
  - refit_and_save_model(): fit one model on the full training set and save it.
  - apply_te_maps(): apply saved TE maps to a test dataframe at inference time.
"""

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from src.config import ID_COL, TARGET_COL, SEED
from src.models import get_model
from src.utils import get_logger

logger = get_logger()

# ------------------------------------------------------------------
# Constants: artifact filenames (all under MODELS_DIR)
# ------------------------------------------------------------------
TE_MAPS_FILENAME = "te_maps.json"
FEATURE_COLS_FILENAME = "feature_cols.json"
ENSEMBLE_CONFIG_FILENAME = "ensemble_config.json"


# ------------------------------------------------------------------
# 1. Full-data feature preparation (mirrors fold TE from validation.py)
# ------------------------------------------------------------------

from src.features import fit_fold_unsupervised_personas, transform_fold_unsupervised_personas

def prepare_full_features(train_fe: pd.DataFrame, test_fe: pd.DataFrame):
    """
    Compute target-encoding fit on the ENTIRE training set (no fold split),
    apply to both train and test, return numeric feature matrices.

    This mirrors the logic in validation.preprocess_fold_features() but uses
    all training rows so the resulting model sees the full distribution.

    Returns
    -------
    X_train : pd.DataFrame   numeric features for training
    X_test  : pd.DataFrame   numeric features for test
    te_maps : dict           {col_name: {category: encoded_value, "__global_mean__": float}}
    feature_cols : list[str] ordered list of feature column names (deterministic)
    """
    X_tr = train_fe.copy()
    X_te = test_fe.copy()

    cat_cols = [
        c for c in X_tr.columns
        if c not in [ID_COL, TARGET_COL]
        and not pd.api.types.is_numeric_dtype(X_tr[c])
    ]

    global_target_mean = float(X_tr[TARGET_COL].mean())
    smooth_weight = 10
    te_maps = {}

    for col in cat_cols:
        stats = X_tr.groupby(col)[TARGET_COL].agg(["count", "mean"])
        smooth_te = (
            (stats["count"] * stats["mean"] + smooth_weight * global_target_mean)
            / (stats["count"] + smooth_weight)
        )
        te_dict = smooth_te.to_dict()

        te_col_name = f"{col}_te"
        X_tr[te_col_name] = X_tr[col].map(te_dict).fillna(global_target_mean).astype(np.float32)
        X_te[te_col_name] = X_te[col].map(te_dict).fillna(global_target_mean).astype(np.float32)

        # Persist map so predict.py can apply it without labels
        te_maps[col] = {str(k): float(v) for k, v in te_dict.items()}
        te_maps[col]["__global_mean__"] = global_target_mean

    # Fit KMeans personas on full training set
    kmeans, scaler_params, avail_cols = fit_fold_unsupervised_personas(X_tr, n_clusters=8, seed=SEED)
    X_tr = transform_fold_unsupervised_personas(X_tr, kmeans, scaler_params, avail_cols)
    X_te = transform_fold_unsupervised_personas(X_te, kmeans, scaler_params, avail_cols)

    feature_cols = [
        c for c in X_tr.select_dtypes(include=[np.number]).columns
        if c not in [ID_COL, TARGET_COL]
    ]

    return X_tr[feature_cols], X_te[feature_cols], te_maps, feature_cols


def apply_te_maps(test_fe: pd.DataFrame, te_maps: dict, feature_cols: list) -> pd.DataFrame:
    """
    Apply pre-fitted TE maps to a test dataframe at inference time.
    Only numeric columns in feature_cols are returned.

    Parameters
    ----------
    test_fe      : output of engineer_features() on test data
    te_maps      : dict loaded from te_maps.json
    feature_cols : list loaded from feature_cols.json

    Returns
    -------
    X_test : pd.DataFrame with only feature_cols columns present
    """
    X_te = test_fe.copy()

    for col, mapping in te_maps.items():
        global_mean = mapping["__global_mean__"]
        te_dict = {k: v for k, v in mapping.items() if k != "__global_mean__"}
        te_col_name = f"{col}_te"
        X_te[te_col_name] = X_te[col].map(te_dict).fillna(global_mean).astype(np.float32)

    # Keep only the exact columns the models were trained on, in the same order
    available = [c for c in feature_cols if c in X_te.columns]
    missing = [c for c in feature_cols if c not in X_te.columns]
    if missing:
        logger.warning(
            f"apply_te_maps: {len(missing)} feature columns missing from test data, "
            f"filling with 0: {missing[:5]}..."
        )
        for c in missing:
            X_te[c] = 0.0

    return X_te[feature_cols]


# ------------------------------------------------------------------
# 2. Model save / load
# ------------------------------------------------------------------

def save_model(model, model_name: str, models_dir: Path) -> Path:
    """
    Save a fitted model to disk.
    - PyTorchMLPWrapper  -> torch.save (models/<name>.pt)
    - sklearn-API models -> joblib.dump (models/<name>.pkl)

    Returns the path where the model was saved.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    if model_name.lower() == "pytorch_mlp":
        import torch
        save_path = models_dir / "pytorch_mlp.pt"
        payload = {
            "params": model.params,
            "input_dim": model.model.input_layer[0].in_features,
            "scaler_mean": model.scaler.mean_.tolist(),
            "scaler_scale": model.scaler.scale_.tolist(),
            "model_state_dict": model.model.state_dict(),
        }
        torch.save(payload, save_path)
        logger.info(f"Saved pytorch_mlp to {save_path}")
    else:
        save_path = models_dir / f"{model_name}.pkl"
        joblib.dump(model, save_path)
        logger.info(f"Saved {model_name} to {save_path}")

    return save_path


def load_model(model_name: str, models_dir: Path):
    """
    Load a persisted model from disk.
    - pytorch_mlp -> reconstruct PyTorchMLPWrapper from .pt payload
    - others      -> joblib.load from .pkl
    """
    models_dir = Path(models_dir)

    if model_name.lower() == "pytorch_mlp":
        import torch
        from sklearn.preprocessing import StandardScaler
        from src.models import PyTorchMLPWrapper, TabularResMLP

        pt_path = models_dir / "pytorch_mlp.pt"
        if not pt_path.exists():
            raise FileNotFoundError(f"pytorch_mlp model not found at {pt_path}")

        payload = torch.load(pt_path, map_location="cpu", weights_only=False)

        wrapper = PyTorchMLPWrapper(params=payload["params"])
        wrapper.model = TabularResMLP(
            input_dim=payload["input_dim"],
            hidden_dim=wrapper.hidden_dim,
            dropout=wrapper.dropout,
        )
        wrapper.model.load_state_dict(payload["model_state_dict"])
        wrapper.model.eval()

        # Restore scaler without re-fitting
        scaler = StandardScaler()
        mean_ = np.array(payload["scaler_mean"])
        scale_ = np.array(payload["scaler_scale"])
        scaler.mean_ = mean_
        scaler.scale_ = scale_
        scaler.var_ = scale_ ** 2
        scaler.n_features_in_ = len(mean_)
        wrapper.scaler = scaler

        logger.info(f"Loaded pytorch_mlp from {pt_path}")
        return wrapper

    else:
        pkl_path = models_dir / f"{model_name}.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(f"Model '{model_name}' not found at {pkl_path}")
        model = joblib.load(pkl_path)
        logger.info(f"Loaded {model_name} from {pkl_path}")
        return model


# ------------------------------------------------------------------
# 3. Ensemble config save / load
# ------------------------------------------------------------------

def save_ensemble_config(
    weight_dict: dict,
    model_names: list,
    strategy: str,
    models_dir: Path,
) -> Path:
    """
    Save ensemble strategy and per-model weights to ensemble_config.json.

    Note: models/*.json is gitignored by the existing .gitignore - this file is
    a local-only artifact, consistent with the project intent that model files
    are not committed to version control.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "strategy": strategy,
        "model_names": model_names,
        "weights": weight_dict,
    }
    config_path = models_dir / ENSEMBLE_CONFIG_FILENAME
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved ensemble config to {config_path}")
    return config_path


def load_ensemble_config(models_dir: Path) -> dict:
    """
    Load ensemble config from models/ensemble_config.json.
    Raises FileNotFoundError with a clear message if not found.
    """
    config_path = Path(models_dir) / ENSEMBLE_CONFIG_FILENAME
    if not config_path.exists():
        raise FileNotFoundError(
            f"Ensemble config not found at {config_path}.\n"
            "Please run 'python train.py' first to train models and generate artifacts."
        )
    with open(config_path) as f:
        config = json.load(f)
    logger.info(f"Loaded ensemble config from {config_path} (strategy={config['strategy']})")
    return config


# ------------------------------------------------------------------
# 4. TE maps + feature columns save / load
# ------------------------------------------------------------------

def save_te_artifacts(te_maps: dict, feature_cols: list, models_dir: Path):
    """Save TE maps and ordered feature column list for inference-time use."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    te_path = models_dir / TE_MAPS_FILENAME
    with open(te_path, "w") as f:
        json.dump(te_maps, f, indent=2)
    logger.info(f"Saved TE maps to {te_path}")

    feat_path = models_dir / FEATURE_COLS_FILENAME
    with open(feat_path, "w") as f:
        json.dump(feature_cols, f, indent=2)
    logger.info(f"Saved feature columns list ({len(feature_cols)} cols) to {feat_path}")


def load_te_artifacts(models_dir: Path):
    """
    Load TE maps and feature columns list from disk.
    Returns (te_maps: dict, feature_cols: list)
    """
    models_dir = Path(models_dir)
    te_path = models_dir / TE_MAPS_FILENAME
    feat_path = models_dir / FEATURE_COLS_FILENAME

    missing = []
    if not te_path.exists():
        missing.append(str(te_path))
    if not feat_path.exists():
        missing.append(str(feat_path))
    if missing:
        raise FileNotFoundError(
            f"Training artifacts missing: {missing}.\n"
            "Please run 'python train.py' first to generate all required artifacts."
        )

    with open(te_path) as f:
        te_maps = json.load(f)
    with open(feat_path) as f:
        feature_cols = json.load(f)

    logger.info(
        f"Loaded TE maps ({len(te_maps)} encoded cols) and {len(feature_cols)} feature columns"
    )
    return te_maps, feature_cols


# ------------------------------------------------------------------
# 5. Full-data refit helper
# ------------------------------------------------------------------

def refit_and_save_model(
    model_name: str,
    params: dict,
    X_train_full: pd.DataFrame,
    y_train: np.ndarray,
    models_dir: Path,
):
    """
    Fit a fresh model of type `model_name` on the COMPLETE training set
    using the same hyperparameters as the CV folds, then persist it.

    This is ONLY for inference -- it does NOT affect OOF/CV metrics.

    Returns the fitted model.
    """
    logger.info(
        f"Refitting {model_name.upper()} on full training data ({len(X_train_full)} rows)..."
    )
    model = get_model(model_name, params=params)
    X_arr = X_train_full.values if hasattr(X_train_full, "values") else X_train_full
    model.fit(X_arr, y_train)
    save_model(model, model_name, models_dir)
    logger.info(f"Full-data refit complete for {model_name.upper()}.")
    return model
