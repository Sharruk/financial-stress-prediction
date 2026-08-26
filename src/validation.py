import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from src.config import N_SPLITS, SEED, TARGET_COL, ID_COL
from src.models import get_model
from src.utils import evaluate_predictions, get_logger

logger = get_logger()

from src.features import fit_fold_unsupervised_personas, transform_fold_unsupervised_personas

def preprocess_fold_features(train_fold, val_fold, test_df, target_col, seed=SEED):
    """
    Fold-aware categorical processing (Target Encoding with Bayesian smoothing & noise injection)
    and KMeans Persona clustering.
    Fit strictly on train_fold only to guarantee zero validation/test leakage.
    """
    X_tr = train_fold.copy()
    X_val = val_fold.copy()
    X_te = test_df.copy()

    # Automatically detect all string/object/categorical columns
    cat_cols = [c for c in X_tr.columns if c not in [ID_COL, target_col] and not pd.api.types.is_numeric_dtype(X_tr[c])]

    # Target Encoding for categorical columns fit ONLY on train_fold with Bayesian smoothing
    global_target_mean = float(train_fold[target_col].mean())
    smooth_weight = 15.0
    rng = np.random.RandomState(seed)

    for col in cat_cols:
        # Group stats strictly from train_fold
        stats = train_fold.groupby(col)[target_col].agg(['count', 'mean'])
        smooth_te = (stats['count'] * stats['mean'] + smooth_weight * global_target_mean) / (stats['count'] + smooth_weight)
        te_dict = smooth_te.to_dict()

        te_col_name = f"{col}_te"
        
        # Training target encoding with subtle Gaussian noise injection to prevent tree memorization
        train_raw_te = X_tr[col].map(te_dict).fillna(global_target_mean).values.astype(np.float32)
        noise = rng.normal(0, 0.005, size=len(train_raw_te)).astype(np.float32)
        X_tr[te_col_name] = np.clip(train_raw_te + noise, 0.0, 1.0)
        
        # Validation & Test remain completely uncorrupted / deterministic
        X_val[te_col_name] = X_val[col].map(te_dict).fillna(global_target_mean).astype(np.float32)
        X_te[te_col_name] = X_te[col].map(te_dict).fillna(global_target_mean).astype(np.float32)

    # Fold-safe Unsupervised KMeans Personas: Fit on train_fold only!
    kmeans, scaler_params, avail_cols = fit_fold_unsupervised_personas(X_tr, n_clusters=8, seed=seed)
    X_tr = transform_fold_unsupervised_personas(X_tr, kmeans, scaler_params, avail_cols)
    X_val = transform_fold_unsupervised_personas(X_val, kmeans, scaler_params, avail_cols)
    X_te = transform_fold_unsupervised_personas(X_te, kmeans, scaler_params, avail_cols)

    # Keep only numeric columns for model input
    feature_cols = [c for c in X_tr.select_dtypes(include=[np.number]).columns if c not in [ID_COL, target_col]]

    return X_tr[feature_cols], X_val[feature_cols], X_te[feature_cols]


def train_cv_model(model_name, train_df, test_df, model_params=None, n_splits=N_SPLITS, seed=SEED):
    """
    Train a model using Stratified K-Fold cross-validation.
    Returns OOF predictions, test predictions, fold metrics, and feature importances.
    """
    logger.info(f"--- Running {n_splits}-Fold Stratified CV for {model_name.upper()} (Seed: {seed}) ---")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_probs = np.zeros(len(train_df), dtype=np.float32)
    test_probs = np.zeros(len(test_df), dtype=np.float32)

    y_train = train_df[TARGET_COL].values

    fold_scores = []
    train_fold_scores = []
    feature_importances = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, y_train), 1):
        tr_fold = train_df.iloc[train_idx]
        val_fold = train_df.iloc[val_idx]

        # Leak-free feature preprocessing per fold
        X_tr, X_val, X_te = preprocess_fold_features(tr_fold, val_fold, test_df, TARGET_COL, seed=seed)
        y_tr = tr_fold[TARGET_COL].values
        y_v = val_fold[TARGET_COL].values

        # Instantiate fresh model
        model = get_model(model_name, params=model_params)
        
        # Fit model with validation set for early stopping if supported
        if model_name.lower() == "catboost":
            model.fit(X_tr, y_tr, eval_set=(X_val, y_v), early_stopping_rounds=150, verbose=False)
        elif model_name.lower() in ["lightgbm", "lightgbm_goss"]:
            try:
                import lightgbm as lgb
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_v)], callbacks=[lgb.early_stopping(100, verbose=False)])
            except Exception:
                model.fit(X_tr, y_tr)
        elif model_name.lower() == "lightgbm_dart":
            # DART booster does not support early stopping due to tree dropouts
            model.fit(X_tr, y_tr)
        elif model_name.lower() == "xgboost":
            try:
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_v)], verbose=False)
            except Exception:
                model.fit(X_tr, y_tr)
        else:
            model.fit(X_tr, y_tr)

        # Predict training probabilities
        train_pred = model.predict_proba(X_tr)[:, 1]
        train_metrics = evaluate_predictions(y_tr, train_pred)
        train_fold_scores.append(train_metrics)

        # Predict validation probabilities
        val_pred = model.predict_proba(X_val)[:, 1]
        oof_probs[val_idx] = val_pred

        # Predict test probabilities
        test_pred = model.predict_proba(X_te)[:, 1]
        test_probs += test_pred / n_splits

        # Fold evaluation
        metrics = evaluate_predictions(y_v, val_pred)
        fold_scores.append(metrics)
        logger.info(f"Fold {fold}/{n_splits} | Log Loss: {metrics['log_loss']:.5f} | ROC-AUC: {metrics['roc_auc']:.5f}")

        # Track feature importance if available
        if hasattr(model, 'feature_importances_'):
            feature_importances.append(model.feature_importances_)

    # Overall OOF evaluation
    oof_metrics = evaluate_predictions(y_train, oof_probs)
    logger.info(f"==> OVERALL OOF {model_name.upper()} | Log Loss: {oof_metrics['log_loss']:.5f} | ROC-AUC: {oof_metrics['roc_auc']:.5f} <==")

    avg_feat_imp = None
    if feature_importances:
        avg_imp = np.mean(feature_importances, axis=0)
        avg_feat_imp = pd.Series(avg_imp, index=X_tr.columns).sort_values(ascending=False)

    avg_train_metrics = {}
    if train_fold_scores:
        for k in train_fold_scores[0].keys():
            avg_train_metrics[k] = float(np.mean([x[k] for x in train_fold_scores]))

    return {
        "model_name": model_name,
        "oof_probs": oof_probs,
        "test_probs": test_probs,
        "oof_metrics": oof_metrics,
        "train_metrics": avg_train_metrics,
        "fold_scores": fold_scores,
        "feature_importances": avg_feat_imp
    }
