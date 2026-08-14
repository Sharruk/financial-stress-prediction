import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from src.config import N_SPLITS, SEED, TARGET_COL, ID_COL
from src.models import get_model
from src.utils import evaluate_predictions, get_logger

logger = get_logger()

def preprocess_fold_features(train_fold, val_fold, test_df, target_col):
    """
    Fold-aware categorical processing (Target Encoding).
    Identifies all non-numeric columns dynamically to guarantee 0 leakage.
    """
    X_tr = train_fold.copy()
    X_val = val_fold.copy()
    X_te = test_df.copy()
    
    # Automatically detect all string/object/categorical columns
    cat_cols = [c for c in X_tr.columns if c not in [ID_COL, target_col] and not pd.api.types.is_numeric_dtype(X_tr[c])]
    
    # Target Encoding for categorical columns fit ONLY on train_fold
    global_target_mean = train_fold[target_col].mean()
    smooth_weight = 10
    
    for col in cat_cols:
        # Group stats
        stats = train_fold.groupby(col)[target_col].agg(['count', 'mean'])
        smooth_te = (stats['count'] * stats['mean'] + smooth_weight * global_target_mean) / (stats['count'] + smooth_weight)
        te_dict = smooth_te.to_dict()
        
        te_col_name = f"{col}_te"
        X_tr[te_col_name] = X_tr[col].map(te_dict).fillna(global_target_mean).astype(np.float32)
        X_val[te_col_name] = X_val[col].map(te_dict).fillna(global_target_mean).astype(np.float32)
        X_te[te_col_name] = X_te[col].map(te_dict).fillna(global_target_mean).astype(np.float32)

    # Keep only numeric columns for model input
    feature_cols = [c for c in X_tr.select_dtypes(include=[np.number]).columns if c not in [ID_COL, target_col]]
    
    return X_tr[feature_cols], X_val[feature_cols], X_te[feature_cols]


def train_cv_model(model_name, train_df, test_df, model_params=None, n_splits=N_SPLITS):
    """
    Train a model using Stratified K-Fold cross-validation.
    Returns OOF predictions, test predictions, fold metrics, and feature importances.
    """
    logger.info(f"--- Running {n_splits}-Fold Stratified CV for {model_name.upper()} ---")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof_probs = np.zeros(len(train_df), dtype=np.float32)
    test_probs = np.zeros(len(test_df), dtype=np.float32)
    
    y_train = train_df[TARGET_COL].values
    
    fold_scores = []
    feature_importances = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, y_train), 1):
        tr_fold = train_df.iloc[train_idx]
        val_fold = train_df.iloc[val_idx]
        
        # Leak-free feature preprocessing per fold
        X_tr, X_val, X_te = preprocess_fold_features(tr_fold, val_fold, test_df, TARGET_COL)
        y_tr = tr_fold[TARGET_COL].values
        y_v = val_fold[TARGET_COL].values
        
        # Instantiate fresh model
        model = get_model(model_name, params=model_params)
        model.fit(X_tr, y_tr)
            
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
        
    return {
        "model_name": model_name,
        "oof_probs": oof_probs,
        "test_probs": test_probs,
        "oof_metrics": oof_metrics,
        "fold_scores": fold_scores,
        "feature_importances": avg_feat_imp
    }
