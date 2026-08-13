import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from src.utils import evaluate_predictions, get_logger

logger = get_logger()

def optimize_ensemble_weights(oof_dict, y_true):
    """
    Find optimal probability weights minimizing OOF Log Loss using SLSQP optimization.
    """
    model_names = list(oof_dict.keys())
    oof_matrix = np.column_stack([oof_dict[m] for m in model_names])
    n_models = len(model_names)
    
    def loss_func(weights):
        # Normalize weights to sum to 1
        weights = weights / np.sum(weights)
        blend_oof = np.dot(oof_matrix, weights)
        metrics = evaluate_predictions(y_true, blend_oof)
        return metrics['log_loss']
        
    initial_weights = np.ones(n_models) / n_models
    bounds = [(0.0, 1.0) for _ in range(n_models)]
    constraints = ({'type': 'eq', 'fun': lambda w: 1.0 - sum(w)})
    
    result = minimize(loss_func, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    best_weights = result.x / np.sum(result.x)
    
    weight_dict = {name: round(float(w), 4) for name, w in zip(model_names, best_weights)}
    logger.info(f"Optimized Ensemble Weights: {weight_dict}")
    return best_weights, weight_dict


def compute_blend_predictions(oof_dict, test_dict, weights):
    """
    Compute weighted average of OOF and Test predictions.
    """
    model_names = list(oof_dict.keys())
    oof_matrix = np.column_stack([oof_dict[m] for m in model_names])
    test_matrix = np.column_stack([test_dict[m] for m in model_names])
    
    blend_oof = np.dot(oof_matrix, weights)
    blend_test = np.dot(test_matrix, weights)
    
    return blend_oof, blend_test


def compute_rank_average(test_dict, weights=None):
    """
    Rank average test predictions to mitigate probability calibration shifts across models.
    """
    model_names = list(test_dict.keys())
    n_models = len(model_names)
    if weights is None:
        weights = np.ones(n_models) / n_models
        
    ranked_matrix = np.column_stack([rankdata(test_dict[m]) / len(test_dict[m]) for m in model_names])
    rank_blend = np.dot(ranked_matrix, weights)
    return rank_blend


def train_stacking_meta_learner(oof_dict, test_dict, y_true):
    """
    Train a LogisticRegression meta-learner on base model OOF probabilities.
    """
    logger.info("Training Stacking Meta-Learner (Logistic Regression)...")
    model_names = list(oof_dict.keys())
    oof_matrix = np.column_stack([oof_dict[m] for m in model_names])
    test_matrix = np.column_stack([test_dict[m] for m in model_names])
    
    meta_model = LogisticRegression(C=1.0, random_state=42)
    meta_model.fit(oof_matrix, y_true)
    
    meta_oof = meta_model.predict_proba(oof_matrix)[:, 1]
    meta_test = meta_model.predict_proba(test_matrix)[:, 1]
    
    metrics = evaluate_predictions(y_true, meta_oof)
    logger.info(f"==> STACKING META-LEARNER OOF | Log Loss: {metrics['log_loss']:.5f} | ROC-AUC: {metrics['roc_auc']:.5f} <==")
    return meta_oof, meta_test, metrics
