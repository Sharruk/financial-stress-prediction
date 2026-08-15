import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV
from src.utils import evaluate_predictions, get_logger

logger = get_logger()

def calibrate_temperature(y_true, oof_probs, initial_temp=1.05):
    """
    Find optimal temperature parameter T minimizing Log Loss on OOF predictions.
    P_calibrated = Sigmoid(logit(P) / T)
    Prevents overconfident penalty on multi-metric leaderboards.
    """
    eps = 1e-6
    raw_logits = logit(np.clip(oof_probs, eps, 1.0 - eps))
    
    def temp_loss(T):
        scaled_probs = expit(raw_logits / T[0])
        metrics = evaluate_predictions(y_true, scaled_probs)
        return metrics['log_loss']
        
    res = minimize(temp_loss, [initial_temp], method='Nelder-Mead', bounds=[(0.5, 2.0)])
    best_temp = float(res.x[0])
    logger.info(f"Optimal Temperature Parameter T: {best_temp:.4f}")
    return best_temp

def apply_temperature_scaling(probs, temp):
    """Apply temperature scaling to probability vector."""
    eps = 1e-6
    raw_logits = logit(np.clip(probs, eps, 1.0 - eps))
    return expit(raw_logits / temp)

def optimize_ensemble_weights(oof_dict, y_true):
    """
    Find optimal probability weights minimizing OOF Multi Score using SLSQP optimization.
    """
    model_names = list(oof_dict.keys())
    oof_matrix = np.column_stack([oof_dict[m] for m in model_names])
    n_models = len(model_names)
    
    def loss_func(weights):
        weights = weights / (np.sum(weights) + 1e-9)
        blend_oof = np.dot(oof_matrix, weights)
        metrics = evaluate_predictions(y_true, blend_oof)
        # Zindi Multi Score objective: lower log loss + higher ROC-AUC
        return metrics['log_loss'] - 0.10 * metrics['roc_auc']
        
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


def compute_logit_blend(oof_dict, test_dict, weights):
    """
    Blend predictions in log-odds (logit) space.
    Preserves probability calibration and high-confidence predictions.
    """
    eps = 1e-6
    model_names = list(oof_dict.keys())
    oof_matrix = np.column_stack([logit(np.clip(oof_dict[m], eps, 1 - eps)) for m in model_names])
    test_matrix = np.column_stack([logit(np.clip(test_dict[m], eps, 1 - eps)) for m in model_names])
    
    blend_oof = expit(np.dot(oof_matrix, weights))
    blend_test = expit(np.dot(test_matrix, weights))
    return blend_oof, blend_test


def compute_rank_average(oof_dict, test_dict, weights=None):
    """
    Rank average test predictions to optimize ROC-AUC ranking consistency.
    """
    model_names = list(oof_dict.keys())
    n_models = len(model_names)
    if weights is None:
        weights = np.ones(n_models) / n_models
        
    ranked_oof_matrix = np.column_stack([rankdata(oof_dict[m]) / len(oof_dict[m]) for m in model_names])
    ranked_test_matrix = np.column_stack([rankdata(test_dict[m]) / len(test_dict[m]) for m in model_names])
    
    rank_oof = np.dot(ranked_oof_matrix, weights)
    rank_test = np.dot(ranked_test_matrix, weights)
    return rank_oof, rank_test


def train_stacking_meta_learner(oof_dict, test_dict, y_true):
    """
    Train a LogisticRegression meta-learner on base model OOF probabilities.
    """
    logger.info("Training Stacking Meta-Learner (Logistic Regression)...")
    model_names = list(oof_dict.keys())
    oof_matrix = np.column_stack([oof_dict[m] for m in model_names])
    test_matrix = np.column_stack([test_dict[m] for m in model_names])
    
    meta_model = LogisticRegression(C=0.5, random_state=42)
    meta_model.fit(oof_matrix, y_true)
    
    meta_oof = meta_model.predict_proba(oof_matrix)[:, 1]
    meta_test = meta_model.predict_proba(test_matrix)[:, 1]
    
    metrics = evaluate_predictions(y_true, meta_oof)
    logger.info(f"==> STACKING META-LEARNER OOF | Log Loss: {metrics['log_loss']:.5f} | ROC-AUC: {metrics['roc_auc']:.5f} <==")
    return meta_oof, meta_test, metrics
