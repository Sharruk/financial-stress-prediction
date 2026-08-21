import numpy as np
import pandas as pd
import pytest
from src.utils import evaluate_predictions, calculate_generalization_gap

def test_evaluate_predictions_binary():
    y_true = np.array([0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.8, 0.4, 0.6])
    
    metrics = evaluate_predictions(y_true, y_prob, threshold=0.5)
    
    assert "log_loss" in metrics
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "accuracy" in metrics
    assert metrics["accuracy"] == 1.0
    assert metrics["tp"] == 3
    assert metrics["tn"] == 2
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0

def test_generalization_gap():
    train_metrics = {
        "log_loss": 0.20,
        "brier_score": 0.10,
        "roc_auc": 0.95,
        "pr_auc": 0.95,
        "accuracy": 0.90,
        "precision": 0.85,
        "recall": 0.80,
        "f1": 0.80,
        "specificity": 0.92,
        "balanced_accuracy": 0.86,
        "mcc": 0.75,
        "tp": 2790,
        "tn": 26844,
        "fp": 355,
        "fn": 2009,
        "model_name": "catboost"
    }
    val_metrics = {
        "log_loss": 0.25,
        "brier_score": 0.15,
        "roc_auc": 0.90,
        "pr_auc": 0.90,
        "accuracy": 0.85,
        "precision": 0.75,
        "recall": 0.70,
        "f1": 0.70,
        "specificity": 0.88,
        "balanced_accuracy": 0.79,
        "mcc": 0.65,
        "tp": 2783,
        "tn": 33117,
        "fp": 883,
        "fn": 3217,
        "model_name": "catboost"
    }
    
    gap = calculate_generalization_gap(train_metrics, val_metrics)
    
    # 1. Loss metrics (Validation - Train -> Positive gap means overfitting)
    assert gap["log_loss"] == pytest.approx(0.05)
    assert gap["brier_score"] == pytest.approx(0.05)
    
    # 2. Score metrics (Train - Validation -> Positive gap means overfitting)
    assert gap["roc_auc"] == pytest.approx(0.05)
    assert gap["pr_auc"] == pytest.approx(0.05)
    assert gap["accuracy"] == pytest.approx(0.05)
    assert gap["precision"] == pytest.approx(0.10)
    assert gap["recall"] == pytest.approx(0.10)
    assert gap["f1"] == pytest.approx(0.10)
    assert gap["specificity"] == pytest.approx(0.04)
    assert gap["balanced_accuracy"] == pytest.approx(0.07)
    assert gap["mcc"] == pytest.approx(0.10)
    
    # 3. Confusion counts must NOT be in generalization_gap
    assert "tp" not in gap
    assert "tn" not in gap
    assert "fp" not in gap
    assert "fn" not in gap
    
    # 4. Non-numeric metadata must NOT be in generalization_gap
    assert "model_name" not in gap

def test_generalization_gap_identical():
    metrics = {"log_loss": 0.25, "roc_auc": 0.90, "f1": 0.60, "tp": 100}
    gap = calculate_generalization_gap(metrics, metrics)
    assert gap["log_loss"] == pytest.approx(0.0)
    assert gap["roc_auc"] == pytest.approx(0.0)
    assert gap["f1"] == pytest.approx(0.0)
    assert "tp" not in gap

def test_generalization_gap_missing_and_robustness():
    # Missing metrics in either train or val should be handled safely without KeyError
    train_m = {"log_loss": 0.20, "extra_meta": "str_val"}
    val_m = {"roc_auc": 0.90, "brier_score": 0.08}
    gap = calculate_generalization_gap(train_m, val_m)
    assert gap == {}

    # Invalid input types should return empty dict
    assert calculate_generalization_gap(None, None) == {}
    assert calculate_generalization_gap([], {}) == {}

