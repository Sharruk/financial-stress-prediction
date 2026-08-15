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
    train_metrics = {"log_loss": 0.3, "roc_auc": 0.85}
    val_metrics = {"log_loss": 0.4, "roc_auc": 0.80}
    
    gap = calculate_generalization_gap(train_metrics, val_metrics)
    
    assert gap["log_loss"] == pytest.approx(-0.1)
    assert gap["roc_auc"] == pytest.approx(0.05)
