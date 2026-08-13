import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from src.config import SEED
from src.utils import get_logger

logger = get_logger()

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# PyTorch Tabular Neural Network Definition
if TORCH_AVAILABLE:
    class TabularMLP(nn.Module):
        def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.2):
            super().__init__()
            layers = []
            in_dim = input_dim
            for h_dim in hidden_dims:
                layers.append(nn.Linear(in_dim, h_dim))
                layers.append(nn.BatchNorm1d(h_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                in_dim = h_dim
            layers.append(nn.Linear(in_dim, 1))
            layers.append(nn.Sigmoid())
            self.network = nn.Sequential(*layers)
            
        def forward(self, x):
            return self.network(x).squeeze(-1)

def get_model(model_name, params=None):
    """
    Factory function to instantiate models with optimal default parameters.
    """
    p = params or {}
    
    if model_name.lower() == "lightgbm":
        default_params = {
            'n_estimators': 800,
            'learning_rate': 0.03,
            'num_leaves': 31,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'scale_pos_weight': 1.5,
            'random_state': SEED,
            'n_jobs': -1,
            'verbose': -1
        }
        default_params.update(p)
        return LGBMClassifier(**default_params)
        
    elif model_name.lower() == "xgboost":
        default_params = {
            'n_estimators': 700,
            'learning_rate': 0.03,
            'max_depth': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'scale_pos_weight': 1.5,
            'random_state': SEED,
            'n_jobs': -1,
            'eval_metric': 'logloss',
            'tree_method': 'hist'
        }
        default_params.update(p)
        return XGBClassifier(**default_params)
        
    elif model_name.lower() == "random_forest":
        default_params = {
            'n_estimators': 300,
            'max_depth': 14,
            'min_samples_split': 10,
            'min_samples_leaf': 4,
            'max_features': 'sqrt',
            'class_weight': 'balanced_subsample',
            'random_state': SEED,
            'n_jobs': -1
        }
        default_params.update(p)
        return RandomForestClassifier(**default_params)
        
    elif model_name.lower() == "extra_trees":
        default_params = {
            'n_estimators': 300,
            'max_depth': 14,
            'min_samples_split': 10,
            'min_samples_leaf': 4,
            'max_features': 'sqrt',
            'class_weight': 'balanced_subsample',
            'random_state': SEED,
            'n_jobs': -1
        }
        default_params.update(p)
        return ExtraTreesClassifier(**default_params)
        
    elif model_name.lower() == "logistic_regression":
        default_params = {
            'C': 0.1,
            'max_iter': 1000,
            'class_weight': 'balanced',
            'random_state': SEED,
            'n_jobs': -1
        }
        default_params.update(p)
        return LogisticRegression(**default_params)
        
    elif model_name.lower() == "pytorch_mlp":
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available for pytorch_mlp model.")
        return PyTorchMLPWrapper(params=p)
        
    else:
        raise ValueError(f"Unknown model name: {model_name}")


class PyTorchMLPWrapper:
    """Scikit-learn compatible wrapper for PyTorch Tabular Neural Network."""
    def __init__(self, params=None):
        self.params = params or {}
        self.epochs = self.params.get('epochs', 25)
        self.lr = self.params.get('lr', 1e-3)
        self.batch_size = self.params.get('batch_size', 256)
        self.hidden_dims = self.params.get('hidden_dims', [128, 64, 32])
        self.dropout = self.params.get('dropout', 0.2)
        self.scaler = StandardScaler()
        self.model = None
        
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(np.nan_to_num(X, nan=0.0))
        y_tensor = torch.tensor(y.values if hasattr(y, 'values') else y, dtype=torch.float32)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model = TabularMLP(input_dim=X.shape[1], hidden_dims=self.hidden_dims, dropout=self.dropout)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.BCELoss()
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                out = self.model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(np.nan_to_num(X, nan=0.0))
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            probs = self.model(X_tensor).numpy()
        return np.column_stack([1 - probs, probs])
