import os
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from src.config import SEED
from src.utils import get_logger

logger = get_logger()

# -------------------------------------------------------------
# Dynamic GPU Auto-Detection
# -------------------------------------------------------------
def is_gpu_available():
    """Check if NVIDIA CUDA GPU is available for acceleration."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "" or os.environ.get("GPU_ENABLED") == "false":
        return False
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return True
    except Exception:
        pass
    return False

GPU_AVAILABLE = is_gpu_available()
if GPU_AVAILABLE:
    logger.info("⚡ NVIDIA CUDA GPU detected! Enabling hardware acceleration.")
else:
    logger.info("💻 GPU not detected/available. Running on multi-core CPU.")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
    TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    TORCH_AVAILABLE = False
    TORCH_DEVICE = "cpu"

# High-Performance PyTorch Tabular Neural Network with Residual connections
if TORCH_AVAILABLE:
    class TabularResMLP(nn.Module):
        def __init__(self, input_dim, hidden_dim=192, dropout=0.2):
            super().__init__()
            self.input_layer = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.Mish(),
                nn.Dropout(dropout)
            )
            # Residual Block 1
            self.res1 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.Mish(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim)
            )
            # Residual Block 2
            self.res2 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.Mish(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2)
            )
            self.proj = nn.Linear(hidden_dim, hidden_dim // 2)
            self.head = nn.Sequential(
                nn.Mish(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
            
        def forward(self, x):
            x0 = self.input_layer(x)
            x1 = x0 + self.res1(x0)
            x2 = self.proj(x1) + self.res2(x1)
            return self.head(x2).squeeze(-1)

def get_model(model_name, params=None):
    """
    Factory function to instantiate models with Grand Master v6 hyperparameters.
    """
    p = params or {}
    
    if model_name.lower() == "lightgbm":
        default_params = {
            'n_estimators': 1800,
            'learning_rate': 0.015,
            'num_leaves': 63,
            'max_depth': 8,
            'subsample': 0.8,
            'colsample_bytree': 0.6,
            'min_child_samples': 25,
            'reg_alpha': 0.5,
            'reg_lambda': 4.0,
            'scale_pos_weight': 1.0,
            'random_state': SEED,
            'n_jobs': -1,
            'verbose': -1
        }
        default_params.update(p)
        return LGBMClassifier(**default_params)

    elif model_name.lower() == "lightgbm_dart":
        default_params = {
            'boosting_type': 'dart',
            'n_estimators': 1400,
            'learning_rate': 0.022,
            'num_leaves': 45,
            'max_depth': 7,
            'subsample': 0.8,
            'colsample_bytree': 0.6,
            'drop_rate': 0.1,
            'skip_drop': 0.5,
            'scale_pos_weight': 1.0,
            'random_state': SEED + 7,
            'n_jobs': -1,
            'verbose': -1
        }
        default_params.update(p)
        return LGBMClassifier(**default_params)

    elif model_name.lower() == "lightgbm_goss":
        default_params = {
            'boosting_type': 'goss',
            'n_estimators': 1400,
            'learning_rate': 0.018,
            'num_leaves': 45,
            'max_depth': 7,
            'colsample_bytree': 0.6,
            'reg_alpha': 0.5,
            'reg_lambda': 4.0,
            'scale_pos_weight': 1.0,
            'random_state': SEED + 13,
            'n_jobs': -1,
            'verbose': -1
        }
        default_params.update(p)
        return LGBMClassifier(**default_params)
        
    elif model_name.lower() == "xgboost":
        default_params = {
            'n_estimators': 1500,
            'learning_rate': 0.015,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.6,
            'min_child_weight': 5,
            'reg_alpha': 0.5,
            'reg_lambda': 5.0,
            'scale_pos_weight': 1.0,
            'random_state': SEED,
            'n_jobs': -1,
            'eval_metric': 'logloss',
            'early_stopping_rounds': 150,
            'tree_method': 'hist',
            'device': 'cuda' if GPU_AVAILABLE else 'cpu'
        }
        default_params.update(p)
        return XGBClassifier(**default_params)

    elif model_name.lower() == "catboost":
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost is not available.")
        
        gpu_detected = is_gpu_available()
        default_params = {
            'iterations': 2500,
            'learning_rate': 0.015,
            'depth': 6,
            'l2_leaf_reg': 8.0,
            'random_strength': 0.8,
            'bagging_temperature': 0.2,
            'border_count': 128,
            'eval_metric': 'Logloss',
            'loss_function': 'Logloss',
            'random_seed': SEED,
            'verbose': 0,
            'task_type': 'GPU' if gpu_detected else 'CPU'
        }
        
        # Check if custom devices passed or in env
        gpu_devices = os.environ.get("GPU_DEVICES", None)
        if default_params['task_type'] == 'GPU' and gpu_devices:
            default_params['devices'] = gpu_devices
            
        default_params.update(p)
        return cb.CatBoostClassifier(**default_params)

    elif model_name.lower() == "hist_gbm":
        default_params = {
            'max_iter': 1000,
            'learning_rate': 0.018,
            'max_leaf_nodes': 45,
            'max_depth': 8,
            'min_samples_leaf': 25,
            'l2_regularization': 3.0,
            'random_state': SEED
        }
        default_params.update(p)
        return HistGradientBoostingClassifier(**default_params)
        
    elif model_name.lower() == "extra_trees":
        default_params = {
            'n_estimators': 600,
            'max_depth': 18,
            'min_samples_split': 6,
            'min_samples_leaf': 2,
            'max_features': 0.25,
            'random_state': SEED,
            'n_jobs': -1
        }
        default_params.update(p)
        return ExtraTreesClassifier(**default_params)

    elif model_name.lower() == "random_forest":
        default_params = {
            'n_estimators': 600,
            'max_depth': 16,
            'min_samples_split': 8,
            'min_samples_leaf': 3,
            'max_features': 0.25,
            'random_state': SEED,
            'n_jobs': -1
        }
        default_params.update(p)
        return RandomForestClassifier(**default_params)
        
    elif model_name.lower() == "logistic_regression":
        default_params = {
            'C': 0.1,
            'max_iter': 1000,
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
    """Scikit-learn compatible wrapper for PyTorch Tabular Neural Network with auto GPU."""
    def __init__(self, params=None):
        self.params = params or {}
        self.epochs = self.params.get('epochs', 25)
        self.lr = self.params.get('lr', 1e-3)
        self.batch_size = self.params.get('batch_size', 256)
        self.hidden_dim = self.params.get('hidden_dim', 192)
        self.dropout = self.params.get('dropout', 0.2)
        self.device = TORCH_DEVICE
        self.scaler = StandardScaler()
        self.model = None
        
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(np.nan_to_num(X, nan=0.0))
        y_tensor = torch.tensor(y.values if hasattr(y, 'values') else y, dtype=torch.float32)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model = TabularResMLP(input_dim=X.shape[1], hidden_dim=self.hidden_dim, dropout=self.dropout).to(self.device)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.BCELoss()
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                out = self.model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(np.nan_to_num(X, nan=0.0))
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            probs = self.model(X_tensor).cpu().numpy()
        return np.column_stack([1 - probs, probs])
