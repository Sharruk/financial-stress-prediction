import streamlit as st
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from api.dependencies import get_all_experiments
from src.utils import calculate_generalization_gap

st.title("Model Leaderboard")

experiments = get_all_experiments()

if not experiments:
    st.info("No experiment data available.")
else:
    rows = []
    for exp in experiments:
        exp_id = exp.get("experiment_id")
        
        # Add ensemble/top level result
        top_metrics = exp.get("oof_metrics", {})
        if top_metrics:
            rows.append({
                "Experiment": exp_id,
                "Model": exp.get("model", "Unknown"),
                "Type": "Primary/Ensemble",
                "Log Loss": top_metrics.get("log_loss"),
                "ROC-AUC": top_metrics.get("roc_auc"),
                "PR-AUC": top_metrics.get("pr_auc"),
                "F1": top_metrics.get("f1"),
                "Accuracy": top_metrics.get("accuracy"),
                "Brier": top_metrics.get("brier_score"),
                "Gap": None
            })
            
        # Add base models
        for bm in exp.get("_base_models", []):
            bm_metrics = bm.get("oof_metrics", {})
            train_metrics = bm.get("training_metrics", {})
            if train_metrics and bm_metrics:
                bm_gap = calculate_generalization_gap(train_metrics, bm_metrics)
            else:
                bm_gap = bm.get("generalization_gap", {})
            rows.append({
                "Experiment": exp_id,
                "Model": bm.get("name"),
                "Type": "Base Model",
                "Log Loss": bm_metrics.get("log_loss"),
                "ROC-AUC": bm_metrics.get("roc_auc"),
                "PR-AUC": bm_metrics.get("pr_auc"),
                "F1": bm_metrics.get("f1"),
                "Accuracy": bm_metrics.get("accuracy"),
                "Brier": bm_metrics.get("brier_score"),
                "Gap": bm_gap.get("log_loss") if bm_gap else None
            })
            
    if rows:
        df = pd.DataFrame(rows)
        # Drop rows with entirely missing metrics
        df = df.dropna(subset=["Log Loss", "ROC-AUC"], how="all")
        
        st.dataframe(
            df.style.highlight_min(subset=["Log Loss", "Brier"], color='lightgreen')
                    .highlight_max(subset=["ROC-AUC", "PR-AUC", "F1", "Accuracy"], color='lightgreen'),
            use_container_width=True
        )
