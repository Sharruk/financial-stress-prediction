import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from api.dependencies import get_project_status, get_all_experiments

st.title("Overview")

experiments = get_all_experiments()

if not experiments:
    st.info("No experiments found in the registry.")
else:
    latest = experiments[0]
    total_exps = len(experiments)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Experiments", total_exps)
    col2.metric("Latest Experiment ID", latest.get("experiment_id", "Unknown"))
    
    # Calculate best metrics
    best_logloss = float('inf')
    best_rocauc = 0.0
    
    for exp in experiments:
        metrics = exp.get("oof_metrics", {})
        if metrics:
            loss = metrics.get("log_loss", float('inf'))
            auc = metrics.get("roc_auc", 0.0)
            if loss < best_logloss: best_logloss = loss
            if auc > best_rocauc: best_rocauc = auc
            
    col3.metric("Best Log Loss", f"{best_logloss:.4f}" if best_logloss != float('inf') else "N/A")
    col4.metric("Best ROC-AUC", f"{best_rocauc:.4f}" if best_rocauc != 0.0 else "N/A")

    st.subheader("Latest Experiment Details")
    st.json(latest)
