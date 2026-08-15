import streamlit as st
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from api.dependencies import get_all_experiments
from app.components.charts import plot_metric_comparison

st.title("Model Metrics Comparison")

experiments = get_all_experiments()

if not experiments:
    st.info("No data available.")
else:
    # Use latest experiment for detailed bar charts
    latest = experiments[0]
    st.write(f"Showing metrics from latest experiment: **{latest.get('experiment_id')}**")
    
    rows = []
    for bm in latest.get("_base_models", []):
        bm_metrics = bm.get("oof_metrics", {})
        row = {"Model": bm.get("name")}
        row.update(bm_metrics)
        rows.append(row)
        
    if rows:
        df = pd.DataFrame(rows)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_metric_comparison(df, "log_loss", "Log Loss (Lower is Better)"), use_container_width=True)
            st.plotly_chart(plot_metric_comparison(df, "f1", "F1 Score (Higher is Better)"), use_container_width=True)
        with col2:
            st.plotly_chart(plot_metric_comparison(df, "roc_auc", "ROC-AUC (Higher is Better)"), use_container_width=True)
            st.plotly_chart(plot_metric_comparison(df, "pr_auc", "PR-AUC (Higher is Better)"), use_container_width=True)
