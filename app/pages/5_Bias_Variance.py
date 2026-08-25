import streamlit as st
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from api.dependencies import get_all_experiments
from app.components.charts import plot_generalization_gap
from src.utils import calculate_generalization_gap

st.title("Bias / Variance Diagnostics")

st.markdown("""
> **Diagnostic Guide**:
> - **Loss Metrics (Log Loss, Brier)**: `Generalization Gap = Validation Loss - Training Loss`
> - **Score Metrics (ROC-AUC, PR-AUC, F1)**: `Generalization Gap = Training Score - Validation Score`
> 
> A **positive gap** (Validation Loss > Train Loss or Train Score > Validation Score) indicates **overfitting / high variance**.
> A **near-zero gap** with strong scores indicates **optimal generalization**.
> Poor performance on both train and validation indicates **underfitting / high bias**.
""")

experiments = get_all_experiments()

if not experiments:
    st.info("No data available.")
else:
    latest = experiments[0]
    st.write(f"Showing diagnostics from latest experiment: **{latest.get('experiment_id')}**")
    
    rows = []
    for bm in latest.get("_base_models", []):
        train_metrics = bm.get("training_metrics", {})
        oof_metrics = bm.get("oof_metrics", {})
        if train_metrics and oof_metrics:
            bm_gap = calculate_generalization_gap(train_metrics, oof_metrics)
        else:
            bm_gap = bm.get("generalization_gap", {})

        if bm_gap:
            train_loss = train_metrics.get("log_loss") if train_metrics else None
            val_loss = oof_metrics.get("log_loss") if oof_metrics else None
            rows.append({
                "Model": bm.get("name"),
                "Train Log Loss": train_loss,
                "Validation Log Loss": val_loss,
                "Generalization Gap": bm_gap.get("log_loss")
            })
            
    if rows:
        df = pd.DataFrame(rows)
        st.plotly_chart(plot_generalization_gap(df), use_container_width=True)
        st.dataframe(df.style.format("{:.5f}", subset=["Train Log Loss", "Validation Log Loss", "Generalization Gap"]), use_container_width=True)
    else:
        st.info("No training metrics were recorded for this experiment. Run a full CV with updated engine to view gaps.")
