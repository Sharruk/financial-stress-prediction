import streamlit as st
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from api.dependencies import get_all_experiments
from app.components.charts import plot_generalization_gap

st.title("Bias / Variance Diagnostics")

st.markdown("""
> **Note**: This is a diagnostic view showing the generalization gap (Train Metric - Validation Metric).
> A large positive gap (e.g., Train loss is much lower than Validation loss) indicates potential overfitting / high variance.
> Poor performance on both train and validation indicates potential underfitting / high bias.
""")

experiments = get_all_experiments()

if not experiments:
    st.info("No data available.")
else:
    latest = experiments[0]
    st.write(f"Showing diagnostics from latest experiment: **{latest.get('experiment_id')}**")
    
    rows = []
    for bm in latest.get("_base_models", []):
        bm_gap = bm.get("generalization_gap", {})
        if bm_gap:
            train_loss = bm.get("training_metrics", {}).get("log_loss")
            val_loss = bm.get("oof_metrics", {}).get("log_loss")
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
