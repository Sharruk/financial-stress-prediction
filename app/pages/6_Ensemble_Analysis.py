import streamlit as st
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from api.dependencies import get_all_experiments

st.title("Ensemble Analysis")

experiments = get_all_experiments()

if not experiments:
    st.info("No data available.")
else:
    latest = experiments[0]
    st.write(f"Showing ensemble data from latest experiment: **{latest.get('experiment_id')}**")
    
    ensemble_info = latest.get("ensemble_information", {})
    if not ensemble_info:
        st.info("No ensemble information recorded for this experiment.")
    else:
        strategy = ensemble_info.get("strategy", "Unknown")
        st.subheader(f"Strategy: {strategy}")
        
        weights = ensemble_info.get("weights", {})
        if weights:
            st.write("### Model Weights")
            df_w = pd.DataFrame(list(weights.items()), columns=["Model", "Weight"])
            st.dataframe(df_w, use_container_width=True)
            
            st.bar_chart(df_w.set_index("Model"))
        else:
            st.write("No specific weight vector associated with this strategy (e.g. Meta-Learner Stacking or Rank Average).")
