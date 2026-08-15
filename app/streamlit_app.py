import streamlit as st

st.set_page_config(
    page_title="Zindi ML Platform",
    page_icon="📈",
    layout="wide",
)

st.title("Zindi Financial Stress Prediction ML Platform")

st.markdown("""
Welcome to the ML Engineering Dashboard for the Zindi Financial Stress Prediction challenge.
Please select a page from the sidebar to view project metrics, models, and progress.

**Available Pages:**
- **Overview**: High-level summary of project status and top models.
- **Project Progress**: Task tracking matrix.
- **Model Leaderboard**: Filterable table of all model results.
- **Model Metrics**: Deep dive into individual metrics.
- **Bias / Variance**: Generalization gap diagnostics.
- **Ensemble Analysis**: View ensemble configurations.
""")
