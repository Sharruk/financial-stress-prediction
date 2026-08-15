import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def plot_metric_comparison(df, metric_col, title):
    if df.empty or metric_col not in df.columns:
        return go.Figure().update_layout(title="Data not available")
    
    fig = px.bar(
        df, 
        x="Model", 
        y=metric_col, 
        color="Model", 
        title=title,
        text_auto=".4f"
    )
    fig.update_layout(xaxis_title="Model", yaxis_title=metric_col, showlegend=False)
    return fig

def plot_generalization_gap(df):
    if df.empty or "Generalization Gap" not in df.columns:
        return go.Figure().update_layout(title="Gap data not available")
        
    df_clean = df.dropna(subset=["Generalization Gap"]).copy()
    if df_clean.empty:
         return go.Figure().update_layout(title="Gap data not available")
         
    df_clean["Color"] = df_clean["Generalization Gap"].apply(lambda x: "red" if x < 0 else "blue")
    
    fig = go.Figure(data=[
        go.Bar(
            name="Generalization Gap",
            x=df_clean["Model"],
            y=df_clean["Generalization Gap"],
            marker_color=df_clean["Color"]
        )
    ])
    fig.update_layout(
        title="Generalization Gap (Train Log Loss - Validation Log Loss)",
        xaxis_title="Model",
        yaxis_title="Gap"
    )
    return fig

def plot_train_vs_val(df, metric_train, metric_val, title):
    if df.empty or metric_train not in df.columns or metric_val not in df.columns:
        return go.Figure().update_layout(title="Train/Val data not available")
        
    df_clean = df.dropna(subset=[metric_train, metric_val]).copy()
    
    fig = go.Figure(data=[
        go.Bar(name='Train', x=df_clean['Model'], y=df_clean[metric_train]),
        go.Bar(name='Validation', x=df_clean['Model'], y=df_clean[metric_val])
    ])
    fig.update_layout(barmode='group', title=title)
    return fig
