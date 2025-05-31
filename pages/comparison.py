import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from model_utils import load_models, predict_with_model, evaluate_model

st.set_page_config(page_title="Model Comparison Dashboard", layout="wide")
st.title("📊 Model Comparison Dashboard")

# Load models and data
models = load_models()
data = pd.read_csv("data/cleaned_data.csv")

def display_model_comparison():
    st.subheader("Select Models to Compare")
    selected_models = st.multiselect(
        "Choose ML models to compare:", list(models.keys()), default=list(models.keys())[:2]
    )

    if len(selected_models) < 2:
        st.warning("Please select at least two models for comparison.")
        return

    st.subheader("Evaluation Results")
    metrics = []
    for model_name in selected_models:
        model = models[model_name]
        y_true, y_pred, mse, rmse, r2 = evaluate_model(model, data)
        metrics.append({
            "Model": model_name,
            "MSE": mse,
            "RMSE": rmse,
            "R² Score": r2
        })

    results_df = pd.DataFrame(metrics)
    st.dataframe(results_df.style.format({"MSE": "{:.2f}", "RMSE": "{:.2f}", "R² Score": "{:.2f}"}))

    st.subheader("📈 Comparison Bar Chart")
    metric_to_plot = st.selectbox("Choose metric to plot:", ["MSE", "RMSE", "R² Score"])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(results_df["Model"], results_df[metric_to_plot], color='skyblue')
    ax.set_ylabel(metric_to_plot)
    ax.set_title(f"{metric_to_plot} by Model")
    st.pyplot(fig)

with st.container():
    display_model_comparison()
