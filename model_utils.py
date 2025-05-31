from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import streamlit as st

def evaluate_model(y_true, y_pred, label="Test"):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    st.markdown(f"""
    **🔍 {label} Evaluation Metrics:**
    - 📉 MSE: `{mse:.2f}`
    - 📏 MAE: `{mae:.2f}`
    - 📈 R² Score: `{r2:.3f}`
    """)
