import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from model_utils import (
    load_data,
    prepare_ml_data,
    train_test_split,
    create_lstm_sequences,
    plot_predictions,
    train_linear_regression,
    train_random_forest,
    train_decision_tree,
    train_ridge_regression,
    train_svr,
    train_lstm_model
)

# Streamlit page config
st.set_page_config(page_title="Carbon Emission Monitor", layout="wide")
st.title("🌍 AI Carbon Emission Monitor")
st.markdown("Monitor and predict global carbon emissions with machine learning.")

# Load data
DATA_PATH = os.path.join("data", "co2_emissions.csv")
df = load_data(DATA_PATH)
st.write("Columns in dataset:", df.columns.tolist())

st.subheader("Raw CO₂ Emission Data")
st.dataframe(df.head(10))

# Country and year selection
countries = df['Entity'].unique()
selected_country = st.selectbox("Select Country", sorted(countries))

min_year = int(df['Year'].min())
max_year = int(df['Year'].max())
year_range = st.slider("Select Year Range", min_year, max_year, (min_year, max_year))

# Filtered view
filtered_df = df[(df['Entity'] == selected_country) & 
                 (df['Year'] >= year_range[0]) & 
                 (df['Year'] <= year_range[1])]
st.subheader(f"Filtered Data for {selected_country} ({year_range[0]} - {year_range[1]})")
st.dataframe(filtered_df)

# Line chart for trend
fig, ax = plt.subplots()
ax.plot(filtered_df['Year'], filtered_df['Annual CO₂ emissions (tonnes )'], marker='o')
ax.set_xlabel('Year')
ax.set_ylabel('Annual CO₂ emissions (tonnes)')
ax.set_title(f'CO₂ Emissions Trend for {selected_country}')
ax.grid(True)
st.pyplot(fig)

# Model choice
model_choice = st.selectbox("Choose ML Model", [
    "Linear Regression", 
    "Random Forest", 
    "Decision Tree", 
    "Ridge Regression", 
    "SVR (RBF Kernel)",
    "LSTM Neural Network"
])

predict_years = st.slider("Select number of years to predict into future", 1, 10, 5)

# Prepare data
X, y = prepare_ml_data(df, selected_country)
X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=5)
last_year = X[-1][0]
X_future = np.array(range(last_year + 1, last_year + 1 + predict_years)).reshape(-1, 1)

# Model routing
if model_choice == "Linear Regression":
    st.subheader("🔹 Linear Regression Prediction")
    y_pred_test, y_pred_future, metrics = train_linear_regression(X_train, y_train, X_test, y_test, X_future)
    st.write(f"Test MSE (log-transformed): {metrics['mse']:.2f}")
    st.write(f"Test MAE (log-transformed): {metrics['mae']:.2f}")
    plot_predictions(X, y, X_test, y_pred_test, X_future, y_pred_future, selected_country, model_choice)

elif model_choice == "Random Forest":
    st.subheader("📦 Random Forest Prediction (Pre-trained)")
    y_pred_test, y_pred_future, metrics, found = train_random_forest(X_train, y_train, X_test, y_test, X_future, selected_country)
    if found:
        st.write(f"Test MSE: {metrics['mse']:.2f}")
        st.write(f"Test MAE: {metrics['mae']:.2f}")
        plot_predictions(X, y, X_test, y_pred_test, X_future, y_pred_future, selected_country, model_choice)
    else:
        st.warning(f"⚠️ Pre-trained Random Forest model for **{selected_country}** not found. Try another model.")

elif model_choice == "Decision Tree":
    st.subheader("🌳 Decision Tree Prediction")
    y_pred_test, y_pred_future, metrics = train_decision_tree(X_train, y_train, X_test, y_test, X_future)
    st.write(f"Test MSE: {metrics['mse']:.2f}")
    st.write(f"Test MAE: {metrics['mae']:.2f}")
    plot_predictions(X, y, X_test, y_pred_test, X_future, y_pred_future, selected_country, model_choice)

elif model_choice == "Ridge Regression":
    st.subheader("🧱 Ridge Regression Prediction")
    y_pred_test, y_pred_future, metrics = train_ridge_regression(X_train, y_train, X_test, y_test, X_future)
    st.write(f"Test MSE: {metrics['mse']:.2f}")
    st.write(f"Test MAE: {metrics['mae']:.2f}")
    plot_predictions(X, y, X_test, y_pred_test, X_future, y_pred_future, selected_country, model_choice)

elif model_choice == "SVR (RBF Kernel)":
    st.subheader("⚙️ SVR (RBF Kernel) Prediction")
    y_pred_test, y_pred_future, metrics = train_svr(X_train, y_train, X_test, y_test, X_future)
    st.write(f"Test MSE: {metrics['mse']:.2f}")
    st.write(f"Test MAE: {metrics['mae']:.2f}")
    plot_predictions(X, y, X_test, y_pred_test, X_future, y_pred_future, selected_country, model_choice)

elif model_choice == "LSTM Neural Network":
    st.subheader("🤖 LSTM Neural Network Prediction")
    y_pred_test, y_pred_future, metrics = train_lstm_model(y, X, predict_years)
    st.write(f"Test MSE: {metrics['mse']:.2f}")
    st.write(f"Test MAE: {metrics['mae']:.2f}")
    X_test_years = X[-5:]
    X_future_years = np.array(range(last_year + 1, last_year + 1 + predict_years)).reshape(-1, 1)
    plot_predictions(X, y, X_test_years, y_pred_test, X_future_years, y_pred_future, selected_country, model_choice)
