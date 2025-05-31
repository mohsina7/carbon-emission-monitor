import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

DATA_PATH = os.path.join("data", "co2_emissions.csv")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

def prepare_ml_data(df, country):
    country_df = df[df['Entity'] == country][['Year', 'Annual CO₂ emissions (tonnes )']].dropna()
    country_df = country_df.sort_values('Year')
    X = country_df['Year'].values.reshape(-1, 1)
    y = country_df['Annual CO₂ emissions (tonnes )'].values
    return X, y

def train_test_split(X, y, test_size=5):
    return X[:-test_size], y[:-test_size], X[-test_size:], y[-test_size:]

def create_lstm_sequences(y_values, seq_length=3):
    Xs, ys = [], []
    for i in range(len(y_values) - seq_length):
        Xs.append(y_values[i:i+seq_length])
        ys.append(y_values[i+seq_length])
    return np.array(Xs), np.array(ys)

def train_lstm(y, epochs=100, seq_length=3):
    X_seq, y_seq = create_lstm_sequences(y, seq_length)
    split_idx = len(X_seq) - 5
    X_train_seq, y_train_seq = X_seq[:split_idx], y_seq[:split_idx]
    X_test_seq, y_test_seq = X_seq[split_idx:], y_seq[split_idx:]

    X_train_seq = X_train_seq.reshape((X_train_seq.shape[0], X_train_seq.shape[1], 1))
    X_test_seq = X_test_seq.reshape((X_test_seq.shape[0], X_test_seq.shape[1], 1))

    model = Sequential([
        LSTM(50, activation='relu', input_shape=(seq_length,1)),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
    model.fit(X_train_seq, y_train_seq, epochs=epochs, verbose=0)

    y_pred_test = model.predict(X_test_seq).flatten()

    return model, y_pred_test, y_test_seq, X_test_seq

def predict_lstm_future(model, y, predict_years, seq_length=3):
    last_seq = y[-seq_length:].tolist()
    y_pred_future = []
    for _ in range(predict_years):
        input_seq = np.array(last_seq[-seq_length:]).reshape(1, seq_length, 1)
        next_pred = model.predict(input_seq)[0,0]
        y_pred_future.append(next_pred)
        last_seq.append(next_pred)
    return y_pred_future

def plot_all_predictions(X, y, X_test, predictions_dict, X_future, future_preds_dict, country):
    plt.figure(figsize=(10,6))
    plt.plot(X, y, label='Actual', marker='o')
    for model_name, y_pred_test in predictions_dict.items():
        plt.plot(X_test, y_pred_test, marker='x', linestyle='--', label=f'{model_name} (Test)')
    for model_name, y_pred_future in future_preds_dict.items():
        plt.plot(X_future, y_pred_future, marker='^', linestyle=':', label=f'{model_name} (Future)')
    plt.title(f'Model Predictions Comparison for {country}')
    plt.xlabel('Year')
    plt.ylabel('CO₂ Emissions')
    plt.grid(True)
    plt.legend()
    st.pyplot(plt.gcf())

def main():
    st.title("🔍 Carbon Emission Models Comparison")

    df = load_data()
    countries = df['Entity'].unique()
    selected_country = st.selectbox("Select Country", sorted(countries))

    min_year = int(df['Year'].min())
    max_year = int(df['Year'].max())
    predict_years = st.slider("Select years to predict into future", 1, 10, 5)

    X, y = prepare_ml_data(df, selected_country)
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=5)
    last_year = X[-1][0]
    X_future = np.array(range(last_year + 1, last_year + 1 + predict_years)).reshape(-1, 1)

    results = []
    predictions_test = {}
    predictions_future = {}

    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_pred_test = lr_model.predict(X_test)
    lr_pred_future = lr_model.predict(X_future)
    lr_mse = mean_squared_error(y_test, lr_pred_test)
    lr_mae = mean_absolute_error(y_test, lr_pred_test)
    results.append(("Linear Regression", lr_mse, lr_mae))
    predictions_test["Linear Regression"] = lr_pred_test
    predictions_future["Linear Regression"] = lr_pred_future

    # Decision Tree
    dt_model = DecisionTreeRegressor(random_state=42)
    dt_model.fit(X_train, y_train)
    dt_pred_test = dt_model.predict(X_test)
    dt_pred_future = dt_model.predict(X_future)
    dt_mse = mean_squared_error(y_test, dt_pred_test)
    dt_mae = mean_absolute_error(y_test, dt_pred_test)
    results.append(("Decision Tree", dt_mse, dt_mae))
    predictions_test["Decision Tree"] = dt_pred_test
    predictions_future["Decision Tree"] = dt_pred_future

    # Ridge Regression
    ridge_model = Ridge()
    ridge_model.fit(X_train, y_train)
    ridge_pred_test = ridge_model.predict(X_test)
    ridge_pred_future = ridge_model.predict(X_future)
    ridge_mse = mean_squared_error(y_test, ridge_pred_test)
    ridge_mae = mean_absolute_error(y_test, ridge_pred_test)
    results.append(("Ridge Regression", ridge_mse, ridge_mae))
    predictions_test["Ridge Regression"] = ridge_pred_test
    predictions_future["Ridge Regression"] = ridge_pred_future

    # SVR
    svr_model = SVR(kernel='rbf')
    svr_model.fit(X_train, y_train)
    svr_pred_test = svr_model.predict(X_test)
    svr_pred_future = svr_model.predict(X_future)
    svr_mse = mean_squared_error(y_test, svr_pred_test)
    svr_mae = mean_absolute_error(y_test, svr_pred_test)
    results.append(("SVR (RBF Kernel)", svr_mse, svr_mae))
    predictions_test["SVR (RBF Kernel)"] = svr_pred_test
    predictions_future["SVR (RBF Kernel)"] = svr_pred_future

    # Random Forest (load if exists)
    rf_model_path = f"models/rf_model_{selected_country.lower()}.pkl"
    if os.path.exists(rf_model_path):
        rf_model = joblib.load(rf_model_path)
        rf_pred_test = rf_model.predict(X_test)
        rf_pred_future = rf_model.predict(X_future)
        rf_mse = mean_squared_error(y_test, rf_pred_test)
        rf_mae = mean_absolute_error(y_test, rf_pred_test)
        results.append(("Random Forest", rf_mse, rf_mae))
        predictions_test["Random Forest"] = rf_pred_test
        predictions_future["Random Forest"] = rf_pred_future
    else:
        st.info(f"Random Forest model for {selected_country} not found. Skipping RF.")

    # LSTM
    lstm_model, lstm_pred_test, y_test_seq, _ = train_lstm(y)
    lstm_pred_future = predict_lstm_future(lstm_model, y, predict_years)
    lstm_mse = mean_squared_error(y_test_seq, lstm_pred_test)
    lstm_mae = mean_absolute_error(y_test_seq, lstm_pred_test)
    results.append(("LSTM Neural Network", lstm_mse, lstm_mae))
    # For plotting, create proper year arrays for test and future predictions:
    X_test_years = X[-5:]
    X_future_years = X_future
    predictions_test["LSTM Neural Network"] = lstm_pred_test
    predictions_future["LSTM Neural Network"] = lstm_pred_future

    # Display comparison table
    results_df = pd.DataFrame(results, columns=["Model", "Test MSE", "Test MAE"])
    st.subheader(f"Model Performance for {selected_country}")
    st.dataframe(results_df.style.format({"Test MSE": "{:.2f}", "Test MAE": "{:.2f}"}))

    if st.checkbox("Show combined predictions plot"):
        plot_all_predictions(X, y, X_test_years, predictions_test, X_future_years, predictions_future, selected_country)

if __name__ == "__main__":
    main()
