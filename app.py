import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error

import joblib
#st.write("Joblib import successful")
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

st.set_page_config(page_title="Carbon Emission Monitor", layout="wide")
st.title("🌍 AI Carbon Emission Monitor")
st.markdown("Monitor and predict global carbon emissions with machine learning.")

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
def plot_predictions(X, y, X_test, y_pred_test, X_future, y_pred_future, country, model_name):
    fig, ax = plt.subplots()
    ax.plot(X, y, label="Actual", marker='o')
    ax.plot(X_test, y_pred_test, label="Predicted (Test)", marker='x')
    ax.plot(X_future, y_pred_future, label="Predicted (Future)", marker='^')
    ax.set_title(f"{model_name} Prediction for {country}")
    ax.set_xlabel("Year")
    ax.set_ylabel("CO₂ Emissions")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

df = load_data()
st.write("Columns in dataset:", df.columns.tolist())

st.subheader("Raw CO₂ Emission Data")
st.dataframe(df.head(10))

countries = df['Entity'].unique()
selected_country = st.selectbox("Select Country", sorted(countries))

min_year = int(df['Year'].min())
max_year = int(df['Year'].max())
year_range = st.slider("Select Year Range", min_year, max_year, (min_year, max_year))

filtered_df = df[(df['Entity'] == selected_country) & 
                 (df['Year'] >= year_range[0]) & 
                 (df['Year'] <= year_range[1])]

st.subheader(f"Filtered Data for {selected_country} ({year_range[0]} - {year_range[1]})")
st.dataframe(filtered_df)

fig, ax = plt.subplots()
ax.plot(filtered_df['Year'], filtered_df['Annual CO₂ emissions (tonnes )'], marker='o')
ax.set_xlabel('Year')
ax.set_ylabel('Annual CO₂ emissions (tonnes)')
ax.set_title(f'CO₂ Emissions Trend for {selected_country}')
ax.grid(True)
st.pyplot(fig)

model_choice = st.selectbox("Choose ML Model", [
    "Linear Regression", 
    "Random Forest", 
    "Decision Tree", 
    "Ridge Regression", 
    "SVR (RBF Kernel)",
    "LSTM Neural Network"
])


predict_years = st.slider("Select number of years to predict into future", 1, 10, 5)

X, y = prepare_ml_data(df, selected_country)
X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=5)
last_year = X[-1][0]
X_future = np.array(range(last_year + 1, last_year + 1 + predict_years)).reshape(-1, 1)


# Model logic
if model_choice == "Linear Regression":
    st.subheader("🔹 Linear Regression Prediction")
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    model = LinearRegression()
    model.fit(X_train, y_train_log)

    y_pred_test_log = model.predict(X_test)
    y_pred_future_log = model.predict(X_future)

    y_pred_test = np.expm1(y_pred_test_log)
    y_pred_future = np.expm1(y_pred_future_log)
    y_test_original = np.expm1(y_test_log)

    mse = mean_squared_error(y_test_original, y_pred_test)
    mae = mean_absolute_error(y_test_original, y_pred_test)

    st.write(f"Test MSE (log-transformed): {mse:.2f}")
    st.write(f"Test MAE (log-transformed): {mae:.2f}")
    plot_predictions(X, y, X_test, y_pred_test, X_future, y_pred_future, selected_country, model_choice)

elif model_choice == "Random Forest":
    st.subheader("📦 Random Forest Prediction (Pre-trained)")
    rf_model_path = f"models/rf_model_{selected_country.lower()}.pkl"

    if os.path.exists(rf_model_path):
        model = joblib.load(rf_model_path)
        y_pred = model.predict(X_test)
        y_future = model.predict(X_future)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        st.write(f"Test MSE: {mse:.2f}")
        st.write(f"Test MAE: {mae:.2f}")
        plot_predictions(X, y, X_test, y_pred, X_future, y_future, selected_country, model_choice)
    else:
        st.warning(f"⚠️ Random Forest model for **{selected_country}** not found. Try another model.")

elif model_choice == "Decision Tree":
    st.subheader("🌳 Decision Tree Prediction")
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_future = model.predict(X_future)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    st.write(f"Test MSE: {mse:.2f}")
    st.write(f"Test MAE: {mae:.2f}")
    plot_predictions(X, y, X_test, y_pred, X_future, y_future, selected_country, model_choice)

elif model_choice == "Ridge Regression":
    st.subheader("🧱 Ridge Regression Prediction")
    model = Ridge()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_future = model.predict(X_future)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    st.write(f"Test MSE: {mse:.2f}")
    st.write(f"Test MAE: {mae:.2f}")
    plot_predictions(X, y, X_test, y_pred, X_future, y_future, selected_country, model_choice)

elif model_choice == "SVR (RBF Kernel)":
    st.subheader("⚙️ SVR (RBF Kernel) Prediction")
    model = SVR(kernel='rbf')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_future = model.predict(X_future)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    st.write(f"Test MSE: {mse:.2f}")
    st.write(f"Test MAE: {mae:.2f}")
    plot_predictions(X, y, X_test, y_pred, X_future, y_future, selected_country, model_choice)

elif model_choice == "LSTM Neural Network":
    st.subheader("🤖 LSTM Neural Network Prediction")

    # Prepare sequences
    seq_length = 3
    # Use entire available data (X and y)
    X_seq, y_seq = create_lstm_sequences(y, seq_length=seq_length)
    
    # Split sequences into train and test
    split_idx = len(X_seq) - 5  # last 5 sequences for test
    X_train_seq, y_train_seq = X_seq[:split_idx], y_seq[:split_idx]
    X_test_seq, y_test_seq = X_seq[split_idx:], y_seq[split_idx:]
    
    # Reshape input to [samples, time_steps, features]
    X_train_seq = X_train_seq.reshape((X_train_seq.shape[0], X_train_seq.shape[1], 1))
    X_test_seq = X_test_seq.reshape((X_test_seq.shape[0], X_test_seq.shape[1], 1))
    
    # Build LSTM model
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(seq_length, 1)),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
    
    # Train model
    model.fit(X_train_seq, y_train_seq, epochs=100, verbose=0)
    
    # Predict test and future
    y_pred_test = model.predict(X_test_seq).flatten()
    
    # For future prediction, iteratively predict next year based on last seq_length years
    last_seq = y[-seq_length:].tolist()
    y_pred_future = []
    for _ in range(predict_years):
        input_seq = np.array(last_seq[-seq_length:]).reshape(1, seq_length, 1)
        next_pred = model.predict(input_seq)[0,0]
        y_pred_future.append(next_pred)
        last_seq.append(next_pred)
    
    # Prepare X_test years and X_future years
    X_test_years = X[-5:]
    X_future_years = np.array(range(last_year + 1, last_year + 1 + predict_years)).reshape(-1, 1)
    
    mse = mean_squared_error(y_test_seq, y_pred_test)
    mae = mean_absolute_error(y_test_seq, y_pred_test)
    st.write(f"Test MSE: {mse:.2f}")
    st.write(f"Test MAE: {mae:.2f}")
    
    # Plot results
    plot_predictions(X, y, X_test_years, y_pred_test, X_future_years, y_pred_future, selected_country, model_choice)
