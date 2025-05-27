import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

st.set_page_config(page_title="Carbon Emission Monitor", layout="wide")

st.title("🌍 AI Carbon Emission Monitor")
st.markdown("Monitor and predict global carbon emissions with machine learning.")

DATA_PATH = os.path.join("data", "co2_emissions.csv")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

df = load_data()

# Show columns for debugging
st.write("Columns in dataset:", df.columns.tolist())

# Show raw data (top 10 rows)
st.subheader("Raw CO₂ Emission Data")
st.dataframe(df.head(10))

# Filter by country (Entity column)
countries = df['Entity'].unique()
selected_country = st.selectbox("Select Country", sorted(countries))

# Filter by year range
min_year = int(df['Year'].min())
max_year = int(df['Year'].max())
year_range = st.slider("Select Year Range", min_year, max_year, (min_year, max_year))

# Filter data accordingly
filtered_df = df[(df['Entity'] == selected_country) & 
                 (df['Year'] >= year_range[0]) & 
                 (df['Year'] <= year_range[1])]

st.subheader(f"Filtered Data for {selected_country} ({year_range[0]} - {year_range[1]})")
st.dataframe(filtered_df)

# Plotting CO2 Emissions over years for selected country
fig, ax = plt.subplots()
ax.plot(filtered_df['Year'], filtered_df['Annual CO₂ emissions (tonnes )'], marker='o')
ax.set_xlabel('Year')
ax.set_ylabel('Annual CO₂ emissions (tonnes)')
ax.set_title(f'CO₂ Emissions Trend for {selected_country}')
ax.grid(True)
st.pyplot(fig)

# --- ML Model Section ---

# Function to prepare data for ML model
def prepare_ml_data(df, country):
    country_df = df[df['Entity'] == country][['Year', 'Annual CO₂ emissions (tonnes )']].dropna()
    country_df = country_df.sort_values('Year')
    
    # Reshape for sklearn
    X = country_df['Year'].values.reshape(-1, 1)
    y = country_df['Annual CO₂ emissions (tonnes )'].values
    return X, y

# Train/test split (last 5 years for test)
def train_test_split(X, y, test_size=5):
    return X[:-test_size], y[:-test_size], X[-test_size:], y[-test_size:]

st.subheader("CO₂ Emission Prediction")

predict_years = st.slider("Select number of years to predict into future", 1, 10, 5)

# Prepare data
X, y = prepare_ml_data(df, selected_country)

# Split train/test
X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=5)

# Log transform target to improve regression
y_train_log = np.log1p(y_train)  # log(1 + y)
y_test_log = np.log1p(y_test)

# Define future years array for prediction (years after last known year)
last_year = X[-1][0]
X_future = np.array(range(last_year + 1, last_year + 1 + predict_years)).reshape(-1, 1)

# Train Linear Regression model on log-transformed data
model = LinearRegression()
model.fit(X_train, y_train_log)

# Predict on test and future years
y_pred_test_log = model.predict(X_test)
y_pred_future_log = model.predict(X_future)

# Inverse transform predictions back to original scale
y_pred_test = np.expm1(y_pred_test_log)
y_pred_future = np.expm1(y_pred_future_log)
y_test_original = np.expm1(y_test_log)

# Calculate metrics on original scale
mse = mean_squared_error(y_test_original, y_pred_test)
mae = mean_absolute_error(y_test_original, y_pred_test)

st.write(f"Test MSE (log-transformed model): {mse:.2f}")
st.write(f"Test MAE (log-transformed model): {mae:.2f}")

# Plot actual, test predictions, and future predictions
fig, ax = plt.subplots()
ax.plot(X, y, label="Actual Emissions", marker='o')
ax.plot(X_test, y_pred_test, label="Predicted (Test)", marker='x')
ax.plot(X_future, y_pred_future, label="Predicted (Future)", marker='^')
ax.set_xlabel("Year")
ax.set_ylabel("Annual CO₂ emissions (tonnes)")
ax.set_title(f"Emission Prediction for {selected_country}")
ax.legend()
ax.grid(True)
st.pyplot(fig)



import joblib

st.subheader("📦 Random Forest Prediction (Pre-trained)")

# Check if model exists
model_path = f"models/rf_model_india.pkl"
if os.path.exists(model_path):
    rf_model = joblib.load(model_path)

    # Prepare input years
    last_year = int(X.max())
    future_years = np.array([last_year + i for i in range(1, predict_years + 1)]).reshape(-1, 1)

    # Predict on full + future
    y_pred_rf_full = rf_model.predict(X)
    y_pred_rf_future = rf_model.predict(future_years)

    y_pred_rf_full = np.expm1(y_pred_rf_full)
    y_pred_rf_future = np.expm1(y_pred_rf_future)

    # Plot
    fig_rf, ax_rf = plt.subplots()
    ax_rf.plot(X, y, label="Actual", marker="o")
    ax_rf.plot(X, y_pred_rf_full, label="RF Prediction (Train)", linestyle="--")
    ax_rf.plot(future_years, y_pred_rf_future, label="RF Prediction (Future)", marker="^")
    ax_rf.set_xlabel("Year")
    ax_rf.set_ylabel("Annual CO₂ emissions (tonnes)")
    ax_rf.set_title(f"Random Forest Emission Prediction for India")
    ax_rf.legend()
    ax_rf.grid(True)
    st.pyplot(fig_rf)
else:
    st.warning("Random Forest model file not found. Please train and save the model for the selected country.")
