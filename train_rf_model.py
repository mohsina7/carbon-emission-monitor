import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
df = pd.read_csv("data/co2_emissions.csv")

# Ensure output folder exists
os.makedirs("models", exist_ok=True)

# Unique countries
countries = df["Entity"].unique()

for country in countries:
    country_df = df[df["Entity"] == country][["Year", "Annual CO₂ emissions (tonnes )"]].dropna()
    
    if len(country_df) < 10:
        print(f"Skipping {country} (not enough data)")
        continue

    country_df = country_df.sort_values("Year")
    X = country_df["Year"].values.reshape(-1, 1)
    y = country_df["Annual CO₂ emissions (tonnes )"].values

    # Split to avoid overfitting
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Save model
    filename = f"models/rf_model_{country.lower().replace(' ', '_')}.pkl"
    joblib.dump(rf, filename)
    print(f"Saved: {filename}")
