import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import numpy as np
import os

# Define base URL for file paths
base_url = "/Users/saleh/Desktop/Projects/AIProject/TaxiOptimization/"

# Load and validate data
data_path = base_url + "data/processed/cleaned_taxi_data.csv"
try:
    df = pd.read_csv(data_path)
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    print(f"Loaded {len(df)} rows from cleaned_taxi_data.csv")
except FileNotFoundError:
    print(f"Error: {data_path} not found")
    raise

# Validate input data
if df["pickup_datetime"].isna().any():
    print("Warning: Missing pickup_datetime values detected")
    df = df.dropna(subset=["pickup_datetime"])

# Aggregate city-wide demand
city_demand = df.groupby(df["pickup_datetime"].dt.floor("h"))["VendorID"].count().reset_index()
city_demand.columns = ["ds", "y"]
print(f"Aggregated to {len(city_demand)} hourly pickup records")

# Check for sufficient data
if len(city_demand) < 2:
    raise ValueError("Insufficient data: Fewer than 2 hourly records")

# Train-test split
train_size = int(0.8 * len(city_demand))
train = city_demand.iloc[:train_size].copy()
test = city_demand.iloc[train_size:].copy()
print(f"Train set: {len(train)} rows, Test set: {len(test)} rows")

# Add weather regressors
weather_cols = ["temperature", "precipitation", "wind_speed"]
weather_data = df[["pickup_datetime", *weather_cols]].drop_duplicates()
weather_data["ds"] = pd.to_datetime(weather_data["pickup_datetime"]).dt.floor("h")
weather_data = weather_data.groupby("ds")[weather_cols].mean().reset_index()
print(f"Weather data: {len(weather_data)} rows")

# Merge weather data
train = train.merge(weather_data, on="ds", how="left")
train[weather_cols] = train[weather_cols].fillna(method="ffill").fillna(method="bfill")
test = test.merge(weather_data, on="ds", how="left")
test[weather_cols] = test[weather_cols].fillna(method="ffill").fillna(method="bfill")
print(f"Train set after merge: {len(train)} rows, NaNs in y: {train['y'].isna().sum()}")

# Validate train data
if train["y"].isna().any() or len(train) < 2:
    raise ValueError("Train data has NaN values in 'y' or fewer than 2 rows")

# Tune Prophet model
param_grid = {
    "changepoint_prior_scale": [0.001, 0.01, 0.05, 0.1],
    "seasonality_mode": ["additive", "multiplicative"],
    "seasonality_prior_scale": [0.1, 1.0, 10.0]
}
best_mae = float("inf")
best_params = {}
for cps in param_grid["changepoint_prior_scale"]:
    for sm in param_grid["seasonality_mode"]:
        for sps in param_grid["seasonality_prior_scale"]:
            print(f"Trying changepoint_prior_scale={cps}, seasonality_mode={sm}, seasonality_prior_scale={sps}")
            model = Prophet(
                daily_seasonality=True,
                changepoint_prior_scale=cps,
                seasonality_mode=sm,
                seasonality_prior_scale=sps
            )
            for col in weather_cols:
                model.add_regressor(col)
            model.fit(train)
            cv_results = cross_validation(
                model,
                initial="7 days",
                period="3 days",
                horizon="1 days",
                parallel="processes"
            )
            metrics = performance_metrics(cv_results, rolling_window=1)
            mae = metrics["mae"].mean()
            print(f"MAE: {mae:.2f}")
            if mae < best_mae:
                best_mae = mae
                best_params = {
                    "changepoint_prior_scale": cps,
                    "seasonality_mode": sm,
                    "seasonality_prior_scale": sps
                }
print(f"Best parameters: {best_params}, MAE: {best_mae:.2f}")

# Train final city-wide model
model = Prophet(daily_seasonality=True, **best_params)
for col in weather_cols:
    model.add_regressor(col)
model.fit(train)

# City-wide forecast and evaluation
future = test[["ds", *weather_cols]].copy()
forecast = model.predict(future)
forecast = forecast[["ds", "yhat"]].merge(test[["ds", "y"]], on="ds", how="left")
mae = mean_absolute_error(forecast["y"], forecast["yhat"])
rmse = np.sqrt(mean_squared_error(forecast["y"], forecast["yhat"]))
mape = np.mean(np.abs((forecast["y"] - forecast["yhat"]) / forecast["y"])) * 100
print(f"City-Wide Forecast Metrics: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")

# Save forecasts
forecast.to_csv(base_url + "data/processed/prophet_forecasts.csv", index=False)
print(f"Saved city-wide forecasts to {base_url}data/processed/prophet_forecasts.csv")

# Visualize
plt.figure(figsize=(10, 6))
plt.plot(forecast["ds"], forecast["y"], label="Actual")
plt.plot(forecast["ds"], forecast["yhat"], label="Predicted")
plt.title("City-Wide Demand Forecast")
plt.legend()
plt.savefig(base_url + "visualizations/demand_forecast.png")
plt.close()

# Save city-wide model
with open(base_url + "models/prophet_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Zone-specific models (top 5 zones)
top_zones = df["PULocationID"].value_counts().head(5).index
for zone in top_zones:
    zone_demand = df[df["PULocationID"] == zone].groupby(df["pickup_datetime"].dt.floor("h"))["VendorID"].count().reset_index()
    zone_demand.columns = ["ds", "y"]
    if len(zone_demand) < 2:
        print(f"Skipping zone {zone}: Insufficient data ({len(zone_demand)} rows)")
        continue
    train_zone = zone_demand.iloc[:int(0.8 * len(zone_demand))].copy()
    if len(train_zone) < 2:
        print(f"Skipping zone {zone}: Insufficient training data ({len(train_zone)} rows)")
        continue
    model_zone = Prophet(daily_seasonality=True, **best_params)
    model_zone.fit(train_zone)
    with open(base_url + f"models/prophet_zone_{zone}.pkl", "wb") as f:
        pickle.dump(model_zone, f)
    print(f"Trained and saved Prophet model for zone {zone}")
print("Completed zone-specific modeling")

