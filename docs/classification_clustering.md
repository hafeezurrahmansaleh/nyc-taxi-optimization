
---

### Step 1: Import Libraries
**What’s Happening?**
We’re bringing in tools (libraries) that help us do different tasks, like working with data, building models, and making pictures (visualizations).

**Code Explanation:**
```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import pickle
import folium
import os
```
- **pandas (pd)**: Think of this as a tool to organize data in tables, like a spreadsheet. We use it to load and work with our taxi data.
- **numpy (np)**: Helps with math calculations, like averages or creating arrays (lists of numbers).
- **RandomForestClassifier**: A machine learning model that makes decisions by combining many small decision trees (like a forest of trees deciding together).
- **train_test_split, GridSearchCV**: Tools to split data into training and testing parts and find the best settings for our models.
- **accuracy_score, precision_score, recall_score, f1_score, roc_auc_score**: Ways to measure how good our models are at predicting high-demand zones.
- **XGBClassifier**: Another model (XGBoost) that’s good at making predictions by learning from data step by step.
- **KMeans, DBSCAN**: Methods to group similar zones together (clustering).
- **MinMaxScaler**: A tool to adjust numbers so they’re on the same scale (e.g., between 0 and 1), which helps clustering work better.
- **silhouette_score**: A score to check how good our clustering is (higher is better).
- **matplotlib.pyplot (plt)**: Helps make graphs and charts, like bar charts or line plots.
- **pickle**: Saves our trained models so we can use them later in the dashboard.
- **folium**: Creates maps to show where zones are in NYC.
- **os**: Helps with file paths, like finding where to save files.

**Why?**
We need these tools to load data, build models, measure their performance, group zones, and create maps and charts to understand our results.

**How?**
We just tell Python to bring these tools in, and we give them short names (like `pd` for `pandas`) to make coding easier.

---

### Step 2: Load Data
**What’s Happening?**
We’re loading the taxi data that was already cleaned in another file (`preprocessing.ipynb`) and saved as `cleaned_taxi_data.csv`.

**Code Explanation:**
```python
data_path = base_url + "data/processed/cleaned_taxi_data.csv"
try:
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows from cleaned_taxi_data.csv")
except FileNotFoundError:
    print(f"Error: {data_path} not found")
    raise
```
- **base_url**: A path to where all our project files are stored (e.g., `/Users/saleh/Desktop/Projects/AIProject/TaxiOptimization/`).
- **data_path**: The full path to our cleaned data file (`cleaned_taxi_data.csv`).
- **pd.read_csv(data_path)**: Loads the data into a table (called a DataFrame) named `df`. The table has rows (like 17,290,240 rows here) and columns (like pickup time, zone ID, etc.).
- **print(f"Loaded {len(df)} rows...")**: Tells us how many rows we loaded (e.g., 17,290,240 rows).
- **try/except**: A safety check. If the file isn’t found, it shows an error message and stops the code.

**Why?**
We need the taxi data to analyze it and build our models. This data has information like where passengers were picked up (`PULocationID`), the time (`hour`), weather (`temperature`, `precipitation`), and more.

**How?**
We use `pandas` to read the CSV file into a table we can work with in Python.

---

### Step 3: Prepare Classification Data
**What’s Happening?**
We’re getting the data ready to predict which zones have a lot of passengers (high-demand zones). We don’t want to cheat by using the number of passengers directly in our predictions, so we use other features like weather and time.

**Code Explanation:**
```python
pickup_counts = df.groupby(["PULocationID", "hour"])["VendorID"].count().reset_index(name="pickup_count")
threshold = pickup_counts["pickup_count"].quantile(0.8)
pickup_counts["is_high_demand"] = (pickup_counts["pickup_count"] > threshold).astype(int)
```
- **df.groupby(["PULocationID", "hour"])**: Groups the data by each zone (`PULocationID`) and hour of the day (e.g., Zone 132 at 8 AM).
- **["VendorID"].count()**: Counts how many trips happened in each group. `VendorID` is the driver ID, so this counts the number of trips (passenger pickups).
- **reset_index(name="pickup_count")**: Turns the counts into a new table with columns `PULocationID`, `hour`, and `pickup_count`.
- **pickup_counts["pickup_count"].quantile(0.8)**: Finds the 80th percentile of pickup counts. This means 80% of the counts are below this number. If a zone-hour has more pickups than this number, we call it “high-demand.”
- **pickup_counts["is_high_demand"]**: Creates a new column. If the pickup count is above the threshold, it’s 1 (high-demand); otherwise, it’s 0 (not high-demand).
- **astype(int)**: Makes sure the values are 0 or 1 (integers).

**Next Part:**
```python
features = df.groupby(["PULocationID", "hour"]).agg({
    "temperature": "mean",
    "precipitation": "mean",
    "wind_speed": "mean",
    "is_holiday": "mean",
    "avg_speed": "mean",
    "day_of_week": "mean",
    "month": "first"
}).reset_index()
features["temp_precip"] = features["temperature"] * features["precipitation"]
data = pickup_counts.merge(features, on=["PULocationID", "hour"])
print("Columns in data after merge:", data.columns.tolist())
```
- **df.groupby(["PULocationID", "hour"])**: Groups the data again by zone and hour.
- **.agg({...})**: Calculates averages (or the first value) for different columns:
  - `temperature`, `precipitation`, `wind_speed`, `is_holiday`, `avg_speed`, `day_of_week`: Takes the average for each zone-hour.
  - `month`: Takes the first month value (since all trips in the same hour should have the same month).
- **reset_index()**: Turns the results into a new table called `features`.
- **features["temp_precip"]**: Creates a new column by multiplying `temperature` and `precipitation`. This shows how temperature and rain together might affect demand (e.g., hot and rainy days might mean more taxi rides).
- **pickup_counts.merge(features, on=["PULocationID", "hour"])**: Combines the `pickup_counts` table (with `pickup_count` and `is_high_demand`) with the `features` table, matching rows by `PULocationID` and `hour`.
- **print("Columns in data after merge:", ...)**: Shows the columns in the combined table (`data`), like `PULocationID`, `hour`, `pickup_count`, `is_high_demand`, `temperature`, etc.

**Next Part:**
```python
try:
    selected_features = pd.read_csv(base_url + "data/processed/selected_features.csv")
    top_features = selected_features[selected_features["ranking"] == 1]["feature"].tolist()
    print(f"Using RFE-selected features: {top_features}")
except FileNotFoundError:
    print("Warning: selected_features.csv not found, using all features")
    top_features = ["temperature", "precipitation", "wind_speed", "is_holiday", "avg_speed", "day_of_week", "hour", "month", "temp_precip"]

X = data[top_features]
y = data["is_high_demand"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
- **pd.read_csv(...)**: Loads a file called `selected_features.csv`, which was created in `feature_selection.ipynb`. This file tells us which features are the best to use (picked by a method called Recursive Feature Elimination, or RFE).
- **top_features = ...**: Gets the best features (e.g., `precipitation`, `is_holiday`, `avg_speed`, `day_of_week`, `temp_precip`).
- **except FileNotFoundError**: If the file isn’t found, we use all features as a backup.
- **X = data[top_features]**: Creates a table (`X`) with only the selected features (e.g., `precipitation`, `is_holiday`, etc.). These are the inputs for our model.
- **y = data["is_high_demand"]**: Creates a list (`y`) of 0s and 1s (high-demand or not), which is what we want to predict.
- **train_test_split(X, y, test_size=0.2, random_state=42)**: Splits the data into training and testing parts:
  - 80% for training (`X_train`, `y_train`): The model learns from this.
  - 20% for testing (`X_test`, `y_test`): We use this to check how good the model is.
  - `random_state=42`: Makes the split the same every time we run the code (for consistency).

**Why?**
- We want to predict high-demand zones without cheating. If we used `pickup_count` directly, the model would just look at the answer, which isn’t fair (this is called “data leakage”).
- We use features like weather (`precipitation`, `temperature`), holidays (`is_holiday`), and traffic speed (`avg_speed`) to guess if a zone will be busy.
- Splitting the data lets us train the model on one part and test it on another to see how well it works on new data.

**How?**
- We group the data to count pickups and calculate features for each zone and hour.
- We decide what “high-demand” means (top 20% of pickups) and label zones as 1 or 0.
- We combine the counts with features and use only the best features (from RFE) to train our model.

---

### Step 4: Train Random Forest
**What’s Happening?**
We’re building a Random Forest model to predict if a zone-hour is high-demand (1) or not (0). We also find the best settings for the model and check how good it is.

**Code Explanation:**
```python
rf = RandomForestClassifier(random_state=42)
rf_params = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}
rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring="f1")
rf_grid.fit(X_train, y_train)
rf_best = rf_grid.best_estimator_
rf_pred = rf_best.predict(X_test)
rf_metrics = {
    "accuracy": accuracy_score(y_test, rf_pred),
    "precision": precision_score(y_test, rf_pred),
    "recall": recall_score(y_test, rf_pred),
    "f1": f1_score(y_test, rf_pred),
    "roc_auc": roc_auc_score(y_test, rf_best.predict_proba(X_test)[:, 1])
}
print("Random Forest Metrics:", rf_metrics)
```
- **RandomForestClassifier(random_state=42)**: Creates a Random Forest model. `random_state=42` ensures we get the same results every time.
- **rf_params**: A list of settings to try for the model:
  - `n_estimators`: Number of trees in the forest (100, 200, or 300).
  - `max_depth`: How deep each tree can go (10 levels, 20 levels, or no limit).
  - `min_samples_split`: Minimum number of data points needed to split a branch (2 or 5).
  - `min_samples_leaf`: Minimum number of data points at the end of a branch (1 or 2).
- **GridSearchCV(rf, rf_params, cv=5, scoring="f1")**: Tests all combinations of settings to find the best one:
  - `cv=5`: Splits the training data into 5 parts, trains on 4, tests on 1, and repeats 5 times to get a good average.
  - `scoring="f1"`: Picks the best settings based on the F1 score (a balance between precision and recall).
- **rf_grid.fit(X_train, y_train)**: Trains the model on the training data with all the settings.
- **rf_best = rf_grid.best_estimator_**: Gets the best model (with the best settings).
- **rf_pred = rf_best.predict(X_test)**: Uses the best model to predict high-demand zones on the test data (0 or 1).
- **rf_metrics**: Measures how good the predictions are:
  - **accuracy_score**: Percentage of correct predictions (e.g., 91.9%).
  - **precision_score**: Of the zones we predicted as high-demand, how many were actually high-demand? (e.g., 81.5%).
  - **recall_score**: Of all the high-demand zones, how many did we find? (e.g., 77.7%).
  - **f1_score**: A balance between precision and recall (e.g., 79.5%).
  - **roc_auc_score**: How well the model separates high-demand from not high-demand (e.g., 97%, which is very good).
- **print("Random Forest Metrics:", rf_metrics)**: Shows the results.

**Why?**
- Random Forest is good at making predictions by combining many decision trees, which reduces mistakes.
- We test different settings to make the model as good as possible.
- We measure performance to see if the model is accurate and reliable for predicting high-demand zones.

**How?**
- We create a model, try different settings, train it on the training data, and test it on the test data.
- We use metrics to check how well it predicts high-demand zones.

---

### Step 4.5: Train Random Forest Regressor for Earnings Prediction
**What’s Happening?**
We’re building another Random Forest model, but this time to predict how much money a driver can earn in each zone-hour (not just 0 or 1, but a number like $50/hour).

**Code Explanation:**
```python
from geopy.distance import geodesic
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

earnings_data = pickup_counts.merge(features, on=["PULocationID", "hour"])
earnings_data["temp_precip"] = earnings_data["temperature"] * earnings_data["precipitation"]
```
- **geopy.distance.geodesic**: A tool to calculate the distance between two locations (in miles).
- **RandomForestRegressor**: A model for predicting numbers (like earnings), not just 0 or 1.
- **mean_squared_error, r2_score**: Ways to measure how good our earnings predictions are.
- **pickup_counts.merge(features, ...)**: Reuses the same data we prepared for classification.
- **earnings_data["temp_precip"]**: Recalculates the `temp_precip` feature (temperature × precipitation).

**Next Part: Compute Additional Features**
```python
supply_data = df.groupby(["PULocationID", "hour"])["VendorID"].nunique().reset_index(name="driver_count")
earnings_data = earnings_data.merge(supply_data, on=["PULocationID", "hour"], how="left")
earnings_data["driver_count"] = earnings_data["driver_count"].fillna(1)
earnings_data["current_demand_supply_ratio"] = earnings_data["pickup_count"] / earnings_data["driver_count"]
```
- **df.groupby(["PULocationID", "hour"])["VendorID"].nunique()**: Counts how many unique drivers (`VendorID`) were in each zone-hour. This tells us the supply of drivers.
- **reset_index(name="driver_count")**: Creates a table with `driver_count`.
- **earnings_data.merge(supply_data, ...)**: Adds the `driver_count` to our data.
- **earnings_data["driver_count"].fillna(1)**: If there are no drivers, we set it to 1 to avoid dividing by zero.
- **earnings_data["current_demand_supply_ratio"]**: Calculates demand ÷ supply (i.e., `pickup_count` ÷ `driver_count`). A high ratio means more passengers per driver (good for earnings).

**Next Part: Add More Features**
```python
zone_features_subset = df.groupby("PULocationID").agg({
    "total_amount": "mean",
    "passenger_count": "mean",
    "trip_duration": "mean"
}).reset_index()
zone_features_subset["avg_fare"] = zone_features_subset["total_amount"].clip(upper=30)
earnings_data = earnings_data.merge(zone_features_subset[["PULocationID", "avg_fare", "passenger_count", "trip_duration"]], on="PULocationID", how="left")
```
- **df.groupby("PULocationID").agg({...})**: For each zone:
  - `total_amount`: Average fare per trip.
  - `passenger_count`: Average number of passengers per trip.
  - `trip_duration`: Average time per trip (in minutes).
- **zone_features_subset["avg_fare"]**: Sets the average fare, but caps it at $30 to remove outliers (very high fares).
- **earnings_data.merge(...)**: Adds these features (`avg_fare`, `passenger_count`, `trip_duration`) to our data.

**Next Part: Calculate Distance**
```python
zone_coords = df.groupby("PULocationID")[["pickup_lat", "pickup_lon"]].mean().reset_index()
earnings_data = earnings_data.merge(zone_coords, on="PULocationID", how="left")
current_location = zone_coords[["pickup_lat", "pickup_lon"]].mean()
earnings_data["distance"] = earnings_data.apply(
    lambda row: geodesic((current_location["pickup_lat"], current_location["pickup_lon"]), (row["pickup_lat"], row["pickup_lon"])).miles
    if pd.notna(row["pickup_lat"]) and pd.notna(row["pickup_lon"]) else float("inf"),
    axis=1
)
```
- **df.groupby("PULocationID")[["pickup_lat", "pickup_lon"]].mean()**: Finds the average latitude and longitude for each zone (like the center of the zone).
- **current_location**: Pretends the driver is at the average location of all zones (a simplification for now).
- **earnings_data["distance"]**: Calculates the distance (in miles) from the driver’s location to each zone using `geodesic`. If coordinates are missing, sets distance to infinity (`float("inf")`).

**Next Part: Compute Earnings**
```python
earnings_data["avg_speed"] = earnings_data["avg_speed"].fillna(earnings_data["avg_speed"].mean())
earnings_data["travel_time"] = (earnings_data["distance"] / earnings_data["avg_speed"] * 60).fillna(float("inf"))
earnings_data["trips_per_hour"] = (60 / earnings_data["trip_duration"]).clip(upper=3)
earnings_data["gross_earnings"] = earnings_data["trips_per_hour"] * earnings_data["avg_fare"]
earnings_data["gross_earnings"] = earnings_data["gross_earnings"] * (1 + 0.1 * earnings_data["passenger_count"])
earnings_data["fuel_cost"] = earnings_data["distance"] * 0.5
idle_time = 15
earnings_data["net_earnings"] = (
    earnings_data["gross_earnings"] * (60 - earnings_data["travel_time"] - idle_time) / 60 - earnings_data["fuel_cost"]
)
earnings_data["net_earnings"] = earnings_data["net_earnings"].clip(lower=0)
earnings_data["preference_score"] = 0  # Placeholder
earnings_data["congestion_factor"] = earnings_data["avg_speed"].std() / earnings_data["avg_speed"]
earnings_data["adjusted_earnings"] = (
    earnings_data["net_earnings"] * (1 + 0.1 * earnings_data["preference_score"]) * (1 - 0.5 * earnings_data["congestion_factor"])
)
```
- **earnings_data["avg_speed"].fillna(...)**: If `avg_speed` is missing, fills it with the average speed.
- **earnings_data["travel_time"]**: Calculates how long it takes to drive to the zone (distance ÷ speed × 60 minutes).
- **earnings_data["trips_per_hour"]**: Estimates how many trips a driver can make in an hour (60 minutes ÷ trip duration), capped at 3 trips.
- **earnings_data["gross_earnings"]**: Money per hour = trips per hour × average fare. Adds a 10% bonus for each passenger.
- **earnings_data["fuel_cost"]**: Cost of driving = distance × $0.5 per mile.
- **idle_time = 15**: Assumes the driver spends 15 minutes waiting between trips.
- **earnings_data["net_earnings"]**: Final earnings = gross earnings × (time available after driving and waiting) - fuel cost. Caps at 0 (no negative earnings).
- **earnings_data["preference_score"]**: Set to 0 (we don’t have real driver preferences in training data).
- **earnings_data["congestion_factor"]**: Measures traffic (standard deviation of speed ÷ speed). Higher means more congestion.
- **earnings_data["adjusted_earnings"]**: Adjusts earnings based on preference and congestion (simplified for training).

**Next Part: Train the Model**
```python
rf_earnings_features = [
    "pickup_count", "driver_count", "temperature", "precipitation", "wind_speed",
    "is_holiday", "avg_speed", "day_of_week", "temp_precip", "travel_time",
    "passenger_count", "avg_fare", "current_demand_supply_ratio"
]
X_earnings = earnings_data[rf_earnings_features]
y_earnings = earnings_data["adjusted_earnings"]
X_earnings_train, X_earnings_test, y_earnings_train, y_earnings_test = train_test_split(X_earnings, y_earnings, test_size=0.2, random_state=42)

rf_earnings = RandomForestRegressor(random_state=42)
rf_earnings_params = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}
rf_earnings_grid = GridSearchCV(rf_earnings, rf_earnings_params, cv=5, scoring="neg_mean_squared_error")
rf_earnings_grid.fit(X_earnings_train, y_earnings_train)
rf_earnings_best = rf_earnings_grid.best_estimator_

rf_earnings_pred = rf_earnings_best.predict(X_earnings_test)
rf_earnings_metrics = {
    "mse": mean_squared_error(y_earnings_test, rf_earnings_pred),
    "r2": r2_score(y_earnings_test, rf_earnings_pred)
}
print("Random Forest Earnings Prediction Metrics:", rf_earnings_metrics)

with open(base_url + "models/rf_earnings_model.pkl", "wb") as f:
    pickle.dump(rf_earnings_best, f)
print("Saved Random Forest earnings model as rf_earnings_model.pkl")
```
- **rf_earnings_features**: The features we use to predict earnings (e.g., number of pickups, drivers, weather, travel time).
- **X_earnings, y_earnings**: Inputs (`X_earnings`) and target (`y_earnings` = adjusted earnings).
- **train_test_split**: Splits data into 80% training and 20% testing.
- **RandomForestRegressor**: Builds a model to predict numbers (earnings).
- **GridSearchCV**: Finds the best settings for the model, using Mean Squared Error (MSE) to measure error.
- **rf_earnings_best.predict(X_earnings_test)**: Predicts earnings on the test data.
- **rf_earnings_metrics**:
  - `mse`: Mean Squared Error (average of squared differences between actual and predicted earnings, e.g., 8.99).
  - `r2`: R² score (how well predictions match actual earnings, e.g., 96%, which is very good).
- **pickle.dump(...)**: Saves the model to `rf_earnings_model.pkl` for use in the dashboard.

**Why?**
- We want to predict earnings so drivers know which zones will make them the most money.
- We use features like demand, supply, and travel time to make realistic predictions.
- Saving the model lets the dashboard use it later.

**How?**
- We calculate earnings based on trips, fares, distance, and congestion, then train a model to predict these earnings using features like weather and demand.

---

### Remaining Steps (5–17)
Since the remaining steps are not directly related to the issue you mentioned earlier (about `prophet_forecasts.csv`), I’ll summarize them briefly to keep this explanation focused. However, I can dive deeper into any step if needed!

#### **Step 5: Train XGBoost**
- Builds another model (XGBoost) to predict high-demand zones, similar to Random Forest.
- Tests different settings and measures performance (accuracy, precision, etc.).
- Saves the model as `xgb_model.pkl`.

#### **Step 6: Feature Importance for Random Forest**
- Shows which features (like `precipitation`, `avg_speed`) were most important for Random Forest’s predictions.
- Creates a bar chart (`feature_importance.png`) to visualize this.

#### **Step 7: Feature Importance for XGBoost**
- Same as Step 6, but for XGBoost.

#### **Step 8: Save Models**
- Saves the Random Forest and XGBoost models for use in the dashboard.

#### **Step 9: Prepare Clustering Data**
- Creates a new table (`zone_features`) with total pickups, fares, and coordinates for each zone.
- Adds the `hour` feature to help with clustering.

#### **Step 10: K-means Clustering**
- Uses K-means to group zones into 4 clusters based on pickups, fares, and location.
- Measures how good the clusters are using a silhouette score (0.62).

#### **Step 11: DBSCAN Clustering**
- Uses DBSCAN to group zones, trying different settings to find the best clusters.
- Measures the silhouette score (0.58).

#### **Step 12: Silhouette Score Comparison**
- Creates a bar chart comparing K-means and DBSCAN silhouette scores (`silhouette_comparison.png`).

#### **Step 13: Cluster Statistics and Labels**
- Calculates stats for each cluster (total pickups, average fare, etc.).
- Labels clusters as “High-Demand Urban,” “High-Fare,” “Residential,” or “Noise.”
- Saves stats to `kmeans_cluster_stats.csv` and `dbscan_cluster_stats.csv`.

#### **Step 14: Verify Cluster Types**
- Checks that cluster IDs are numbers (integers), which is important for mapping.

#### **Step 15: Visualize K-means Clusters**
- Creates a map (`kmeans_cluster_map.html`) showing zones in different colors based on their cluster.

#### **Step 16: Visualize DBSCAN Clusters**
- Creates another map (`dbscan_cluster_map.html`) for DBSCAN clusters, excluding noise points.

#### **Step 17: Conclusion**
- Summarizes the work: we predicted high-demand zones, grouped zones, and saved everything for the dashboard.

---

### Summary of the Notebook
This notebook does two main things:
1. **Classification**: Predicts which zones are busy using Random Forest and XGBoost, based on features like weather and time. It also predicts earnings using another Random Forest model.
2. **Clustering**: Groups zones into similar types (like busy or quiet areas) using K-means and DBSCAN, then shows these groups on maps.

**Why It’s Useful:**
- Drivers can use the predictions to go to busy zones and earn more money.
- The clusters help understand which zones are similar (e.g., all busy downtown zones).
- The maps and charts help visualize the results, and the saved models are used in the dashboard to give real-time advice to drivers.

**How It Fits in the Project:**
- The high-demand predictions and earnings estimates feed into the dashboard to recommend zones to drivers.
- The clusters help the dashboard show patterns, like which areas are always busy.
