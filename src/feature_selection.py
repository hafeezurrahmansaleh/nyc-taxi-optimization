import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
import os

# Define base URL for file paths
base_url = "/Users/saleh/Desktop/Projects/AIProject/TaxiOptimization/"

# Load processed data from preprocessing
data_path = base_url + "data/processed/cleaned_taxi_data.csv"
try:
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows from cleaned_taxi_data.csv")
except FileNotFoundError:
    print(f"Error: {data_path} not found")
    raise

# Prepare data for classification (same as classification_clustering.py)
pickup_counts = df.groupby(["PULocationID", "hour"])["VendorID"].count().reset_index(name="pickup_count")
threshold = pickup_counts["pickup_count"].quantile(0.8)
pickup_counts["is_high_demand"] = (pickup_counts["pickup_count"] > threshold).astype(int)
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
print("Columns in data:", data.columns.tolist())

# Prepare features and target
X = data[["temperature", "precipitation", "wind_speed", "is_holiday", "avg_speed", "day_of_week", "hour", "month", "temp_precip"]]
y = data["is_high_demand"]

# Recursive Feature Elimination (RFE) with Random Forest
# Note: RFE ranks features by importance, helping select the best for classification
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(estimator=rf, n_features_to_select=5)  # Select top 5 features
rfe.fit(X, y)
# Get feature rankings (1 = selected, higher = eliminated earlier)
ranking = pd.DataFrame({
    "feature": X.columns,
    "ranking": rfe.ranking_
}).sort_values("ranking")
print("\nFeature Rankings (1 = selected, higher = less important):")
print(ranking)

# Save rankings for classification and dashboard
ranking.to_csv(base_url + "data/processed/selected_features.csv", index=False)

# Visualize feature rankings
plt.figure(figsize=(8, 6))
plt.bar(ranking["feature"], ranking["ranking"])
plt.title("RFE Feature Ranking (Lower = More Important)")
plt.xticks(rotation=45)
plt.xlabel("Feature")
plt.ylabel("Ranking (1 = Selected)")
plt.tight_layout()
plt.savefig(base_url + "visualizations/rfe_feature_ranking.png")
plt.close()

# Note for presentation: RFE improves classification by selecting the most impactful features
# (e.g., hour, temperature), reducing noise and enhancing model performance
print("Saved feature rankings to data/processed/selected_features.csv and plot to visualizations/rfe_feature_ranking.png")