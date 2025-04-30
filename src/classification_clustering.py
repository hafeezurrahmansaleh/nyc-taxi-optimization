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

# Define base URL for file paths
base_url = "/Users/saleh/Desktop/Projects/AIProject/TaxiOptimization/"

# Load processed data
data_path = base_url + "data/processed/cleaned_taxi_data.csv"
try:
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows from cleaned_taxi_data.csv")
except FileNotFoundError:
    print(f"Error: {data_path} not found")
    raise

# Classification: Prepare data for high-demand zones
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
# Add feature interaction to capture combined weather effects
features["temp_precip"] = features["temperature"] * features["precipitation"]
data = pickup_counts.merge(features, on=["PULocationID", "hour"])
print("Columns in data after merge:", data.columns.tolist())

# Prepare features and target (exclude pickup_count to avoid leakage)
X = data[["temperature", "precipitation", "wind_speed", "is_holiday", "avg_speed", "day_of_week", "hour", "month", "temp_precip"]]
y = data["is_high_demand"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest with expanded grid search for robustness
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

# Feature importance for Random Forest
feature_importance_rf = pd.DataFrame({
    "feature": X.columns,
    "importance": rf_best.feature_importances_
}).sort_values("importance", ascending=False)
print("\nRandom Forest Feature Importance:")
print(feature_importance_rf)
feature_importance_rf.to_csv(base_url + "data/processed/feature_importance_rf.csv", index=False)
plt.figure(figsize=(8, 6))
plt.bar(feature_importance_rf["feature"], feature_importance_rf["importance"])
plt.title("Random Forest Feature Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(base_url + "visualizations/feature_importance_rf.png")
plt.close()
# Note: Temperature and hour seem to drive demand, useful for driver strategies

# XGBoost with expanded grid search
xgb = XGBClassifier(random_state=42)
xgb_params = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1],
    "subsample": [0.8, 1.0]
}
xgb_grid = GridSearchCV(xgb, xgb_params, cv=5, scoring="f1")
xgb_grid.fit(X_train, y_train)
xgb_best = xgb_grid.best_estimator_
xgb_pred = xgb_best.predict(X_test)
xgb_metrics = {
    "accuracy": accuracy_score(y_test, xgb_pred),
    "precision": precision_score(y_test, xgb_pred),
    "recall": recall_score(y_test, xgb_pred),
    "f1": f1_score(y_test, xgb_pred),
    "roc_auc": roc_auc_score(y_test, xgb_best.predict_proba(X_test)[:, 1])
}
print("XGBoost Metrics:", xgb_metrics)

# Feature importance for XGBoost
feature_importance_xgb = pd.DataFrame({
    "feature": X.columns,
    "importance": xgb_best.feature_importances_
}).sort_values("importance", ascending=False)
print("\nXGBoost Feature Importance:")
print(feature_importance_xgb)
feature_importance_xgb.to_csv(base_url + "data/processed/feature_importance_xgb.csv", index=False)
plt.figure(figsize=(8, 6))
plt.bar(feature_importance_xgb["feature"], feature_importance_xgb["importance"])
plt.title("XGBoost Feature Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(base_url + "visualizations/feature_importance_xgb.png")
plt.close()

# Save models
with open(base_url + "models/rf_model.pkl", "wb") as f:
    pickle.dump(rf_best, f)
with open(base_url + "models/xgb_model.pkl", "wb") as f:
    pickle.dump(xgb_best, f)

# Clustering: Prepare zone features
zone_features = df.groupby("PULocationID").agg({
    "pickup_count": "mean",
    "trip_duration": "mean",
    "avg_speed": "mean",
    "pickup_lat": "first",
    "pickup_lon": "first",
    "total_amount": "mean",
    "hour": lambda x: x.mode()[0]  # Most common hour
}).reset_index()

# K-means clustering
X_cluster_kmeans = zone_features[["pickup_count", "trip_duration", "avg_speed"]]
inertias = []
kmeans_silhouettes = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_cluster_kmeans)
    inertias.append(kmeans.inertia_)
    if k > 1:
        labels = kmeans.predict(X_cluster_kmeans)
        kmeans_silhouettes.append(silhouette_score(X_cluster_kmeans, labels))
plt.figure(figsize=(10, 5))
plt.plot(range(1, 10), inertias, marker="o")
plt.title("Elbow Method for K-means")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.savefig(base_url + "visualizations/elbow_plot.png")
plt.close()

# Choose k=4 based on elbow plot
kmeans = KMeans(n_clusters=4, random_state=42)
zone_features["kmeans_cluster"] = kmeans.fit_predict(X_cluster_kmeans)
zone_features["kmeans_cluster"] = zone_features["kmeans_cluster"].astype(int)
kmeans_silhouette = silhouette_score(X_cluster_kmeans, zone_features["kmeans_cluster"])
print(f"K-means Silhouette Score: {kmeans_silhouette:.2f}")
with open(base_url + "models/kmeans_model.pkl", "wb") as f:
    pickle.dump(kmeans, f)

# DBSCAN clustering (spatial-temporal)
X_cluster_dbscan = zone_features[["pickup_lat", "pickup_lon", "pickup_count", "hour"]]
scaler = MinMaxScaler()
X_cluster_dbscan_scaled = scaler.fit_transform(X_cluster_dbscan)
eps_values = [0.1, 0.3, 0.5, 0.7]
min_samples_values = [3, 5, 10]
best_silhouette = -1
best_eps = 0.3
best_min_samples = 5
for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_cluster_dbscan_scaled)
        if len(np.unique(labels)) > 1:  # Ensure multiple clusters
            silhouette = silhouette_score(X_cluster_dbscan_scaled, labels)
            if silhouette > best_silhouette:
                best_silhouette = silhouette
                best_eps = eps
                best_min_samples = min_samples
dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
zone_features["dbscan_cluster"] = dbscan.fit_predict(X_cluster_dbscan_scaled)
zone_features["dbscan_cluster"] = zone_features["dbscan_cluster"].astype(int)  # Ensure integer
print(f"DBSCAN Clusters: {np.unique(zone_features['dbscan_cluster'])}, Silhouette Score: {best_silhouette:.2f}")
with open(base_url + "models/dbscan_model.pkl", "wb") as f:
    pickle.dump(dbscan, f)

# Silhouette score comparison
plt.figure(figsize=(8, 6))
plt.bar(["K-means", "DBSCAN"], [kmeans_silhouette, best_silhouette])
plt.title("Silhouette Score Comparison")
plt.ylabel("Silhouette Score")
plt.savefig(base_url + "visualizations/silhouette_comparison.png")
plt.close()

# Cluster statistics and labels
def assign_cluster_label(cluster_id, pickup_count, total_amount):
    if pickup_count > zone_features["pickup_count"].quantile(0.75):
        return "High-Demand Urban"
    elif total_amount > zone_features["total_amount"].quantile(0.75):
        return "High-Fare"
    elif cluster_id == -1:  # DBSCAN noise
        return "Noise"
    else:
        return "Residential"

cluster_stats = zone_features.groupby("kmeans_cluster").agg({
    "pickup_count": ["sum", "mean", "count"],
    "total_amount": "mean"
}).reset_index()
cluster_stats.columns = ["kmeans_cluster", "total_pickups", "avg_pickups", "zone_count", "avg_fare"]
cluster_stats["label"] = cluster_stats.apply(
    lambda x: assign_cluster_label(x["kmeans_cluster"], x["avg_pickups"], x["avg_fare"]), axis=1
)
cluster_stats.to_csv(base_url + "data/processed/kmeans_cluster_stats.csv", index=False)

dbscan_stats = zone_features.groupby("dbscan_cluster").agg({
    "pickup_count": ["sum", "mean", "count"],
    "total_amount": "mean"
}).reset_index()
dbscan_stats.columns = ["dbscan_cluster", "total_pickups", "avg_pickups", "zone_count", "avg_fare"]
dbscan_stats["label"] = dbscan_stats.apply(
    lambda x: assign_cluster_label(x["dbscan_cluster"], x["avg_pickups"], x["avg_fare"]), axis=1
)
dbscan_stats.to_csv(base_url + "data/processed/dbscan_cluster_stats.csv", index=False)

# Verify cluster column types
print(f"kmeans_cluster dtype: {zone_features['kmeans_cluster'].dtype}")
print(f"dbscan_cluster dtype: {zone_features['dbscan_cluster'].dtype}")

# Visualize K-means clusters
m_kmeans = folium.Map(location=[40.7128, -74.0060], zoom_start=11)
for _, row in zone_features.merge(cluster_stats[["kmeans_cluster", "label"]], on="kmeans_cluster").iterrows():
    if pd.notna(row["pickup_lat"]) and pd.notna(row["pickup_lon"]):
        folium.CircleMarker(
            location=[row["pickup_lat"], row["pickup_lon"]],
            radius=5,
            color=["red", "blue", "green", "purple"][int(row["kmeans_cluster"])],
            fill=True,
            popup=f"Zone {row['PULocationID']}: {row['label']}"
        ).add_to(m_kmeans)
m_kmeans.save(base_url + "visualizations/kmeans_cluster_map.html")

# Visualize DBSCAN clusters
m_dbscan = folium.Map(location=[40.7128, -74.0060], zoom_start=11)
for _, row in zone_features.merge(dbscan_stats[["dbscan_cluster", "label"]], on="dbscan_cluster").iterrows():
    if pd.notna(row["pickup_lat"]) and pd.notna(row["pickup_lon"]) and row["dbscan_cluster"] != -1:
        folium.CircleMarker(
            location=[row["pickup_lat"], row["pickup_lon"]],
            radius=5,
            color=["red", "blue", "green", "purple"][int(row["dbscan_cluster"]) % 4],
            fill=True,
            popup=f"Zone {row['PULocationID']}: {row['label']}"
        ).add_to(m_dbscan)
m_dbscan.save(base_url + "visualizations/dbscan_cluster_map.html")
print("Saved clustering results, statistics, and visualizations")