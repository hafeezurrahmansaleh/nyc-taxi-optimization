import pandas as pd
import dask.dataframe as dd
import glob
import numpy as np
from datetime import datetime
import holidays
import geopandas as gpd
import os
import json

# Define base URL for file paths
base_url = "/Users/saleh/Desktop/Projects/AIProject/TaxiOptimization/"

# Track row counts for debugging
row_counts = []

# Load taxi data with Dask for scalability
parquet_files = glob.glob(base_url + "data/raw/taxi_data/*.parquet")
if not parquet_files:
    raise FileNotFoundError(f"No Parquet files found at {base_url}data/raw/taxi_data/")
ddf = dd.read_parquet(parquet_files)
taxi_df = ddf.compute()  # Convert to Pandas for feature engineering
print(f"Loaded {len(taxi_df)} trip records from {len(parquet_files)} files.")
row_counts.append({"step": "Initial Load", "rows": len(taxi_df)})

# Load zone information
zones_path = base_url + "data/raw/taxi_zones.csv"
try:
    zones_df = pd.read_csv(zones_path)
    print(f"Loaded {len(zones_df)} zones.")
except FileNotFoundError:
    print(f"Error: taxi_zones.csv not found at {zones_path}")
    raise

# Clean zone data
initial_zones = len(zones_df)
zones_df = zones_df.dropna(subset=["LocationID", "Borough", "Zone"])
zones_df = zones_df[zones_df["LocationID"].isin(range(1, 266))]  # Valid LocationID: 1-265
print(f"Kept {len(zones_df)} zones after cleaning ({initial_zones - len(zones_df)} dropped).")

# Load GeoJSON for zone centroids
geojson_path = base_url + "data/raw/taxi_zones.geojson"
print(f"Attempting to load GeoJSON from: {geojson_path}")

# Validate GeoJSON file
try:
    with open(geojson_path, "r") as f:
        json.load(f)  # Check if file is valid JSON
    print("GeoJSON file is valid JSON.")
except json.JSONDecodeError as e:
    print(f"Error: Invalid GeoJSON file - JSON parsing failed: {str(e)}")
    print("Download from: https://gist.github.com/erikig/8beeca9be47713f196b721b9f48a8e5e")
    raise
except FileNotFoundError:
    print(f"Error: GeoJSON file not found at {geojson_path}")
    print("Download from: https://gist.github.com/erikig/8beeca9be47713f196b721b9f48a8e5e")
    raise

# Load GeoJSON with geopandas
try:
    gdf = gpd.read_file(geojson_path)
    zones_df["lat"] = gdf.geometry.centroid.y
    zones_df["lon"] = gdf.geometry.centroid.x
    # Broader NYC bounds to include all zones (e.g., JFK, Staten Island)
    zones_df = zones_df[
        (zones_df["lat"].between(40.4, 41.1)) & (zones_df["lon"].between(-74.5, -73.5))
    ]
    print(f"Loaded GeoJSON with {len(zones_df)} valid centroids.")
except Exception as e:
    print(f"Error loading GeoJSON: {str(e)}")
    print("Ensure valid GeoJSON: https://gist.github.com/erikig/8beeca9be47713f196b721b9f48a8e5e")
    raise

# Merge pickup and dropoff locations
taxi_df = taxi_df.merge(zones_df, left_on="PULocationID", right_on="LocationID", how="inner") \
                 .rename(columns={"Borough": "Pickup_Borough", "Zone": "Pickup_Zone", "lat": "pickup_lat", "lon": "pickup_lon"}) \
                 .drop(columns=["LocationID", "service_zone"])
taxi_df = taxi_df.merge(zones_df, left_on="DOLocationID", right_on="LocationID", how="inner") \
                 .rename(columns={"Borough": "Dropoff_Borough", "Zone": "Dropoff_Zone", "lat": "dropoff_lat", "lon": "dropoff_lon"}) \
                 .drop(columns=["LocationID", "service_zone"])
print(f"After zone merge: {len(taxi_df)} rows.")
row_counts.append({"step": "Zone Merge", "rows": len(taxi_df)})

# Clean taxi data
# Convert timestamps
taxi_df["pickup_datetime"] = pd.to_datetime(taxi_df["tpep_pickup_datetime"], errors="coerce")
taxi_df["dropoff_datetime"] = pd.to_datetime(taxi_df["tpep_dropoff_datetime"], errors="coerce")
taxi_df = taxi_df.dropna(subset=["pickup_datetime", "dropoff_datetime"])
print(f"After timestamp conversion: {len(taxi_df)} rows.")
row_counts.append({"step": "Timestamp Conversion", "rows": len(taxi_df)})

# Dynamic date range
min_date = taxi_df["pickup_datetime"].min().floor("D")
max_date = taxi_df["pickup_datetime"].max().ceil("D")
print(f"Dataset date range: {min_date} to {max_date}")
taxi_df = taxi_df[
    (taxi_df["pickup_datetime"] >= min_date) &
    (taxi_df["pickup_datetime"] <= max_date) &
    (taxi_df["dropoff_datetime"] >= min_date) &
    (taxi_df["dropoff_datetime"] <= max_date) &
    (taxi_df["pickup_datetime"] <= taxi_df["dropoff_datetime"])
]
print(f"After date range filter: {len(taxi_df)} rows.")
row_counts.append({"step": "Date Range Filter", "rows": len(taxi_df)})

# Filter trip details
taxi_df = taxi_df[
    (taxi_df["trip_distance"] > 0) & (taxi_df["trip_distance"] < 100) &
    (taxi_df["total_amount"] > 0) & (taxi_df["total_amount"] < 500) &
    (taxi_df["passenger_count"] > 0) & (taxi_df["passenger_count"] < 10)
]
print(f"After trip filters: {len(taxi_df)} rows.")
row_counts.append({"step": "Trip Filters", "rows": len(taxi_df)})

# Validate fares and surcharges
fare_cols = ["fare_amount", "extra", "mta_tax", "tip_amount", "tolls_amount", "improvement_surcharge", "congestion_surcharge", "Airport_fee"]
for col in fare_cols:
    if col in taxi_df.columns:
        taxi_df = taxi_df[taxi_df[col] >= 0]
print(f"After fare validation: {len(taxi_df)} rows.")
row_counts.append({"step": "Fare Validation", "rows": len(taxi_df)})

# Validate RatecodeID
if "RatecodeID" in taxi_df.columns:
    taxi_df = taxi_df[taxi_df["RatecodeID"].isin([1, 2, 3, 4, 5, 6])]
print(f"After RatecodeID filter: {len(taxi_df)} rows.")
row_counts.append({"step": "RatecodeID Filter", "rows": len(taxi_df)})

# Remove duplicates
taxi_df = taxi_df.drop_duplicates(subset=["VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime"])
print(f"After duplicate removal: {len(taxi_df)} rows.")
row_counts.append({"step": "Duplicate Removal", "rows": len(taxi_df)})

# Feature engineering: temporal
taxi_df["hour"] = taxi_df["pickup_datetime"].dt.hour
taxi_df["day_of_week"] = taxi_df["pickup_datetime"].dt.dayofweek
taxi_df["month"] = taxi_df["pickup_datetime"].dt.month
us_holidays = holidays.US()
taxi_df["is_holiday"] = taxi_df["pickup_datetime"].apply(lambda x: 1 if x in us_holidays else 0)

# Feature engineering: trip-based
taxi_df["trip_duration"] = (taxi_df["dropoff_datetime"] - taxi_df["pickup_datetime"]).dt.total_seconds() / 60
taxi_df = taxi_df[taxi_df["trip_duration"] > 0]
taxi_df["avg_speed"] = taxi_df["trip_distance"] / (taxi_df["trip_duration"] / 60)
taxi_df = taxi_df[taxi_df["avg_speed"] < 100]
print(f"After trip feature filters: {len(taxi_df)} rows.")
row_counts.append({"step": "Trip Feature Filters", "rows": len(taxi_df)})

# Feature engineering: demand-based
pickup_counts = taxi_df.groupby(["PULocationID", "hour"])["VendorID"].count().reset_index(name="pickup_count")
if "pickup_count" in taxi_df.columns:
    taxi_df = taxi_df.drop(columns=["pickup_count"])
taxi_df = taxi_df.merge(pickup_counts, on=["PULocationID", "hour"], how="left")
print(f"After demand features: {len(taxi_df)} rows.")
row_counts.append({"step": "Demand Features", "rows": len(taxi_df)})
print("Added demand-based features.")

# Load NOAA weather data
weather_file = base_url + "data/external/noaa_weather_2023_2024.csv"
if not os.path.exists(weather_file):
    print(f"Error: Weather data not found at {weather_file}")
    print("Download JFK 2023-2024 data from:")
    print("2023: https://www.ncei.noaa.gov/data/global-hourly/access/2023/947890.csv")
    print("2024: https://www.ncei.noaa.gov/data/global-hourly/access/2024/947890.csv")
    raise FileNotFoundError
else:
    weather_data = pd.read_csv(weather_file)
    print("Available weather columns:", weather_data.columns.tolist())
    # Parse datetime
    weather_data["date"] = pd.to_datetime(weather_data["DATE"], errors="coerce")
    # Extract temperature (°C to °F)
    weather_data["temperature"] = weather_data["TMP"].str.split(",", expand=True)[0].replace(r"[^0-9\-+]", "", regex=True).astype(float) / 10 * 9/5 + 32
    # Extract precipitation (mm to inches)
    if "AA1" in weather_data.columns:
        weather_data["precipitation"] = weather_data["AA1"].str.split(",", expand=True)[0].replace(r"[^0-9]", "", regex=True).astype(float) / 25.4
    else:
        print("Warning: AA1 column not found, setting precipitation to 0.")
        weather_data["precipitation"] = 0
    # Extract wind speed (m/s to mph)
    weather_data["wind_speed"] = weather_data["WND"].str.split(",", expand=True)[3].replace(r"[^0-9]", "", regex=True).astype(float) / 10 * 2.23694
    # Select columns
    weather_data = weather_data[["date", "temperature", "precipitation", "wind_speed"]]
    weather_data = weather_data.dropna(subset=["date"])
    weather_data[["temperature", "precipitation", "wind_speed"]] = weather_data[["temperature", "precipitation", "wind_speed"]].fillna(method="ffill").fillna(0)
    print("Loaded and processed NOAA weather data.")

# Merge weather data
taxi_df["date"] = taxi_df["pickup_datetime"].dt.floor("h")
taxi_df = taxi_df.merge(weather_data, on="date", how="left")
taxi_df[["temperature", "precipitation", "wind_speed"]] = taxi_df[["temperature", "precipitation", "wind_speed"]].fillna(method="ffill").fillna(0)
print(f"After weather merge: {len(taxi_df)} rows.")
row_counts.append({"step": "Weather Merge", "rows": len(taxi_df)})
print("Integrated real weather data.")

# Print row counts for debugging
print("\nRow counts after each step:\\")
for count in row_counts:
    print(f"{count['step']}: {count['rows']} rows")

# Save processed data
taxi_df.to_csv(base_url + "data/processed/cleaned_taxi_data.csv", index=False)
print(f"Saved processed dataset to {base_url}data/processed/cleaned_taxi_data.csv")