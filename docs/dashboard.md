Let’s break down the `dashboard.py` code in a simple and easy-to-understand way. I’ll explain what the code does, why it’s structured that way, the logic behind each section, and the calculations involved. This will give you a clear understanding of everything happening in the dashboard so you can explain it confidently to others.

---

### Overview of the Dashboard
The `dashboard.py` script creates a web-based dashboard using Streamlit for NYC taxi drivers. The goal is to help drivers find the best zones to maximize earnings by providing real-time insights, recommendations, and visualizations. The dashboard has two main pages: **Driver Dashboard** (for drivers) and **Technical Dashboard** (for technical users). The Driver Dashboard is the main focus, with three tabs: **Demand Insights**, **Recommendations**, and **Visualizations**.

### Key Features
1. **Demand Insights Tab**: Shows trends and statistics about demand in different zones.
2. **Recommendations Tab**: Suggests the best zone for the driver to go to, with details on earnings, travel time, and other factors.
3. **Visualizations Tab**: Displays a map of high-demand zones and a chart of demand distribution.
4. **Technical Dashboard**: Provides detailed model performance metrics for technical users (less relevant for drivers).

---

### Step-by-Step Explanation of the Code

#### 1. Importing Libraries
```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import folium
from streamlit_folium import folium_static
import pickle
from geopy.distance import geodesic
from datetime import datetime, timedelta
import pytz
import time
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
```
- **What**: These are the tools (libraries) the code uses.
- **Why**: 
  - `streamlit`: Creates the web dashboard.
  - `pandas` and `numpy`: Handle data (tables and numbers).
  - `plotly` and `folium`: Create charts and maps.
  - `prophet`: Forecasts future demand (like predicting how many rides will happen).
  - `pickle`: Loads saved machine learning models.
  - `geopy`: Calculates distances between locations.
  - `datetime`, `timedelta`, `pytz`, `time`: Work with dates and times.
  - `xgboost` and `sklearn`: Machine learning models to predict high-demand zones and earnings.
- **How**: These libraries are standard tools for data analysis, visualization, and machine learning in Python.

---

#### 2. Setting Up the Dashboard
```python
BASE_URL = "/Users/saleh/Desktop/Projects/AIProject/TaxiOptimization/"
st.set_page_config(page_title="NYC Taxi Optimization Dashboard", layout="wide")
```
- **What**: Sets the base path for files and configures the dashboard.
- **Why**: 
  - `BASE_URL`: Points to the folder where your data and models are stored.
  - `st.set_page_config`: Makes the dashboard wide (uses the full screen) and sets the title.
- **How**: This is the starting point to ensure the dashboard knows where to find files and how to display itself.

#### 3. Styling the Dashboard with CSS
```python
st.markdown("""
    <style>
    .stButton>button {
        background-color: #FFD700;
        color: black;
        border-radius: 5px;
        border: none;
        padding: 8px 16px;
        font-weight: bold;
        margin-right: 5px;
    }
    .stButton>button:hover {
        background-color: #FFC107;
        color: black;
    }
    .card {
        background-color: #2E2E2E;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        margin-bottom: 15px;
        color: white;
    }
    ...
    </style>
""", unsafe_allow_html=True)
```
- **What**: Adds custom styling to make the dashboard look nice.
- **Why**: 
  - Makes buttons yellow (`#FFD700`) with a hover effect (`#FFC107`).
  - Creates styled cards (boxes) for recommendations with a dark background (`#2E2E2E`), white text, and a shadow effect.
  - Defines styles for stats cards, gauges, and other elements.
- **How**: Uses CSS (a styling language) inside `st.markdown` to change the appearance of the dashboard.

---

#### 4. Loading and Preparing Data
```python
@st.cache_data
def load_and_aggregate_data():
    test_data_path = BASE_URL + "data/processed/cleaned_taxi_data.csv"
    test_df = pd.read_csv(test_data_path)
    test_df["pickup_datetime"] = pd.to_datetime(test_df["pickup_datetime"])
    test_df["hour"] = test_df["pickup_datetime"].dt.hour
    test_df["day_of_week"] = test_df["pickup_datetime"].dt.dayofweek

    pickup_counts = test_df.groupby(["PULocationID", "hour", "day_of_week", "is_holiday"])["VendorID"].count().reset_index(name="pickup_count")
    features = test_df.groupby(["PULocationID", "hour", "day_of_week", "is_holiday"]).agg({
        "temperature": "mean",
        "precipitation": "mean",
        "wind_speed": "mean",
        "avg_speed": "mean",
        "month": "mean",
        "trip_duration": "mean",
        "passenger_count": "mean"
    }).reset_index()
    speed_stats = test_df.groupby(["PULocationID", "hour", "day_of_week"])["avg_speed"].agg(["mean", "std"]).reset_index()
    speed_stats = speed_stats.rename(columns={"mean": "avg_speed_mean", "std": "avg_speed_std"})
    features = features.merge(speed_stats[["PULocationID", "hour", "day_of_week", "avg_speed_std"]], on=["PULocationID", "hour", "day_of_week"], how="left")
    supply_data = test_df.groupby(["PULocationID", "hour", "day_of_week"])["VendorID"].nunique().reset_index(name="driver_count")
    zone_features = test_df.groupby("PULocationID").agg({
        "pickup_count": "sum",
        "total_amount": "mean",
        "pickup_lat": "mean",
        "pickup_lon": "mean",
        "trip_duration": "mean",
        "passenger_count": "mean"
    }).reset_index()
    zone_features["avg_fare"] = zone_features["total_amount"].clip(upper=30)
    return pickup_counts, features, supply_data, zone_features, test_df

pickup_counts, features, supply_data, zone_features, test_df = load_and_aggregate_data()
```
- **What**: Loads the main data file and organizes it into different tables.
- **Why**: 
  - The data contains taxi trip information (e.g., pickup locations, times, fares, weather).
  - We need to summarize this data to analyze demand, supply, and zone characteristics.
- **How**:
  - **Main Data (`test_df`)**: Loads `cleaned_taxi_data.csv`, which has columns like `PULocationID` (pickup zone), `pickup_datetime`, `VendorID` (driver ID), `temperature`, `precipitation`, etc.
  - **Extracts Time Info**: Adds `hour` and `day_of_week` columns from `pickup_datetime` to analyze patterns by time.
  - **Summarizes Data**:
    - `pickup_counts`: Counts the number of pickups (`pickup_count`) per zone, hour, day of week, and holiday status.
    - `features`: Averages weather and trip features (e.g., `temperature`, `avg_speed`) per zone, hour, day of week, and holiday status.
    - `speed_stats`: Calculates the mean and standard deviation of `avg_speed` per zone, hour, and day of week.
    - `supply_data`: Counts unique drivers (`driver_count`) per zone, hour, and day of week.
    - `zone_features`: Summarizes overall zone info (total pickups, average fare, latitude, longitude, etc.).
  - **Capping Fares**: Clips `avg_fare` at $30 to avoid outliers (e.g., very expensive trips).
- **Logic**: Grouping and summarizing data helps us analyze patterns (e.g., which zones are busy at certain times) and prepare features for machine learning models.

---

#### 5. Loading Additional Data and Models
```python
prophet_path = BASE_URL + "data/processed/prophet_forecasts.csv"
try:
    prophet_df = pd.read_csv(prophet_path)
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
    required_columns = ["ds", "y", "yhat"]
    if not all(col in prophet_df.columns for col in required_columns):
        st.error(f"prophet_forecasts.csv is missing required columns. Expected: {required_columns}, Found: {prophet_df.columns.tolist()}")
        st.stop()
except FileNotFoundError:
    st.error(f"Error: {prophet_path} not found")
    st.stop()

top_features = ['precipitation', 'is_holiday', 'avg_speed', 'day_of_week', 'temp_precip']
rf_earnings_features = [
    "pickup_count", "driver_count", "temperature", "precipitation", "wind_speed",
    "is_holiday", "avg_speed", "day_of_week", "temp_precip", "travel_time",
    "passenger_count", "avg_fare", "current_demand_supply_ratio"
]

@st.cache_resource
def load_xgb_model():
    with open(BASE_URL + "models/xgb_model.pkl", "rb") as f:
        return pickle.load(f)

xgb_model = load_xgb_model()

@st.cache_resource
def load_rf_classifier_model():
    try:
        with open(BASE_URL + "models/rf_model.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

rf_classifier_model = load_rf_classifier_model()

@st.cache_resource
def load_rf_earnings_model():
    try:
        with open(BASE_URL + "models/rf_earnings_model.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

rf_earnings_model = load_rf_earnings_model()

@st.cache_resource
def load_prophet_models():
    prophet_model = None
    zone_models = {}
    try:
        with open(BASE_URL + "models/prophet_model.pkl", "rb") as f:
            prophet_model = pickle.load(f)
    except FileNotFoundError:
        st.error(f"Error: City-wide Prophet model not found")
        st.stop()

    top_zones = test_df["PULocationID"].value_counts().head(10).index.tolist()
    for zone in top_zones:
        try:
            with open(BASE_URL + f"models/prophet_zone_{zone}.pkl", "rb") as f:
                zone_models[zone] = pickle.load(f)
        except FileNotFoundError:
            pass  # Silently skip missing models
    return prophet_model, zone_models, top_zones

prophet_model, zone_models, top_zones = load_prophet_models()

kmeans_stats_path = BASE_URL + "data/processed/kmeans_cluster_stats.csv"
try:
    kmeans_stats = pd.read_csv(kmeans_stats_path)
except FileNotFoundError:
    st.error(f"Error: {kmeans_stats_path} not found")
    st.stop()

zone_features = zone_features.merge(kmeans_stats[["kmeans_cluster", "label"]], left_on="PULocationID", right_on="kmeans_cluster", how="left")
zone_features["kmeans_cluster"] = zone_features["kmeans_cluster"].fillna(-1)
zone_features["label"] = zone_features["label"].fillna("Low-Demand")
```
- **What**: Loads additional data (forecasts, clustering stats) and machine learning models.
- **Why**: 
  - `prophet_df`: Contains city-wide demand forecasts (actual vs. predicted) for the Technical Dashboard.
  - `top_features` and `rf_earnings_features`: Lists of features used by the machine learning models.
  - `xgb_model`, `rf_classifier_model`, `rf_earnings_model`: Pre-trained models to predict high-demand zones and earnings.
  - `prophet_model`, `zone_models`: Models to forecast demand (city-wide and per zone).
  - `kmeans_stats`: Clustering data to label zones (e.g., "High-Demand Urban").
- **How**:
  - **Prophet Forecasts**: Loads `prophet_forecasts.csv` for city-wide demand predictions.
  - **Feature Lists**: Defines which columns (e.g., `precipitation`, `avg_speed`) the models use.
  - **Models**: Loads pre-trained models from `.pkl` files using `pickle`. If a model is missing, it either falls back (e.g., Random Forest to XGBoost) or stops.
  - **K-means Clustering**: Merges clustering labels into `zone_features` to classify zones (e.g., "High-Demand Urban").
- **Logic**: These resources are essential for making predictions and visualizing patterns. The `@st.cache_resource` decorator ensures models are loaded only once for efficiency.

---

#### 6. Setting Up Initial Parameters
```python
current_utc = datetime.now(pytz.UTC)
nyc_tz = pytz.timezone("America/New_York")
current_nyc = current_utc.astimezone(nyc_tz)
default_day_of_week = current_nyc.strftime("%A")
default_hour = current_nyc.hour

us_holidays_2025 = {
    "2025-01-01": "New Year's Day",
    "2025-01-20": "Martin Luther King Jr. Day",
    "2025-02-17": "Presidents' Day",
    "2025-05-26": "Memorial Day",
    "2025-07-04": "Independence Day",
    "2025-09-01": "Labor Day",
    "2025-10-13": "Columbus Day",
    "2025-11-11": "Veterans Day",
    "2025-11-27": "Thanksgiving Day",
    "2025-12-25": "Christmas Day",
}
current_date = current_nyc.strftime("%Y-%m-%d")
default_is_holiday = current_date in us_holidays_2025
default_weather = "Clear"

day_of_week_options = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

current_day_of_week_num = day_of_week_mapping[default_day_of_week]
current_supply = supply_data[(supply_data["day_of_week"] == current_day_of_week_num) & (supply_data["hour"] == default_hour)]
if current_supply.empty:
    day_supply = supply_data[supply_data["day_of_week"] == current_day_of_week_num]
    if not day_supply.empty:
        current_supply = day_supply.groupby("PULocationID")["driver_count"].mean().reset_index()
    else:
        current_supply = supply_data.groupby("PULocationID")["driver_count"].mean().reset_index()
np.random.seed(42)
current_supply["driver_count"] = current_supply["driver_count"] * np.random.uniform(0.9, 1.1, size=len(current_supply))
current_supply = current_supply[["PULocationID", "driver_count"]]
```
- **What**: Sets up the current date, time, and other default settings.
- **Why**: 
  - Determines the current time in NYC to set default values for user inputs.
  - Checks if today is a holiday to adjust demand predictions.
  - Estimates the current number of drivers (supply) in each zone.
- **How**:
  - **Current Time**: Uses `datetime` and `pytz` to get the current time in NYC (e.g., Thursday, 16:00).
  - **Holidays**: Defines a list of 2025 US holidays and checks if today is a holiday.
  - **Default Weather**: Sets to "Clear" unless the user changes it.
  - **Supply Estimation**: 
    - Looks for the number of drivers (`driver_count`) in `supply_data` for the current day and hour.
    - If data is missing, falls back to the average for the day or overall average.
    - Adds a small random variation (0.9 to 1.1) to make the supply more realistic.
- **Logic**: These settings provide a starting point for the dashboard and ensure we have realistic supply data for calculations like demand-to-supply ratio.

---

#### 7. Initializing Session State
```python
if "driver_preferences" not in st.session_state:
    st.session_state.driver_preferences = {zone: 0 for zone in test_df["PULocationID"].unique()}
if "performance_history" not in st.session_state:
    st.session_state.performance_history = []
if "last_recommendation_time" not in st.session_state:
    st.session_state.last_recommendation_time = 0
```
- **What**: Sets up variables to track user actions across sessions.
- **Why**: 
  - `driver_preferences`: Tracks which zones the driver likes (increases when they accept a recommendation, decreases when they reject).
  - `performance_history`: Records earnings from accepted recommendations to show average performance.
  - `last_recommendation_time`: Ensures drivers don’t reposition too frequently (e.g., waits 5 minutes).
- **How**: Uses `st.session_state` to store these variables so they persist while the user interacts with the dashboard.
- **Logic**: This helps personalize the experience (e.g., recommending preferred zones) and provides feedback on how well recommendations are working.

---

#### 8. Creating the Sidebar
```python
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page:", ["Driver Dashboard", "Technical Dashboard"])

with st.sidebar:
    st.header("⚙️ Your Inputs")
    day_of_week = st.selectbox("Day of the Week", options=day_of_week_options, index=day_of_week_options.index(default_day_of_week), key="driver_day")
    is_holiday = st.checkbox("Is Holiday", value=default_is_holiday, key="driver_holiday")
    current_time = st.slider("Current Hour", 0, 23, default_hour, key="driver_time")
    weather_condition = st.selectbox("Current Weather", options=["Clear", "Rainy", "Snowy"], index=["Clear", "Rainy", "Snowy"].index(default_weather), key="driver_weather")
    current_zone = st.selectbox("Current Zone (Your Location)", options=sorted(test_df["PULocationID"].unique()), key="driver_zone")
    classifier_choice = st.selectbox("Choose Classifier for High-Demand Zones:", ["XGBoost", "Random Forest"], key="classifier_choice")
```
- **What**: Creates a sidebar for navigation and user inputs.
- **Why**: 
  - Lets the user switch between the Driver Dashboard and Technical Dashboard.
  - Allows the driver to customize the scenario (e.g., change the day, time, weather).
- **How**:
  - **Navigation**: A dropdown (`selectbox`) to choose the page.
  - **Inputs**:
    - `day_of_week`: Choose the day (e.g., Thursday).
    - `is_holiday`: Check if it’s a holiday.
    - `current_time`: Slider to select the hour (0 to 23).
    - `weather_condition`: Choose weather ("Clear", "Rainy", "Snowy").
    - `current_zone`: Select the driver’s current zone.
    - `classifier_choice`: Choose between XGBoost or Random Forest for predicting high-demand zones (more relevant for technical users).
- **Logic**: These inputs let the driver simulate different scenarios (e.g., "What if it’s raining on Friday at 8 PM?") and customize the dashboard to their current location.

---

#### 9. Driver Dashboard: Main Structure
```python
if page == "Driver Dashboard":
    st.title("🚖 NYC Taxi Driver Dashboard")
    st.markdown("Find the best zones to maximize your earnings with real-time insights and recommendations.")

    selected_day_of_week = day_of_week_mapping[day_of_week]

    # Define current_location early
    current_location = test_df[test_df["PULocationID"] == current_zone][["pickup_lat", "pickup_lon"]].mean()

    # Compute demand forecasts and distances for zone_features early
    future_dates = pd.DataFrame({
        "ds": [current_nyc + timedelta(hours=i) for i in range(1, 7)]  # Forecast for next 6 hours
    })
    future_dates["ds"] = future_dates["ds"].dt.tz_localize(None)
    future_dates["precipitation"] = 0.0 if weather_condition == "Clear" else (0.5 if weather_condition == "Rainy" else 1.0)
    future_dates["temperature"] = 15.0
    future_dates["temp_precip"] = future_dates["temperature"] * future_dates["precipitation"]

    zone_forecasts = {}
    zone_forecasts_no_weather = {}
    top_zones_forecast = top_zones
    for zone in top_zones_forecast:
        if zone in zone_models:
            zone_forecast = zone_models[zone].predict(future_dates)
            zone_forecast["PULocationID"] = zone
            zone_forecasts[zone] = zone_forecast
            future_dates_no_weather = future_dates.copy()
            future_dates_no_weather["precipitation"] = 0.0
            future_dates_no_weather["temp_precip"] = 0.0
            zone_forecast_no_weather = zone_models[zone].predict(future_dates_no_weather)
            zone_forecast_no_weather["PULocationID"] = zone
            zone_forecasts_no_weather[zone] = zone_forecast_no_weather

    if zone_forecasts:
        all_zone_forecasts = pd.concat([df for df in zone_forecasts.values()], ignore_index=True)
        next_hour_forecast = all_zone_forecasts[all_zone_forecasts["ds"] == future_dates["ds"][0]].copy()
        next_hour_forecast = next_hour_forecast[["PULocationID", "yhat", "trend", "yhat_lower", "yhat_upper"]].rename(columns={"yhat": "forecasted_demand"})
        next_hour_forecast["demand_slope"] = all_zone_forecasts.groupby("PULocationID")["yhat"].diff().fillna(0)
        next_hour_forecast["prophet_confidence"] = (next_hour_forecast["yhat_upper"] - next_hour_forecast["yhat_lower"]) / next_hour_forecast["forecasted_demand"]
        next_hour_forecast["prophet_confidence"] = next_hour_forecast["prophet_confidence"].clip(lower=0, upper=1.0)
        all_zone_forecasts_no_weather = pd.concat([df for df in zone_forecasts_no_weather.values()], ignore_index=True)
        next_hour_forecast_no_weather = all_zone_forecasts_no_weather[all_zone_forecasts_no_weather["ds"] == future_dates["ds"][0]].copy()
        next_hour_forecast_no_weather = next_hour_forecast_no_weather[["PULocationID", "yhat"]].rename(columns={"yhat": "forecasted_demand_no_weather"})
    else:
        next_hour_forecast = pd.DataFrame(columns=["PULocationID", "forecasted_demand", "demand_slope", "prophet_confidence"])
        next_hour_forecast_no_weather = pd.DataFrame(columns=["PULocationID", "forecasted_demand_no_weather"])

    zone_features = zone_features.merge(next_hour_forecast[["PULocationID", "forecasted_demand"]], on="PULocationID", how="left")
    zone_features["forecasted_demand"] = zone_features["forecasted_demand"].fillna(0)
    if pd.notna(current_location["pickup_lat"]) and pd.notna(current_location["pickup_lon"]):
        coords_current = (current_location["pickup_lat"], current_location["pickup_lon"])
        zone_features["distance"] = zone_features.apply(
            lambda row: geodesic(coords_current, (row["pickup_lat"], row["pickup_lon"])).miles
            if pd.notna(row["pickup_lat"]) and pd.notna(row["pickup_lon"]) else float("inf"),
            axis=1
        )
    else:
        zone_features["distance"] = float("inf")

    tab1, tab2, tab3 = st.tabs(["📊 Demand Insights", "🚕 Recommendations", "🗺️ Visualizations"])
```
- **What**: Sets up the Driver Dashboard with three tabs.
- **Why**: 
  - Provides a user-friendly interface with separate sections for insights, recommendations, and maps.
  - Prepares data for use in all tabs (e.g., forecasts, distances).
- **How**:
  - **Title and Description**: Sets the title and a brief description for the dashboard.
  - **Current Location**: Calculates the driver’s current coordinates (latitude, longitude) by averaging the coordinates of their current zone.
  - **Demand Forecasts**:
    - Creates a list of future times (`future_dates`) for the next 6 hours.
    - Sets weather conditions (`precipitation` = 0 for "Clear", 0.5 for "Rainy", 1 for "Snowy") and a fixed temperature (15°C).
    - Uses Prophet models (`zone_models`) to forecast demand for the top 10 zones (`top_zones`) over the next 6 hours.
    - Stores forecasts in `zone_forecasts` (with weather) and `zone_forecasts_no_weather` (without weather).
    - Combines all forecasts into `all_zone_forecasts` and extracts the next hour’s forecast (`next_hour_forecast`).
    - Calculates `demand_slope` (trend: positive means increasing demand, negative means decreasing).
  - **Distances**: Adds a `distance` column to `zone_features` by calculating the distance (in miles) from the driver’s current location to each zone.
  - **Tabs**: Creates three tabs using `st.tabs` for organizing the content.
- **Logic**: This section prepares the data needed for all tabs, such as demand forecasts and distances, to avoid repeating calculations later.

---

#### 10. Demand Insights Tab
```python
with tab1:
    st.header(f"📅 Demand Forecast for the Next Hour ({day_of_week}, {current_time}:00)")
    if zone_forecasts:
        st.write("**Demand-to-Supply Ratio Trends for Top Zones (Next 6 Hours)**")
        top_5_zones = next_hour_forecast.nlargest(5, "forecasted_demand")["PULocationID"].tolist()
        supply_forecasts = {}
        for zone in top_5_zones:
            zone_supply = supply_data[supply_data["PULocationID"] == zone].groupby("hour")["driver_count"].mean().reset_index()
            zone_supply = zone_supply.rename(columns={"driver_count": "forecasted_driver_count"})
            zone_supply["hour"] = zone_supply["hour"].astype(int)
            zone_supply["future_hour"] = (zone_supply["hour"] - current_time) % 24
            supply_forecasts[zone] = zone_supply
        ratio_data = []
        for zone in top_5_zones:
            zone_data = all_zone_forecasts[all_zone_forecasts["PULocationID"] == zone].copy()
            zone_data["hour"] = zone_data["ds"].dt.hour
            zone_supply = supply_forecasts[zone]
            zone_data = zone_data.merge(zone_supply[["hour", "forecasted_driver_count"]], on="hour", how="left")
            zone_data["forecasted_driver_count"] = zone_data["forecasted_driver_count"].fillna(1)
            zone_data["demand_supply_ratio"] = zone_data["yhat"] / zone_data["forecasted_driver_count"]
            zone_data["demand_supply_ratio"] = zone_data["demand_supply_ratio"].clip(lower=0, upper=10)
            ratio_data.append(zone_data[["ds", "PULocationID", "demand_supply_ratio"]])
        ratio_df = pd.concat(ratio_data, ignore_index=True)
        fig_ratios = go.Figure()
        for zone in top_5_zones:
            zone_data = ratio_df[ratio_df["PULocationID"] == zone]
            fig_ratios.add_trace(go.Scatter(
                x=zone_data["ds"],
                y=zone_data["demand_supply_ratio"],
                name=f"Zone {zone}",
                mode="lines",
                stackgroup='one',
                fill='tonexty'
            ))
        fig_ratios.update_layout(
            title="Demand-to-Supply Ratio Trends for Top 5 Zones",
            xaxis_title="Time",
            yaxis_title="Demand-to-Supply Ratio",
            hovermode="x unified",
            template="plotly_dark",
            title_font_size=16,
            height=400
        )
        st.plotly_chart(fig_ratios, use_container_width=True)

        st.dataframe(
            next_hour_forecast[["PULocationID", "forecasted_demand", "demand_slope"]].style.format({
                "forecasted_demand": "{:.0f}",
                "demand_slope": "{:.2f}"
            }).set_table_styles([
                {'selector': 'th', 'props': [('background-color', '#FFD700'), ('color', 'black'), ('font-weight', 'bold')]},
                {'selector': 'tr:nth-child(even)', 'props': [('background-color', '#f2f2f2')]}
            ])
        )
    else:
        st.write("No zone-specific forecasts available.")

    st.markdown("---")
```
- **What**: Shows demand trends and high-demand zones.
- **Why**: Helps drivers understand where demand is high relative to supply, so they can plan their next move.
- **How**:
  - **Header**: Displays the current scenario (e.g., "Demand Forecast for the Next Hour (Thursday, 16:00)").
  - **Chart: Demand-to-Supply Ratio Trends**:
    - Selects the top 5 zones by `forecasted_demand`.
    - Estimates future driver supply (`forecasted_driver_count`) by averaging historical supply data (`supply_data`) for each hour.
    - Calculates the demand-to-supply ratio (`yhat / forecasted_driver_count`) for each hour over the next 6 hours.
    - Clips the ratio between 0 and 10 to avoid extreme values for visualization.
    - Creates a stacked area chart showing how the ratio changes over time for each zone.
  - **Table**: Displays a table with `PULocationID`, `forecasted_demand`, and `demand_slope` for the next hour.
- **Logic**:
  - The demand-to-supply ratio is a key metric because a high ratio means more passengers per driver, which is good for earnings.
  - The chart shows trends over time, helping drivers see if a zone will stay busy or get less busy.

---

#### 11. High-Demand Zones Section
```python
st.header(f"🔥 High-Demand Zones Right Now ({day_of_week}, {current_time}:00)")
high_demand_zones = pd.DataFrame()
filtered_pickup_counts = pickup_counts[
    (pickup_counts["day_of_week"] == selected_day_of_week) &
    (pickup_counts["is_holiday"] == int(is_holiday)) &
    (pickup_counts["hour"] == current_time)
].copy()

filtered_features = features[
    (features["day_of_week"] == selected_day_of_week) &
    (features["is_holiday"] == int(is_holiday)) &
    (features["hour"] == current_time)
].copy()

if filtered_pickup_counts.empty:
    st.write("No pickup data available for the selected parameters. Try adjusting the day, time, or holiday status.")
elif filtered_features.empty:
    st.write("No feature data available for the selected parameters. Proceeding with available pickup data.")
    data = filtered_pickup_counts.copy()
    for col in ['temperature', 'precipitation', 'wind_speed', 'avg_speed', 'trip_duration', 'passenger_count', 'avg_speed_std']:
        if col not in data.columns:
            data[col] = 0.0
else:
    data = filtered_pickup_counts.merge(
        filtered_features,
        on=["PULocationID", "hour", "day_of_week", "is_holiday"],
        how="left"
    )
    data['temperature'] = data['temperature'].fillna(0.0)
    data['precipitation'] = data['precipitation'].fillna(0.0)
    data['wind_speed'] = data['wind_speed'].fillna(0.0)
    data['avg_speed'] = data['avg_speed'].fillna(data['avg_speed'].mean() if not data['avg_speed'].isna().all() else 0.0)
    data['trip_duration'] = data['trip_duration'].fillna(0.0)
    data['passenger_count'] = data['passenger_count'].fillna(0.0)
    data['avg_speed_std'] = data['avg_speed_std'].fillna(0.0)

if data.empty:
    st.write("No data available after merging. Try adjusting the inputs.")
else:
    data["temp_precip"] = data["temperature"] * data["precipitation"]
    data["temp_precip"] = data["temp_precip"].fillna(0.0)
    data = data.reset_index(drop=True)

    missing_features = [feature for feature in top_features if feature not in data.columns]
    if missing_features:
        st.error(f"Missing required features in data: {missing_features}")
        st.stop()

    X = data[top_features]
    if classifier_choice == "Random Forest" and rf_classifier_model:
        probs = rf_classifier_model.predict_proba(X)[:, 1]
    else:
        probs = xgb_model.predict_proba(X)[:, 1]
    
    threshold = max(np.percentile(probs, 50) if len(probs) > 0 else 0.3, 0.001)
    data["is_high_demand"] = (probs >= threshold).astype(int)
    data["xgb_confidence"] = probs

    filtered_data = data[data["is_high_demand"] == 1]
    high_demand_zones = filtered_data[
        ["PULocationID", "pickup_count", "temperature", "wind_speed", "precipitation", "is_holiday", "avg_speed", "temp_precip", "day_of_week", "trip_duration", "passenger_count", "xgb_confidence", "avg_speed_std"]
    ].copy()

    if high_demand_zones.empty:
        data["combined_score"] = data["pickup_count"] * data["xgb_confidence"]
        high_demand_zones = data.nlargest(5, "combined_score")[
            ["PULocationID", "pickup_count", "temperature", "wind_speed", "precipitation", "is_holiday", "avg_speed", "temp_precip", "day_of_week", "trip_duration", "passenger_count", "xgb_confidence", "avg_speed_std"]
        ].copy()
        high_demand_zones["is_high_demand"] = 1

    if high_demand_zones.empty:
        st.write("No high-demand zones predicted for the current time.")
    else:
        if high_demand_zones.columns.duplicated().any():
            st.error(f"Duplicate columns found in high_demand_zones: {high_demand_zones.columns[high_demand_zones.columns.duplicated()]}")
            st.stop()

        high_demand_zones = high_demand_zones.merge(current_supply, on="PULocationID", how="left")
        high_demand_zones["driver_count"] = high_demand_zones["driver_count"].fillna(1)
        high_demand_zones["current_demand_supply_ratio"] = high_demand_zones["pickup_count"] / high_demand_zones["driver_count"]

        high_demand_zones = high_demand_zones.merge(next_hour_forecast, on="PULocationID", how="left")
        high_demand_zones["forecasted_demand"] = high_demand_zones["forecasted_demand"].fillna(0)
        high_demand_zones["demand_slope"] = high_demand_zones["demand_slope"].fillna(0)
        high_demand_zones["prophet_confidence"] = high_demand_zones["prophet_confidence"].fillna(0.5)

        high_demand_zones = high_demand_zones.merge(
            test_df[["PULocationID", "Pickup_Zone", "pickup_lat", "pickup_lon"]].drop_duplicates(),
            on="PULocationID",
            how="left"
        )

        high_demand_zones["ensemble_score"] = (
            0.5 * high_demand_zones["current_demand_supply_ratio"] +
            0.5 * high_demand_zones["forecasted_demand"] / high_demand_zones["driver_count"]
        )
        high_demand_zones["ensemble_confidence"] = (
            0.5 * high_demand_zones["xgb_confidence"] + 0.5 * high_demand_zones["prophet_confidence"]
        )
        historical_accuracy_weight = 1 - (644.59 / high_demand_zones["pickup_count"].mean() if high_demand_zones["pickup_count"].mean() != 0 else 1)
        historical_accuracy_weight = max(0, min(1, historical_accuracy_weight))
        high_demand_zones["ensemble_confidence"] = high_demand_zones["ensemble_confidence"] * historical_accuracy_weight

        if pd.notna(current_location["pickup_lat"]) and pd.notna(current_location["pickup_lon"]):
            coords_current = (current_location["pickup_lat"], current_location["pickup_lon"])
            high_demand_zones["distance"] = high_demand_zones.apply(
                lambda row: geodesic(coords_current, (row["pickup_lat"], row["pickup_lon"])).miles
                if pd.notna(row["pickup_lat"]) and pd.notna(row["pickup_lon"]) else float("inf"),
                axis=1
            )
        else:
            high_demand_zones["distance"] = float("inf")

        avg_speed = data["avg_speed"].mean()
        high_demand_zones = high_demand_zones.merge(zone_features[["PULocationID", "avg_fare"]], on="PULocationID", how="left")
        high_demand_zones["travel_time"] = (high_demand_zones["distance"] / high_demand_zones["avg_speed"] * 60).fillna(float("inf"))
        high_demand_zones["potential_trips"] = high_demand_zones["forecasted_demand"] / high_demand_zones["driver_count"]
        high_demand_zones["trips_per_hour"] = (60 / high_demand_zones["trip_duration"]).clip(upper=3)
        high_demand_zones["gross_earnings"] = high_demand_zones["trips_per_hour"] * high_demand_zones["avg_fare"]
        high_demand_zones["gross_earnings"] = high_demand_zones["gross_earnings"] * (1 + 0.1 * high_demand_zones["passenger_count"])
        high_demand_zones["fuel_cost"] = high_demand_zones["distance"] * 0.5
        idle_time = 15
        high_demand_zones["net_earnings"] = (
            high_demand_zones["gross_earnings"] * (60 - high_demand_zones["travel_time"] - idle_time) / 60 - high_demand_zones["fuel_cost"]
        )
        high_demand_zones["net_earnings"] = high_demand_zones["net_earnings"].clip(lower=0)
        high_demand_zones["risk_score"] = high_demand_zones["demand_slope"].apply(lambda x: "Low" if x >= 0 else "High")
        high_demand_zones["preference_score"] = high_demand_zones["PULocationID"].map(st.session_state.driver_preferences)
        high_demand_zones["congestion_factor"] = high_demand_zones["avg_speed_std"].fillna(0) / high_demand_zones["avg_speed"]

        if rf_earnings_model:
            missing_rf_features = [feature for feature in rf_earnings_features if feature not in high_demand_zones.columns]
            if missing_rf_features:
                st.error(f"Missing features for Random Forest earnings prediction: {missing_rf_features}")
                st.stop()
            X_rf = high_demand_zones[rf_earnings_features]
            high_demand_zones["adjusted_earnings"] = rf_earnings_model.predict(X_rf)
        else:
            high_demand_zones["adjusted_earnings"] = (
                high_demand_zones["net_earnings"] * (1 + 0.1 * high_demand_zones["preference_score"]) * (1 - 0.5 * high_demand_zones["congestion_factor"])
            )

        high_demand_zones = high_demand_zones.nlargest(5, "adjusted_earnings")

        st.markdown(f"""
            <div class="stats-card">
                <div><strong>Total Forecasted Demand:</strong> {int(high_demand_zones["forecasted_demand"].sum())} pickups</div>
                <div><strong>High-Demand Zones:</strong> {len(high_demand_zones)}</div>
            </div>
        """, unsafe_allow_html=True)

        st.write(f"**Top 5 High-Demand Zones (Weather: {weather_condition})**")
        display_cols = [
            "PULocationID", "Pickup_Zone", "pickup_count", "driver_count",
            "current_demand_supply_ratio", "forecasted_demand", "adjusted_earnings",
            "travel_time", "risk_score"
        ]

        fig_pie = go.Figure(data=[
            go.Pie(
                labels=high_demand_zones["PULocationID"].astype(str),
                values=high_demand_zones["current_demand_supply_ratio"],
                hole=0.4,
                marker_colors=px.colors.sequential.YlOrRd,
                textinfo='label+percent',
                hoverinfo='label+value'
            )
        ])
        fig_pie.update_layout(
            title="Demand-to-Supply Ratio Distribution",
            template="plotly_dark",
            title_font_size=16,
            height=400
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        st.dataframe(
            high_demand_zones[display_cols].style.format({
                "pickup_count": "{:.0f}",
                "driver_count": "{:.0f}",
                "current_demand_supply_ratio": "{:.2f}",
                "forecasted_demand": "{:.0f}",
                "adjusted_earnings": "${:.2f}",
                "travel_time": "{:.1f} min"
            }).set_table_styles([
                {'selector': 'th', 'props': [('background-color', '#FFD700'), ('color', 'black'), ('font-weight', 'bold')]},
                {'selector': 'tr:nth-child(even)', 'props': [('background-color', '#f2f2f2')]}
            ])
        )
```
- **What**: Identifies and displays the top 5 high-demand zones right now.
- **Why**: Shows drivers the busiest zones so they can decide where to go.
- **How**:
  - **Filter Data**:
    - Filters `pickup_counts` and `features` for the selected day, time, and holiday status.
    - Merges them into `data`, filling missing values with 0 or averages.
  - **Predict High-Demand Zones**:
    - Uses the classifier (XGBoost or Random Forest) to predict the probability (`probs`) that each zone is high-demand.
    - Sets a threshold (50th percentile of `probs` or 0.001) to mark zones as high-demand (`is_high_demand = 1`).
    - If no zones are high-demand, falls back to a hybrid approach: calculates a `combined_score` (`pickup_count * xgb_confidence`) and picks the top 5.
  - **Add More Data**:
    - Merges with `current_supply` to add `driver_count`.
    - Calculates `current_demand_supply_ratio` (`pickup_count / driver_count`).
    - Adds forecast data (`next_hour_forecast`).
    - Adds zone details (`Pickup_Zone`, `pickup_lat`, `pickup_lon`).
  - **Calculations**:
    - `ensemble_score`: Combines current and forecasted demand-to-supply ratios.
    - `distance`: Distance from the driver’s current location to each zone.
    - `travel_time`: Time to travel to the zone (`distance / avg_speed * 60` minutes).
    - `potential_trips`: Estimated trips per driver (`forecasted_demand / driver_count`).
    - `trips_per_hour`: Max trips per hour (`60 / trip_duration`, capped at 3).
    - `gross_earnings`: Earnings per hour (`trips_per_hour * avg_fare * (1 + 0.1 * passenger_count)`).
    - `fuel_cost`: Cost to travel (`distance * 0.5`).
    - `net_earnings`: Final earnings after costs (`gross_earnings * (60 - travel_time - idle_time) / 60 - fuel_cost`).
    - `risk_score`: "Low" if demand is increasing (`demand_slope >= 0`), "High" if decreasing.
    - `congestion_factor`: Traffic congestion (`avg_speed_std / avg_speed`).
    - `adjusted_earnings`: If `rf_earnings_model` exists, predicts earnings; otherwise, adjusts `net_earnings` based on preferences and congestion.
  - **Display**:
    - **Stats Card**: Shows total forecasted demand and number of high-demand zones.
    - **Pie Chart**: Shows the distribution of `current_demand_supply_ratio` across zones.
    - **Table**: Displays key metrics for the top 5 zones (sorted by `adjusted_earnings`).
- **Logic**:
  - The classifier tries to identify high-demand zones, but since it’s not working well (low probabilities), the hybrid approach ensures we still get reasonable zones.
  - Calculations like `net_earnings` and `adjusted_earnings` help rank zones by profitability, considering travel time, fuel costs, and traffic.

---

#### 12. Recommendations Tab
```python
with tab2:
    st.header("🚖 Where Should You Go Next?")
    current_time_secs = time.time()
    if current_time_secs - st.session_state.last_recommendation_time < 300:
        st.info("⏳ You've recently followed a recommendation. Consider waiting a bit before repositioning again to maximize earnings.")
    
    if not high_demand_zones.empty:
        if pd.notna(current_location["pickup_lat"]) and pd.notna(current_location["pickup_lon"]):
            recommended_zones = high_demand_zones["PULocationID"].tolist()
            zones_to_forecast = list(set(top_zones_forecast + recommended_zones))
            for zone in zones_to_forecast:
                if zone not in zone_forecasts and zone in zone_models:
                    zone_forecast = zone_models[zone].predict(future_dates)
                    zone_forecast["PULocationID"] = zone
                    zone_forecasts[zone] = zone_forecast
                    future_dates_no_weather = future_dates.copy()
                    future_dates_no_weather["precipitation"] = 0.0
                    future_dates_no_weather["temp_precip"] = 0.0
                    zone_forecast_no_weather = zone_models[zone].predict(future_dates_no_weather)
                    zone_forecast_no_weather["PULocationID"] = zone
                    zone_forecasts_no_weather[zone] = zone_forecast_no_weather

            preferred_nearby = high_demand_zones[high_demand_zones["preference_score"] > 0].copy()
            if not preferred_nearby.empty:
                preferred_nearby = preferred_nearby.loc[[preferred_nearby["distance"].idxmin()]]
                if preferred_nearby.iloc[0]["distance"] < 5:
                    recommended_zone = preferred_nearby
                    st.success("🌟 **Recommendation Note**: Prioritizing a nearby zone you prefer!")
                else:
                    filtered_zones = high_demand_zones[high_demand_zones["travel_time"] < 30].copy()
                    if not filtered_zones.empty:
                        idx = filtered_zones["adjusted_earnings"].idxmax()
                        recommended_zone = filtered_zones.loc[[idx]]
                    else:
                        recommended_zone = pd.DataFrame()
            else:
                filtered_zones = high_demand_zones[high_demand_zones["travel_time"] < 30].copy()
                if not filtered_zones.empty:
                    idx = filtered_zones["adjusted_earnings"].idxmax()
                    recommended_zone = filtered_zones.loc[[idx]]
                else:
                    recommended_zone = pd.DataFrame()

            if not recommended_zone.empty:
                recommended_zone["day_of_week"] = selected_day_of_week
                zone_id = recommended_zone.iloc[0]["PULocationID"]
                zone_name = recommended_zone.iloc[0]["Pickup_Zone"]
                travel_time = recommended_zone.iloc[0]["travel_time"]
                earnings = recommended_zone.iloc[0]["adjusted_earnings"]
                risk_score = recommended_zone.iloc[0]["risk_score"]
                ratio = recommended_zone.iloc[0]["current_demand_supply_ratio"]
                forecasted_demand = recommended_zone.iloc[0]["forecasted_demand"]
                potential_trips = recommended_zone.iloc[0]["potential_trips"]
                avg_fare = recommended_zone.iloc[0]["avg_fare"]
                congestion_factor = recommended_zone.iloc[0]["congestion_factor"]

                st.markdown(f"""
                    <div class="card high-demand">
                        <div class="card-header">
                            <span style="font-size: 24px;">🚕</span> Primary Recommendation: Zone {zone_id} ({zone_name})
                        </div>
                        <div class="metric">
                            <span style="font-size: 24px;">🕒</span> Travel Time: {travel_time:.1f} minutes
                        </div>
                        <div class="metric">
                            <span style="font-size: 24px;">💰</span> Net Earnings (Next Hour): ${earnings:.2f}
                        </div>
                        <div class="metric">
                            <span style="font-size: 24px;">👥</span> Estimated Passengers: {int(potential_trips)} per hour
                        </div>
                        <div class="metric">
                            <span style="font-size: 24px;">💵</span> Average Fare: ${avg_fare:.2f}
                        </div>
                        <div class="metric">
                            <span style="font-size: 24px;">⚠️</span> Risk of Demand Decreasing: <span style="color: {'green' if risk_score == 'Low' else 'red'}">{risk_score}</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                if recommended_zone.iloc[0]["demand_slope"] > 50:
                    st.warning("📈 **Demand Surge Alert**: This zone is experiencing a rapid demand increase!")

                st.markdown("<div style='display: flex; gap: 10px;'>", unsafe_allow_html=True)
                if st.button("✔️ Accept", key=f"accept_{zone_id}"):
                    st.session_state.driver_preferences[zone_id] += 1
                    st.session_state.performance_history.append(earnings)
                    st.session_state.last_recommendation_time = current_time_secs
                    st.success(f"Preference for Zone {zone_id} updated. You prefer this zone!")
                if st.button("❌ Reject", key=f"reject_{zone_id}"):
                    st.session_state.driver_preferences[zone_id] -= 1
                    st.info(f"Preference for Zone {zone_id} updated. Looking for alternatives...")
                st.markdown("</div>", unsafe_allow_html=True)

                with st.expander("🔍 See Details"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown("**Demand-to-Supply Ratio (Higher is Better)**")
                        ratio_value = min(ratio * 10, 100)
                        fig_ratio = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=ratio_value,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "#FFD700"},
                                'steps': [
                                    {'range': [0, 33], 'color': "red"},
                                    {'range': [33, 66], 'color': "yellow"},
                                    {'range': [66, 100], 'color': "green"}
                                ]
                            }
                        ))
                        fig_ratio.update_layout(height=150, margin=dict(l=20, r=20, t=20, b=20))
                        st.plotly_chart(fig_ratio, use_container_width=True)

                    with col2:
                        st.markdown("**Congestion Level (Lower is Better)**")
                        congestion_value = min(congestion_factor * 100, 100)
                        fig_congestion = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=congestion_value,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "#FFD700"},
                                'steps': [
                                    {'range': [0, 33], 'color': "green"},
                                    {'range': [33, 66], 'color': "yellow"},
                                    {'range': [66, 100], 'color': "red"}
                                ]
                            }
                        ))
                        fig_congestion.update_layout(height=150, margin=dict(l=20, r=20, t=20, b=20))
                        st.plotly_chart(fig_congestion, use_container_width=True)

                    with col3:
                        st.markdown("**Risk Level (Lower is Better)**")
                        risk_value = 0 if risk_score == "Low" else 100
                        fig_risk = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=risk_value,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "#FFD700"},
                                'steps': [
                                    {'range': [0, 50], 'color': "green"},
                                    {'range': [50, 100], 'color': "red"}
                                ]
                            }
                        ))
                        fig_risk.update_layout(height=150, margin=dict(l=20, r=20, t=20, b=20))
                        st.plotly_chart(fig_risk, use_container_width=True)

                    st.markdown("**Why This Zone? Top Contributing Factors:**")
                    feature_contributions = recommended_zone[top_features].iloc[0]
                    feature_contributions = feature_contributions.sort_values(ascending=False).head(3)
                    fig_factors = go.Figure()
                    fig_factors.add_trace(go.Bar(
                        x=feature_contributions.values,
                        y=feature_contributions.index,
                        orientation='h',
                        marker_color="#FFD700",
                        text=[f"{val:.2f}" for val in feature_contributions.values],
                        textposition='auto'
                    ))
                    fig_factors.update_layout(
                        title="Top Factors Influencing Recommendation",
                        xaxis_title="Value",
                        yaxis_title="Factor",
                        template="plotly_dark",
                        title_font_size=16,
                        height=300
                    )
                    st.plotly_chart(fig_factors, use_container_width=True)

                    for feature, value in feature_contributions.items():
                        explanation = {
                            "precipitation": "Rain or snow increases taxi demand.",
                            "is_holiday": "Holidays often see higher demand.",
                            "avg_speed": "Traffic speed impacts pickup efficiency.",
                            "day_of_week": "Day of the week affects demand patterns.",
                            "temp_precip": "Interaction of temperature and precipitation affects demand."
                        }.get(feature, "This factor contributes to demand prediction.")
                        st.markdown(f"- {feature}: {value:.2f} ({explanation})")

                    st.markdown(f"- **Current Demand-to-Supply Ratio**: {ratio:.2f}")
                    st.markdown(f"- **Forecasted Demand (Next Hour)**: {forecasted_demand:.0f} pickups")

                    st.markdown("**Historical Demand Trend for This Zone:**")
                    zone_history = pickup_counts[pickup_counts["PULocationID"] == zone_id].groupby("hour")["pickup_count"].mean().reset_index()
                    if not zone_history.empty:
                        fig_history = go.Figure()
                        fig_history.add_trace(go.Scatter(
                            x=zone_history["hour"],
                            y=zone_history["pickup_count"],
                            name="Average Pickups",
                            mode="lines+markers",
                            line=dict(color="#FFD700"),
                            fill="tozeroy",
                            fillcolor="rgba(255, 215, 0, 0.2)"
                        ))
                        fig_history.update_layout(
                            title=f"Historical Demand for Zone {zone_id}",
                            xaxis_title="Hour of Day",
                            yaxis_title="Average Pickups",
                            hovermode="x unified",
                            template="plotly_dark",
                            title_font_size=16,
                            height=300
                        )
                        st.plotly_chart(fig_history, use_container_width=True)
                    else:
                        st.write("No historical demand data available for this zone.")

                if risk_score == "High" or f"reject_{zone_id}" in st.session_state:
                    st.subheader("🔄 Alternative Recommendation")
                    alternative_zone = high_demand_zones[
                        (high_demand_zones["travel_time"] < 30) &
                        (high_demand_zones["PULocationID"] != zone_id) &
                        (high_demand_zones["risk_score"] == "Low")
                    ].copy()
                    if not alternative_zone.empty:
                        idx = alternative_zone["adjusted_earnings"].idxmax()
                        alternative_zone = alternative_zone.loc[[idx]]
                        alternative_zone["day_of_week"] = selected_day_of_week
                        alt_zone_id = alternative_zone.iloc[0]["PULocationID"]
                        alt_zone_name = alternative_zone.iloc[0]["Pickup_Zone"]
                        alt_travel_time = alternative_zone.iloc[0]["travel_time"]
                        alt_earnings = alternative_zone.iloc[0]["adjusted_earnings"]
                        alt_risk_score = alternative_zone.iloc[0]["risk_score"]
                        alt_potential_trips = alternative_zone.iloc[0]["potential_trips"]
                        alt_avg_fare = alternative_zone.iloc[0]["avg_fare"]

                        st.markdown(f"""
                            <div class="card alternative">
                                <div class="card-header">
                                    <span style="font-size: 24px;">🚖</span> Alternative: Zone {alt_zone_id} ({alt_zone_name})
                                </div>
                                <div class="metric">
                                    <span style="font-size: 24px;">🕒</span> Travel Time: {alt_travel_time:.1f} minutes
                                </div>
                                <div class="metric">
                                    <span style="font-size: 24px;">💰</span> Net Earnings (Next Hour): ${alt_earnings:.2f}
                                </div>
                                <div class="metric">
                                    <span style="font-size: 24px;">👥</span> Estimated Passengers: {int(alt_potential_trips)} per hour
                                </div>
                                <div class="metric">
                                    <span style="font-size: 24px;">💵</span> Average Fare: ${alt_avg_fare:.2f}
                                </div>
                                <div class="metric">
                                    <span style="font-size: 24px;">⚠️</span> Risk of Demand Decreasing: <span style="color: {'green' if alt_risk_score == 'Low' else 'red'}">{alt_risk_score}</span>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

                        with st.expander("🔍 See Details"):
                            alt_ratio = alternative_zone.iloc[0]["current_demand_supply_ratio"]
                            alt_forecasted_demand = alternative_zone.iloc[0]["forecasted_demand"]
                            st.markdown(f"- **Current Demand-to-Supply Ratio**: {alt_ratio:.2f}")
                            st.markdown(f"- **Forecasted Demand (Next Hour)**: {alt_forecasted_demand:.0f} pickups")

                st.subheader("📈 Demand Trend for Recommended Zone")
                zone_forecast = zone_forecasts.get(zone_id)
                if zone_forecast is not None:
                    fig_trend = go.Figure()
                    fig_trend.add_trace(go.Scatter(
                        x=zone_forecast["ds"],
                        y=zone_forecast["yhat"],
                        name=f"Forecasted Demand ({weather_condition})",
                        mode="lines",
                        line=dict(color="#FFD700")
                    ))
                    zone_forecast_no_weather = zone_forecasts_no_weather.get(zone_id)
                    if zone_forecast_no_weather is not None:
                        fig_trend.add_trace(go.Scatter(
                            x=zone_forecast_no_weather["ds"],
                            y=zone_forecast_no_weather["yhat"],
                            name="Forecasted Demand (Clear Weather)",
                            mode="lines",
                            line=dict(dash="dash", color="#A9A9A9")
                        ))
                        weather_impact = zone_forecast["yhat"].iloc[0] - zone_forecast_no_weather["yhat"].iloc[0]
                        weather_explanation = (
                            f"Rain or snow increases taxi demand due to passengers seeking shelter." if weather_impact > 0
                            else "Clear weather typically aligns with normal demand patterns."
                        )
                        st.markdown(f"**Weather Impact**: {weather_condition} weather changes demand by {weather_impact:.0f} pickups. *{weather_explanation}*")
                    fig_trend.update_layout(
                        title=f"Demand Forecast for Zone {zone_id}",
                        xaxis_title="Time",
                        yaxis_title="Pickups",
                        hovermode="x unified",
                        template="plotly_dark",
                        title_font_size=16
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)
                else:
                    st.write("No forecast available for this zone.")

                st.subheader("💸 Earnings Comparison")
                current_zone_data = high_demand_zones[high_demand_zones["PULocationID"] == current_zone] if not high_demand_zones.empty else pd.DataFrame()
                if not current_zone_data.empty:
                    stay_earnings = current_zone_data.iloc[0]["adjusted_earnings"]
                else:
                    stay_earnings = 0
                earnings_diff = earnings - stay_earnings

                fig_earnings = go.Figure()
                fig_earnings.add_trace(go.Bar(
                    x=[stay_earnings, earnings],
                    y=["Stay in Current Zone", "Move to Recommended Zone"],
                    orientation="h",
                    marker_color=["#A9A9A9", "#FFD700"],
                    text=[f"${stay_earnings:.2f}", f"${earnings:.2f}"],
                    textposition="auto"
                ))
                fig_earnings.update_layout(
                    title=f"Earnings Gain: ${earnings_diff:.2f}/hour",
                    xaxis_title="Earnings per Hour ($)",
                    yaxis_title="",
                    template="plotly_dark",
                    title_font_size=16,
                    height=300
                )
                st.plotly_chart(fig_earnings, use_container_width=True)
```
- **What**: Recommends the best zone for the driver to go to next.
- **Why**: Helps the driver decide where to go to maximize earnings, considering factors like distance, earnings, and risk.
- **How**:
  - **Cooldown Check**: If the driver recently followed a recommendation (within 5 minutes), shows a message to wait.
  - **Ensure Forecasts**: Adds forecasts for the high-demand zones if not already present.
  - **Select Recommended Zone**:
    - Prefers a nearby zone (within 5 miles) that the driver likes (`preference_score > 0`).
    - Otherwise, picks the zone with the highest `adjusted_earnings` within 30 minutes travel time.
  - **Recommendation Card**:
    - Shows the zone ID, name, travel time, net earnings, estimated passengers, average fare, and risk.
  - **Accept/Reject Buttons**:
    - "Accept": Increases the preference score for the zone, logs the earnings, and updates the last recommendation time.
    - "Reject": Decreases the preference score and looks for an alternative.
  - **Details Section**:
    - **Gauge Charts**:
      - Demand-to-Supply Ratio: Scaled to 0-100 (higher is better).
      - Congestion Level: Scaled to 0-100 (lower is better).
      - Risk Level: 0 for "Low", 100 for "High".
    - **Bar Chart**: Shows the top 3 contributing factors (e.g., `precipitation`, `avg_speed`).
    - **Historical Demand Trend**: Line chart of average pickups per hour for the zone.
  - **Alternative Recommendation**:
    - If the risk is "High" or the driver rejects the recommendation, suggests another zone with "Low" risk within 30 minutes.
  - **Demand Trend Chart**: Shows the forecasted demand for the recommended zone over the next 6 hours, with and without weather effects.
  - **Earnings Comparison**: Bar chart comparing earnings if the driver stays vs. moves to the recommended zone.
- **Logic**:
  - Prioritizes nearby preferred zones to personalize recommendations.
  - Uses `adjusted_earnings` to pick the best zone, considering distance, traffic, and driver preferences.
  - Gauge charts provide a quick visual summary of key factors.

---

#### 13. Personalized Tips Section
```python
st.header("📋 Personalized Tips to Maximize Your Earnings")
st.markdown("Here are some insights tailored to your preferences and current location:")
peak_hours = prophet_df[prophet_df["yhat"] > prophet_df["yhat"].quantile(0.8)]["ds"].dt.hour.value_counts().head(3)
st.write("**Peak Demand Hours (City-Wide):**")
for hour, count in peak_hours.items():
    st.write(f"- {hour}:00: High demand expected.")
if not high_demand_zones.empty:
    st.write("**Top High-Demand Zones Right Now:**")
    top_zones_driver = high_demand_zones.nlargest(3, "adjusted_earnings")
    for _, row in top_zones_driver.iterrows():
        st.write(f"- Zone {row['PULocationID']} ({row['Pickup_Zone']}): Potential Earnings ${row['adjusted_earnings']:.2f}/hour (Risk: {row['risk_score']})")
nearby_high_demand = high_demand_zones[high_demand_zones["travel_time"] < 15].nlargest(3, "adjusted_earnings") if not high_demand_zones.empty else pd.DataFrame()
if not nearby_high_demand.empty:
    st.write(f"- **Nearby High-Demand Zones**: The closest high-demand zone is Zone {nearby_high_demand.iloc[0]['PULocationID']} ({nearby_high_demand.iloc[0]['Pickup_Zone']}), just {nearby_high_demand.iloc[0]['travel_time']:.1f} minutes away.")
preferred_zones = high_demand_zones[high_demand_zones["preference_score"] > 0].nlargest(3, "adjusted_earnings") if not high_demand_zones.empty else pd.DataFrame()
if not preferred_zones.empty:
    st.write(f"- **Preferred Zones**: Based on your preferences, consider Zone {preferred_zones.iloc[0]['PULocationID']} ({preferred_zones.iloc[0]['Pickup_Zone']}) with earnings of ${preferred_zones.iloc[0]['adjusted_earnings']:.2f}/hour.")
else:
    st.write("- **Explore New Zones**: You haven’t shown a strong preference for any zones yet. Try accepting recommendations to find zones you like!")
nearby_hotspots = zone_features[(zone_features["label"] == "High-Demand Urban") & (zone_features["distance"] < 10)].nlargest(3, "forecasted_demand")
st.write("**Nearby High-Demand Hotspots (Within 10 Miles):**")
for _, row in nearby_hotspots.iterrows():
    st.write(f"- Zone {row['PULocationID']}: Classified as {row['label']}, Forecasted Demand: {row['forecasted_demand']:.0f}, Distance: {row['distance']:.1f} miles")

st.write("**Your Top Preferred Zones:**")
preference_df = pd.DataFrame.from_dict(st.session_state.driver_preferences, orient="index", columns=["Preference Score"])
preference_df = preference_df.nlargest(3, "Preference Score").reset_index().rename(columns={"index": "Zone"})
if not preference_df.empty:
    preference_df = preference_df.merge(zone_features[["PULocationID", "pickup_lat", "pickup_lon"]], left_on="Zone", right_on="PULocationID", how="left")
    preference_df["distance"] = preference_df.apply(
        lambda row: geodesic(
            (current_location["pickup_lat"], current_location["pickup_lon"]),
            (row["pickup_lat"], row["pickup_lon"])
        ).miles if pd.notna(row["pickup_lat"]) and pd.notna(row["pickup_lon"]) else float("inf"),
        axis=1
    )
    preference_df = preference_df[["Zone", "Preference Score", "distance"]]
    st.dataframe(
        preference_df.style.format({"distance": "{:.1f} miles"}).set_table_styles([
            {'selector': 'th', 'props': [('background-color', '#FFD700'), ('color', 'black'), ('font-weight', 'bold')]},
            {'selector': 'tr:nth-child(even)', 'props': [('background-color', '#f2f2f2')]}
        ])
    )
    if preference_df["distance"].min() < 10:
        closest_preferred = preference_df.nsmallest(1, "distance")
        st.write(f"- **Visit a Preferred Zone**: Zone {closest_preferred.iloc[0]['Zone']} is nearby ({closest_preferred.iloc[0]['distance']:.1f} miles) and matches your preferences!")
else:
    st.write("- No preferred zones yet. Try accepting recommendations!")

st.write("**Recommendation Performance:**")
if st.session_state.performance_history:
    avg_gain = np.mean(st.session_state.performance_history)
    st.write(f"- Average Earnings from Following Recommendations: ${avg_gain:.2f}/hour (based on {len(st.session_state.performance_history)} recommendations)")
else:
    st.write("- Follow recommendations to track your earnings performance!")
st.markdown("**Tip**: Follow the repositioning recommendation above to maximize earnings, or head to nearby hotspots during peak hours.")
```
- **What**: Provides personalized tips to help the driver earn more.
- **Why**: Gives additional guidance beyond the main recommendation.
- **How**:
  - **Peak Hours**: Identifies the top 3 hours with high demand (80th percentile of `yhat` in `prophet_df`).
  - **Top High-Demand Zones**: Lists the top 3 zones by `adjusted_earnings`.
  - **Nearby High-Demand Zones**: Finds zones within 15 minutes travel time.
  - **Preferred Zones**: Suggests zones the driver likes (positive `preference_score`).
  - **Nearby Hotspots**: Lists zones within 10 miles labeled "High-Demand Urban" by K-means clustering.
  - **Preferred Zones Table**: Shows the top 3 zones by `preference_score` with their distances.
  - **Recommendation Performance**: Shows the average earnings from following recommendations.
- **Logic**: Combines data-driven insights (e.g., peak hours) with personalized info (e.g., preferred zones) to give actionable advice.

---

#### 14. Visualizations Tab
```python
with tab3:
    st.header("🗺️ Hotspot Map")
    st.write("This map shows high-demand zones (red) and other zones (gray, blue, green, purple based on clustering). Marker size reflects forecasted demand.")

    col1, col2 = st.columns([3, 1])
    with col1:
        m = folium.Map(location=[40.7128, -74.0060], zoom_start=11)
        if not next_hour_forecast.empty:
            max_demand = zone_features["forecasted_demand"].max() if zone_features["forecasted_demand"].max() > 0 else 1
            if pd.notna(current_location["pickup_lat"]) and pd.notna(current_location["pickup_lon"]):
                high_demand_zone_ids = high_demand_zones["PULocationID"].tolist() if not high_demand_zones.empty else []
                
                for _, row in zone_features.iterrows():
                    if pd.notna(row["pickup_lat"]) and pd.notna(row["pickup_lon"]):
                        is_high_demand = row["PULocationID"] in high_demand_zone_ids
                        cluster_id = int(row["kmeans_cluster"])
                        
                        if is_high_demand:
                            color = "red"
                        else:
                            color = "gray" if cluster_id == -1 else ["blue", "green", "purple", "orange"][cluster_id % 4]
                        
                        marker_size = 5 + (row["forecasted_demand"] / max_demand * 10)
                        folium.CircleMarker(
                            location=[row["pickup_lat"], row["pickup_lon"]],
                            radius=marker_size,
                            color=color,
                            fill=True,
                            popup=f"Zone {row['PULocationID']}: {row['label']}, Forecasted Demand: {row['forecasted_demand']:.0f}"
                        ).add_to(m)
                folium_static(m)
            else:
                st.write("Current location coordinates are not available. Please select a different zone.")
        else:
            st.write("No forecasted demand data available for the hotspot map.")
    
    with col2:
        st.markdown("**Legend**")
        st.markdown("- 🟥 **Red**: High-Demand Zones (Current)")
        st.markdown("- ⬜ **Gray**: Low-Demand Zones")
        st.markdown("- 🟦 **Blue**, 🟩 **Green**, 🟪 **Purple**, 🟧 **Orange**: Clustered Zones")
        if not high_demand_zones.empty:
            st.markdown(f"**High-Demand Zones**: {len(high_demand_zones)}")
        st.markdown(f"**Total Forecasted Demand**: {int(zone_features['forecasted_demand'].sum())} pickups")

        st.markdown("**Demand Distribution Across Zones**")
        demand_dist = zone_features.groupby("label")["forecasted_demand"].sum().reset_index()
        fig_dist = go.Figure(data=[
            go.Pie(
                labels=demand_dist["label"],
                values=demand_dist["forecasted_demand"],
                hole=0.4,
                marker_colors=px.colors.sequential.YlOrRd,
                textinfo='label+percent',
                hoverinfo='label+value'
            )
        ])
        fig_dist.update_layout(
            title="Demand by Zone Type",
            template="plotly_dark",
            title_font_size=16,
            height=300
        )
        st.plotly_chart(fig_dist, use_container_width=True)
```
- **What**: Displays a map and a chart to visualize demand.
- **Why**: Helps drivers see where demand is high on a map and understand demand distribution.
- **How**:
  - **Map**:
    - Centers on NYC (coordinates 40.7128, -74.0060).
    - Marks each zone with a circle:
      - Red for high-demand zones (from `high_demand_zones`).
      - Gray for low-demand zones, or colored (blue, green, purple, orange) based on K-means clusters.
      - Size of the circle reflects `forecasted_demand` (scaled relative to the max demand).
  - **Legend**: Explains the colors used on the map.
  - **Stats**: Shows the number of high-demand zones and total forecasted demand.
  - **Pie Chart**: Shows the distribution of `forecasted_demand` across zone types (e.g., "High-Demand Urban").
- **Logic**: The map provides a visual way to see high-demand areas, while the pie chart gives an overview of demand by zone type.

---

#### 15. Technical Dashboard (Continued)
The **Technical Dashboard** is designed for technical users (e.g., data scientists or developers) to analyze the performance of the AI models used in the project. It provides detailed visualizations, metrics, and simulations to evaluate how well the models are working.

```python
elif page == "Technical Dashboard":
    st.title("Technical Dashboard 📊")
    st.markdown("This page provides detailed results, visualizations, and impact analysis of the AI models used in the NYC Taxi Optimization project.")

    st.sidebar.header("Visualization Options")
    forecast_date_range = st.sidebar.date_input(
        "Select Forecast Date Range",
        [prophet_df["ds"].min().date(), prophet_df["ds"].max().date()],
        key="tech_date_range"
    )
    selected_zone = st.sidebar.selectbox("Select Zone for Forecast", options=["City-Wide"] + list(top_zones), key="tech_zone")

    start_date, end_date = pd.to_datetime(forecast_date_range[0]), pd.to_datetime(forecast_date_range[1])
    filtered_prophet_df = prophet_df[(prophet_df["ds"] >= start_date) & (prophet_df["ds"] <= end_date)]

    st.header("City-Wide Demand Forecast (Prophet)")
    st.write(f"Prophet Model Metrics: MAE=644.59, MAPE=27.22%, Median APE=15.28%")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_prophet_df["ds"], y=filtered_prophet_df["y"], name="Actual", mode="lines"))
    fig.add_trace(go.Scatter(x=filtered_prophet_df["ds"], y=filtered_prophet_df["yhat"], name="Predicted", mode="lines", line=dict(dash="dash")))
    fig.update_layout(title="City-Wide Demand: Actual vs Predicted", xaxis_title="Date", yaxis_title="Pickups", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.header(f"Zone-Specific Demand Forecast: {selected_zone}")
    if selected_zone == "City-Wide":
        st.write("Showing city-wide forecast above.")
    else:
        zone = int(selected_zone)
        if zone in zone_models:
            zone_model = zone_models[zone]
            zone_data = test_df[test_df["PULocationID"] == zone].groupby(test_df["pickup_datetime"].dt.floor("h"))["VendorID"].count().reset_index()
            zone_data.columns = ["ds", "y"]
            zone_data = zone_data[(zone_data["ds"] >= start_date) & (zone_data["ds"] <= end_date)]
            if not zone_data.empty:
                zone_forecast = zone_model.predict(zone_data[["ds"]])
                zone_forecast = zone_forecast[["ds", "yhat"]].merge(zone_data, on="ds", how="left")
                fig_zone = go.Figure()
                fig_zone.add_trace(go.Scatter(x=zone_forecast["ds"], y=zone_forecast["y"], name="Actual", mode="lines"))
                fig_zone.add_trace(go.Scatter(x=zone_forecast["ds"], y=zone_forecast["yhat"], name="Predicted", mode="lines", line=dict(dash="dash")))
                fig_zone.update_layout(title=f"Demand Forecast for Zone {zone}", xaxis_title="Date", yaxis_title="Pickups", hovermode="x unified")
                st.plotly_chart(fig_zone, use_container_width=True)
            else:
                st.write("No data available for the selected date range.")
        else:
            st.write("Zone model not available.")

    st.header("Demand Hotspot Clusters (K-means)")
    st.write("K-means Clustering: Silhouette Score = 0.86, 4 Clusters")
    m = folium.Map(location=[40.7128, -74.0060], zoom_start=11)
    for _, row in zone_features.iterrows():
        if pd.notna(row["pickup_lat"]) and pd.notna(row["pickup_lon"]):
            cluster_id = int(row["kmeans_cluster"])
            color = "gray" if cluster_id == -1 else ["red", "blue", "green", "purple"][cluster_id % 4]
            folium.CircleMarker(
                location=[row["pickup_lat"], row["pickup_lon"]],
                radius=5,
                color=color,
                fill=True,
                popup=f"Zone {row['PULocationID']}: {row['label']}"
            ).add_to(m)
    folium_static(m)

    st.header("Model Comparison Summary")
    st.write("Performance metrics for all models used in the project:")
    st.dataframe(comparison_df)

    st.header("Impact Simulation")
    st.write("Simulating the impact of repositioning on passenger wait times and driver earnings (baseline vs. AI-guided):")
    high_demand_zones_sim = pickup_counts[pickup_counts["is_holiday"] == 1][["PULocationID", "pickup_count"]] if "is_holiday" in pickup_counts else pd.DataFrame()
    if not high_demand_zones_sim.empty:
        high_demand_zones_sim = high_demand_zones_sim.merge(current_supply, on="PULocationID", how="left")
        high_demand_zones_sim["driver_count"] = high_demand_zones_sim["driver_count"].fillna(1)
        high_demand_zones_sim["demand_supply_ratio"] = high_demand_zones_sim["pickup_count"] / high_demand_zones_sim["driver_count"]
        baseline_earnings = high_demand_zones_sim["pickup_count"].sum() * zone_features["total_amount"].mean() / current_supply["driver_count"].mean()
        ai_guided_earnings = high_demand_zones_sim.nlargest(3, "demand_supply_ratio")["pickup_count"].sum() * zone_features["total_amount"].mean() / 3
        wait_time_reduction = (high_demand_zones_sim["demand_supply_ratio"].mean() - 1) * 5
        st.write(f"- **Baseline Earnings (Random Positioning)**: ${baseline_earnings:.2f} per driver")
        st.write(f"- **AI-Guided Earnings (Top 3 Zones)**: ${ai_guided_earnings:.2f} per driver")
        st.write(f"- **Estimated Wait Time Reduction**: {wait_time_reduction:.2f} minutes per passenger")
    else:
        st.write("No high-demand zones available for impact simulation.")

    st.markdown("---")
    st.markdown("**Conclusion**: The AI models (Prophet, XGBoost, K-means, Random Forest) provide robust predictions and clustering for demand optimization. The impact simulation demonstrates significant improvements in earnings and wait times.")
```
- **What**: Displays technical details about the models used in the dashboard.
- **Why**: Helps technical users evaluate the performance of the AI models (e.g., Prophet, XGBoost, K-means) and understand their impact.
- **How**:
  - **Sidebar Inputs**:
    - **Date Range**: Allows the user to select a date range to filter the data (defaults to the full range in `prophet_df`).
    - **Zone Selection**: Choose between "City-Wide" or a specific zone to view forecasts.
  - **City-Wide Demand Forecast**:
    - Shows a line chart comparing actual (`y`) vs. predicted (`yhat`) demand from `prophet_df`.
    - Displays metrics: MAE (Mean Absolute Error = 644.59), MAPE (Mean Absolute Percentage Error = 27.22%), Median APE (Median Absolute Percentage Error = 15.28%).
  - **Zone-Specific Demand Forecast**:
    - If "City-Wide" is selected, refers to the chart above.
    - Otherwise, aggregates historical data for the selected zone, forecasts demand using `zone_models`, and shows actual vs. predicted demand.
  - **Demand Hotspot Clusters**:
    - Displays a map with zones colored by K-means clusters (red, blue, green, purple, or gray for unclustered zones).
    - Shows a silhouette score (0.86, indicating good clustering quality) and the number of clusters (4).
  - **Model Comparison Summary**:
    - Displays a table (`comparison_df`) with performance metrics for all models (e.g., accuracy, MAE).
  - **Impact Simulation**:
    - Filters `pickup_counts` for holidays and calculates demand-to-supply ratio.
    - **Baseline Earnings**: Total pickups * average fare / average drivers (random positioning).
    - **AI-Guided Earnings**: Total pickups in top 3 zones * average fare / 3 (AI-guided positioning).
    - **Wait Time Reduction**: Estimates reduced wait time for passengers (`(demand_supply_ratio - 1) * 5` minutes).
  - **Conclusion**: Summarizes that the models improve earnings and wait times.
- **Logic**:
  - The Technical Dashboard is for debugging and evaluation, showing raw model outputs and simulations.
  - Metrics like MAE and MAPE quantify prediction accuracy, while the simulation shows real-world impact.

---

### Summary of All Features, Logic, and Calculations

#### Features of the Driver Dashboard
1. **Demand Insights Tab**:
   - **Demand-to-Supply Ratio Trends**: A stacked area chart showing how the demand-to-supply ratio changes over the next 6 hours for the top 5 zones by forecasted demand.
   - **High-Demand Zones**:
     - Lists the top 5 zones by `adjusted_earnings`.
     - Shows a pie chart of demand-to-supply ratio distribution.
     - Displays a table with zone details (e.g., `pickup_count`, `driver_count`, `adjusted_earnings`).

2. **Recommendations Tab**:
   - **Primary Recommendation**: Suggests the best zone based on `adjusted_earnings`, prioritizing nearby preferred zones.
   - **Details Section**:
     - Gauge charts for demand-to-supply ratio, congestion level, and risk level.
     - Bar chart of top contributing factors.
     - Historical demand trend for the zone.
   - **Alternative Recommendation**: Suggests another zone if the risk is high or the driver rejects the primary recommendation.
   - **Demand Trend**: Line chart of forecasted demand for the recommended zone.
   - **Earnings Comparison**: Bar chart comparing earnings if the driver stays vs. moves.
   - **Personalized Tips**: Provides additional guidance (e.g., peak hours, nearby hotspots).

3. **Visualizations Tab**:
   - **Hotspot Map**: Shows zones on a map, colored by demand and clustering, with marker size reflecting forecasted demand.
   - **Demand Distribution**: Pie chart of forecasted demand by zone type.

#### Features of the Technical Dashboard
- **City-Wide Forecast**: Compares actual vs. predicted demand with error metrics.
- **Zone-Specific Forecast**: Shows demand forecasts for a selected zone.
- **K-means Clusters**: Map of zones colored by clusters.
- **Model Comparison**: Table of model performance metrics.
- **Impact Simulation**: Compares earnings and wait times with vs. without AI guidance.

#### Key Logic and Calculations
1. **Demand Forecasting**:
   - Uses Prophet models to forecast demand (`yhat`) for the next 6 hours.
   - Calculates `demand_slope` to determine if demand is increasing or decreasing.

2. **High-Demand Zone Selection**:
   - Attempts to use a classifier (XGBoost or Random Forest) to predict high-demand zones.
   - Falls back to a hybrid approach (`pickup_count * xgb_confidence`) due to low classifier probabilities.

3. **Earnings Calculations**:
   - `trips_per_hour = 60 / trip_duration` (capped at 3).
   - `gross_earnings = trips_per_hour * avg_fare * (1 + 0.1 * passenger_count)`.
   - `fuel_cost = distance * 0.5`.
   - `net_earnings = gross_earnings * (60 - travel_time - idle_time) / 60 - fuel_cost`.
   - `adjusted_earnings`: Either predicted by `rf_earnings_model` or adjusted based on preferences and congestion.

4. **Demand-to-Supply Ratio**:
   - `current_demand_supply_ratio = pickup_count / driver_count`.
   - Forecasted ratio: `forecasted_demand / forecasted_driver_count`.

5. **Distance and Travel Time**:
   - `distance`: Calculated using `geopy.distance.geodesic` (in miles).
   - `travel_time = distance / avg_speed * 60` (in minutes).

6. **Risk and Congestion**:
   - `risk_score`: "Low" if `demand_slope >= 0`, "High" otherwise.
   - `congestion_factor = avg_speed_std / avg_speed`.

#### Why This Structure?
- **Modularity**: Each tab focuses on a specific task (insights, recommendations, visualizations), making the dashboard easy to navigate.
- **User-Centric Design**: The Driver Dashboard prioritizes driver-relevant info (earnings, travel time), while the Technical Dashboard provides detailed metrics for analysis.
- **Efficiency**: Uses caching (`@st.cache_data`, `@st.cache_resource`) to avoid reloading data or models unnecessarily.
- **Personalization**: Tracks driver preferences and performance to tailor recommendations.

#### How It All Works Together
- **Data Preparation**: Loads and summarizes data to create tables like `pickup_counts`, `features`, and `zone_features`.
- **Model Predictions**: Uses Prophet for demand forecasts, XGBoost/Random Forest for high-demand zones, and K-means for clustering.
- **User Inputs**: Allows the driver to customize the scenario (e.g., day, time, weather).
- **Visualizations**: Presents data in charts, maps, and tables to make it easy to understand.
- **Recommendations**: Combines all data to suggest the best zone, considering earnings, distance, and risk.

---

### Answering Common Questions
1. **Why are the classifier probabilities so low?**
   - The classifier (XGBoost/Random Forest) was likely trained on data with different feature scales or distributions. For example, the `precipitation` values in the dashboard (0.0, 0.5, 1.0) might not match the training data (e.g., 0 to 100 mm). This mismatch causes the classifier to output low probabilities. The hybrid approach (`pickup_count * xgb_confidence`) ensures we still get reasonable high-demand zones.

2. **How are earnings calculated?**
   - Earnings are calculated in multiple steps:
     - Estimate how many trips a driver can make per hour (`trips_per_hour`).
     - Calculate gross earnings per trip (`avg_fare * (1 + 0.1 * passenger_count)`).
     - Adjust for travel time and idle time (15 minutes assumed per hour).
     - Subtract fuel costs (`distance * 0.5`).
     - Adjust further based on preferences and congestion if `rf_earnings_model` is unavailable.

3. **What does the demand-to-supply ratio mean?**
   - It’s the number of pickups (or forecasted pickups) divided by the number of drivers in a zone. A high ratio (e.g., 5) means there are many passengers per driver, making it easier to find rides and earn more.

4. **Why do we forecast driver supply?**
   - To calculate the demand-to-supply ratio over time, we need to estimate how many drivers will be in each zone in the future. We use historical averages (from `supply_data`) and assume the daily pattern repeats.

5. **What’s the purpose of the gauge charts?**
   - The gauge charts provide a quick visual summary of key factors:
     - **Demand-to-Supply Ratio**: Shows how busy the zone is (higher is better).
     - **Congestion Level**: Shows traffic congestion (lower is better).
     - **Risk Level**: Indicates if demand might decrease (lower is better).

---

### Final Thoughts
The `dashboard.py` script is a comprehensive tool for NYC taxi drivers, combining data analysis, machine learning, and visualization to provide actionable insights. Despite the classifier issue, the hybrid approach ensures usability, and the personalized features (e.g., preferences, tips) make it driver-friendly. The Technical Dashboard adds value for developers by providing model evaluation tools.
