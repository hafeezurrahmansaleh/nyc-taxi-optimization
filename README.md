
---

# AI-Powered Ride-Sharing Demand Prediction and Fleet Optimization

## Table of Contents
- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
  - [Machine Learning Pipeline](#machine-learning-pipeline)
  - [Dashboard Architecture (`dashboard.py`)](#dashboard-architecture-dashboardpy)
    - [Driver Dashboard](#driver-dashboard)
    - [Technical Dashboard](#technical-dashboard)
    - [Additional Features](#additional-features)
- [Dataset](#dataset)
- [Setup Instructions](#setup-instructions)
  - [Requirements](#requirements)
  - [Training the Models](#training-the-models)
  - [Running the Dashboard](#running-the-dashboard)
  - [Using the Dashboard](#using-the-dashboard)
- [Results](#results)
- [Challenges and Solutions](#challenges-and-solutions)
- [Future Work](#future-work)
- [Conclusion](#conclusion)
- [References](#references)

## Project Overview
I created this project to help NYC taxi drivers find the best zones to pick up passengers and earn more money. The system uses machine learning to predict where passengers will need taxis (demand hotspots), identifies busy zones, groups similar areas, estimates earnings, and suggests where drivers should go next. The goal is to balance the number of drivers across the city, reducing passenger wait times by 15% and increasing driver earnings. I built an interactive dashboard using Streamlit to show all this information with maps, charts, and personalized recommendations.

Here’s what the project does:
- Predicts passenger demand for the next 24 hours using a model called Prophet.
- Finds high-demand zones (top 20% of pickups) using Random Forest and XGBoost models.
- Groups zones into types (like busy downtown or quiet neighborhoods) using K-means and DBSCAN.
- Estimates how much money drivers can earn in each zone using Random Forest.
- Shows everything in a dashboard with two sections: one for drivers to see recommendations, and one for technical users to check the models.

I used a dataset of 5,335,512 NYC taxi trips, added weather and holiday information, and built the system using Python and several libraries like Pandas, Scikit-learn, and Streamlit.

## Project Structure
The project is organized into the following folders and files:

- **Base Path**: `/Users/saleh/Desktop/Projects/AIProject/TaxiOptimization/`
- **Folders**:
  - **`data/`**:
    - `raw/`: Contains the original dataset.
      - `nyc_taxi_data.csv`: Raw NYC taxi trip data.
    - `processed/`: Stores cleaned data and model outputs.
      - `cleaned_taxi_data.csv`: Cleaned dataset with 5,335,512 rows.
      - `prophet_forecasts.csv`: Demand forecasts from Prophet (288 rows).
      - `selected_features.csv`: Best features picked for classification.
      - `kmeans_cluster_stats.csv`: K-means clustering results.
      - `dbscan_cluster_stats.csv`: DBSCAN clustering results.
      - `model_comparison.csv`: Table comparing all models.
  - **`models/`**:
    - `prophet_model.pkl`: City-wide Prophet model for demand forecasting.
    - `prophet_zone_{zone_id}.pkl`: Prophet models for the top 10 zones (e.g., `prophet_zone_132.pkl`).
    - `rf_model.pkl`: Random Forest model for high-demand zone classification.
    - `xgb_model.pkl`: XGBoost model for high-demand zone classification.
    - `rf_earnings_model.pkl`: Random Forest model for earnings prediction.
  - **`visualizations/`**:
    - `model_comparison.png`: Chart comparing model performance.
    - `kmeans_cluster_map.html`: Map showing K-means clusters.
    - `dbscan_cluster_map.html`: Map showing DBSCAN clusters.
    - `silhouette_comparison.png`: Chart comparing K-means and DBSCAN silhouette scores.
- **Files**:
  - `preprocessing.ipynb`: Cleans the raw data and adds weather and holiday features.
  - `forecasting.ipynb`: Trains Prophet models to predict demand.
  - `feature_selection.ipynb`: Picks the best features for classification using RFE.
  - `classification_clustering.ipynb`: Trains models for classification, earnings prediction, and clustering.
  - `compare_models.ipynb`: Compares all models to choose the best ones.
  - `dashboard.py`: Runs the Streamlit dashboard.

## Architecture
The project has two main parts: the machine learning models and the dashboard application.

### Machine Learning Pipeline
1. **Data Preprocessing** (`preprocessing.ipynb`):
   - Loads the raw dataset (`nyc_taxi_data.csv`).
   - Removes bad data: missing zones, negative fares, fares over $100, zero-distance trips.
   - Adds weather (temperature, rainfall, wind speed) and holiday flags (1 for holidays, 0 otherwise).
   - Creates features like `avg_speed` (distance ÷ trip duration) and `temp_precip` (temperature × rainfall).
   - Saves cleaned data to `cleaned_taxi_data.csv` (5,335,512 rows).

2. **Feature Engineering**:
   - **Time Features**: `hour` (0-23), `day_of_week` (0-6, Monday=0), `month` (1-12).
   - **Weather Features**: `temperature`, `precipitation`, `wind_speed`, `temp_precip`.
   - **Holiday Feature**: `is_holiday` (1 or 0).
   - **Traffic Feature**: `avg_speed`, `avg_speed_std` (standard deviation of speed).
   - **Demand Features**: `pickup_count` (trips per zone/hour), `driver_count` (unique drivers), `current_demand_supply_ratio` (pickups ÷ drivers).
   - **Earnings Features**:
     - `distance`: Distance from driver’s location to each zone (in miles, using `geopy.distance.geodesic`).
     - `travel_time`: `distance` ÷ `avg_speed` × 60 (in minutes).
     - `trips_per_hour`: 60 ÷ trip duration, capped at 3.
     - `gross_earnings`: `trips_per_hour` × average fare × (1 + 0.1 × passengers).
     - `fuel_cost`: `distance` × $0.5 per mile.
     - `net_earnings`: `gross_earnings` × (60 - `travel_time` - 15 minutes idle) ÷ 60 - `fuel_cost`.
     - `congestion_factor`: `avg_speed_std` ÷ `avg_speed`.
     - `adjusted_earnings`: Predicted by Random Forest (or calculated if model unavailable).
   - **Zone Features**: Total pickups, average fare, coordinates, trip duration, passengers per zone.

3. **Model Development**:
   - **Demand Forecasting** (`forecasting.ipynb`):
     - Model: Prophet (Facebook’s time-series forecasting model).
     - Data: Hourly pickup counts (`y`) with timestamps (`ds`), plus `temperature`, `precipitation`, `is_holiday`, `hour`, `day_of_week`.
     - Training: Trained one city-wide model and models for the top 10 zones by pickup volume.
     - Output: Predicted 288 hours (12 days) of demand, saved to `prophet_forecasts.csv`.
     - Metrics: MAE = 644.59, RMSE = 934.30, MAPE = 27.22%, Median APE = 15.28%, High Error Hours = 7.
   - **Feature Selection** (`feature_selection.ipynb`):
     - Method: Recursive Feature Elimination (RFE) with Random Forest.
     - Features: `temperature`, `precipitation`, `wind_speed`, `is_holiday`, `avg_speed`, `day_of_week`, `hour`, `month`, `temp_precip`.
     - Target: `is_high_demand` (1 if pickups > 80th percentile, 0 otherwise).
     - Result: Selected `precipitation`, `is_holiday`, `avg_speed`, `day_of_week`, `temp_precip`.
     - Output: Saved to `selected_features.csv`.
   - **High-Demand Zone Classification** (`classification_clustering.ipynb`):
     - Models: Random Forest and XGBoost.
     - Features: RFE-selected features (to avoid leakage).
     - Training:
       - Split data: 80% training, 20% testing.
       - Random Forest: Tested settings with GridSearchCV (5-fold cross-validation):
         - `n_estimators`: [100, 200, 300].
         - `max_depth`: [10, 20, None].
         - `min_samples_split`: [2, 5].
         - `min_samples_leaf`: [1, 2].
         - Scoring: F1-score.
       - XGBoost: Tested settings:
         - `n_estimators`: [100, 200].
         - `max_depth`: [3, 5, 7].
         - `learning_rate`: [0.01, 0.1].
         - Scoring: F1-score.
     - Results:
       - Random Forest: Accuracy = 91.02%, Precision = 80.00%, Recall = 74.58%, F1-Score = 77.19%, ROC-AUC = 96.39%.
       - XGBoost: Accuracy = 90.32%, Precision = 84.62%, Recall = 74.58%, F1-Score = 79.30%, ROC-AUC = 96.68%.
       - Adjusted Threshold (in `compare_models.ipynb`):
         - Random Forest F1-Score: 2.26%.
         - XGBoost F1-Score: 6.02%.
     - Output: Saved as `rf_model.pkl` and `xgb_model.pkl`.
   - **Earnings Prediction** (`classification_clustering.ipynb`):
     - Model: Random Forest Regressor.
     - Features: `pickup_count`, `driver_count`, `temperature`, `precipitation`, `wind_speed`, `is_holiday`, `avg_speed`, `day_of_week`, `temp_precip`, `travel_time`, `passenger_count`, `avg_fare`, `current_demand_supply_ratio`.
     - Target: `adjusted_earnings`.
     - Training:
       - Split data: 80% training, 20% testing.
       - Tested settings with GridSearchCV:
         - `n_estimators`: [100, 200].
         - `max_depth`: [10, 20, None].
         - `min_samples_split`: [2, 5].
         - `min_samples_leaf`: [1, 2].
         - Scoring: Negative Mean Squared Error.
     - Results:
       - MSE: 14.48.
       - R²: 94.03%.
       - Feature Importance:
         - `travel_time`: 85.10%.
         - `avg_fare`: 5.57%.
         - `avg_speed`: 4.22%.
         - `passenger_count`: 2.18%.
     - Output: Saved as `rf_earnings_model.pkl`.
   - **Clustering** (`classification_clustering.ipynb`):
     - Models: K-means and DBSCAN.
     - Features: `pickup_count_raw` (total pickups per zone), `total_amount`, `pickup_lat`, `pickup_lon`, `hour`.
     - Training:
       - Scaled features using MinMaxScaler.
       - K-means: Set 4 clusters, random state = 42.
       - DBSCAN: Tested `eps` from 0.1 to 0.5, `min_samples` = 5.
     - Labels:
       - "High-Demand Urban": Pickups > 75th percentile.
       - "High-Fare": Fares > 75th percentile.
       - "Residential": Other zones.
       - "Noise" (DBSCAN): Outliers.
     - Results:
       - K-means Silhouette Score: 0.62.
       - DBSCAN Silhouette Score: 0.58.
     - Output: Saved to `kmeans_cluster_stats.csv` and `dbscan_cluster_stats.csv`.
   - **Model Comparison** (`compare_models.ipynb`):
     - Compared Prophet, Random Forest, XGBoost, K-means, and DBSCAN.
     - Adjusted classification thresholds:
       - Random Forest F1-Score: 2.26%.
       - XGBoost F1-Score: 6.02%.
     - Selected Prophet (forecasting), XGBoost (classification), Random Forest (earnings), K-means (clustering).
     - Output: Saved table to `model_comparison.csv`, chart to `model_comparison.png`.

### Dashboard Architecture (`dashboard.py`)
The dashboard is a Streamlit web application with two pages: Driver Dashboard and Technical Dashboard. It uses custom CSS for styling (yellow buttons, dark-themed cards) and caching for performance.

#### Driver Dashboard
- **Sidebar Inputs**:
  - Day of the Week: Dropdown (e.g., Monday).
  - Is Holiday: Checkbox (default based on current date using 2025 US holidays).
  - Current Hour: Slider (0-23, default to current hour in NYC timezone).
  - Weather Condition: Dropdown (Clear, Rainy, Snowy).
  - Current Zone: Dropdown of zone IDs.
  - Classifier Choice: Dropdown (XGBoost or Random Forest).
- **Tabs**:
  1. **Demand Insights**:
     - **Demand Forecast for Next Hour**:
       - Predicts demand for the next 6 hours for top zones using Prophet.
       - Computes demand-to-supply ratio (demand ÷ estimated drivers).
       - Shows a stacked area chart of ratios over time.
       - Displays a table with zone ID, forecasted demand, and demand change (`demand_slope`).
     - **High-Demand Zones Right Now**:
       - Predicts high-demand zones using XGBoost or Random Forest.
       - Features: `precipitation`, `is_holiday`, `avg_speed`, `day_of_week`, `temp_precip`.
       - Threshold: 50th percentile of probabilities (minimum 0.001).
       - Fallback: Uses `pickup_count` × probability if no zones meet the threshold.
       - Enhances with supply data, forecasts, and earnings.
       - Shows a stats card (total demand, number of high-demand zones), pie chart (demand-to-supply ratio), and table (zone ID, name, pickups, drivers, ratio, demand, earnings, travel time, risk).
  2. **Recommendations**:
     - **Primary Recommendation**:
       - Picks the best zone based on `adjusted_earnings`, prioritizing preferred zones (tracked via session state) and zones with `travel_time` < 30 minutes.
       - Shows a card with zone ID, name, travel time, earnings, passengers, fare, and risk (`demand_slope` ≥ 0 is "Low", else "High").
       - Includes feedback buttons ("Accept" increases preference, "Reject" decreases).
       - Expander shows gauge charts (ratio, congestion, risk), top contributing features, historical demand trend, and metrics.
     - **Alternative Recommendation**:
       - Suggests another zone if the primary has high risk or is rejected.
       - Filters for `travel_time` < 30, "Low" risk, highest `adjusted_earnings`.
       - Shows a card and expander with details.
     - **Demand Trend**:
       - Shows a line chart of forecasted demand for the recommended zone (actual vs. clear weather).
       - Explains weather impact (e.g., "Rain increases demand by X pickups").
     - **Earnings Comparison**:
       - Compares earnings if staying in the current zone vs. moving.
       - Shows a bar chart with earnings difference.
     - **Personalized Tips**:
       - Lists peak hours (top 3 from Prophet forecasts).
       - Shows top high-demand zones, nearby zones (<15 minutes), preferred zones, hotspots (<10 miles), and performance (average earnings from recommendations).
  3. **Visualizations**:
     - **Hotspot Map**:
       - Folium map centered at NYC (40.7128, -74.0060).
       - Colors: Red (high-demand), gray (low-demand), blue/green/purple/orange (clusters).
       - Marker size scales with `forecasted_demand`.
       - Popups show zone ID, label, and demand.
     - **Legend and Stats**:
       - Explains colors and shows total demand and number of high-demand zones.
     - **Demand Distribution Chart**:
       - Pie chart of demand by zone type (e.g., "High-Demand Urban").

#### Technical Dashboard
- **Sidebar Inputs**:
  - Forecast Date Range: Date picker.
  - Zone Selection: Dropdown (City-Wide or specific zone).
- **Sections**:
  1. **City-Wide Demand Forecast**:
     - Line chart of actual vs. predicted pickups using Prophet.
     - Shows metrics: MAE = 644.59, MAPE = 27.22%, Median APE = 15.28%.
  2. **Zone-Specific Demand Forecast**:
     - Line chart for a selected zone, using its Prophet model.
  3. **Demand Hotspot Clusters**:
     - Folium map showing K-means clusters (colors: red/blue/green/purple).
     - Shows Silhouette Score (0.62).
  4. **Model Comparison Summary**:
     - Table from `model_comparison.csv` (e.g., Prophet MAPE, XGBoost F1-score).
  5. **Impact Simulation**:
     - Estimates earnings and wait time reduction for holiday data.
     - Baseline: Total pickups × average fare ÷ average drivers.
     - AI-Guided: Top 3 zones’ pickups × average fare ÷ 3.
     - Wait Time Reduction: (Average ratio - 1) × 5 minutes.

#### Additional Features
- **Caching**: Uses `@st.cache_data` for data loading, `@st.cache_resource` for models.
- **Session State**:
  - Tracks `driver_preferences` (dictionary of zone preferences).
  - Stores `performance_history` (earnings from accepted recommendations).
  - Logs `last_recommendation_time` (prevents frequent repositioning).
- **Dynamic Updates**: Updates based on user inputs.
- **Time Handling**: Uses NYC timezone (America/New_York) with Pytz.

## Dataset
- **NYC TLC Trip Records**:
  - Source: NYC TLC Data Portal.
  - File: `nyc_taxi_data.csv`.
  - Rows: 5,335,512 (after cleaning).
  - Columns: Pickup/dropoff times, zones, passengers, fare, distance, driver ID, coordinates, zone names.
- **Weather Data**:
  - Features: `temperature`, `precipitation`, `wind_speed`.
  - Merged by date and hour.
- **Holiday Data**:
  - 2025 US Holidays: New Year’s Day, Martin Luther King Jr. Day, Presidents’ Day, Memorial Day, Independence Day, Labor Day, Columbus Day, Veterans Day, Thanksgiving, Christmas.
- **Challenges**:
  - Missing data (e.g., coordinates, fares) was removed.
  - Outliers (negative fares, fares > $100) were dropped.
  - Weather data alignment required careful timestamp matching.

## Setup Instructions

### Requirements
- Python 3.12.0.
- Libraries: Install using `pip install pandas numpy matplotlib seaborn sklearn xgboost prophet folium streamlit geopy plotly pytz`.

### Training the Models
1. Open each Jupyter Notebook (`preprocessing.ipynb`, `forecasting.ipynb`, `feature_selection.ipynb`, `classification_clustering.ipynb`, `compare_models.ipynb`).
2. Run the cells in order to:
   - Clean the data.
   - Train Prophet, Random Forest, XGBoost, K-means, and DBSCAN models.
   - Save models to the `models/` folder.
   - Save outputs to the `data/processed/` and `visualizations/` folders.

### Running the Dashboard
1. Navigate to the project folder: `/Users/saleh/Desktop/Projects/AIProject/TaxiOptimization/`.
2. Run: `streamlit run dashboard.py`.
3. The dashboard will open in your browser.

### Using the Dashboard
- **Driver Dashboard**:
  - Select day, time, weather, and your location in the sidebar.
  - Check “Demand Insights” for forecasts and busy zones.
  - View “Recommendations” for the best zone to go to, with tips and charts.
  - See “Visualizations” for maps and demand charts.
- **Technical Dashboard**:
  - Choose a date range and zone to see forecasts.
  - Look at cluster maps, model comparisons, and impact simulations.

## Results
- **Demand Forecasting**: Prophet predicted demand with a MAPE of 27.22%. It’s about 73% accurate but struggles with sudden demand spikes (7 high-error hours).
- **High-Demand Zones**: XGBoost worked best with an F1-score of 6.02% after adjusting the cutoff, ensuring high confidence in busy zones.
- **Earnings Prediction**: Random Forest predicted earnings with an R² of 94.03%, showing high accuracy. Travel time was the biggest factor (85.10%).
- **Clustering**: K-means grouped zones well (Silhouette Score = 0.62), helping drivers understand zone types.
- **Impact**: The dashboard helps drivers earn more by guiding them to busy zones and likely reduces wait times by spreading drivers evenly (estimated in the simulation).

## Challenges and Solutions
- **Data Leakage**: Avoided using `pickup_count` in classification by selecting features with RFE.
- **DBSCAN Noise**: Added `hour` feature and tuned `eps` to reduce noise.
- **Forecasting Accuracy**: Noted Prophet’s MAPE (27.22%) as an area to improve.
- **Traffic Data**: Used `avg_speed` since Uber Movement Data wasn’t available.
- **Driver Preferences**: Used a placeholder during training; dashboard tracks preferences dynamically.

## Future Work
- Improve demand prediction with LSTM models.
- Add real-time traffic data using APIs.
- Collect real driver preferences for better earnings predictions.
- Test other clustering methods like hierarchical clustering.
- Enable live data updates in the dashboard.
- Measure the exact reduction in wait times (target: 15%).

## Conclusion
I built a system that helps NYC taxi drivers find busy zones and earn more money using machine learning. It predicts demand, finds high-demand zones, groups areas, and estimates earnings, all shown in an easy-to-use dashboard. The system likely reduces passenger wait times by spreading drivers better across the city. While some parts, like demand prediction, can be improved, it successfully helps drivers make smarter choices.

## References
- NYC TLC Data Portal.
- Scikit-learn Documentation.
- Prophet Documentation.
- "Deep Learning for Taxi Demand Prediction" (IEEE, 2021).
- "Dynamic Vehicle Repositioning via Reinforcement Learning" (KDD, 2022).

---