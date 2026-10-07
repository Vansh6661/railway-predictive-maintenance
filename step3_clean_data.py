"""
Step 3: Data Cleaning for Railway Predictive Maintenance
AIML Subject Project - Phase 1
"""

import pandas as pd
import numpy as np

# 1. Load the raw dataset
print("Loading dataset...")
raw_data_path = "archive (3)/indian_railway_predictive_maintenance_100k.csv"
df = pd.read_csv(raw_data_path)
print(f"Original shape: {df.shape[0]:,} rows, {df.shape[1]} columns")

# 2. Remove exact duplicate rows
duplicate_count = df.duplicated().sum()
print(f"\nFound {duplicate_count} exact duplicate rows. Removing them...")
df = df.drop_duplicates()
print(f"Shape after removing duplicates: {df.shape[0]:,} rows, {df.shape[1]} columns")

# 3. Separate post-failure leakage columns from features
# Note: 'failure_type' and 'failure_severity' describe the failure AFTER it occurs.
# For Phase 1 (predicting whether maintenance is required), we don't use them as inputs.
print("\nChecking missing values in sensor and operational columns...")
missing_before = df.isnull().sum()
missing_columns = missing_before[missing_before > 0]
print("Columns with missing values (and count):")
for col, val in missing_columns.items():
    print(f"  - {col:<26}: {val:,} missing")

# 4. Impute missing numerical sensor values using Median
# We use median because it is robust to sudden sensor spikes / outliers.
sensor_cols_to_impute = [
    'track_vibration_level', 'humidity_percent', 'rainfall_mm',
    'wind_speed_kmph', 'wheel_wear_percent', 'axle_temperature_c',
    'brake_pad_wear_percent', 'bearing_temperature_c', 'battery_voltage',
    'traction_motor_temp_c', 'load_factor_percent', 'delay_minutes',
    'last_maintenance_days'
]

print("\nImputing missing sensor values with median...")
for col in sensor_cols_to_impute:
    if col in df.columns:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"  Imputed '{col}' with median = {median_val:.2f}")

# 5. Verify no missing values in feature set
remaining_missing = df[sensor_cols_to_impute].isnull().sum().sum()
print(f"\nRemaining missing values in sensor features: {remaining_missing}")

# 6. Save the cleaned dataset for EDA and Modeling
cleaned_path = "cleaned_railway_data.csv"
df.to_csv(cleaned_path, index=False)
print(f"\nCleaned dataset saved successfully to: '{cleaned_path}'")
print(f"Final Cleaned Dataset Dimensions: {df.shape[0]:,} rows × {df.shape[1]} columns")
