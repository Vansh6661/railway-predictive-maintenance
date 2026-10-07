"""
Static metadata, preset profiles, and precomputed model evaluation benchmarks
"""

PRESET_PROFILES = [
    {
        "id": "pristine_express",
        "name": "Vande Bharat Express",
        "category": "High-Speed Passenger",
        "status_tag": "Healthy (Optimal)",
        "description": "Newly commissioned rake with minimal wheel wear (15%), normal bearing heat (48°C), and full 102 PSI brake pressure.",
        "data": {
            "region": "Northern Railway", "season": "Winter", "train_type": "Express",
            "train_age_years": 2.0, "average_speed_kmph": 110.0, "distance_travelled_km": 65000.0,
            "track_temperature_c": 22.0, "rail_wear_mm": 2.5, "track_vibration_level": 1.6,
            "ballast_condition": "Good", "track_curvature_degree": 1.5, "ambient_temperature_c": 18.0,
            "humidity_percent": 48.0, "rainfall_mm": 0.0, "wind_speed_kmph": 10.0,
            "wheel_wear_percent": 15.0, "axle_temperature_c": 42.0, "brake_pressure_psi": 102.0,
            "brake_pad_wear_percent": 14.0, "bearing_temperature_c": 48.0, "battery_voltage": 26.5,
            "traction_motor_temp_c": 58.0, "signal_system_status": "Normal", "power_consumption_kw": 580.0,
            "load_factor_percent": 70.0, "daily_trips": 8.0, "delay_minutes": 2.0,
            "last_maintenance_days": 15.0, "inspection_score": 98.0, "sensor_health_index": 96.0,
            "asset_id": "VB-EXP-204"
        }
    },
    {
        "id": "critical_freight",
        "name": "Heavy Mineral Freight Rake",
        "category": "Heavy Freight",
        "status_tag": "Critical Wear Alert",
        "description": "High mileage freight rake with severe rail wear (19.2 mm), hot bearings (116°C), and degraded braking pressure.",
        "data": {
            "region": "Central Railway", "season": "Summer", "train_type": "Freight",
            "train_age_years": 27.0, "average_speed_kmph": 88.0, "distance_travelled_km": 1420000.0,
            "track_temperature_c": 56.0, "rail_wear_mm": 19.2, "track_vibration_level": 8.8,
            "ballast_condition": "Poor", "track_curvature_degree": 14.8, "ambient_temperature_c": 42.0,
            "humidity_percent": 65.0, "rainfall_mm": 35.0, "wind_speed_kmph": 28.0,
            "wheel_wear_percent": 88.0, "axle_temperature_c": 105.0, "brake_pressure_psi": 48.0,
            "brake_pad_wear_percent": 91.0, "bearing_temperature_c": 116.0, "battery_voltage": 18.2,
            "traction_motor_temp_c": 128.0, "signal_system_status": "Fault", "power_consumption_kw": 1420.0,
            "load_factor_percent": 98.0, "daily_trips": 20.0, "delay_minutes": 55.0,
            "last_maintenance_days": 320.0, "inspection_score": 46.0, "sensor_health_index": 42.0,
            "asset_id": "BOXN-FRT-918"
        }
    },
    {
        "id": "monsoon_warning",
        "name": "Monsoon Coastal Passenger",
        "category": "Intercity Passenger",
        "status_tag": "Track Vibration Warning",
        "description": "Operating on wet coastal tracks with mud-pumped ballast, elevated lateral vibration (7.2 G), and intermittent signal warnings.",
        "data": {
            "region": "Southern Railway", "season": "Monsoon", "train_type": "Passenger",
            "train_age_years": 16.0, "average_speed_kmph": 65.0, "distance_travelled_km": 780000.0,
            "track_temperature_c": 32.0, "rail_wear_mm": 12.8, "track_vibration_level": 7.2,
            "ballast_condition": "Fair", "track_curvature_degree": 8.5, "ambient_temperature_c": 28.0,
            "humidity_percent": 92.0, "rainfall_mm": 85.0, "wind_speed_kmph": 34.0,
            "wheel_wear_percent": 58.0, "axle_temperature_c": 76.0, "brake_pressure_psi": 82.0,
            "brake_pad_wear_percent": 62.0, "bearing_temperature_c": 84.0, "battery_voltage": 22.8,
            "traction_motor_temp_c": 89.0, "signal_system_status": "Warning", "power_consumption_kw": 890.0,
            "load_factor_percent": 82.0, "daily_trips": 14.0, "delay_minutes": 28.0,
            "last_maintenance_days": 195.0, "inspection_score": 68.0, "sensor_health_index": 72.0,
            "asset_id": "SR-PSG-441"
        }
    },
    {
        "id": "summer_emu_thermal",
        "name": "Suburban EMU Commuter Rake",
        "category": "Suburban Metro/EMU",
        "status_tag": "Traction Motor Heat",
        "description": "Intensive multi-stop commuter rake in peak summer suffering from traction motor thermal spikes (118°C) and battery drain.",
        "data": {
            "region": "Western Railway", "season": "Summer", "train_type": "Metro",
            "train_age_years": 12.0, "average_speed_kmph": 52.0, "distance_travelled_km": 540000.0,
            "track_temperature_c": 52.0, "rail_wear_mm": 8.4, "track_vibration_level": 4.6,
            "ballast_condition": "Good", "track_curvature_degree": 4.2, "ambient_temperature_c": 43.0,
            "humidity_percent": 70.0, "rainfall_mm": 0.0, "wind_speed_kmph": 15.0,
            "wheel_wear_percent": 64.0, "axle_temperature_c": 86.0, "brake_pressure_psi": 88.0,
            "brake_pad_wear_percent": 74.0, "bearing_temperature_c": 88.0, "battery_voltage": 19.8,
            "traction_motor_temp_c": 118.0, "signal_system_status": "Normal", "power_consumption_kw": 1150.0,
            "load_factor_percent": 110.0, "daily_trips": 24.0, "delay_minutes": 14.0,
            "last_maintenance_days": 140.0, "inspection_score": 75.0, "sensor_health_index": 82.0,
            "asset_id": "WR-EMU-1207"
        }
    },
    {
        "id": "high_speed_wheel_wear",
        "name": "Rajdhani Superfast Corridor Rake",
        "category": "Superfast Express",
        "status_tag": "Wheel Flange Thinning",
        "description": "High-velocity rake with advancing wheel tread wear (76%) and high speed continuous duty requiring wheel profiling.",
        "data": {
            "region": "Eastern Railway", "season": "Winter", "train_type": "Express",
            "train_age_years": 9.0, "average_speed_kmph": 125.0, "distance_travelled_km": 920000.0,
            "track_temperature_c": 19.0, "rail_wear_mm": 9.1, "track_vibration_level": 3.8,
            "ballast_condition": "Good", "track_curvature_degree": 3.0, "ambient_temperature_c": 15.0,
            "humidity_percent": 55.0, "rainfall_mm": 0.0, "wind_speed_kmph": 18.0,
            "wheel_wear_percent": 76.0, "axle_temperature_c": 68.0, "brake_pressure_psi": 94.0,
            "brake_pad_wear_percent": 68.0, "bearing_temperature_c": 72.0, "battery_voltage": 24.5,
            "traction_motor_temp_c": 74.0, "signal_system_status": "Normal", "power_consumption_kw": 780.0,
            "load_factor_percent": 88.0, "daily_trips": 6.0, "delay_minutes": 5.0,
            "last_maintenance_days": 175.0, "inspection_score": 84.0, "sensor_health_index": 90.0,
            "asset_id": "ER-RAJ-12301"
        }
    },
    {
        "id": "shunter_signaling_anomaly",
        "name": "Yard Shunter & Pilot Loco",
        "category": "Yard Shunter",
        "status_tag": "Low Voltage & Signal Fault",
        "description": "Depot switcher loco with degraded battery voltage (16.4V), pneumatic leaks (58 PSI), and intermittent cab signal disconnects.",
        "data": {
            "region": "South Central Railway", "season": "Winter", "train_type": "Passenger",
            "train_age_years": 22.0, "average_speed_kmph": 30.0, "distance_travelled_km": 310000.0,
            "track_temperature_c": 26.0, "rail_wear_mm": 11.0, "track_vibration_level": 5.4,
            "ballast_condition": "Fair", "track_curvature_degree": 9.0, "ambient_temperature_c": 22.0,
            "humidity_percent": 60.0, "rainfall_mm": 0.0, "wind_speed_kmph": 8.0,
            "wheel_wear_percent": 48.0, "axle_temperature_c": 54.0, "brake_pressure_psi": 58.0,
            "brake_pad_wear_percent": 56.0, "bearing_temperature_c": 60.0, "battery_voltage": 16.4,
            "traction_motor_temp_c": 66.0, "signal_system_status": "Fault", "power_consumption_kw": 420.0,
            "load_factor_percent": 40.0, "daily_trips": 12.0, "delay_minutes": 38.0,
            "last_maintenance_days": 260.0, "inspection_score": 58.0, "sensor_health_index": 60.0,
            "asset_id": "SCR-WDS-602"
        }
    }
]

MODEL_EVALUATION_METRICS = {
    "comparison": [
        {
            "model": "Logistic Regression (Baseline)",
            "accuracy": 60.58,
            "precision": 22.95,
            "recall": 40.41,
            "f1": 29.28,
            "roc_auc": 53.86,
            "role": "Linear Baseline Benchmark"
        },
        {
            "model": "Random Forest Classifier",
            "accuracy": 79.97,
            "precision": 51.46,
            "recall": 13.96,
            "f1": 21.97,
            "roc_auc": 55.02,
            "role": "Production Non-Linear Ensemble"
        }
    ],
    "top_features": [
        {"name": "rail_wear_mm", "importance": 8.62, "category": "Track", "description": "Track rail crown thinning and vertical wear thickness"},
        {"name": "wheel_wear_percent", "importance": 6.43, "category": "Rolling Stock", "description": "Wheel flange angle and tread profile deterioration"},
        {"name": "track_vibration_level", "importance": 5.50, "category": "Track", "description": "Dynamic acceleration caused by track alignment defects"},
        {"name": "ballast_condition (Poor)", "importance": 4.76, "category": "Infrastructure", "description": "Eroded stone sleeper bedding and poor track elasticity"},
        {"name": "battery_voltage", "importance": 4.15, "category": "Electrical", "description": "Locomotive auxiliary battery voltage drop anomalies"},
        {"name": "bearing_temperature_c", "importance": 3.53, "category": "Mechanical", "description": "Thermal friction rise inside axle roller bearings"},
        {"name": "brake_pressure_psi", "importance": 3.51, "category": "Pneumatics", "description": "Pneumatic air pipe pressure drops indicating valve leaks"},
        {"name": "axle_temperature_c", "importance": 3.49, "category": "Mechanical", "description": "Journal axle box thermal readings"},
        {"name": "delay_minutes", "importance": 3.41, "category": "Operations", "description": "Operational schedule delay correlated with component sluggishness"},
        {"name": "ambient_temperature_c", "importance": 3.38, "category": "Weather", "description": "High ambient temperatures accelerating thermal fatigue"},
        {"name": "track_curvature_degree", "importance": 3.31, "category": "Track", "description": "Sharp curves amplifying lateral flange force"},
        {"name": "traction_motor_temp_c", "importance": 3.29, "category": "Electrical", "description": "Electric motor stator coil operating temperature"},
        {"name": "brake_pad_wear_percent", "importance": 3.24, "category": "Pneumatics", "description": "Friction block composite thickness remaining"},
        {"name": "average_speed_kmph", "importance": 3.20, "category": "Operations", "description": "Sustained high-velocity mechanical stress cycles"},
        {"name": "power_consumption_kw", "importance": 3.19, "category": "Electrical", "description": "Locomotive power draw anomalies"}
    ],
    "dataset_summary": {
        "total_records": 100000,
        "train_records": 80000,
        "test_records": 20000,
        "healthy_count": 79804,
        "healthy_pct": 79.8,
        "maintenance_count": 20196,
        "maintenance_pct": 20.2,
        "features_count": 30
    },
    "confusion_matrices": {
        "logistic_regression": {
            "true_negatives": 9680,
            "false_positives": 6281,
            "false_negatives": 2407,
            "true_positives": 1632
        },
        "random_forest": {
            "true_negatives": 15429,
            "false_positives": 532,
            "false_negatives": 3475,
            "true_positives": 564
        }
    },
    "roc_curve": {
        "points_rf": [
            {"fpr": 0.0, "tpr": 0.0},
            {"fpr": 0.033, "tpr": 0.140},
            {"fpr": 0.120, "tpr": 0.260},
            {"fpr": 0.280, "tpr": 0.450},
            {"fpr": 0.520, "tpr": 0.680},
            {"fpr": 0.780, "tpr": 0.880},
            {"fpr": 1.0, "tpr": 1.0}
        ],
        "points_lr": [
            {"fpr": 0.0, "tpr": 0.0},
            {"fpr": 0.393, "tpr": 0.404},
            {"fpr": 0.600, "tpr": 0.620},
            {"fpr": 0.820, "tpr": 0.840},
            {"fpr": 1.0, "tpr": 1.0}
        ]
    }
}
