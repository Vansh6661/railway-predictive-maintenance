"""
Railway Predictive Maintenance Web Application
Flask Backend Server providing REST API & UI
"""

import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, render_template

app = Flask(__name__, static_folder="static", template_folder="templates")

# Paths to models & resources
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "preprocessor.joblib")
RF_MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "random_forest.joblib")
LR_MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "logistic_regression.joblib")

# Load models at startup
print("Loading preprocessor and ML models...")
preprocessor = joblib.load(PREPROCESSOR_PATH)
rf_model = joblib.load(RF_MODEL_PATH)
lr_model = joblib.load(LR_MODEL_PATH)
print("Models loaded successfully!")

# Feature names and defaults
FEATURE_ORDER = [
    'region', 'season', 'train_type', 'train_age_years', 'average_speed_kmph',
    'distance_travelled_km', 'track_temperature_c', 'rail_wear_mm',
    'track_vibration_level', 'ballast_condition', 'track_curvature_degree',
    'ambient_temperature_c', 'humidity_percent', 'rainfall_mm', 'wind_speed_kmph',
    'wheel_wear_percent', 'axle_temperature_c', 'brake_pressure_psi',
    'brake_pad_wear_percent', 'bearing_temperature_c', 'battery_voltage',
    'traction_motor_temp_c', 'signal_system_status', 'power_consumption_kw',
    'load_factor_percent', 'daily_trips', 'delay_minutes',
    'last_maintenance_days', 'inspection_score', 'sensor_health_index'
]

# Preset asset configurations
PRESET_ASSETS = {
    "pristine_express": {
        "name": "Vande Bharat Express (Healthy / Low Wear)",
        "description": "Newly commissioned train with low mileage, optimal bearing temperatures, and fresh brake pads.",
        "data": {
            'region': 'Northern Railway', 'season': 'Winter', 'train_type': 'Express',
            'train_age_years': 2, 'average_speed_kmph': 110.0, 'distance_travelled_km': 65000,
            'track_temperature_c': 22.0, 'rail_wear_mm': 2.5, 'track_vibration_level': 1.6,
            'ballast_condition': 'Good', 'track_curvature_degree': 1.5, 'ambient_temperature_c': 18.0,
            'humidity_percent': 48.0, 'rainfall_mm': 0.0, 'wind_speed_kmph': 10.0,
            'wheel_wear_percent': 15.0, 'axle_temperature_c': 42.0, 'brake_pressure_psi': 102.0,
            'brake_pad_wear_percent': 14.0, 'bearing_temperature_c': 48.0, 'battery_voltage': 26.5,
            'traction_motor_temp_c': 58.0, 'signal_system_status': 'Normal', 'power_consumption_kw': 580.0,
            'load_factor_percent': 70.0, 'daily_trips': 8, 'delay_minutes': 2.0,
            'last_maintenance_days': 15.0, 'inspection_score': 98.0, 'sensor_health_index': 96.0
        }
    },
    "critical_freight": {
        "name": "Heavy Freight Rake (Critical Wear & Overheating)",
        "description": "High mileage freight rake with severe rail & wheel wear, elevated bearing heat, and worn brake pads.",
        "data": {
            'region': 'Central Railway', 'season': 'Summer', 'train_type': 'Freight',
            'train_age_years': 27, 'average_speed_kmph': 88.0, 'distance_travelled_km': 1420000,
            'track_temperature_c': 56.0, 'rail_wear_mm': 19.2, 'track_vibration_level': 8.8,
            'ballast_condition': 'Poor', 'track_curvature_degree': 14.8, 'ambient_temperature_c': 42.0,
            'humidity_percent': 65.0, 'rainfall_mm': 35.0, 'wind_speed_kmph': 28.0,
            'wheel_wear_percent': 88.0, 'axle_temperature_c': 105.0, 'brake_pressure_psi': 48.0,
            'brake_pad_wear_percent': 91.0, 'bearing_temperature_c': 116.0, 'battery_voltage': 18.2,
            'traction_motor_temp_c': 128.0, 'signal_system_status': 'Fault', 'power_consumption_kw': 1420.0,
            'load_factor_percent': 98.0, 'daily_trips': 20, 'delay_minutes': 55.0,
            'last_maintenance_days': 320.0, 'inspection_score': 46.0, 'sensor_health_index': 42.0
        }
    },
    "monsoon_passenger": {
        "name": "Monsoon Passenger (High Vibration & Track Degradation)",
        "description": "Passenger train operating during heavy monsoon with compromised track ballast and elevated vibration.",
        "data": {
            'region': 'Southern Railway', 'season': 'Monsoon', 'train_type': 'Passenger',
            'train_age_years': 16, 'average_speed_kmph': 65.0, 'distance_travelled_km': 780000,
            'track_temperature_c': 32.0, 'rail_wear_mm': 12.8, 'track_vibration_level': 7.2,
            'ballast_condition': 'Fair', 'track_curvature_degree': 8.5, 'ambient_temperature_c': 28.0,
            'humidity_percent': 92.0, 'rainfall_mm': 85.0, 'wind_speed_kmph': 34.0,
            'wheel_wear_percent': 58.0, 'axle_temperature_c': 76.0, 'brake_pressure_psi': 82.0,
            'brake_pad_wear_percent': 62.0, 'bearing_temperature_c': 84.0, 'battery_voltage': 22.8,
            'traction_motor_temp_c': 89.0, 'signal_system_status': 'Warning', 'power_consumption_kw': 890.0,
            'load_factor_percent': 82.0, 'daily_trips': 14, 'delay_minutes': 28.0,
            'last_maintenance_days': 195.0, 'inspection_score': 68.0, 'sensor_health_index': 72.0
        }
    }
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/presets", methods=["GET"])
def get_presets():
    return jsonify(PRESET_ASSETS)

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        
        # Build dataframe in exact order
        input_dict = {}
        for feat in FEATURE_ORDER:
            val = data.get(feat, PRESET_ASSETS['pristine_express']['data'].get(feat))
            # Convert numerical
            if feat not in ['region', 'season', 'train_type', 'ballast_condition', 'signal_system_status']:
                val = float(val)
            input_dict[feat] = [val]
            
        input_df = pd.DataFrame(input_dict)
        
        # Transform via preprocessor
        processed_data = preprocessor.transform(input_df)
        
        # Run inference
        rf_pred = int(rf_model.predict(processed_data)[0])
        rf_prob = rf_model.predict_proba(processed_data)[0]
        
        lr_pred = int(lr_model.predict(processed_data)[0])
        lr_prob = lr_model.predict_proba(processed_data)[0]
        
        maint_prob_rf = float(rf_prob[1] * 100)
        healthy_prob_rf = float(rf_prob[0] * 100)
        
        maint_prob_lr = float(lr_prob[1] * 100)
        
        # Calculate Subsystem Health Indices (0-100%)
        # 1. Track Infrastructure Health
        rail_wear = float(input_df['rail_wear_mm'].iloc[0])
        vib = float(input_df['track_vibration_level'].iloc[0])
        ballast = str(input_df['ballast_condition'].iloc[0])
        track_health = max(10, 100 - (rail_wear * 3.5) - (vib * 4.0) - (20 if ballast == 'Poor' else (10 if ballast == 'Fair' else 0)))
        
        # 2. Wheel & Mechanical Health
        wheel_wear = float(input_df['wheel_wear_percent'].iloc[0])
        bearing_temp = float(input_df['bearing_temperature_c'].iloc[0])
        axle_temp = float(input_df['axle_temperature_c'].iloc[0])
        mech_health = max(10, 100 - (wheel_wear * 0.5) - max(0, (bearing_temp - 60) * 0.6) - max(0, (axle_temp - 60) * 0.5))
        
        # 3. Braking & Pneumatic Health
        brake_wear = float(input_df['brake_pad_wear_percent'].iloc[0])
        brake_psi = float(input_df['brake_pressure_psi'].iloc[0])
        brake_health = max(10, 100 - (brake_wear * 0.6) - max(0, (95 - brake_psi) * 0.8))
        
        # 4. Electrical & Signal Health
        batt_volt = float(input_df['battery_voltage'].iloc[0])
        motor_temp = float(input_df['traction_motor_temp_c'].iloc[0])
        signal_status = str(input_df['signal_system_status'].iloc[0])
        elec_health = max(10, 100 - max(0, (24 - batt_volt) * 8.0) - max(0, (motor_temp - 75) * 0.5) - (25 if signal_status == 'Fault' else (10 if signal_status == 'Warning' else 0)))
        
        # Urgency classification (Phase 2 Preview)
        if maint_prob_rf >= 65 or track_health < 40 or mech_health < 40 or brake_health < 40:
            urgency = "CRITICAL"
            urgency_color = "#ef4444"
            rec_window = "Immediate Track/Yard Inspection (01:00 - 03:00 hrs)"
        elif maint_prob_rf >= 50 or track_health < 60 or mech_health < 60:
            urgency = "HIGH"
            urgency_color = "#f97316"
            rec_window = "Next Scheduled Depot Halt (02:00 - 04:30 hrs)"
        elif maint_prob_rf >= 35:
            urgency = "MEDIUM"
            urgency_color = "#eab308"
            rec_window = "Routine Weekly Maintenance Block (Weekend Slot)"
        else:
            urgency = "LOW"
            urgency_color = "#22c55e"
            rec_window = "Standard Inspection Cycle"
            
        # Key Contributing Factors
        factors = []
        if rail_wear > 12.0:
            factors.append(f"Excessive Rail Wear ({rail_wear:.1f} mm, normal < 10 mm)")
        if wheel_wear > 65.0:
            factors.append(f"Severe Wheel Tread Wear ({wheel_wear:.1f}%)")
        if bearing_temp > 90.0:
            factors.append(f"Bearing Overheating ({bearing_temp:.1f} °C, normal < 75 °C)")
        if brake_psi < 70.0:
            factors.append(f"Low Braking Air Pressure ({brake_psi:.1f} PSI)")
        if brake_wear > 75.0:
            factors.append(f"Brake Pad Thinning ({brake_wear:.1f}%)")
        if vib > 6.0:
            factors.append(f"High Track Vibration ({vib:.2f} G-level)")
        if ballast == 'Poor':
            factors.append("Degraded Track Ballast Bed Foundation")
        if signal_status == 'Fault':
            factors.append("Signaling Interlock Sensor Fault")
        if float(input_df['last_maintenance_days'].iloc[0]) > 250:
            factors.append(f"Extended Time Since Last Overhaul ({int(input_df['last_maintenance_days'].iloc[0])} days)")
            
        if not factors:
            factors.append("All physical parameters and sensor telemetry within nominal railway safety thresholds.")
            
        return jsonify({
            "status": "success",
            "prediction": {
                "maintenance_required": rf_pred == 1,
                "label": "MAINTENANCE REQUIRED" if rf_pred == 1 else "HEALTHY (NORMAL OPERATION)",
                "rf_risk_probability": round(maint_prob_rf, 1),
                "rf_healthy_probability": round(healthy_prob_rf, 1),
                "lr_risk_probability": round(maint_prob_lr, 1),
                "urgency": urgency,
                "urgency_color": urgency_color,
                "recommended_block_window": rec_window
            },
            "subsystems": {
                "track_health": round(track_health, 1),
                "mechanical_health": round(mech_health, 1),
                "braking_health": round(brake_health, 1),
                "electrical_health": round(elec_health, 1)
            },
            "contributing_factors": factors
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/api/model_stats", methods=["GET"])
def get_model_stats():
    return jsonify({
        "comparison": [
            {"model": "Logistic Regression (Baseline)", "accuracy": 60.58, "precision": 22.95, "recall": 40.41, "f1": 29.28, "roc_auc": 53.86},
            {"model": "Random Forest (Main Model)", "accuracy": 79.97, "precision": 51.46, "recall": 13.96, "f1": 21.97, "roc_auc": 55.02}
        ],
        "top_features": [
            {"name": "rail_wear_mm", "importance": 8.62, "category": "Track"},
            {"name": "wheel_wear_percent", "importance": 6.43, "category": "Rolling Stock"},
            {"name": "track_vibration_level", "importance": 5.50, "category": "Track"},
            {"name": "ballast_condition (Poor)", "importance": 4.76, "category": "Infrastructure"},
            {"name": "battery_voltage", "importance": 4.15, "category": "Electrical"},
            {"name": "bearing_temperature_c", "importance": 3.53, "category": "Mechanical"},
            {"name": "brake_pressure_psi", "importance": 3.51, "category": "Pneumatics"},
            {"name": "axle_temperature_c", "importance": 3.49, "category": "Mechanical"},
            {"name": "delay_minutes", "importance": 3.41, "category": "Operations"},
            {"name": "ambient_temperature_c", "importance": 3.38, "category": "Weather"}
        ],
        "dataset_summary": {
            "total_records": 100000,
            "features_count": 35,
            "healthy_records": 79804,
            "maintenance_records": 20196,
            "train_records": 80000,
            "test_records": 20000
        }
    })

# Serve generated plots
@app.route("/plots/<path:filename>")
def serve_plot(filename):
    if os.path.exists(os.path.join(BASE_DIR, "eda_plots", filename)):
        return send_from_directory(os.path.join(BASE_DIR, "eda_plots"), filename)
    elif os.path.exists(os.path.join(BASE_DIR, "model_eval_plots", filename)):
        return send_from_directory(os.path.join(BASE_DIR, "model_eval_plots"), filename)
    return "Plot not found", 404

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)
