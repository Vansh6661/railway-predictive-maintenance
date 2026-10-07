"""
Inference engine handling scikit-learn preprocessing and prediction
"""

import os
import io
import csv
import joblib
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from backend.schemas import AssetTelemetryInput, PredictionResult, SubsystemHealth, BatchPredictionResponse

# Paths to models
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "preprocessor.joblib")
RF_MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "random_forest.joblib")
LR_MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "logistic_regression.joblib")

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

class ModelManager:
    def __init__(self):
        self.preprocessor = None
        self.rf_model = None
        self.lr_model = None
        self.is_loaded = False

    def load_models(self):
        if not self.is_loaded:
            print("Loading ML models and preprocessor pipeline into memory...")
            self.preprocessor = joblib.load(PREPROCESSOR_PATH)
            self.rf_model = joblib.load(RF_MODEL_PATH)
            self.lr_model = joblib.load(LR_MODEL_PATH)
            self.is_loaded = True
            print("ML models loaded successfully!")

    def predict_single(self, telemetry: AssetTelemetryInput) -> PredictionResult:
        self.load_models()

        # Convert to dictionary matching FEATURE_ORDER
        raw_dict = telemetry.model_dump()
        asset_id = raw_dict.pop('asset_id', None) or "ASSET-LIVE"

        input_row = {k: [raw_dict.get(k)] for k in FEATURE_ORDER}
        df_row = pd.DataFrame(input_row)

        # Preprocess features
        X_proc = self.preprocessor.transform(df_row)

        # Predict
        rf_pred = int(self.rf_model.predict(X_proc)[0])
        rf_probs = self.rf_model.predict_proba(X_proc)[0]
        lr_probs = self.lr_model.predict_proba(X_proc)[0]

        rf_risk_prob = float(round(rf_probs[1] * 100, 1))
        rf_healthy_prob = float(round(rf_probs[0] * 100, 1))
        lr_risk_prob = float(round(lr_probs[1] * 100, 1))

        # Subsystems Health computation
        subsystems = self._calculate_subsystems(raw_dict)

        # Urgency & Recommended Action
        urgency, urgency_color, action = self._determine_urgency(rf_risk_prob, subsystems)

        # Contributing Root Causes
        top_factors = self._extract_contributing_factors(raw_dict)

        return PredictionResult(
            asset_id=asset_id,
            maintenance_required=(rf_pred == 1 or rf_risk_prob >= 50.0),
            label="MAINTENANCE REQUIRED" if (rf_pred == 1 or rf_risk_prob >= 50.0) else "HEALTHY (NORMAL OPERATION)",
            rf_risk_probability=rf_risk_prob,
            rf_healthy_probability=rf_healthy_prob,
            lr_risk_probability=lr_risk_prob,
            urgency=urgency,
            urgency_color=urgency_color,
            recommended_action=action,
            subsystems=subsystems,
            top_contributing_features=top_factors
        )

    def predict_batch_df(self, df: pd.DataFrame) -> BatchPredictionResponse:
        self.load_models()

        results: List[PredictionResult] = []
        clean_rows = []
        asset_ids = []

        for idx, row in df.iterrows():
            aid = str(row.get('asset_id') or row.get('train_id') or f"ASSET-{idx+1:04d}")
            asset_ids.append(aid)

            row_data = {}
            for col in FEATURE_ORDER:
                if col in row:
                    val = row[col]
                    if col not in ['region', 'season', 'train_type', 'ballast_condition', 'signal_system_status']:
                        try:
                            val = float(val) if pd.notna(val) else 0.0
                        except:
                            val = 0.0
                    row_data[col] = val
                else:
                    # default fallback
                    row_data[col] = "Good" if "condition" in col else ("Normal" if "status" in col else ("Passenger" if col == "train_type" else 0.0))
            clean_rows.append(row_data)

        batch_df = pd.DataFrame(clean_rows)
        X_proc = self.preprocessor.transform(batch_df)

        rf_preds = self.rf_model.predict(X_proc)
        rf_probs = self.rf_model.predict_proba(X_proc)
        lr_probs = self.lr_model.predict_proba(X_proc)

        critical_count = 0
        high_count = 0
        medium_count = 0
        low_count = 0
        total_risk_sum = 0.0

        for i in range(len(clean_rows)):
            rf_risk_prob = float(round(rf_probs[i][1] * 100, 1))
            rf_healthy_prob = float(round(rf_probs[i][0] * 100, 1))
            lr_risk_prob = float(round(lr_probs[i][1] * 100, 1))
            total_risk_sum += rf_risk_prob

            subsystems = self._calculate_subsystems(clean_rows[i])
            urgency, urgency_color, action = self._determine_urgency(rf_risk_prob, subsystems)
            top_factors = self._extract_contributing_factors(clean_rows[i])

            if urgency == "CRITICAL":
                critical_count += 1
            elif urgency == "HIGH":
                high_count += 1
            elif urgency == "MEDIUM":
                medium_count += 1
            else:
                low_count += 1

            results.append(PredictionResult(
                asset_id=asset_ids[i],
                maintenance_required=(rf_preds[i] == 1 or rf_risk_prob >= 50.0),
                label="MAINTENANCE REQUIRED" if (rf_preds[i] == 1 or rf_risk_prob >= 50.0) else "HEALTHY (NORMAL OPERATION)",
                rf_risk_probability=rf_risk_prob,
                rf_healthy_probability=rf_healthy_prob,
                lr_risk_probability=lr_risk_prob,
                urgency=urgency,
                urgency_color=urgency_color,
                recommended_action=action,
                subsystems=subsystems,
                top_contributing_features=top_factors
            ))

        # Sort ranked by risk probability descending (highest risk first)
        results.sort(key=lambda r: r.rf_risk_probability, reverse=True)

        avg_risk = round(total_risk_sum / max(1, len(results)), 1)

        return BatchPredictionResponse(
            total_assets=len(results),
            critical_count=critical_count,
            high_count=high_count,
            medium_count=medium_count,
            low_count=low_count,
            average_risk_pct=avg_risk,
            results=results
        )

    def _calculate_subsystems(self, d: Dict[str, Any]) -> SubsystemHealth:
        # 1. Track Health
        rail_wear = float(d.get('rail_wear_mm', 2.5))
        vib = float(d.get('track_vibration_level', 1.6))
        ballast = str(d.get('ballast_condition', 'Good'))
        track_h = max(10.0, 100.0 - (rail_wear * 3.5) - (vib * 4.0) - (20.0 if ballast == 'Poor' else (10.0 if ballast == 'Fair' else 0.0)))

        # 2. Mechanical Health
        wheel_wear = float(d.get('wheel_wear_percent', 15.0))
        bearing_temp = float(d.get('bearing_temperature_c', 48.0))
        axle_temp = float(d.get('axle_temperature_c', 42.0))
        mech_h = max(10.0, 100.0 - (wheel_wear * 0.5) - max(0.0, (bearing_temp - 60.0) * 0.6) - max(0.0, (axle_temp - 60.0) * 0.5))

        # 3. Braking Health
        brake_wear = float(d.get('brake_pad_wear_percent', 14.0))
        brake_psi = float(d.get('brake_pressure_psi', 102.0))
        brake_h = max(10.0, 100.0 - (brake_wear * 0.6) - max(0.0, (95.0 - brake_psi) * 0.8))

        # 4. Electrical Health
        batt_volt = float(d.get('battery_voltage', 26.5))
        motor_temp = float(d.get('traction_motor_temp_c', 58.0))
        signal_status = str(d.get('signal_system_status', 'Normal'))
        elec_h = max(10.0, 100.0 - max(0.0, (24.0 - batt_volt) * 8.0) - max(0.0, (motor_temp - 75.0) * 0.5) - (25.0 if signal_status == 'Fault' else (10.0 if signal_status == 'Warning' else 0.0)))

        return SubsystemHealth(
            track_health=round(track_h, 1),
            mechanical_health=round(mech_h, 1),
            braking_health=round(brake_h, 1),
            electrical_health=round(elec_h, 1)
        )

    def _determine_urgency(self, risk_pct: float, subs: SubsystemHealth) -> Tuple[str, str, str]:
        if risk_pct >= 65.0 or subs.track_health < 40.0 or subs.mechanical_health < 40.0 or subs.braking_health < 40.0:
            return "CRITICAL", "#ef4444", "Immediate Depot Halt / Emergency Track Inspection Window (01:00 - 03:00 hrs)"
        elif risk_pct >= 50.0 or subs.track_health < 60.0 or subs.mechanical_health < 60.0:
            return "HIGH", "#f97316", "Schedule Priority Maintenance within 24 Hours (Next Yard Halt: 02:00 - 04:30 hrs)"
        elif risk_pct >= 35.0:
            return "MEDIUM", "#eab308", "Routine Weekly Inspection Block (Weekend Depot Allocation)"
        else:
            return "LOW", "#10b981", "Standard Operating Condition (Next Periodic Overhaul Cycle)"

    def _extract_contributing_factors(self, d: Dict[str, Any]) -> List[str]:
        factors = []
        rail_wear = float(d.get('rail_wear_mm', 0))
        wheel_wear = float(d.get('wheel_wear_percent', 0))
        bearing_temp = float(d.get('bearing_temperature_c', 0))
        brake_psi = float(d.get('brake_pressure_psi', 100))
        brake_wear = float(d.get('brake_pad_wear_percent', 0))
        vib = float(d.get('track_vibration_level', 0))
        batt_volt = float(d.get('battery_voltage', 24))
        motor_temp = float(d.get('traction_motor_temp_c', 0))
        ballast = str(d.get('ballast_condition', 'Good'))
        signal = str(d.get('signal_system_status', 'Normal'))
        days = float(d.get('last_maintenance_days', 0))

        if rail_wear > 12.0:
            factors.append(f"Excessive Rail Crown Wear ({rail_wear:.1f} mm, normal < 10 mm)")
        if wheel_wear > 65.0:
            factors.append(f"Severe Wheel Flange & Tread Wear ({wheel_wear:.1f}%)")
        if bearing_temp > 85.0:
            factors.append(f"Axle Bearing Overheating ({bearing_temp:.1f} °C, normal < 75 °C)")
        if brake_psi < 72.0:
            factors.append(f"Pneumatic Brake Line Pressure Loss ({brake_psi:.1f} PSI)")
        if brake_wear > 75.0:
            factors.append(f"Friction Brake Pad Thickness Worn ({brake_wear:.1f}%)")
        if vib > 6.0:
            factors.append(f"High Dynamic Track Vibration ({vib:.2f} G-level)")
        if ballast == 'Poor':
            factors.append("Degraded Track Ballast Bed Foundation")
        if batt_volt < 19.0:
            factors.append(f"Low Auxiliary Battery Voltage ({batt_volt:.1f} V)")
        if motor_temp > 105.0:
            factors.append(f"Traction Motor Stator Thermal Spike ({motor_temp:.1f} °C)")
        if signal == 'Fault':
            factors.append("Signaling Interlock Status in Fault Condition")
        if days > 250.0:
            factors.append(f"Extended Time Since Last Depot Overhaul ({int(days)} days)")

        if not factors:
            factors.append("All mechanical wear, thermal, and electrical telemetry parameters within nominal safety margins.")

        return factors[:3]

# Global singleton
model_manager = ModelManager()
