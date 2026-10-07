"""
Pydantic Schemas for FastAPI Railway Predictive Maintenance Service
"""

from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field, field_validator


class AssetTelemetryInput(BaseModel):
    # Categorical fields with default fallbacks
    region: str = Field(default="Northern Railway", description="Railway operational zone")
    season: str = Field(default="Winter", description="Operating season (Winter, Summer, Monsoon)")
    train_type: str = Field(default="Express", description="Train classification (Express, Passenger, Freight, Metro)")
    ballast_condition: str = Field(default="Good", description="Track ballast state (Good, Fair, Poor)")
    signal_system_status: str = Field(default="Normal", description="Signaling interlock status (Normal, Warning, Fault)")

    # Numerical Track & Infrastructure Features
    rail_wear_mm: float = Field(default=2.5, ge=0.0, le=30.0, description="Rail head wear thickness in mm")
    track_vibration_level: float = Field(default=1.6, ge=0.0, le=20.0, description="Track vibration level (G)")
    track_temperature_c: float = Field(default=22.0, ge=-20.0, le=90.0, description="Track rail temperature in °C")
    track_curvature_degree: float = Field(default=1.5, ge=0.0, le=45.0, description="Track curvature angle in degrees")

    # Numerical Rolling Stock Mechanical & Thermal Features
    wheel_wear_percent: float = Field(default=15.0, ge=0.0, le=100.0, description="Wheel tread/flange wear percentage")
    brake_pad_wear_percent: float = Field(default=14.0, ge=0.0, le=100.0, description="Brake pad wear percentage")
    brake_pressure_psi: float = Field(default=102.0, ge=0.0, le=200.0, description="Pneumatic brake air pressure in PSI")
    bearing_temperature_c: float = Field(default=48.0, ge=0.0, le=180.0, description="Axle box bearing temperature in °C")
    axle_temperature_c: float = Field(default=42.0, ge=0.0, le=160.0, description="Wheel axle temperature in °C")
    traction_motor_temp_c: float = Field(default=58.0, ge=0.0, le=200.0, description="Locomotive traction motor temperature in °C")

    # Numerical Operational & Electrical Features
    train_age_years: float = Field(default=2.0, ge=0.0, le=50.0, description="Train age in years")
    average_speed_kmph: float = Field(default=110.0, ge=0.0, le=250.0, description="Average operating speed in km/h")
    distance_travelled_km: float = Field(default=65000.0, ge=0.0, description="Cumulative distance in km")
    battery_voltage: float = Field(default=26.5, ge=0.0, le=50.0, description="Auxiliary control battery voltage in V")
    power_consumption_kw: float = Field(default=580.0, ge=0.0, le=5000.0, description="Power consumption in kW")
    load_factor_percent: float = Field(default=70.0, ge=0.0, le=150.0, description="Capacity utilization percentage")
    daily_trips: float = Field(default=8.0, ge=0.0, le=100.0, description="Trips per day")
    delay_minutes: float = Field(default=2.0, ge=0.0, le=500.0, description="Recorded schedule delay in minutes")
    last_maintenance_days: float = Field(default=15.0, ge=0.0, le=1000.0, description="Days elapsed since last overhaul")
    inspection_score: float = Field(default=98.0, ge=0.0, le=100.0, description="Manual inspection health score (0-100)")
    sensor_health_index: float = Field(default=96.0, ge=0.0, le=100.0, description="Telemetry sensor calibration index (0-100)")

    # Weather Features
    ambient_temperature_c: float = Field(default=18.0, ge=-30.0, le=65.0, description="Ambient air temperature in °C")
    humidity_percent: float = Field(default=48.0, ge=0.0, le=100.0, description="Atmospheric humidity percentage")
    rainfall_mm: float = Field(default=0.0, ge=0.0, le=500.0, description="Precipitation rainfall in mm")
    wind_speed_kmph: float = Field(default=10.0, ge=0.0, le=200.0, description="Wind speed in km/h")

    # Optional Asset ID for tracking in batch uploads
    asset_id: Optional[str] = Field(default=None, description="Optional Asset Identifier (e.g. RAKE-1042)")


class SubsystemHealth(BaseModel):
    track_health: float
    mechanical_health: float
    braking_health: float
    electrical_health: float


class PredictionResult(BaseModel):
    asset_id: Optional[str] = None
    maintenance_required: bool
    label: str
    rf_risk_probability: float
    rf_healthy_probability: float
    lr_risk_probability: float
    urgency: str  # CRITICAL, HIGH, MEDIUM, LOW
    urgency_color: str
    recommended_action: str
    subsystems: SubsystemHealth
    top_contributing_features: List[str]


class BatchPredictionResponse(BaseModel):
    total_assets: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    average_risk_pct: float
    results: List[PredictionResult]


class PresetProfile(BaseModel):
    id: str
    name: str
    category: str
    status_tag: str
    description: str
    data: Dict[str, Any]


class ModelMetricItem(BaseModel):
    model: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    role: str


class FeatureImportanceItem(BaseModel):
    name: str
    importance: float
    category: str
    description: str


class ModelInfoResponse(BaseModel):
    comparison: List[ModelMetricItem]
    top_features: List[FeatureImportanceItem]
    dataset_summary: Dict[str, Any]
    confusion_matrices: Dict[str, Any]
    roc_curve: Dict[str, Any]
