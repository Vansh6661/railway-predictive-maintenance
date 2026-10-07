"""
Railway Predictive Maintenance — AIML Project (Phase 1: Steps 7 to 12)
- Step 7: Logistic Regression (Baseline Model)
- Step 8: Random Forest Classifier (Ensemble Model)
- Step 9: Model Evaluation (Metrics, Confusion Matrix, ROC-AUC)
- Step 10: Model Comparison
- Step 11: Feature Importance Analysis
- Step 12: Real-time Sample Prediction
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)

# Output directory for evaluation plots
eval_dir = "model_eval_plots"
models_dir = "saved_models"
os.makedirs(eval_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# 1. Load prepared data
print("="*70)
print("LOADING PROCESSED DATA FOR MODEL TRAINING")
print("="*70)
data = np.load("train_test_data.npz", allow_pickle=True)
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']
feature_names = data['feature_names']

print(f"X_train shape: {X_train.shape} | y_train shape: {y_train.shape}")
print(f"X_test  shape: {X_test.shape}  | y_test  shape: {y_test.shape}")
print(f"Total Features: {len(feature_names)}")

# ==============================================================================
# STEP 7: FIRST ML MODEL — LOGISTIC REGRESSION (BASELINE)
# ==============================================================================
print("\n" + "="*70)
print("STEP 7: TRAINING LOGISTIC REGRESSION (BASELINE MODEL)")
print("="*70)
lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr_model.fit(X_train, y_train)

# Predictions
y_pred_lr = lr_model.predict(X_test)
y_prob_lr = lr_model.predict_proba(X_test)[:, 1]

print("✓ Logistic Regression training completed!")

# ==============================================================================
# STEP 8: SECOND ML MODEL — RANDOM FOREST CLASSIFIER
# ==============================================================================
print("\n" + "="*70)
print("STEP 8: TRAINING RANDOM FOREST CLASSIFIER (ENSEMBLE MODEL)")
print("="*70)
# Use n_estimators=100 trees, random_state=42, n_jobs=-1 for fast parallel execution on Mac
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

print("✓ Random Forest Classifier training completed!")

# Save trained models
joblib.dump(lr_model, os.path.join(models_dir, "logistic_regression.joblib"))
joblib.dump(rf_model, os.path.join(models_dir, "random_forest.joblib"))
print("✓ Both models serialized and saved to 'saved_models/'")

# ==============================================================================
# STEP 9: MODEL EVALUATION & METRICS CALCULATION
# ==============================================================================
print("\n" + "="*70)
print("STEP 9: DETAILED EVALUATION ON UNSEEN TEST DATA (20,000 ASSETS)")
print("="*70)

def calculate_metrics(name, y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    return {
        "Model": name,
        "Accuracy (%)": acc * 100,
        "Precision (%)": prec * 100,
        "Recall (%)": rec * 100,
        "F1-Score (%)": f1 * 100,
        "ROC-AUC (%)": roc_auc * 100
    }

lr_metrics = calculate_metrics("Logistic Regression", y_test, y_pred_lr, y_prob_lr)
rf_metrics = calculate_metrics("Random Forest", y_test, y_pred_rf, y_prob_rf)

# ==============================================================================
# STEP 10: MODEL COMPARISON TABLE
# ==============================================================================
print("\n" + "="*70)
print("STEP 10: MODEL COMPARISON SUMMARY")
print("="*70)
comparison_df = pd.DataFrame([lr_metrics, rf_metrics])
print(comparison_df.to_string(index=False))

# Confusion Matrix Visualizations
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt=',d', cmap='Blues', ax=ax1, cbar=False,
            xticklabels=['Healthy (0)', 'Maintenance (1)'],
            yticklabels=['Healthy (0)', 'Maintenance (1)'])
ax1.set_title("Confusion Matrix: Logistic Regression", fontsize=13, fontweight='bold')
ax1.set_xlabel("Predicted Label", fontsize=11)
ax1.set_ylabel("Actual Ground Truth", fontsize=11)

cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt=',d', cmap='Greens', ax=ax2, cbar=False,
            xticklabels=['Healthy (0)', 'Maintenance (1)'],
            yticklabels=['Healthy (0)', 'Maintenance (1)'])
ax2.set_title("Confusion Matrix: Random Forest", fontsize=13, fontweight='bold')
ax2.set_xlabel("Predicted Label", fontsize=11)
ax2.set_ylabel("Actual Ground Truth", fontsize=11)

plt.tight_layout()
cm_plot_path = os.path.join(eval_dir, "confusion_matrices.png")
plt.savefig(cm_plot_path, dpi=300)
plt.close()
print(f"\n✓ Confusion matrices saved to: '{cm_plot_path}'")

# ROC Curve Comparison
plt.figure(figsize=(8, 6))
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {lr_metrics['ROC-AUC (%)']:.1f}%)", color='#2980b9', lw=2)
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {rf_metrics['ROC-AUC (%)']:.1f}%)", color='#27ae60', lw=2)
plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7, label="Random Guess Line")
plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
plt.title("ROC Curves Comparison (Receiver Operating Characteristic)", fontsize=13, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

roc_plot_path = os.path.join(eval_dir, "roc_curves_comparison.png")
plt.savefig(roc_plot_path, dpi=300)
plt.close()
print(f"✓ ROC curves saved to: '{roc_plot_path}'")

# ==============================================================================
# STEP 11: FEATURE IMPORTANCE (RANDOM FOREST)
# ==============================================================================
print("\n" + "="*70)
print("STEP 11: FEATURE IMPORTANCE ANALYSIS (RANDOM FOREST)")
print("="*70)

importances = rf_model.feature_importances_
feat_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("\nTop 15 Most Influential Features:")
print(feat_imp_df.head(15).to_string(index=False))

# Plot top 15 features
plt.figure(figsize=(10, 7))
sns.barplot(
    data=feat_imp_df.head(15),
    x='Importance',
    y='Feature',
    palette='viridis'
)
plt.title("Top 15 Predictive Features (Random Forest Importance)", fontsize=13, fontweight='bold')
plt.xlabel("Gini Feature Importance Score", fontsize=11)
plt.ylabel("Railway Sensor / Asset Feature", fontsize=11)
plt.tight_layout()

feat_plot_path = os.path.join(eval_dir, "top_feature_importances.png")
plt.savefig(feat_plot_path, dpi=300)
plt.close()
print(f"✓ Feature importance chart saved to: '{feat_plot_path}'")

# ==============================================================================
# STEP 12: REAL-TIME SAMPLE PREDICTION DEMONSTRATION
# ==============================================================================
print("\n" + "="*70)
print("STEP 12: MAKING A REAL-TIME PREDICTION ON A NEW RAILWAY ASSET")
print("="*70)

# Load preprocessor
preprocessor = joblib.load("preprocessor.joblib")

# Create 2 realistic sample railway asset profiles:
# Asset A: Normal healthy passenger rake
# Asset B: Degraded rake with high wear, elevated bearing temperature, low inspection score
sample_assets = pd.DataFrame([
    {
        'region': 'Western Railway',
        'season': 'Winter',
        'train_type': 'Passenger',
        'train_age_years': 4,
        'average_speed_kmph': 72.0,
        'distance_travelled_km': 120000,
        'track_temperature_c': 28.0,
        'rail_wear_mm': 3.2,
        'track_vibration_level': 1.8,
        'ballast_condition': 'Good',
        'track_curvature_degree': 2.1,
        'ambient_temperature_c': 24.0,
        'humidity_percent': 50.0,
        'rainfall_mm': 2.0,
        'wind_speed_kmph': 12.0,
        'wheel_wear_percent': 22.0,
        'axle_temperature_c': 48.0,
        'brake_pressure_psi': 98.0,
        'brake_pad_wear_percent': 20.0,
        'bearing_temperature_c': 52.0,
        'battery_voltage': 25.2,
        'traction_motor_temp_c': 62.0,
        'signal_system_status': 'Normal',
        'power_consumption_kw': 620.0,
        'load_factor_percent': 65.0,
        'daily_trips': 10,
        'delay_minutes': 4.0,
        'last_maintenance_days': 25.0,
        'inspection_score': 95.0,
        'sensor_health_index': 92.0
    },
    {
        'region': 'Central Railway',
        'season': 'Summer',
        'train_type': 'Freight',
        'train_age_years': 26,
        'average_speed_kmph': 98.0,
        'distance_travelled_km': 1350000,
        'track_temperature_c': 58.0,
        'rail_wear_mm': 18.5,
        'track_vibration_level': 8.9,
        'ballast_condition': 'Poor',
        'track_curvature_degree': 16.4,
        'ambient_temperature_c': 44.0,
        'humidity_percent': 78.0,
        'rainfall_mm': 65.0,
        'wind_speed_kmph': 38.0,
        'wheel_wear_percent': 89.0,
        'axle_temperature_c': 108.0,
        'brake_pressure_psi': 46.0,
        'brake_pad_wear_percent': 92.0,
        'bearing_temperature_c': 118.0,
        'battery_voltage': 17.5,
        'traction_motor_temp_c': 132.0,
        'signal_system_status': 'Fault',
        'power_consumption_kw': 1480.0,
        'load_factor_percent': 95.0,
        'daily_trips': 22,
        'delay_minutes': 64.0,
        'last_maintenance_days': 330.0,
        'inspection_score': 45.0,
        'sensor_health_index': 40.0
    }
])

# Preprocess sample assets using saved preprocessor
sample_processed = preprocessor.transform(sample_assets)

# Run Inference with both models
pred_rf = rf_model.predict(sample_processed)
prob_rf = rf_model.predict_proba(sample_processed)

print("\n--- SAMPLE INFERENCE RESULTS (Random Forest) ---")
for i, asset_desc in enumerate(["Asset #1 (Healthy Passenger Train)", "Asset #2 (Critical Heavy Freight Train)"]):
    decision = "🚨 YES (MAINTENANCE REQUIRED)" if pred_rf[i] == 1 else "✅ NO (HEALTHY - NORMAL OPERATION)"
    maint_prob = prob_rf[i][1] * 100
    healthy_prob = prob_rf[i][0] * 100
    print(f"\n{asset_desc}:")
    print(f"  -> Prediction          : {decision}")
    print(f"  -> Probability of Need : {maint_prob:.2f}% (Healthy Confidence: {healthy_prob:.2f}%)")

print("\n" + "="*70)
print("PHASE 1 ALL STEPS (1-12) COMPLETED SUCCESSFULLY!")
print("="*70)
