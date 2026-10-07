"""
Step 4: Exploratory Data Analysis (EDA) for Railway Predictive Maintenance
AIML Subject Project - Phase 1
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set clean aesthetic styling for plots
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({'font.sans-serif': 'Helvetica', 'axes.edgecolor': '#cccccc'})

# Create directory to store visualizations
output_dir = "eda_plots"
os.makedirs(output_dir, exist_ok=True)

# 1. Load the cleaned dataset
print("Loading cleaned dataset for EDA...")
df = pd.read_csv("cleaned_railway_data.csv")
print(f"Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ==============================================================================
# PLOT 1: Target Variable & Failure Severity Distribution
# ==============================================================================
print("\n1. Generating Plot 1: Target Variable & Severity Distribution...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Target count plot
target_counts = df['maintenance_required'].value_counts()
target_labels = ['0 (Normal / No Action)', '1 (Maintenance Required)']
colors = ['#2ecc71', '#e74c3c']
bars = ax1.bar(target_labels, target_counts, color=colors, edgecolor='black', alpha=0.85)
ax1.set_title('Target Distribution: Maintenance Required', fontsize=14, fontweight='bold')
ax1.set_ylabel('Number of Assets', fontsize=12)
for bar in bars:
    height = bar.get_height()
    ax1.annotate(f'{height:,}\n({height/len(df)*100:.1f}%)',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 4), textcoords="offset points",
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

# Failure Severity breakdown (for assets needing maintenance)
failing_df = df[df['maintenance_required'] == 1]
severity_counts = failing_df['failure_severity'].value_counts()
ax2.pie(severity_counts, labels=severity_counts.index, autopct='%1.1f%%',
        colors=['#f39c12', '#e67e22', '#e74c3c', '#9b59b6'], startangle=140,
        wedgeprops=dict(edgecolor='black', alpha=0.85))
ax2.set_title('Failure Severity Breakdown (When Target = 1)', fontsize=14, fontweight='bold')

plt.tight_layout()
p1_path = os.path.join(output_dir, "01_target_distribution.png")
plt.savefig(p1_path, dpi=300)
plt.close()
print(f"   Saved to: {p1_path}")

# ==============================================================================
# PLOT 2: Key Sensor Readings by Maintenance Requirement (Boxplots)
# ==============================================================================
print("2. Generating Plot 2: Key Physical Sensors vs Maintenance...")
key_sensors = [
    ('rail_wear_mm', 'Rail Wear (mm)'),
    ('wheel_wear_percent', 'Wheel Wear (%)'),
    ('brake_pad_wear_percent', 'Brake Pad Wear (%)'),
    ('track_vibration_level', 'Track Vibration Level'),
    ('bearing_temperature_c', 'Bearing Temp (°C)'),
    ('axle_temperature_c', 'Axle Temp (°C)')
]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, (col, title) in enumerate(key_sensors):
    sns.boxplot(
        data=df,
        x='maintenance_required',
        y=col,
        hue='maintenance_required',
        palette=['#2ecc71', '#e74c3c'],
        legend=False,
        ax=axes[idx]
    )
    axes[idx].set_title(f'{title} by Maintenance Need', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Maintenance Required (0 = No, 1 = Yes)', fontsize=10)
    axes[idx].set_ylabel(title, fontsize=10)

plt.tight_layout()
p2_path = os.path.join(output_dir, "02_sensor_distributions_by_target.png")
plt.savefig(p2_path, dpi=300)
plt.close()
print(f"   Saved to: {p2_path}")

# ==============================================================================
# PLOT 3: Maintenance Rate across Railway Zones and Train Types
# ==============================================================================
print("3. Generating Plot 3: Maintenance Rate across Zones and Train Types...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

# By Region
region_rates = df.groupby('region')['maintenance_required'].mean().reset_index()
region_rates['maintenance_rate_%'] = region_rates['maintenance_required'] * 100
sns.barplot(
    data=region_rates.sort_values('maintenance_rate_%', ascending=False),
    x='maintenance_rate_%',
    y='region',
    hue='region',
    palette='Blues_r',
    legend=False,
    ax=ax1
)
ax1.set_title('Maintenance Requirement Rate by Railway Zone', fontsize=13, fontweight='bold')
ax1.set_xlabel('Maintenance Required Rate (%)', fontsize=11)
ax1.set_ylabel('Railway Zone', fontsize=11)

# By Train Type
type_rates = df.groupby('train_type')['maintenance_required'].mean().reset_index()
type_rates['maintenance_rate_%'] = type_rates['maintenance_required'] * 100
sns.barplot(
    data=type_rates.sort_values('maintenance_rate_%', ascending=False),
    x='train_type',
    y='maintenance_rate_%',
    hue='train_type',
    palette='Purples_r',
    legend=False,
    ax=ax2
)
ax2.set_title('Maintenance Requirement Rate by Train Type', fontsize=13, fontweight='bold')
ax2.set_ylabel('Maintenance Required Rate (%)', fontsize=11)
ax2.set_xlabel('Train Type', fontsize=11)

plt.tight_layout()
p3_path = os.path.join(output_dir, "03_categorical_maintenance_rates.png")
plt.savefig(p3_path, dpi=300)
plt.close()
print(f"   Saved to: {p3_path}")

# ==============================================================================
# PLOT 4: Correlation Matrix of Key Numeric Features
# ==============================================================================
print("4. Generating Plot 4: Correlation Heatmap...")
corr_features = [
    'maintenance_required', 'risk_score', 'inspection_score',
    'sensor_health_index', 'last_maintenance_days', 'rail_wear_mm',
    'wheel_wear_percent', 'brake_pad_wear_percent', 'track_vibration_level',
    'bearing_temperature_c', 'axle_temperature_c', 'traction_motor_temp_c',
    'power_consumption_kw', 'delay_minutes'
]

corr_matrix = df[corr_features].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1,
            linewidths=0.5, cbar_kws={'label': 'Pearson Correlation Coefficient'})
plt.title('Correlation Matrix of Key Railway Features', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)

plt.tight_layout()
p4_path = os.path.join(output_dir, "04_correlation_heatmap.png")
plt.savefig(p4_path, dpi=300)
plt.close()
print(f"   Saved to: {p4_path}")

# ==============================================================================
# Compute and print key statistical insights
# ==============================================================================
print("\n" + "="*70)
print("KEY STATISTICAL INSIGHTS DISCOVERED")
print("="*70)

# 1. Target balance
n_healthy = (df['maintenance_required'] == 0).sum()
n_maint = (df['maintenance_required'] == 1).sum()
print(f"1. Target Balance: Healthy = {n_healthy:,} (79.8%), Maintenance = {n_maint:,} (20.2%)")

# 2. Key sensor comparisons (Mean healthy vs Mean maintenance)
print("\n2. Comparing Sensor Means (Healthy vs Needing Maintenance):")
for col, name in key_sensors:
    mean_0 = df[df['maintenance_required'] == 0][col].mean()
    mean_1 = df[df['maintenance_required'] == 1][col].mean()
    diff_pct = ((mean_1 - mean_0) / mean_0) * 100
    print(f"  - {name:<25}: Healthy = {mean_0:.2f} | Needing Maintenance = {mean_1:.2f} (Shift: {diff_pct:+.1f}%)")

# 3. Correlation with maintenance_required
print("\n3. Top Correlations with Target (maintenance_required):")
target_corr = corr_matrix['maintenance_required'].sort_values(ascending=False)
for col, val in target_corr.items():
    if col != 'maintenance_required':
        print(f"  - {col:<26}: {val:+.3f}")
print("="*70)
