"""
Step 5 & 6: Feature Selection & Data Preparation for Machine Learning
AIML Subject Project - Phase 1
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

# 1. Load the cleaned dataset
print("Loading cleaned dataset...")
df = pd.read_csv("cleaned_railway_data.csv")

# ==============================================================================
# STEP 5: DEFINE TARGET (y) AND INPUT FEATURES (X)
# ==============================================================================
# Target: maintenance_required (Binary Classification: 0 or 1)
y = df['maintenance_required']

# Drop columns that are IDs or Post-Failure Leakage
drop_cols = [
    'train_id',              # Identifier only (arbitrary integer)
    'maintenance_required',  # This is the target itself
    'failure_type',          # Post-failure outcome (Data Leakage)
    'failure_severity',      # Post-failure outcome (Data Leakage)
    'risk_score'             # Derived output score
]

X = df.drop(columns=drop_cols)

# Separate Numerical and Categorical feature columns
categorical_features = ['region', 'season', 'train_type', 'ballast_condition', 'signal_system_status']
numerical_features = [col for col in X.columns if col not in categorical_features]

print("\n" + "="*60)
print("FEATURE MATRIX (X) AND TARGET VECTOR (y) SUMMARY")
print("="*60)
print(f"Target (y) shape: {y.shape} (Binary: {y.nunique()} classes)")
print(f"Features (X) shape: {X.shape} ({len(numerical_features)} numerical, {len(categorical_features)} categorical)")
print(f"\nCategorical Features ({len(categorical_features)}): {categorical_features}")
print(f"Numerical Features ({len(numerical_features)}): {numerical_features[:6]} ... (+ {len(numerical_features)-6} more)")

# ==============================================================================
# STEP 6: TRAIN / TEST SPLIT (80% Train, 20% Test)
# ==============================================================================
# stratify=y ensures both Train and Test sets have the exact same 80% / 20% class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n" + "="*60)
print("TRAIN / TEST SPLIT SUMMARY (80% / 20%)")
print("="*60)
print(f"X_train: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(X)*100:.0f}%)")
print(f"X_test : {X_test.shape[0]:,} samples ({X_test.shape[0]/len(X)*100:.0f}%)")
print(f"y_train class distribution: 0: {(y_train == 0).sum():,} | 1: {(y_train == 1).sum():,}")
print(f"y_test  class distribution: 0: {(y_test == 0).sum():,}  | 1: {(y_test == 1).sum():,}")

# ==============================================================================
# PREPROCESSING PIPELINE DEFINITION
# ==============================================================================
# Numerical features -> StandardScaler (mean=0, std=1)
# Categorical features -> OneHotEncoder (converts strings like 'Summer', 'Winter' to binary columns)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ]
)

print("\nFitting preprocessor on Training Data only (preventing Data Leakage)...")
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Get feature names after one-hot encoding
cat_encoded_names = list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
all_processed_feature_names = numerical_features + cat_encoded_names

print(f"Total processed feature dimensions: {X_train_processed.shape[1]} columns")
print(f"Sample encoded features: {all_processed_feature_names[:8]}")

# Save the split datasets and preprocessor for Step 7 and Step 8
joblib.dump(preprocessor, "preprocessor.joblib")
np.savez_compressed(
    "train_test_data.npz",
    X_train=X_train_processed,
    X_test=X_test_processed,
    y_train=y_train.values,
    y_test=y_test.values,
    feature_names=all_processed_feature_names
)
print("\nPreprocessor and split data successfully saved to disk!")
print("="*60)
