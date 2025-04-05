# 011_TrainPromotionModel.py
# Purpose: Trains a classifier to predict Promotion Status and saves the model.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np
import os

print("Starting Promotion Predictor Model Training...")

# --- Configuration ---
DATA_FILE = 'k12_tutoring_dataset.csv'
MODEL_SAVE_FILE = 'promotion_predictor_model.pkl' # Separate file for this model

# Features relevant for general promotion prediction (excluding direct score)
# Earning_Class added as it might correlate with resources/support affecting promotion
ORIGINAL_FEATURES_PROMO = ['Age', 'IQ', 'Time_Per_Day', 'Level_Student', 'Earning_Class']
CATEGORICAL_FEATURES_PROMO = ['Level_Student', 'Earning_Class'] # Features needing encoding
TARGET_PROMO = 'Promotion_Status'

# --- Load Data ---
if not os.path.exists(DATA_FILE):
    print(f"FATAL ERROR: Data file '{DATA_FILE}' not found.")
    print("Please run '007_DataGenerator.py' first.")
    exit()
try:
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded data from {DATA_FILE} ({len(df)} rows)")
except Exception as e:
    print(f"FATAL ERROR: Could not load data file {DATA_FILE}: {e}")
    exit()

# --- Feature Engineering & Preprocessing ---
print("\nPreprocessing data for promotion prediction...")

# 1. Encode the TARGET variable ('Yes'/'No' to 1/0)
target_encoder = LabelEncoder()
if TARGET_PROMO not in df.columns:
    print(f"FATAL ERROR: Target column '{TARGET_PROMO}' not found in the data.")
    exit()

df[TARGET_PROMO + '_Encoded'] = target_encoder.fit_transform(df[TARGET_PROMO])
print(f"Encoded target '{TARGET_PROMO}'. Mappings: {dict(zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_)))}")
# Store the mapping for decoding later: 0 -> No, 1 -> Yes (based on alphabetical order usually)
target_mapping = dict(zip(target_encoder.transform(target_encoder.classes_), target_encoder.classes_))


# 2. Encode CATEGORICAL FEATURES using LabelEncoder
encoders_promo = {}
df_processed = df.copy()

for col in CATEGORICAL_FEATURES_PROMO:
    if col in df_processed.columns:
        le = LabelEncoder()
        df_processed[col + '_Encoded'] = le.fit_transform(df_processed[col])
        encoders_promo[col] = le # Store encoder
        print(f"  Encoded feature '{col}'. Mappings: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    else:
        print(f"  Warning: Categorical feature '{col}' not found.")

# 3. Prepare feature matrix (X) and target vector (y)
encoded_feature_names_promo = []
for feature in ORIGINAL_FEATURES_PROMO:
    encoded_col = feature + '_Encoded'
    if encoded_col in df_processed.columns:
        encoded_feature_names_promo.append(encoded_col) # Use the encoded version
    elif feature in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[feature]):
         encoded_feature_names_promo.append(feature) # Keep original numeric features
    else:
        print(f"  Warning: Feature '{feature}' (or encoded) not found/numeric. Skipping.")

if not encoded_feature_names_promo:
    print("FATAL ERROR: No valid features selected for promotion model.")
    exit()

print(f"\nUsing final features for promotion training: {encoded_feature_names_promo}")

X_promo = df_processed[encoded_feature_names_promo]
y_promo = df_processed[TARGET_PROMO + '_Encoded'] # Use the encoded target

# --- Train-Test Split ---
X_train_promo, X_test_promo, y_train_promo, y_test_promo = train_test_split(
    X_promo, y_promo, test_size=0.2, random_state=42, stratify=y_promo # Stratify for classification
)
print(f"\nData split: Train={len(X_train_promo)}, Test={len(X_test_promo)}")

# --- Model Training (RandomForestClassifier) ---
print("\nTraining RandomForestClassifier model for promotion...")
promo_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
promo_model.fit(X_train_promo, y_train_promo)
print("Model training complete.")

# --- Evaluation ---
print("\nEvaluating promotion model performance...")
y_pred_promo_test = promo_model.predict(X_test_promo)

accuracy = accuracy_score(y_test_promo, y_pred_promo_test)
report = classification_report(y_test_promo, y_pred_promo_test, target_names=target_encoder.classes_)

print(f"  Test Accuracy: {accuracy:.3f}")
print("  Classification Report (Test Set):\n", report)

# --- Save Model, Features, Encoders, and Target Mapping ---
print(f"\nSaving promotion model artifacts to {MODEL_SAVE_FILE}...")
joblib.dump({
    'model': promo_model,
    'feature_names_encoded': encoded_feature_names_promo, # Encoded feature names
    'encoders': encoders_promo,                 # Feature encoders
    'target_encoder': target_encoder,           # Target encoder (to get class names)
    'target_mapping': target_mapping          # Direct mapping (e.g., {0: 'No', 1: 'Yes'})
    }, MODEL_SAVE_FILE)
print("Promotion model artifacts saved successfully.")

# --- Feature Importance Plot (Optional) ---
print("\nGenerating feature importance plot for promotion model...")
try:
    import matplotlib.pyplot as plt
    importances = promo_model.feature_importances_
    indices = np.argsort(importances)[::-1] # Descending order

    plt.figure(figsize=(8, 5))
    plt.title('Feature Importances for Promotion Prediction')
    plt.bar(range(X_promo.shape[1]), importances[indices], align='center')
    plt.xticks(range(X_promo.shape[1]), [encoded_feature_names_promo[i] for i in indices], rotation=45, ha="right")
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig('promotion_predictor_feature_importance.png') # Save the plot
    print("Feature importance plot saved as promotion_predictor_feature_importance.png")
    # plt.show()
except ImportError:
    print("  Warning: Matplotlib not found. Skipping feature importance plot generation.")
except Exception as e:
    print(f"  Error generating plot: {e}")

print("\n✅ Promotion predictor training finished.")