

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np
import matplotlib.pyplot as plt 

print("Starting Score Predictor Model Training...")

DATA_FILE = 'k12_tutoring_dataset.csv'
MODEL_SAVE_FILE = 'score_predictor_model.pkl'

ORIGINAL_FEATURES = ['Age', 'IQ', 'Time_Per_Day', 'Level_Student', 'Course_Name', 'Material_Level']
TARGET = 'Assessment_Score'
CATEGORICAL_FEATURES = ['Level_Student', 'Course_Name', 'Material_Level'] 


try:
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded data from {DATA_FILE} ({len(df)} rows)")
except FileNotFoundError:
    print(f"Error: {DATA_FILE} not found. Please run 007_DataGenerator.py first.")
    exit()
print("\nPreprocessing data and encoding features...")

encoders = {}
df_processed = df.copy()

for col in CATEGORICAL_FEATURES:
    if col in df_processed.columns:
        le = LabelEncoder()
        
        df_processed[col + '_Encoded'] = le.fit_transform(df_processed[col])
        encoders[col] = le 
        print(f"  Encoded '{col}'. Mappings: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    else:
        print(f"  Warning: Categorical feature '{col}' not found in DataFrame.")


encoded_feature_names = []
for feature in ORIGINAL_FEATURES:
    encoded_col = feature + '_Encoded'
    if encoded_col in df_processed.columns:
        encoded_feature_names.append(encoded_col) 
    elif feature in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[feature]):
         encoded_feature_names.append(feature) 
    else:
        print(f"  Warning: Feature '{feature}' or its encoded version not found or not numeric. Skipping.")

if not encoded_feature_names:
    print("Error: No valid features selected for modeling. Check ORIGINAL_FEATURES and CATEGORICAL_FEATURES.")
    exit()

print(f"\nUsing final features for training: {encoded_feature_names}")

X = df_processed[encoded_feature_names]
y = df_processed[TARGET]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nData split: Train={len(X_train)}, Test={len(X_test)}")


print("\nTraining RandomForestRegressor model...")

model = RandomForestRegressor(n_estimators=150, 
                              random_state=42,
                              n_jobs=-1,         
                              max_depth=15,     
                              min_samples_split=5, 
                              min_samples_leaf=3) 
model.fit(X_train, y_train)
print("Model training complete.")


print("\nEvaluating model performance...")
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
r2_train = r2_score(y_train, y_pred_train)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_test = r2_score(y_test, y_pred_test)

print(f"  Train RMSE: {rmse_train:.2f}, R2: {r2_train:.2f}")
print(f"  Test  RMSE: {rmse_test:.2f}, R2: {r2_test:.2f}")
print(f"  (RMSE = Root Mean Squared Error - Lower is better)")
print(f"  (R2 = Coefficient of Determination - Closer to 1 is better)")


print(f"\nSaving model artifacts to {MODEL_SAVE_FILE}...")
joblib.dump({
    'model': model,
    'feature_names_encoded': encoded_feature_names,
    'encoders': encoders,                
    'original_features': ORIGINAL_FEATURES 
    }, MODEL_SAVE_FILE)
print("Model artifacts saved successfully.")


print("\nGenerating feature importance plot...")
try:
    importances = model.feature_importances_
    
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title('Feature Importances for Score Prediction')
    plt.bar(range(X_train.shape[1]), importances[indices], align='center')
    plt.xticks(range(X_train.shape[1]), [encoded_feature_names[i] for i in indices], rotation=45, ha="right")
    plt.xlabel('Feature')
    plt.ylabel('Relative Importance')
    plt.tight_layout()
    plt.savefig('score_predictor_feature_importance.png') 
    print("Feature importance plot saved as score_predictor_feature_importance.png")

except ImportError:
    print("  Warning: Matplotlib not found. Skipping feature importance plot generation.")
except Exception as e:
    print(f"  Error generating plot: {e}")

print("\n✅ Score predictor training finished.")