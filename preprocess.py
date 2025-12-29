import os
import pandas as pd
import numpy as np
import joblib
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

PROJECT_NAME = os.getenv("PROJECT_NAME", "diabetes").lower()
DATA_URL = os.getenv("DATA_URL", None)
OUT_DIR = os.getenv("DATA_DIR", "data")
os.makedirs(OUT_DIR, exist_ok=True)

# --- Feature Engineering Logic ---

def engineer_diabetes_features(df):
    """Adds domain-specific features for Diabetes prediction (Total 13)."""
    # Original 3
    df['BMI_Age_Interaction'] = df['BMI'] * df['Age']
    df['Glucose_Insulin_Ratio'] = df['Glucose'] / (df['Insulin'] + 0.1)
    df['Is_Senior'] = (df['Age'] > 50).astype(int)
    
    # NEW: Add 2 more to satisfy the 13-feature requirement of your model
    df['Glucose_Age_Interaction'] = df['Glucose'] * df['Age']
    df['BMI_Age_Ratio'] = df['BMI'] / (df['Age'] + 1)
    
    return df

def engineer_heart_features(df):
    """Adds domain-specific features for Heart Disease prediction."""
    # 1. Risk Score: Combined high blood pressure and high cholesterol
    # Using common medical thresholds (e.g., Systolic > 130 or Chol > 240)
    # Note: These thresholds depend on your dataset's specific units
    df['High_Risk_Combo'] = ((df['trestbps'] > 130) & (df['chol'] > 240)).astype(int)
   
    # 2. Maximum Heart Rate relative to Age
    # Estimated max HR is often 220 - age
    df['HR_Efficiency'] = df['thalach'] / (220 - df['age'])
   
    return df

# --- Main Preprocessing Functions ---

def preprocess_diabetes(data_url=None, out_dir=OUT_DIR):
    print("[preprocess] Running diabetes preprocessing with Feature Engineering")
    columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
    url = data_url or os.getenv("DATA_URL", "https://raw.githubusercontent.com/Sh123-max/mlops-agp/main/diabetes_new.csv")
    df = pd.read_csv(url, names=columns, header=0)
   
    na_columns = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
    df[na_columns] = df[na_columns].replace(0, np.nan)

    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # Scale before KNN-impute
    scaler_before_impute = StandardScaler()
    X_scaled_for_knn = scaler_before_impute.fit_transform(X)

    imputer = KNNImputer(n_neighbors=5)
    X_imputed_scaled = imputer.fit_transform(X_scaled_for_knn)
    X_imputed = pd.DataFrame(scaler_before_impute.inverse_transform(X_imputed_scaled), columns=X.columns)

    # Sensible clipping
    X_imputed['BloodPressure'] = X_imputed['BloodPressure'].clip(40,140)
    X_imputed['BMI'] = X_imputed['BMI'].clip(15,50)
    X_imputed['Glucose'] = X_imputed['Glucose'].clip(50,200)

    # --- FEATURE ENGINEERING ---
    X_engineered = engineer_diabetes_features(X_imputed)

    X_train, X_test, y_train, y_test = train_test_split(X_engineered, y, test_size=0.2, random_state=42, stratify=y)
   
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save artifacts
    joblib.dump(X_train_scaled, os.path.join(out_dir, "X_train.pkl"))
    joblib.dump(X_test_scaled, os.path.join(out_dir, "X_test.pkl"))
    joblib.dump(y_train.reset_index(drop=True), os.path.join(out_dir, "y_train.pkl"))
    joblib.dump(y_test.reset_index(drop=True), os.path.join(out_dir, "y_test.pkl"))
    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))
    joblib.dump(imputer, os.path.join(out_dir, "imputer.pkl"))
    joblib.dump(X_train.reset_index(drop=True), os.path.join(out_dir, "X_train_unscaled_df.pkl"))
    joblib.dump(X_test.reset_index(drop=True), os.path.join(out_dir, "X_test_unscaled_df.pkl"))

    print("[preprocess] Diabetes preprocessing completed.")

def preprocess_heart(data_url=None, out_dir=OUT_DIR):
    print("[preprocess] Running heart-disease preprocessing with Feature Engineering")
    url = data_url or os.getenv("DATA_URL", "https://raw.githubusercontent.com/Sh123-max/mlops-agp/main/heart_new.csv")
    df = pd.read_csv(url)

    if "target" in df.columns: label_col = "target"
    elif "Outcome" in df.columns: label_col = "Outcome"
    else: label_col = "y"

    X = df.drop(columns=[label_col])
    y = df[label_col]

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X_num = X[numeric_cols].copy()

    imputer = SimpleImputer(strategy="median")
    X_num_imputed = pd.DataFrame(imputer.fit_transform(X_num), columns=numeric_cols)

    # --- FEATURE ENGINEERING ---
    X_engineered = engineer_heart_features(X_num_imputed)

    # Categorical handling
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if cat_cols:
        X_cat = pd.get_dummies(X[cat_cols].astype(str), drop_first=True)
        X_processed = pd.concat([X_engineered.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
    else:
        X_processed = X_engineered

    try:
        X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save artifacts
    joblib.dump(X_train_scaled, os.path.join(out_dir, "X_train.pkl"))
    joblib.dump(X_test_scaled, os.path.join(out_dir, "X_test.pkl"))
    joblib.dump(y_train.reset_index(drop=True), os.path.join(out_dir, "y_train.pkl"))
    joblib.dump(y_test.reset_index(drop=True), os.path.join(out_dir, "y_test.pkl"))
    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))
    joblib.dump(imputer, os.path.join(out_dir, "imputer.pkl"))
    joblib.dump(X_train.reset_index(drop=True), os.path.join(out_dir, "X_train_unscaled_df.pkl"))
    joblib.dump(X_test.reset_index(drop=True), os.path.join(out_dir, "X_test_unscaled_df.pkl"))

    print("[preprocess] Heart preprocessing completed.")

def run_preprocess():
    data_url = os.getenv("DATA_URL", None)
    if PROJECT_NAME in ("heart", "heart-disease", "heart_disease"):
        preprocess_heart(data_url=data_url, out_dir=OUT_DIR)
    else:
        preprocess_diabetes(data_url=data_url, out_dir=OUT_DIR)

if __name__ == "__main__":
    run_preprocess()
