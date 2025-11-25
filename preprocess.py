# preprocess.py
"""
Unified preprocessing script supporting multiple projects (diabetes, heart).
It produces artifacts in DATA_DIR:
  X_train.pkl, X_test.pkl, y_train.pkl, y_test.pkl,
  scaler.pkl, imputer.pkl, X_train_unscaled_df.pkl, X_test_unscaled_df.pkl

Behavior:
- PROJECT_NAME environment variable selects which preprocessing to run
  - 'diabetes' (default) uses the original diabetes flow
  - 'heart' uses a heart-disease friendly preprocessing
"""
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

def preprocess_diabetes(data_url=None, out_dir=OUT_DIR):
    print("[preprocess] Running diabetes preprocessing")
    columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
    url = data_url or os.getenv("DATA_URL", "https://raw.githubusercontent.com/Sh123-max/mlops-agp/main/diabetes.csv")
    df = pd.read_csv(url, names=columns, header=0)
    na_columns = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
    df[na_columns] = df[na_columns].replace(0, np.nan)

    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # scale before knn-impute
    scaler_before_impute = StandardScaler()
    X_scaled_for_knn = scaler_before_impute.fit_transform(X)

    imputer = KNNImputer(n_neighbors=5)
    X_imputed_scaled = imputer.fit_transform(X_scaled_for_knn)
    X_imputed = pd.DataFrame(scaler_before_impute.inverse_transform(X_imputed_scaled), columns=X.columns)

    # sensible clipping
    X_imputed['BloodPressure'] = X_imputed['BloodPressure'].clip(40,140)
    X_imputed['BMI'] = X_imputed['BMI'].clip(15,50)
    X_imputed['Glucose'] = X_imputed['Glucose'].clip(50,200)

    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(X_train_scaled, os.path.join(out_dir, "X_train.pkl"))
    joblib.dump(X_test_scaled, os.path.join(out_dir, "X_test.pkl"))
    joblib.dump(y_train.reset_index(drop=True), os.path.join(out_dir, "y_train.pkl"))
    joblib.dump(y_test.reset_index(drop=True), os.path.join(out_dir, "y_test.pkl"))
    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))
    joblib.dump(imputer, os.path.join(out_dir, "imputer.pkl"))

    # keep unscaled train/test DataFrames for drift baselines
    joblib.dump(X_train.reset_index(drop=True), os.path.join(out_dir, "X_train_unscaled_df.pkl"))
    joblib.dump(X_test.reset_index(drop=True), os.path.join(out_dir, "X_test_unscaled_df.pkl"))

    print("[preprocess] Diabetes preprocessing completed and saved to", out_dir)

def preprocess_heart(data_url=None, out_dir=OUT_DIR):
    """
    Basic heart-disease preprocessing (generic). Expects a CSV with a label column
    named 'target' or 'Outcome'. Handles simple imputation + scaling.
    Typical heart csv columns: age, sex, cp, trestbps, chol, fbs, restecg,
    thalach, exang, oldpeak, slope, ca, thal, target
    """
    print("[preprocess] Running heart-disease preprocessing")
    url = data_url or os.getenv("DATA_URL", "https://raw.githubusercontent.com/Sh123-max/mlops-agp/main/heart.csv")
    df = pd.read_csv(url)

    # detect label
    if "target" in df.columns:
        label_col = "target"
    elif "Outcome" in df.columns:
        label_col = "Outcome"
    elif "y" in df.columns:
        label_col = "y"
    else:
        raise RuntimeError("Heart dataset must contain a 'target' or 'Outcome' column")

    X = df.drop(columns=[label_col])
    y = df[label_col]

    # Simple numeric-only pipeline: impute (median) then scale
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X_num = X[numeric_cols].copy()

    # Fill simple missing values with median (works for many heart datasets)
    imputer = SimpleImputer(strategy="median")
    X_num_imputed = imputer.fit_transform(X_num)
    X_num_imputed = pd.DataFrame(X_num_imputed, columns=numeric_cols)

    # If there are categorical non-numeric columns, one-hot encode them
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if cat_cols:
        X_cat = pd.get_dummies(X[cat_cols].astype(str), drop_first=True)
        X_processed = pd.concat([X_num_imputed.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
    else:
        X_processed = X_num_imputed

    # Train-test split with stratify if possible
    try:
        X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save artifacts with same filenames used by training
    joblib.dump(X_train_scaled, os.path.join(out_dir, "X_train.pkl"))
    joblib.dump(X_test_scaled, os.path.join(out_dir, "X_test.pkl"))
    joblib.dump(y_train.reset_index(drop=True), os.path.join(out_dir, "y_train.pkl"))
    joblib.dump(y_test.reset_index(drop=True), os.path.join(out_dir, "y_test.pkl"))
    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))
    joblib.dump(imputer, os.path.join(out_dir, "imputer.pkl"))

    # Keep unscaled train/test frames (useful for drift)
    joblib.dump(X_train.reset_index(drop=True), os.path.join(out_dir, "X_train_unscaled_df.pkl"))
    joblib.dump(X_test.reset_index(drop=True), os.path.join(out_dir, "X_test_unscaled_df.pkl"))

    print("[preprocess] Heart preprocessing completed and saved to", out_dir)

def run_preprocess():
    data_url = os.getenv("DATA_URL", None)
    if PROJECT_NAME in ("heart", "heart-disease", "heart_disease"):
        preprocess_heart(data_url=data_url, out_dir=OUT_DIR)
    else:
        preprocess_diabetes(data_url=data_url, out_dir=OUT_DIR)

if __name__ == "__main__":
    run_preprocess()
