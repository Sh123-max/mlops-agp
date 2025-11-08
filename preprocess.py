# preprocess.py
import pandas as pd
import numpy as np
import os
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

DATA_URL = os.getenv("DATA_URL", "https://raw.githubusercontent.com/madhav481010/Diabetes-Prediction/main/diabetes.csv")
OUT_DIR = os.getenv("DATA_DIR", "data")
os.makedirs(OUT_DIR, exist_ok=True)

def run_preprocess():
    columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
    df = pd.read_csv(DATA_URL, names=columns, header=0)

    na_columns = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
    df[na_columns] = df[na_columns].replace(0, np.nan)

    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    scaler_before_impute = StandardScaler()
    X_scaled_for_knn = scaler_before_impute.fit_transform(X)

    imputer = KNNImputer(n_neighbors=5)
    X_imputed_scaled = imputer.fit_transform(X_scaled_for_knn)
    X_imputed = pd.DataFrame(scaler_before_impute.inverse_transform(X_imputed_scaled), columns=X.columns)

    X_imputed['BloodPressure'] = X_imputed['BloodPressure'].clip(40,140)
    X_imputed['BMI'] = X_imputed['BMI'].clip(15,50)
    X_imputed['Glucose'] = X_imputed['Glucose'].clip(50,200)

    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(X_train_scaled, os.path.join(OUT_DIR, "X_train.pkl"))
    joblib.dump(X_test_scaled, os.path.join(OUT_DIR, "X_test.pkl"))
    joblib.dump(y_train, os.path.join(OUT_DIR, "y_train.pkl"))
    joblib.dump(y_test, os.path.join(OUT_DIR, "y_test.pkl"))
    joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.pkl"))
    joblib.dump(imputer, os.path.join(OUT_DIR, "imputer.pkl"))

    print("Preprocessing completed and saved to", OUT_DIR)

if __name__ == "__main__":
    run_preprocess()
