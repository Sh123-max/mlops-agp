# app_heart.py
import os
import json
import joblib
import time
import socket
import numpy as np
import pandas as pd # Added for easier engineering
from flask import Flask, request, render_template
from prometheus_client import Gauge, Counter, Histogram
from prometheus_flask_exporter import PrometheusMetrics
import glob

PROJECT = os.getenv("PROJECT_NAME", "heart")
MODEL_BASE_DIR = os.getenv("MODEL_DIR", "models")
PROJECT_MODEL_DIR = os.path.join(MODEL_BASE_DIR, PROJECT)
meta_path = os.path.join(PROJECT_MODEL_DIR, "model_metadata.json")

model = None
model_name = "unknown"
model_metrics = {}
expected_feature_order = None
expected_feature_count = None
start_time = time.time()

def load_project_model():
    global model, model_name, model_metrics, expected_feature_order, expected_feature_count
    try:
        if os.path.exists(meta_path):
            try:
                meta = json.load(open(meta_path))
                model_name = meta.get("model_name", model_name)
                model_metrics = meta.get("metrics", meta.get("best", {}))
                expected_feature_order = meta.get("feature_order")
                expected_feature_count = meta.get("feature_count")
                if expected_feature_count is not None:
                    expected_feature_count = int(expected_feature_count)
            except Exception as me:
                print(f"[{PROJECT}] Failed reading metadata: {me}")

        deployed_dir = os.path.join(PROJECT_MODEL_DIR, "deployed_model")
        if os.path.exists(deployed_dir):
            candidates = sorted(glob.glob(os.path.join(deployed_dir, "*.pkl"))) + \
                         sorted(glob.glob(os.path.join(deployed_dir, "*.joblib")))
            if not candidates:
                candidates += sorted(glob.glob(os.path.join(deployed_dir, "**", "*.pkl"), recursive=True))
           
            if candidates:
                model_path = candidates[0]
                model = joblib.load(model_path)
                if expected_feature_count is None:
                    expected_feature_count = getattr(model, "n_features_in_", None)
                print(f"[{PROJECT}] Loaded model. expected_features={expected_feature_count}")
    except Exception as e:
        print(f"[{PROJECT}] Failed loading model: {e}")

load_project_model()

app = Flask(__name__, template_folder="templates")
metrics = PrometheusMetrics(app, path="/metrics")
HOSTNAME = socket.gethostname()

MODEL_INFO = Gauge("heart_model_version_info", "Info", ["project", "model_name", "model_version", "host"])
PREDICTION_COUNT = Counter("heart_prediction_requests_total", "Total", ["project", "outcome"])
INPUT_VALIDATION_ERRORS = Counter("heart_input_validation_errors_total", "Errors", ["project", "field"])
PREDICTION_LATENCY = Histogram("heart_inference_latency_ms", "Latency", buckets=(1,5,10,20,50,100,200,500,1000))
MODEL_ACCURACY = Gauge("heart_current_model_accuracy", "Accuracy", ["project"])
MODEL_UPTIME = Gauge("heart_model_service_uptime_seconds", "Uptime", ["project"])

RAW_FEATURES = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
VALID_RANGES = {
    "age": (18,100), "sex": (0,1), "cp": (0,4), "trestbps": (80,220),
    "chol": (100,600), "fbs": (0,1), "restecg": (0,2), "thalach": (50,260),
    "exang": (0,1), "oldpeak": (0,10), "slope": (0,3), "ca": (0,4), "thal": (0,3)
}

try:
    MODEL_ACCURACY.labels(PROJECT).set(float(model_metrics.get("accuracy", 0.0)))
    MODEL_INFO.labels(PROJECT, model_name, str(model_metrics.get("version", "")), HOSTNAME).set(1)
except: pass

@app.route('/')
def home():
    return render_template('form_heart.html', prediction=None, model_name=model_name, model_metrics=model_metrics)

@app.route('/predict', methods=['POST'])
def predict():
    start = time.time()
    if model is None:
        return render_template('form_heart.html', error_messages=["Model missing."], model_name=model_name)

    raw_data = {}
    errors = []
   
    # Collect & Validate Raw Features
    for key in RAW_FEATURES:
        raw_val = request.form.get(key, "")
        if raw_val == "":
            errors.append(f"{key} is required")
            continue
        try:
            val = float(raw_val)
            if key in VALID_RANGES:
                lo, hi = VALID_RANGES[key]
                if not (lo <= val <= hi):
                    errors.append(f"{key} must be {lo}-{hi}")
            raw_data[key] = val
        except:
            errors.append(f"{key} must be numeric")

    if errors:
        for err in errors: INPUT_VALIDATION_ERRORS.labels(PROJECT, "validation").inc()
        return render_template('form_heart.html', error_messages=errors, model_name=model_name, model_metrics=model_metrics)

    try:
        #  FEATURE ENGINEERING ( matching preprocess.py)
        # Create the engineered features from raw dictionary
        high_risk_combo = 1.0 if (raw_data['trestbps'] > 130 and raw_data['chol'] > 240) else 0.0
        hr_efficiency = raw_data['thalach'] / (220 - raw_data['age'])

        # 3. Construct Final Feature List (15 features)
        final_inputs = [raw_data[k] for k in RAW_FEATURES]
        final_inputs.append(high_risk_combo)               
        final_inputs.append(hr_efficiency)                 

        sample = np.array([final_inputs])
       
        # 4. Inference
        pred = model.predict(sample)[0]
        proba = model.predict_proba(sample)[0][1] if hasattr(model, "predict_proba") else 0.5
       
        latency_ms = (time.time() - start) * 1000.0
        outcome = "disease" if pred == 1 else "no_disease"
        PREDICTION_COUNT.labels(PROJECT, outcome).inc()
        PREDICTION_LATENCY.observe(latency_ms)

        result = "Heart Disease Detected" if pred == 1 else "No Heart Disease Detected"
        return render_template('form_heart.html', prediction=result, probability=f"{proba:.1%}", model_name=model_name, model_metrics=model_metrics)
    except Exception as e:
        return render_template('form_heart.html', error_messages=[f"Inference error: {e}"], model_name=model_name)

@app.route('/health')
def health():
    uptime = time.time() - start_time
    MODEL_UPTIME.labels(PROJECT).set(uptime)
    return {"status": "healthy", "uptime": uptime}

if __name__ == '__main__':
    port = int(os.getenv("HEART_APP_PORT", "5005"))
    app.run(host='0.0.0.0', port=port)
