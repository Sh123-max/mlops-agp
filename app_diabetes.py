# app_diabetes.py
import os
import json
import joblib
import time
import socket
import numpy as np
from flask import Flask, request, render_template
from prometheus_client import Gauge, Counter, Histogram
from prometheus_flask_exporter import PrometheusMetrics
import glob

# ==========================================================
# CONFIG
# ==========================================================
PROJECT = os.getenv("PROJECT_NAME", "diabetes")
MODEL_BASE_DIR = os.getenv("MODEL_DIR", "models")

PROJECT_MODEL_DIR = os.path.join(MODEL_BASE_DIR, PROJECT)
DEPLOYED_DIR = os.path.join(PROJECT_MODEL_DIR, "deployed_model")

# IMPORTANT FILES
LAST_SUMMARY_PATH = os.path.join(MODEL_BASE_DIR, "last_run_summary.json")
META_PATH = os.path.join(PROJECT_MODEL_DIR, "model_metadata.json")

model = None
model_name = "unknown"
model_metrics = {}
expected_feature_order = None
expected_feature_count = None
start_time = time.time()

# ==========================================================
# LOAD MODEL + METRICS
# ==========================================================
def load_project_model():
    global model, model_name, model_metrics
    global expected_feature_order, expected_feature_count

    # ---------- 1. LOAD LATEST METRICS ----------
    model_metrics = {}

    if os.path.exists(LAST_SUMMARY_PATH):
        try:
            summary = json.load(open(LAST_SUMMARY_PATH))
            if summary.get("project") == PROJECT:
                best_name = summary["best"]["name"]
                model_name = best_name

                best_metrics = next(
                    r for r in summary["results"]
                    if r["name"] == best_name
                )
                model_metrics = best_metrics
                print(f"[{PROJECT}] Loaded metrics from last_run_summary.json")
        except Exception as e:
            print(f"[{PROJECT}] Failed loading last summary: {e}")

    # ---------- 2. FALLBACK TO METADATA ----------
    if not model_metrics and os.path.exists(META_PATH):
        try:
            meta = json.load(open(META_PATH))
            model_name = meta.get("model_name", model_name)
            model_metrics = meta.get("metrics", {})
            expected_feature_order = meta.get("feature_order")
            expected_feature_count = meta.get("feature_count")
            print(f"[{PROJECT}] Loaded metrics from model_metadata.json")
        except Exception as e:
            print(f"[{PROJECT}] Failed reading metadata: {e}")

    # ---------- 3. LOAD DEPLOYED MODEL ----------
    candidates = []
    if os.path.exists(DEPLOYED_DIR):
        candidates += glob.glob(f"{DEPLOYED_DIR}/model/*.pkl")
        candidates += glob.glob(f"{DEPLOYED_DIR}/model/*.joblib")
        candidates += glob.glob(f"{DEPLOYED_DIR}/*.pkl")
        candidates += glob.glob(f"{DEPLOYED_DIR}/*.joblib")

    if candidates:
        model_path = sorted(candidates)[0]
        try:
            model = joblib.load(model_path)
            expected_feature_count = getattr(model, "n_features_in_", expected_feature_count)
            print(f"[{PROJECT}] Loaded model: {model_path}")
        except Exception as e:
            print(f"[{PROJECT}] Model load failed: {e}")
            model = None
    else:
        print(f"[{PROJECT}] No deployed model found")

# Load once on startup
load_project_model()

# ==========================================================
# FLASK + PROMETHEUS
# ==========================================================
app = Flask(__name__, template_folder="templates")
metrics = PrometheusMetrics(app, path="/metrics")
HOSTNAME = socket.gethostname()

MODEL_INFO = Gauge(
    "model_version_info",
    "Loaded model info",
    ["project", "model_name", "host"]
)

PREDICTION_COUNT = Counter(
    "prediction_requests_total",
    "Prediction requests",
    ["project", "outcome"]
)

PREDICTION_LATENCY = Histogram(
    "inference_latency_ms",
    "Inference latency",
    buckets=(1,5,10,20,50,100,200,500,1000)
)

MODEL_ACCURACY = Gauge(
    "ml_current_model_accuracy",
    "Accuracy of deployed model",
    ["project"]
)

MODEL_UPTIME = Gauge(
    "model_service_uptime_seconds",
    "Service uptime",
    ["project"]
)

# Set static metrics
if model_metrics:
    try:
        MODEL_ACCURACY.labels(PROJECT).set(float(model_metrics.get("accuracy", 0.0)))
    except: pass
MODEL_INFO.labels(PROJECT, model_name, HOSTNAME).set(1)

# ==========================================================
# UI ROUTES
# ==========================================================
@app.route("/")
def home():
    return render_template(
        "form.html",
        prediction=None,
        probability=None,
        error_messages=[],
        model_name=model_name,
        model_metrics=model_metrics
    )

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template(
            "form.html",
            prediction="Error",
            probability=None,
            error_messages=["Model not loaded"],
            model_name=model_name,
            model_metrics=model_metrics
        )

    try:
        # 1. Collect the 8 raw inputs from the HTML form
        raw_inputs = {
            'Pregnancies': float(request.form.get('Pregnancies', 0)),
            'Glucose': float(request.form.get('Glucose', 0)),
            'BloodPressure': float(request.form.get('BloodPressure', 0)),
            'SkinThickness': float(request.form.get('SkinThickness', 0)),
            'Insulin': float(request.form.get('Insulin', 0)),
            'BMI': float(request.form.get('BMI', 0)),
            'DiabetesPedigreeFunction': float(request.form.get('DiabetesPedigreeFunction', 0)),
            'Age': float(request.form.get('Age', 0))
        }

        # 2. FEATURE ENGINEERING (Sync with preprocess.py)
        # We transform the 8 inputs into 11 features
        processed_data = [
            raw_inputs['Pregnancies'],
            raw_inputs['Glucose'],
            raw_inputs['BloodPressure'],
            raw_inputs['SkinThickness'],
            raw_inputs['Insulin'],
            raw_inputs['BMI'],
            raw_inputs['DiabetesPedigreeFunction'],
            raw_inputs['Age'],
            # New Engineered features:
            raw_inputs['BMI'] * raw_inputs['Age'],                   # BMI_Age_Interaction
            raw_inputs['Glucose'] / (raw_inputs['Insulin'] + 0.1),   # Glucose_Insulin_Ratio
            1.0 if raw_inputs['Age'] > 50 else 0.0                  # Is_Senior
        ]

        sample = np.array([processed_data])

        # 3. INFERENCE
        t0 = time.time()
        pred = model.predict(sample)[0]
        proba = model.predict_proba(sample)[0][1] if hasattr(model, "predict_proba") else 0.5

        latency = (time.time() - t0) * 1000
        outcome = "positive" if pred == 1 else "negative"

        PREDICTION_COUNT.labels(PROJECT, outcome).inc()
        PREDICTION_LATENCY.observe(latency)

        return render_template(
            "form.html",
            prediction="Diabetic" if pred == 1 else "Not Diabetic",
            probability=f"{proba:.2%}",
            error_messages=[],
            model_name=model_name,
            model_metrics=model_metrics
        )
    except Exception as e:
        return render_template(
            "form.html",
            prediction=None,
            probability=None,
            error_messages=[f"Processing error: {str(e)}"],
            model_name=model_name,
            model_metrics=model_metrics
        )

@app.route("/health")
def health():
    uptime = time.time() - start_time
    MODEL_UPTIME.labels(PROJECT).set(uptime)
    return {"status": "healthy", "uptime": uptime}

@app.route("/reload", methods=["POST"])
def reload_model():
    load_project_model()
    return {"reloaded": True, "model": model_name}

# ==========================================================
if __name__ == "__main__":
    port = int(os.getenv("DIABETES_APP_PORT", "5000"))
    print(f"[app_diabetes] Running on port {port}")
    app.run(host="0.0.0.0", port=port)
