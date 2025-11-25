# app_heart.py
import os
import json
import joblib
import time
import socket
import numpy as np
from flask import Flask, request, render_template
from prometheus_client import Gauge, Counter, Histogram
from prometheus_flask_exporter import PrometheusMetrics

app = Flask(__name__, template_folder="templates")
metrics = PrometheusMetrics(app, path="/metrics")
HOSTNAME = socket.gethostname()

PROJECT_NAME = os.getenv("PROJECT_NAME", "heart")

# Metrics (use project label where possible)
MODEL_INFO = Gauge("heart_model_version_info", "Info about loaded heart model", ["project", "model_name", "model_version", "host"])
PREDICTION_COUNT = Counter("heart_prediction_requests_total", "Total heart prediction requests", ["project", "outcome"])
INPUT_VALIDATION_ERRORS = Counter("heart_input_validation_errors_total", "Input validation errors counted", ["project", "field"])
PREDICTION_LATENCY = Histogram("heart_inference_latency_ms", "Heart model inference latency (ms)", buckets=(1,5,10,20,50,100,200,500,1000))
MODEL_ACCURACY = Gauge("heart_current_model_accuracy", "Accuracy of deployed heart ML model", ["project"])
MODEL_UPTIME = Gauge("heart_model_service_uptime_seconds", "Uptime of heart model service in seconds", ["project"])

MODEL_DIR = os.getenv("MODEL_DIR", "models")
meta_path = os.path.join(MODEL_DIR, "model_metadata.json")
model = None
model_name = "unknown"
model_metrics = {}
start_time = time.time()

# Feature order and validation (best-effort)
FEATURE_ORDER = [
    "age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"
]
VALID_RANGES = {
    "age": (18,100),
    "sex": (0,1),
    "cp": (0,4),
    "trestbps": (80,220),
    "chol": (100,600),
    "fbs": (0,1),
    "restecg": (0,2),
    "thalach": (50,260),
    "exang": (0,1),
    "oldpeak": (0,10),
    "slope": (0,3),
    "ca": (0,4),
    "thal": (0,3)
}

def load_model():
    global model, model_name, model_metrics
    try:
        if os.path.exists(meta_path):
            meta = json.load(open(meta_path))
            model_name = meta.get("model_name", model_name)
            model_metrics = meta.get("metrics", meta.get("best", {}))
            try:
                MODEL_ACCURACY.labels(PROJECT_NAME).set(float(model_metrics.get("accuracy", 0.0)))
            except Exception:
                pass
        deployed_dir = os.path.join(MODEL_DIR, "deployed_model")
        if os.path.exists(deployed_dir):
            candidates = [os.path.join(deployed_dir, f) for f in os.listdir(deployed_dir) if f.endswith(".pkl") or f.endswith(".joblib")]
            if candidates:
                model = joblib.load(candidates[0])
                print("[app_heart] Loaded model from:", candidates[0])
                try:
                    MODEL_INFO.labels(PROJECT_NAME, model_name, str(meta.get("version", "")), HOSTNAME).set(1)
                except Exception:
                    pass
    except Exception as e:
        print("[app_heart] Failed to load model:", e)

load_model()

@app.route('/')
def home():
    return render_template('form_heart.html', prediction=None, probability=None, error_messages=[], model_name=model_name, model_metrics=model_metrics)

@app.route('/predict', methods=['POST'])
def predict():
    start = time.time()
    if model is None:
        return render_template('form_heart.html', prediction="Error", probability="Model not loaded", error_messages=["Model missing."], model_name=model_name, model_metrics=model_metrics)

    input_vals = []
    errors = []
    for key in FEATURE_ORDER:
        raw = request.form.get(key, "")
        if raw == "":
            errors.append(f"{key} is required")
            try:
                INPUT_VALIDATION_ERRORS.labels(PROJECT_NAME, key).inc()
            except Exception:
                INPUT_VALIDATION_ERRORS.labels("unknown", key).inc()
            continue
        try:
            val = float(raw)
            if key in VALID_RANGES:
                lo, hi = VALID_RANGES[key]
                if not (lo <= val <= hi):
                    errors.append(f"{key} must be between {lo} and {hi}")
                    try:
                        INPUT_VALIDATION_ERRORS.labels(PROJECT_NAME, key).inc()
                    except Exception:
                        INPUT_VALIDATION_ERRORS.labels("unknown", key).inc()
            input_vals.append(val)
        except Exception:
            errors.append(f"{key} must be numeric")
            try:
                INPUT_VALIDATION_ERRORS.labels(PROJECT_NAME, key).inc()
            except Exception:
                INPUT_VALIDATION_ERRORS.labels("unknown", key).inc()

    if errors:
        return render_template('form_heart.html', prediction=None, probability=None, error_messages=errors, model_name=model_name, model_metrics=model_metrics)

    try:
        sample = np.array([input_vals])
        t0 = time.time()
        pred = model.predict(sample)[0]
        try:
            proba = model.predict_proba(sample)[0][1] if hasattr(model, "predict_proba") else 0.5
        except Exception:
            proba = 0.5
        latency_ms = (time.time() - start) * 1000.0
        outcome = "disease" if pred == 1 else "no_disease"
        try:
            PREDICTION_COUNT.labels(PROJECT_NAME, outcome).inc()
        except Exception:
            PREDICTION_COUNT.labels("unknown", outcome).inc()
        try:
            PREDICTION_LATENCY.observe(latency_ms)
        except Exception:
            pass
        result = "Heart Disease" if pred == 1 else "No Heart Disease"
        return render_template('form_heart.html', prediction=result, probability=f"{proba:.1%}", error_messages=[], model_name=model_name, model_metrics=model_metrics)
    except Exception as e:
        return render_template('form_heart.html', prediction=None, probability=None, error_messages=[f"Prediction error: {e}"], model_name=model_name, model_metrics=model_metrics)

@app.route('/health')
def health():
    uptime = time.time() - start_time
    try:
        MODEL_UPTIME.labels(PROJECT_NAME).set(uptime)
    except Exception:
        pass
    return {"status": "healthy", "uptime_seconds": uptime}

@app.route('/reload', methods=['POST'])
def reload_model():
    try:
        load_model()
        return {"reloaded": True, "model": model_name}
    except Exception as e:
        return {"reloaded": False, "error": str(e)}, 500

if __name__ == '__main__':
    port = int(os.getenv("HEART_APP_PORT", "5005"))
    print(f"[app_heart] Starting Flask server on port {port} (MODEL_DIR={MODEL_DIR})...")
    app.run(host='0.0.0.0', port=port)
