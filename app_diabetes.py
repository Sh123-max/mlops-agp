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

app = Flask(__name__, template_folder="templates")
metrics = PrometheusMetrics(app, path="/metrics")
HOSTNAME = socket.gethostname()

# use PROJECT_NAME env if provided (for labels)
PROJECT_NAME = os.getenv("PROJECT_NAME", "diabetes")

# Metric definitions (include project as label where useful)
MODEL_INFO = Gauge("model_version_info", "Info about loaded model", ["project", "model_name", "model_version", "host"])
PREDICTION_COUNT = Counter("prediction_requests_total", "Total prediction requests", ["project", "outcome"])
INPUT_VALIDATION_ERRORS = Counter("input_validation_errors_total", "Input validation errors counted", ["project", "field"])
PREDICTION_LATENCY = Histogram("inference_latency_ms", "Model inference latency (ms)", buckets=(1,5,10,20,50,100,200,500,1000), labels=["project"]) if False else None
# Note: prometheus_client.Histogram doesn't accept labels at construction time without wrapping, so we'll record by value
PREDICTION_LATENCY_NO_LABEL = Histogram("inference_latency_ms", "Model inference latency (ms)", buckets=(1,5,10,20,50,100,200,500,1000))
MODEL_ACCURACY = Gauge("ml_current_model_accuracy", "Accuracy of deployed ML model", ["project"])
MODEL_UPTIME = Gauge("model_service_uptime_seconds", "Uptime of model service in seconds", ["project"])

MODEL_DIR = os.getenv("MODEL_DIR", "models")
meta_path = os.path.join(MODEL_DIR, "model_metadata.json")
model = None
model_name = "unknown"
model_metrics = {}
start_time = time.time()

# features for diabetes form (same as preprocess expectation)
FEATURE_ORDER = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]
VALID_RANGES = {
    "Pregnancies": (0,20),
    "Glucose": (50,200),
    "BloodPressure": (40,140),
    "SkinThickness": (10,100),
    "Insulin": (15,846),
    "BMI": (15,50),
    "DiabetesPedigreeFunction": (0.1,2.5),
    "Age": (15,100)
}

def load_model():
    global model, model_name, model_metrics
    try:
        if os.path.exists(meta_path):
            meta = json.load(open(meta_path))
            model_name = meta.get("model_name", model_name)
            model_metrics = meta.get("metrics", meta.get("best", {}))
            # set model accuracy gauge if available
            try:
                MODEL_ACCURACY.labels(PROJECT_NAME).set(float(model_metrics.get("accuracy", 0.0)))
            except Exception:
                pass
        # load deployed model artifact (deployed_model folder inside MODEL_DIR)
        deployed_dir = os.path.join(MODEL_DIR, "deployed_model")
        if os.path.exists(deployed_dir):
            candidates = [os.path.join(deployed_dir, f) for f in os.listdir(deployed_dir) if f.endswith(".pkl") or f.endswith(".joblib")]
            if candidates:
                model = joblib.load(candidates[0])
                print("[app_diabetes] Loaded model from:", candidates[0])
                # set version info metric (best effort)
                try:
                    MODEL_INFO.labels(PROJECT_NAME, model_name, str(meta.get("version", "")), HOSTNAME).set(1)
                except Exception:
                    pass
    except Exception as e:
        print("[app_diabetes] Failed to load model:", e)

load_model()

@app.route('/')
def home():
    return render_template('form.html', prediction=None, probability=None, error_messages=[], model_name=model_name, model_metrics=model_metrics)

@app.route('/predict', methods=['POST'])
def predict():
    start = time.time()
    if model is None:
        return render_template('form.html', prediction="Error", probability="Model not loaded", error_messages=["Model missing."], model_name=model_name, model_metrics=model_metrics)

    input_data = []
    error_messages = []
    for key in FEATURE_ORDER:
        raw = request.form.get(key, "")
        try:
            value = float(raw)
            lo, hi = VALID_RANGES.get(key, (None, None))
            if lo is not None and hi is not None and not (lo <= value <= hi):
                error_messages.append(f"{key} must be between {lo} and {hi}")
                try:
                    INPUT_VALIDATION_ERRORS.labels(PROJECT_NAME, key).inc()
                except Exception:
                    INPUT_VALIDATION_ERRORS.labels("unknown", key).inc()
            input_data.append(value)
        except Exception:
            error_messages.append(f"{key} must be numeric")
            try:
                INPUT_VALIDATION_ERRORS.labels(PROJECT_NAME, key).inc()
            except Exception:
                INPUT_VALIDATION_ERRORS.labels("unknown", key).inc()

    if error_messages:
        return render_template('form.html', prediction=None, probability=None, error_messages=error_messages, model_name=model_name, model_metrics=model_metrics)

    try:
        sample = np.array([input_data])
        t0 = time.time()
        pred = model.predict(sample)[0]
        try:
            proba = model.predict_proba(sample)[0][1] if hasattr(model, "predict_proba") else 0.5
        except Exception:
            proba = 0.5
        latency_ms = (time.time() - start) * 1000.0
        outcome = "diabetic" if pred == 1 else "not_diabetic"
        try:
            PREDICTION_COUNT.labels(PROJECT_NAME, outcome).inc()
        except Exception:
            PREDICTION_COUNT.labels("unknown", outcome).inc()
        # observe latency
        try:
            PREDICTION_LATENCY_NO_LABEL.observe(latency_ms)
        except Exception:
            pass
        result = "Diabetic" if pred == 1 else "Not Diabetic"
        return render_template('form.html', prediction=result, probability=f"{proba:.1%}", error_messages=[], model_name=model_name, model_metrics=model_metrics)
    except Exception as e:
        return render_template('form.html', prediction=None, probability=None, error_messages=[f"Prediction error: {e}"], model_name=model_name, model_metrics=model_metrics)

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
    # allow remote reload of model artifact (useful after deploy.py finishes)
    try:
        load_model()
        return {"reloaded": True, "model": model_name}
    except Exception as e:
        return {"reloaded": False, "error": str(e)}, 500

if __name__ == '__main__':
    port = int(os.getenv("DIABETES_APP_PORT", "5000"))
    print(f"[app_diabetes] Starting Flask server on port {port} (MODEL_DIR={MODEL_DIR})...")
    app.run(host='0.0.0.0', port=port)
