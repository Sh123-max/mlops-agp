# app_diabetes.py
# Updated: prefer deployed_model/model/ before top-level files to avoid loading wrong artifact
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

# Per-project config
PROJECT = os.getenv("PROJECT_NAME", "diabetes")
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
    """
    Load the deployed model and its metadata for the given project.
    Preference order:
      1) files inside deployed_model/model/ (common MLflow layout)
      2) files at deployed_model/ (top-level)
      3) recursive search under deployed_model/ as fallback
    """
    global model, model_name, model_metrics, expected_feature_order, expected_feature_count
    try:
        # Load metadata if exists
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
                print(f"[{PROJECT}] Failed reading metadata {meta_path}: {me}")

        deployed_dir = os.path.join(PROJECT_MODEL_DIR, "deployed_model")
        if os.path.exists(deployed_dir):
            candidates = []

            # 1) Prefer files in deployed_model/model/
            model_subdir = os.path.join(deployed_dir, "model")
            if os.path.exists(model_subdir):
                candidates += sorted(glob.glob(os.path.join(model_subdir, "*.pkl")))
                candidates += sorted(glob.glob(os.path.join(model_subdir, "*.joblib")))

            # 2) Then prefer top-level files under deployed_model/
            candidates += sorted(glob.glob(os.path.join(deployed_dir, "*.pkl")))
            candidates += sorted(glob.glob(os.path.join(deployed_dir, "*.joblib")))

            # 3) Recursive fallback if nothing found yet
            if not candidates:
                candidates += sorted(glob.glob(os.path.join(deployed_dir, "**", "*.pkl"), recursive=True))
                candidates += sorted(glob.glob(os.path.join(deployed_dir, "**", "*.joblib"), recursive=True))

            if candidates:
                model_path = candidates[0]
                try:
                    model = joblib.load(model_path)
                except Exception as le:
                    print(f"[{PROJECT}] Failed loading model file {model_path}: {le}")
                    model = None
                    return
                # infer features if not provided
                if expected_feature_count is None:
                    expected_feature_count = getattr(model, "n_features_in_", None)
                print(f"[{PROJECT}] Loaded model from {model_path}. expected_features={expected_feature_count}")
                return
            else:
                print(f"[{PROJECT}] No model files in {deployed_dir}")
        else:
            print(f"[{PROJECT}] deployed_model directory not found: {deployed_dir}")
    except Exception as e:
        print(f"[{PROJECT}] Failed loading model: {e}")

# load once at process start
load_project_model()

app = Flask(__name__, template_folder="templates")
metrics = PrometheusMetrics(app, path="/metrics")
HOSTNAME = socket.gethostname()

# Metrics with project label
MODEL_INFO = Gauge("model_version_info", "Info about loaded model", ["project", "model_name", "model_version", "host"])
PREDICTION_COUNT = Counter("prediction_requests_total", "Total prediction requests", ["project", "outcome"])
INPUT_VALIDATION_ERRORS = Counter("input_validation_errors_total", "Input validation errors counted", ["project", "field"])
PREDICTION_LATENCY = Histogram("inference_latency_ms", "Model inference latency (ms)", buckets=(1,5,10,20,50,100,200,500,1000))
MODEL_ACCURACY = Gauge("ml_current_model_accuracy", "Accuracy of deployed ML model", ["project"])
MODEL_UPTIME = Gauge("model_service_uptime_seconds", "Uptime of model service in seconds", ["project"])

# Defaults for diabetes features
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

# set accuracy metric if available
if model_metrics:
    try:
        MODEL_ACCURACY.labels(PROJECT).set(float(model_metrics.get("accuracy", 0.0)))
    except Exception:
        pass
try:
    MODEL_INFO.labels(PROJECT, model_name, str(model_metrics.get("version", "")), HOSTNAME).set(1)
except Exception:
    pass

@app.route('/')
def home():
    return render_template('form.html', prediction=None, probability=None, error_messages=[], model_name=model_name, model_metrics=model_metrics)

@app.route('/predict', methods=['POST'])
def predict():
    start = time.time()
    if model is None:
        return render_template('form.html', prediction="Error", probability="Model not loaded",
                               error_messages=[f"Model missing for project '{PROJECT}'. Deploy first."],
                               model_name=model_name, model_metrics=model_metrics)
    # determine order
    order = expected_feature_order if expected_feature_order else FEATURE_ORDER
    inputs = []
    errors = []
    for key in order:
        raw = request.form.get(key, "")
        if raw == "":
            errors.append(f"{key} is required")
            INPUT_VALIDATION_ERRORS.labels(PROJECT, key).inc()
            continue
        try:
            val = float(raw)
            if key in VALID_RANGES:
                lo, hi = VALID_RANGES[key]
                if not (lo <= val <= hi):
                    errors.append(f"{key} must be between {lo} and {hi}")
                    INPUT_VALIDATION_ERRORS.labels(PROJECT, key).inc()
            inputs.append(val)
        except Exception:
            errors.append(f"{key} must be numeric")
            INPUT_VALIDATION_ERRORS.labels(PROJECT, key).inc()
    if errors:
        return render_template('form.html', prediction=None, probability=None, error_messages=errors, model_name=model_name, model_metrics=model_metrics)
    # check feature count
    if expected_feature_count is not None and len(inputs) != expected_feature_count:
        return render_template('form.html', prediction=None, probability=None,
                               error_messages=[f"Input length {len(inputs)} != model expects {expected_feature_count} features (check model metadata)."],
                               model_name=model_name, model_metrics=model_metrics)
    try:
        sample = np.array([inputs])
        t0 = time.time()
        pred = model.predict(sample)[0]
        try:
            proba = model.predict_proba(sample)[0][1] if hasattr(model, "predict_proba") else 0.5
        except Exception:
            proba = 0.5
        latency_ms = (time.time() - start) * 1000.0
        outcome = "positive" if pred == 1 else "negative"
        PREDICTION_COUNT.labels(PROJECT, outcome).inc()
        try:
            PREDICTION_LATENCY.observe(latency_ms)
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
        MODEL_UPTIME.labels(PROJECT).set(uptime)
    except Exception:
        pass
    return {"status": "healthy", "uptime_seconds": uptime}

@app.route('/reload', methods=['POST'])
def reload_model():
    try:
        load_project_model()
        return {"reloaded": True, "model": model_name}
    except Exception as e:
        return {"reloaded": False, "error": str(e)}, 500

if __name__ == '__main__':
    port = int(os.getenv("DIABETES_APP_PORT", "5000"))
    print(f"[app_diabetes] Starting Flask server on port {port} (PROJECT={PROJECT} MODEL_DIR={MODEL_BASE_DIR})...")
    app.run(host='0.0.0.0', port=port)
