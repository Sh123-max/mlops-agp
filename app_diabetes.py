# app_diabetes.py
from flask import Flask, request, render_template
import os, json, joblib, numpy as np, time, socket
from prometheus_client import Gauge, Counter, Histogram
from prometheus_flask_exporter import PrometheusMetrics
import math

app = Flask(__name__, template_folder="templates")
metrics = PrometheusMetrics(app, path="/metrics")
HOSTNAME = socket.gethostname()

MODEL_INFO = Gauge("model_version_info", "Info about loaded model", ["model_name", "model_version", "host"])
PREDICTION_COUNT = Counter("prediction_requests_total", "Total prediction requests", ["outcome"])
INPUT_VALIDATION_ERRORS = Counter("input_validation_errors_total", "Input validation errors counted", ["field"])
PREDICTION_LATENCY = Histogram("inference_latency_ms", "Model inference latency (ms)", buckets=(1,5,10,20,50,100,200,500,1000))
MODEL_ACCURACY = Gauge("ml_current_model_accuracy", "Accuracy of deployed ML model")
MODEL_UPTIME = Gauge("model_service_uptime_seconds", "Uptime of model service in seconds")

# load model metadata and model
MODEL_DIR = os.getenv("MODEL_DIR", "models")
meta_path = os.path.join(MODEL_DIR, "model_metadata.json")
model = None
model_name = "unknown"
model_metrics = {}
start_time = time.time()

def load_model():
    global model, model_name, model_metrics
    try:
        if os.path.exists(meta_path):
            meta = json.load(open(meta_path))
            model_name = meta.get("model_name", model_name)
            model_metrics = meta.get("metrics", meta.get("best", {}))
        # load deployed model artifact (deployed_model folder)
        deployed_dir = os.path.join(MODEL_DIR, "deployed_model")
        if os.path.exists(deployed_dir):
            candidates = [os.path.join(deployed_dir, f) for f in os.listdir(deployed_dir) if f.endswith(".pkl") or f.endswith(".joblib")]
            if candidates:
                model = joblib.load(candidates[0])
                print("[app_diabetes] Loaded model from:", candidates[0])
    except Exception as e:
        print("[app_diabetes] Failed to load model:", e)

load_model()

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
FEATURE_ORDER = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]

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
        try:
            value = float(request.form.get(key, ""))
            lo, hi = VALID_RANGES.get(key, (None, None))
            if lo is not None and hi is not None and not (lo <= value <= hi):
                error_messages.append(f"{key} must be between {lo} and {hi}")
                INPUT_VALIDATION_ERRORS.labels(field=key).inc()
            input_data.append(value)
        except Exception:
            error_messages.append(f"{key} must be numeric")
            INPUT_VALIDATION_ERRORS.labels(field=key).inc()
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
        PREDICTION_COUNT.labels(outcome=outcome).inc()
        with PREDICTION_LATENCY.time():
            pass  # the decorator already measures duration here if you prefer
        result = "Diabetic" if pred == 1 else "Not Diabetic"
        return render_template('form.html', prediction=result, probability=f"{proba:.1%}", error_messages=[], model_name=model_name, model_metrics=model_metrics)
    except Exception as e:
        return render_template('form.html', prediction=None, probability=None, error_messages=[f"Prediction error: {e}"], model_name=model_name, model_metrics=model_metrics)

@app.route('/health')
def health():
    uptime = time.time() - start_time
    MODEL_UPTIME.set(uptime)
    return {"status": "healthy", "uptime_seconds": uptime}

if __name__ == '__main__':
    print("[app_diabetes] Starting Flask server on port 5000...")
    app.run(host='0.0.0.0', port=int(os.getenv("DIABETES_APP_PORT", "5000")))
