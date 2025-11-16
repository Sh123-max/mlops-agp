# app.py
from flask import Flask, request, render_template
import os, json, joblib, numpy as np, time, socket
from prometheus_client import Gauge, Counter, Histogram
from prometheus_flask_exporter import PrometheusMetrics

app = Flask(__name__)
# Expose metrics at /metrics
metrics = PrometheusMetrics(app, path="/metrics")
HOSTNAME = socket.gethostname()

# Existing metrics
MODEL_INFO = Gauge(
    "model_version_info",
    "Info about loaded model",
    ["model_name", "model_version", "host"]
)
PREDICTION_COUNT = Counter(
    "prediction_requests_total",
    "Total prediction requests",
    ["outcome"]
)
INPUT_VALIDATION_ERRORS = Counter(
    "input_validation_errors_total",
    "Input validation errors counted",
    ["field"]
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction latency seconds",
    buckets=(0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0)
)
MODEL_ACCURACY = Gauge("ml_current_model_accuracy", "Accuracy of deployed ML model")

# === NEW: low-cardinality per-request metrics ===
# Stores the epoch timestamp of the most recent prediction request (one series)
PREDICTION_TIMESTAMP = Gauge(
    "prediction_request_timestamp_seconds",
    "Timestamp (epoch seconds) of latest prediction request"
)

# Stores the predicted probability (labelled by low-cardinality 'outcome')
# outcome will be "diabetic" or "not_diabetic"
MODEL_PRED_PROB = Gauge(
    "model_prediction_probability",
    "Latest predicted probability (labelled by outcome)",
    ["outcome"]
)

# Model & metadata loading
model = None
model_name = "unknown"
model_metrics = {}

try:
    metadata_path = os.path.join("models", "model_metadata.json")
    if os.path.exists(metadata_path):
        meta = json.load(open(metadata_path))
        model_name = meta.get("model_name", "unknown")
        model_version = str(meta.get("version", "v1"))
        MODEL_INFO.labels(model_name=model_name, model_version=model_version, host=HOSTNAME).set(1)

        deploy_dir = os.path.join("models", "deployed_model")
        found = False
        for root, _, files in os.walk(deploy_dir):
            for f in files:
                if f.endswith(".pkl") or f.endswith(".joblib"):
                    model = joblib.load(os.path.join(root, f))
                    found = True
                    break
            if found:
                break

        if "metrics" in meta:
            model_metrics = meta["metrics"]
            if "accuracy" in model_metrics:
                try:
                    MODEL_ACCURACY.set(float(model_metrics["accuracy"]))
                except Exception:
                    # safe fallback if stored metric is not numeric
                    pass
    else:
        print("model_metadata.json not found. App runs but model not loaded.")
except Exception as e:
    print("Error loading model:", e)

# Valid ranges for input validation
valid_ranges = {
    "Pregnancies": (0, 20),
    "Glucose": (50, 200),
    "BloodPressure": (40, 140),
    "SkinThickness": (10, 100),
    "Insulin": (15, 846),
    "BMI": (15, 50),
    "DiabetesPedigreeFunction": (0.1, 2.5),
    "Age": (15, 100)
}

@app.route('/')
def home():
    return render_template(
        'form.html',
        prediction=None,
        probability=None,
        error_messages=[],
        non_diabetic_warnings=[],
        model_name=model_name,
        model_metrics=model_metrics
    )

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    if model is None:
        return render_template(
            'form.html',
            prediction="Error",
            probability="Model not loaded",
            error_messages=["Model missing."],
            non_diabetic_warnings=[],
            model_name=model_name,
            model_metrics=model_metrics
        )

    input_data = []
    error_messages = []
    non_diabetic_warnings = []

    # Validate inputs and increment validation error counters on failure
    for key in valid_ranges.keys():
        try:
            value = float(request.form[key])
            if not (valid_ranges[key][0] <= value <= valid_ranges[key][1]):
                error_messages.append(
                    f"{key} must be between {valid_ranges[key][0]} and {valid_ranges[key][1]}"
                )
                INPUT_VALIDATION_ERRORS.labels(field=key).inc()
            input_data.append(value)
        except Exception:
            error_messages.append(f"{key} must be numeric")
            INPUT_VALIDATION_ERRORS.labels(field=key).inc()

    if error_messages:
        return render_template(
            'form.html',
            prediction=None,
            probability=None,
            error_messages=error_messages,
            non_diabetic_warnings=[],
            model_name=model_name,
            model_metrics=model_metrics
        )

    try:
        sample = np.array([input_data])
        with PREDICTION_LATENCY.time():
            # model inference
            prediction = model.predict(sample)[0]
            probability = model.predict_proba(sample)[0][1] if hasattr(model, 'predict_proba') else 0.5

        outcome = "diabetic" if prediction == 1 else "not_diabetic"
        result = "Diabetic" if prediction == 1 else "Not Diabetic"

        # === Set new metrics (low-cardinality) ===
        now_ts = time.time()
        # single-series timestamp (will be overwritten on each request; useful for ordering)
        PREDICTION_TIMESTAMP.set(now_ts)
        # set probability under the low-cardinality label 'outcome'
        try:
            MODEL_PRED_PROB.labels(outcome=outcome).set(float(probability))
        except Exception:
            # safe fallback if probability cannot be converted
            MODEL_PRED_PROB.labels(outcome=outcome).set(0.0)

        # increment request counter (kept for totals / rate)
        PREDICTION_COUNT.labels(outcome=outcome).inc()

        return render_template(
            'form.html',
            prediction=result,
            probability=f"{probability:.1%}",
            error_messages=[],
            non_diabetic_warnings=non_diabetic_warnings,
            model_name=model_name,
            model_metrics=model_metrics
        )
    except Exception as e:
        return render_template(
            'form.html',
            prediction=None,
            probability=None,
            error_messages=[f"Prediction error: {e}"],
            non_diabetic_warnings=[],
            model_name=model_name,
            model_metrics=model_metrics
        )

if __name__ == '__main__':
    # Bind to 0.0.0.0 so container and host can reach the app
    app.run(host='0.0.0.0', port=5000)

