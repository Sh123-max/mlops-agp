# app.py
from flask import Flask, request, render_template
import os
import json
import joblib
import numpy as np
import time
import socket
from prometheus_client import Gauge, Counter, Histogram
from prometheus_flask_exporter import PrometheusMetrics

# Additional imports for drift detection / stats
from scipy.stats import ks_2samp
import math

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
baseline = None

try:
    metadata_path = os.path.join("models", "model_metadata.json")
    if os.path.exists(metadata_path):
        meta = json.load(open(metadata_path))
        # model_name and version fallbacks to support both previous and new metadata formats
        model_name = meta.get("model_name", meta.get("best", {}).get("name", "unknown"))
        model_version = str(meta.get("version", meta.get("best", {}).get("registry", {}).get("version", "v1")))
        MODEL_INFO.labels(model_name=model_name, model_version=model_version, host=HOSTNAME).set(1)

        # Try deployed_model folder first
        deploy_dir = os.path.join("models", "deployed_model")
        found = False
        if os.path.exists(deploy_dir):
            for root, _, files in os.walk(deploy_dir):
                for f in files:
                    if f.endswith(".pkl") or f.endswith(".joblib"):
                        try:
                            model = joblib.load(os.path.join(root, f))
                            found = True
                            break
                        except Exception as e:
                            print("Failed to load candidate deployed model:", e)
                if found:
                    break

        # fallback: try best_name_model.pkl or any *_model.pkl in models/
        if model is None:
            try:
                candidates = [p for p in os.listdir("models") if p.endswith("_model.pkl") or p.endswith("_model.joblib")]
                if candidates:
                    # pick first candidate
                    model = joblib.load(os.path.join("models", candidates[0]))
            except Exception as e:
                print("Fallback model load failed:", e)

        # load model metrics if present in metadata (older format uses "metrics")
        if "metrics" in meta:
            model_metrics = meta["metrics"]
            if "accuracy" in model_metrics:
                try:
                    MODEL_ACCURACY.set(float(model_metrics["accuracy"]))
                except Exception:
                    pass

        # new format: results array may have metrics keyed by model name
        if "results" in meta and not model_metrics:
            for r in meta.get("results", []):
                if r.get("name") == meta.get("best", {}).get("name"):
                    model_metrics = r
                    break
            if "accuracy" in model_metrics:
                try:
                    MODEL_ACCURACY.set(float(model_metrics["accuracy"]))
                except Exception:
                    pass

        # baseline used for PSI/KS/uncertainty routines
        if "baseline" in meta:
            baseline = meta["baseline"]
    else:
        print("model_metadata.json not found. App runs but model not loaded.")
except Exception as e:
    print("Error loading model metadata or model:", e)

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

# Because the original form likely sends keys exactly matching column names used in preprocess,
# we will define the expected order here (must match preprocess / training order)
FEATURE_ORDER = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                 "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]


def calculate_psi(expected, actual, buckets=10):
    """
    Population Stability Index between two arrays.
    expected: list/np.array (baseline)
    actual: list/np.array (current)
    """
    try:
        expected = np.array(expected).astype(float)
        actual = np.array(actual).astype(float)
        if len(expected) == 0:
            return float("nan")
        # define bins using expected percentiles to be robust
        quantiles = np.percentile(expected, np.linspace(0, 100, buckets + 1))
        # ensure monotonic bins
        quantiles[0] = quantiles[0] - 1e-6
        quantiles[-1] = quantiles[-1] + 1e-6
        exp_counts, _ = np.histogram(expected, bins=quantiles)
        act_counts, _ = np.histogram(actual, bins=quantiles)
        # avoid zero percentages
        exp_perc = np.where(exp_counts == 0, 1e-8, exp_counts / max(1, len(expected)))
        act_perc = np.where(act_counts == 0, 1e-8, act_counts / max(1, len(actual)))
        psi = np.sum((exp_perc - act_perc) * np.log(exp_perc / act_perc))
        return float(psi)
    except Exception:
        return float("nan")


def approx_uncertainty_via_perturbation(model_obj, x_sample, baseline_stds, n_rounds=30):
    """
    Approximate model predictive uncertainty by perturbing the input
    with gaussian noise using baseline_stds and computing std of predicted probabilities.
    Returns dict: {"mean_proba":.., "std_proba":.., "entropy":..}
    """
    try:
        probs = []
        x = np.array(x_sample).astype(float).reshape(1, -1)
        # if no predict_proba, return neutral values
        if not hasattr(model_obj, "predict_proba"):
            return {"mean_proba": 0.5, "std_proba": 0.0, "entropy": 0.0}

        # baseline_stds is a dict keyed by feature name; fallback to small positive values
        stds = np.ones(x.shape[1])
        if baseline_stds:
            try:
                stds = np.array([baseline_stds.get(fn, 1.0) for fn in baseline.get("feature_names", [])])
                if stds.shape[0] != x.shape[1]:
                    stds = np.ones(x.shape[1])
            except Exception:
                stds = np.ones(x.shape[1])

        stds = np.where(np.array(stds) <= 0, 1e-6, np.array(stds))
        # small relative perturbation (1% of baseline std)
        noise_scale = 0.01 * stds
        for _ in range(n_rounds):
            noise = np.random.normal(loc=0.0, scale=noise_scale).reshape(1, -1)
            xp = x + noise
            try:
                p = model_obj.predict_proba(xp)[0][1]
            except Exception:
                p = 0.5
            probs.append(float(p))

        probs = np.array(probs)
        mean_p = float(np.mean(probs))
        std_p = float(np.std(probs))
        # approximate entropy for Bernoulli
        if mean_p <= 0 or mean_p >= 1:
            entropy = 0.0
        else:
            entropy = - (mean_p * math.log(mean_p + 1e-12) + (1 - mean_p) * math.log(1 - mean_p + 1e-12))
        return {"mean_proba": mean_p, "std_proba": std_p, "entropy": entropy}
    except Exception:
        return {"mean_proba": 0.5, "std_proba": 0.0, "entropy": 0.0}


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

    # collect inputs in FEATURE_ORDER
    for key in FEATURE_ORDER:
        try:
            # use .get to avoid KeyError; will be caught if missing/invalid
            raw_val = request.form.get(key, "")
            value = float(raw_val)
            # optionally validate range if available
            if key in valid_ranges:
                lo, hi = valid_ranges.get(key, (None, None))
                if lo is not None and hi is not None and not (lo <= value <= hi):
                    error_messages.append(f"{key} must be between {lo} and {hi}")
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

        # --- Drift detection & uncertainty ---
        drift_report = {"psi": {}, "ks": {}, "uncertainty": {}}
        # Prepare baseline arrays
        if baseline and "feature_distributions" in baseline:
            for i, fname in enumerate(baseline.get("feature_names", [])):
                expected = np.array(baseline["feature_distributions"].get(fname, []))
                # actual distribution for single-sample — use array of that single value
                actual = np.array([input_data[i]])
                # For PSI we need a distribution for actual; single sample gives noisy PSI but we'll still compute it
                try:
                    psi = calculate_psi(expected, actual, buckets=min(10, max(2, len(expected) // 5)))
                except Exception:
                    psi = float("nan")
                drift_report["psi"][fname] = psi
                try:
                    ks_stat, pval = ks_2samp(expected, actual)
                    drift_report["ks"][fname] = {"ks_stat": float(ks_stat), "pval": float(pval)}
                except Exception:
                    drift_report["ks"][fname] = {"ks_stat": float("nan"), "pval": float("nan")}
        else:
            # no baseline available
            drift_report["psi"] = {}
            drift_report["ks"] = {}

        # uncertainty approximation using perturbation
        baseline_stds = baseline.get("feature_stds", {}) if baseline else {}
        unc = approx_uncertainty_via_perturbation(model, input_data, baseline_stds, n_rounds=30)
        drift_report["uncertainty"] = unc

        with PREDICTION_LATENCY.time():
            prediction = model.predict(sample)[0]
            try:
                probability = model.predict_proba(sample)[0][1] if hasattr(model, 'predict_proba') else 0.5
            except Exception:
                probability = 0.5

        outcome = "diabetic" if prediction == 1 else "not_diabetic"
        PREDICTION_COUNT.labels(outcome=outcome).inc()
        result = "Diabetic" if prediction == 1 else "Not Diabetic"

        # === Set new low-cardinality metrics ===
        now_ts = time.time()
        PREDICTION_TIMESTAMP.set(now_ts)
        try:
            MODEL_PRED_PROB.labels(outcome=outcome).set(float(probability))
        except Exception:
            MODEL_PRED_PROB.labels(outcome=outcome).set(0.0)

        # Write drift report into model metadata (augment)
        try:
            meta_path = os.path.join("models", "model_metadata.json")
            meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
            meta.setdefault("drift_reports", [])
            entry = {"ts": int(time.time()), "input": input_data, "drift": drift_report}
            meta["drift_reports"].append(entry)
            # cap drift_reports to last 50 entries to avoid file blowup
            meta["drift_reports"] = meta["drift_reports"][-50:]
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            print("Failed to write drift report:", e)

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

