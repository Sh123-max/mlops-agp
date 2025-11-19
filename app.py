# app.py
"""
Flask app that serves predictions and exposes Prometheus metrics.
It loads a deployed model (from models/deployed_model or fallback local model),
records inference latency, input validation errors, prediction counts,
and writes drift report entries to models/model_metadata.json.
"""
from flask import Flask, request, render_template
import os, json, joblib, numpy as np, time, socket
from prometheus_client import Gauge, Counter, Histogram
from prometheus_flask_exporter import PrometheusMetrics
from scipy.stats import ks_2samp
import math

app = Flask(__name__)
metrics = PrometheusMetrics(app, path="/metrics")
HOSTNAME = socket.gethostname()

# Prometheus metrics
MODEL_INFO = Gauge("model_version_info", "Info about loaded model", ["model_name", "model_version", "host"])
PREDICTION_COUNT = Counter("prediction_requests_total", "Total prediction requests", ["outcome"])
INPUT_VALIDATION_ERRORS = Counter("input_validation_errors_total", "Input validation errors counted", ["field"])
PREDICTION_LATENCY = Histogram("inference_latency_ms", "Model inference latency (ms)", buckets=(1,5,10,20,50,100,200,500,1000))
MODEL_ACCURACY = Gauge("ml_current_model_accuracy", "Accuracy of deployed ML model")
MODEL_UPTIME = Gauge("model_service_uptime_seconds", "Uptime of model service in seconds")

# global state
model = None
model_name = "unknown"
model_metrics = {}
baseline = None
start_time = time.time()

# path & reload state
meta_path = os.path.join("models", "model_metadata.json")
last_meta_mtime = None

def load_model_and_metadata(force=False):
    """
    Load model_metadata.json and model artifact if available.
    If force is False, function will check mtime to avoid unnecessary reloads.
    """
    global model, model_name, model_metrics, baseline, last_meta_mtime

    try:
        if not os.path.exists(meta_path):
            return
        mtime = os.path.getmtime(meta_path)
        if not force and last_meta_mtime is not None and mtime == last_meta_mtime:
            return  # no change
        last_meta_mtime = mtime

        meta = json.load(open(meta_path))

        # update model_name and model_metrics
        model_name = meta.get("model_name", meta.get("best", {}).get("name", model_name))
        model_metrics = {}
        if "results" in meta:
            for r in meta.get("results", []):
                if r.get("name") == meta.get("best", {}).get("name"):
                    model_metrics = r
                    break

        if "baseline" in meta:
            baseline = meta.get("baseline")

        # set prometheus info label (version or 'unknown')
        version = meta.get("version") or (meta.get("best", {}).get("registry") or {}).get("version") or "unknown"
        try:
            MODEL_INFO.labels(model_name=model_name, model_version=str(version), host=HOSTNAME).set(1)
        except Exception:
            pass
        if "accuracy" in model_metrics:
            try:
                MODEL_ACCURACY.set(model_metrics.get("accuracy"))
            except Exception:
                pass

        # Try to load deployed model from models/deployed_model first, fallback to local model files
        deploy_dir = os.path.join("models", "deployed_model")
        loaded = False
        if os.path.exists(deploy_dir):
            for root, _, files in os.walk(deploy_dir):
                for f in files:
                    if f.endswith(".pkl") or f.endswith(".joblib"):
                        try:
                            new_model = joblib.load(os.path.join(root, f))
                            model = new_model
                            loaded = True
                            print("[INFO] Reloaded deployed model from", os.path.join(root, f))
                            break
                        except Exception as e:
                            print("Failed to load candidate deployed model during reload:", e)
                if loaded:
                    break

        # fallback to model files in models/ e.g., {name}_model.pkl
        if not loaded:
            candidates = [p for p in os.listdir("models") if p.endswith("_model.pkl") or p.endswith("_model.joblib")]
            # prefer file that matches model_name
            pref = None
            for c in candidates:
                if c.startswith(str(model_name)):
                    pref = c
                    break
            choice = pref or (candidates[0] if candidates else None)
            if choice:
                try:
                    model = joblib.load(os.path.join("models", choice))
                    loaded = True
                    print("[INFO] Reloaded fallback model:", choice)
                except Exception as e:
                    print("Failed to reload fallback model during metadata update:", e)

    except Exception as e:
        print("Error while reloading model and metadata:", e)

# initial load at startup
load_model_and_metadata(force=True)

# Reload metadata (and model) check before each request (lightweight: checks mtime)
@app.before_request
def reload_metadata_if_changed():
    try:
        load_model_and_metadata(force=False)
    except Exception:
        pass

# validation ranges (same as earlier)
valid_ranges = {
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

def calculate_psi(expected, actual, buckets=10):
    try:
        expected = np.array(expected).astype(float)
        actual = np.array(actual).astype(float)
        quantiles = np.percentile(expected, np.linspace(0,100,buckets+1))
        quantiles[0] -= 1e-6
        quantiles[-1] += 1e-6
        exp_counts, _ = np.histogram(expected, bins=quantiles)
        act_counts, _ = np.histogram(actual, bins=quantiles)
        exp_perc = np.where(exp_counts == 0, 1e-8, exp_counts / max(1, len(expected)))
        act_perc = np.where(act_counts == 0, 1e-8, act_counts / max(1, len(actual)))
        psi = np.sum((exp_perc - act_perc) * np.log(exp_perc / act_perc))
        return float(psi)
    except Exception:
        return float("nan")

def approx_uncertainty_via_perturbation(model, x_sample, baseline_stds, n_rounds=30):
    try:
        probs = []
        x = np.array(x_sample).astype(float).reshape(1, -1)
        if not hasattr(model, "predict_proba"):
            return {"mean_proba": 0.5, "std_proba": 0.0, "entropy": 0.0}
        stds_arr = np.array([baseline_stds.get(fn, 1.0) for fn in baseline.get("feature_names", [])]) if baseline else np.ones(x.shape[1])
        stds_arr = np.where(stds_arr <= 0, 1e-6, stds_arr)
        for _ in range(n_rounds):
            noise = np.random.normal(loc=0.0, scale=0.01*stds_arr).reshape(1, -1)
            xp = x + noise
            try:
                p = model.predict_proba(xp)[0][1]
            except Exception:
                p = 0.5
            probs.append(float(p))
        probs = np.array(probs)
        mean_p = float(np.mean(probs))
        std_p = float(np.std(probs))
        if mean_p <= 0 or mean_p >= 1:
            entropy = 0.0
        else:
            entropy = - (mean_p * math.log(mean_p + 1e-12) + (1-mean_p) * math.log(1-mean_p + 1e-12))
        return {"mean_proba": mean_p, "std_proba": std_p, "entropy": entropy}
    except Exception:
        return {"mean_proba": 0.5, "std_proba": 0.0, "entropy": 0.0}

@app.route('/')
def home():
    return render_template('form.html', prediction=None, probability=None, error_messages=[], non_diabetic_warnings=[], model_name=model_name, model_metrics=model_metrics)

@app.route('/predict', methods=['POST'])
def predict():
    start = time.time()
    if model is None:
        return render_template('form.html', prediction="Error", probability="Model not loaded", error_messages=["Model missing."], non_diabetic_warnings=[], model_name=model_name, model_metrics=model_metrics)

    input_data = []
    error_messages = []
    for key in FEATURE_ORDER:
        try:
            value = float(request.form.get(key, ""))
            if key in valid_ranges:
                lo, hi = valid_ranges[key]
                if not (lo <= value <= hi):
                    error_messages.append(f"{key} must be between {lo} and {hi}")
                    INPUT_VALIDATION_ERRORS.labels(field=key).inc()
            input_data.append(value)
        except Exception:
            error_messages.append(f"{key} must be numeric")
            INPUT_VALIDATION_ERRORS.labels(field=key).inc()

    if error_messages:
        return render_template('form.html', prediction=None, probability=None, error_messages=error_messages, non_diabetic_warnings=[], model_name=model_name, model_metrics=model_metrics)

    try:
        sample = np.array([input_data])
        # Drift detection (single-sample: noisy; for production use mini-batches)
        drift_report = {"psi": {}, "ks": {}, "uncertainty": {}}
        if baseline and "feature_distributions" in baseline:
            for i, fname in enumerate(baseline.get("feature_names", [])):
                expected = np.array(baseline["feature_distributions"].get(fname, []))
                actual = np.array([input_data[i]])
                psi = calculate_psi(expected, actual, buckets=min(10, max(2, len(expected)//5)))
                drift_report["psi"][fname] = psi
                try:
                    ks_stat, pval = ks_2samp(expected, actual)
                    drift_report["ks"][fname] = {"ks_stat": float(ks_stat), "pval": float(pval)}
                except Exception:
                    drift_report["ks"][fname] = {"ks_stat": float("nan"), "pval": float("nan")}
        else:
            drift_report["psi"] = {}
            drift_report["ks"] = {}

        unc = approx_uncertainty_via_perturbation(model, input_data, baseline.get("feature_stds", {}) if baseline else {}, n_rounds=30)
        drift_report["uncertainty"] = unc

        # prediction + latency metric
        with PREDICTION_LATENCY.time():
            pred = model.predict(sample)[0]
            try:
                proba = model.predict_proba(sample)[0][1] if hasattr(model, "predict_proba") else 0.5
            except Exception:
                proba = 0.5

        latency_ms = (time.time() - start) * 1000.0
        # prom histogram observed via context manager; additionally log as custom field in metadata
        outcome = "diabetic" if pred == 1 else "not_diabetic"
        PREDICTION_COUNT.labels(outcome=outcome).inc()

        # Append drift report to metadata (cap to last 50 entries)
        try:
            meta_path_local = os.path.join("models", "model_metadata.json")
            meta = json.load(open(meta_path_local)) if os.path.exists(meta_path_local) else {}
            meta.setdefault("drift_reports", [])
            entry = {"ts": int(time.time()), "input": input_data, "latency_ms": latency_ms, "drift": drift_report}
            meta["drift_reports"].append(entry)
            meta["drift_reports"] = meta["drift_reports"][-50:]
            with open(meta_path_local, "w") as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            print("Failed to write drift report:", e)

        result = "Diabetic" if pred == 1 else "Not Diabetic"
        return render_template('form.html', prediction=result, probability=f"{proba:.1%}", error_messages=[], non_diabetic_warnings=[], model_name=model_name, model_metrics=model_metrics)
    except Exception as e:
        return render_template('form.html', prediction=None, probability=None, error_messages=[f"Prediction error: {e}"], non_diabetic_warnings=[], model_name=model_name, model_metrics=model_metrics)

@app.route('/health')
def health():
    uptime = time.time() - start_time
    MODEL_UPTIME.set(uptime)
    return {"status": "healthy", "uptime_seconds": uptime}

if __name__ == '__main__':
    import sys
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass
    print("[INFO] Starting Flask server on port 5000...")
    app.run(host='0.0.0.0', port=5000)
