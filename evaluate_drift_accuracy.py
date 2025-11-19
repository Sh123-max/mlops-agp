#!/usr/bin/env python3
"""
evaluate_drift_full.py

Compute:
 - accuracy_after_drift (on labeled new data)
 - per-feature PSI (Population Stability Index)
 - per-feature KS test (statistic + p-value)
 - dataset-level uncertainty via perturbation (entropy)
 - aggregate drift score (weighted)
Log metrics to MLflow, push key gauges to Pushgateway, and append report to models/model_metadata.json.

Usage:
python3 evaluate_drift_full.py --data data/recent_7_days.csv \
    --pushgateway http://localhost:9091 --mlflow http://localhost:5001

"""

import os, sys, json, argparse, glob, joblib, math, time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import mlflow
from prometheus_client import Gauge, CollectorRegistry, push_to_gateway
from scipy.stats import ks_2samp

# Metric names
METRIC_ACC = "accuracy_after_drift"
METRIC_MEAN_ENTROPY = "mean_prediction_entropy"
METRIC_AGG_DRIFT = "aggregate_drift_score"

# Create process-level gauges (only used if pushgateway not supplied)
_g_acc = Gauge(METRIC_ACC, "Model accuracy after drift")
_g_entropy = Gauge(METRIC_MEAN_ENTROPY, "Mean prediction entropy after perturbation")
_g_agg = Gauge(METRIC_AGG_DRIFT, "Aggregate drift score")

# PSI helper
def calculate_psi(expected, actual, buckets=10):
    try:
        expected = np.asarray(expected).astype(float)
        actual = np.asarray(actual).astype(float)
        # build quantile bins based on expected
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

def approx_uncertainty_via_perturbation(model, X, baseline_stds, n_rounds=30):
    """
    For each sample compute predictive probability distribution by perturbing inputs by
    small noise proportional to baseline_stds. Returns mean entropy across samples and
    per-sample summary.
    """
    n_samples = X.shape[0]
    entropies = []
    mean_probs = []
    std_probs = []
    if not hasattr(model, "predict_proba"):
        # fallback: use decision_function to approximate probabilities if possible? else constant
        return {"mean_entropy": 0.0, "mean_proba": 0.5, "std_proba": 0.0, "per_sample": []}

    # prepare stds array shape=(n_features,)
    stds_arr = np.array([baseline_stds.get(fname, 1.0) for fname in baseline_feature_names]) if baseline_feature_names else np.ones(X.shape[1])
    stds_arr = np.where(stds_arr <= 0, 1e-6, stds_arr)

    per_sample = []
    for i in range(n_samples):
        probs = []
        x = X[i].reshape(1, -1).astype(float)
        for _ in range(n_rounds):
            noise = np.random.normal(loc=0.0, scale=0.01*stds_arr).reshape(1, -1)
            xp = x + noise
            try:
                p = model.predict_proba(xp)[0][1]
            except Exception:
                p = 0.5
            probs.append(float(p))
        probs = np.array(probs)
        mean_p = float(probs.mean())
        std_p = float(probs.std())
        # entropy for Bernoulli
        if mean_p <= 0 or mean_p >= 1:
            entropy = 0.0
        else:
            entropy = - (mean_p * math.log(mean_p + 1e-12) + (1-mean_p) * math.log(1-mean_p + 1e-12))
        per_sample.append({"mean_proba": mean_p, "std_proba": std_p, "entropy": entropy})
        entropies.append(entropy)
        mean_probs.append(mean_p)
        std_probs.append(std_p)
    return {"mean_entropy": float(np.mean(entropies)), "mean_proba": float(np.mean(mean_probs)), "std_proba": float(np.mean(std_probs)), "per_sample": per_sample}

# Model / artifact discovery helpers (reuse your existing logic)
def find_deployed_model(models_dir="models"):
    deployed_dir = os.path.join(models_dir, "deployed_model")
    if os.path.exists(deployed_dir):
        for ext in ("*.pkl", "*.joblib"):
            files = glob.glob(os.path.join(deployed_dir, ext))
            if files:
                return files[0]
    fallback = glob.glob(os.path.join(models_dir, "*_model.pkl"))
    if fallback:
        return fallback[0]
    return None

def load_model(model_path=None):
    if model_path and os.path.exists(model_path):
        p = model_path
    else:
        p = find_deployed_model()
    if not p:
        raise FileNotFoundError("No model artifact found in models/deployed_model or models/*.pkl")
    print("[INFO] Loading model from:", p)
    model = joblib.load(p)
    return model, p

def load_scaler_and_baseline(data_dir="data", models_dir="models"):
    # try model_metadata.json first for baseline distributions
    baseline = None
    md_path = os.path.join(models_dir, "model_metadata.json")
    if os.path.exists(md_path):
        try:
            md = json.load(open(md_path))
            baseline = md.get("baseline", None)
            if baseline:
                print("[INFO] Loaded baseline from models/model_metadata.json")
        except Exception as e:
            print("[WARN] Failed to load models/model_metadata.json:", e)
    # fallback to data/scaler.pkl or models/scaler.pkl and X_train_unscaled_df
    scaler = None
    scaler_paths = [os.path.join(data_dir, "scaler.pkl"), os.path.join(models_dir, "scaler.pkl")]
    for p in scaler_paths:
        if os.path.exists(p):
            try:
                scaler = joblib.load(p)
                print("[INFO] Loaded scaler from", p)
                break
            except Exception as e:
                print("[WARN] Failed to load scaler at", p, e)
    # try X_train_unscaled_df
    X_train_unscaled = None
    p_unscaled = os.path.join(data_dir, "X_train_unscaled_df.pkl")
    if os.path.exists(p_unscaled):
        try:
            X_train_unscaled = joblib.load(p_unscaled)
            print("[INFO] Loaded X_train_unscaled_df from data")
        except Exception:
            X_train_unscaled = None
    return scaler, baseline, X_train_unscaled

def load_data(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "Outcome" not in df.columns:
        raise ValueError("CSV must contain 'Outcome' column with labels")
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"].values
    return X, y, df

def push_metric(pushgateway, metric_name, value, job="drift-eval"):
    try:
        registry = CollectorRegistry()
        g_local = Gauge(metric_name, "drift metric", registry=registry)
        g_local.set(float(value))
        push_to_gateway(pushgateway, job=job, registry=registry)
        print(f"[INFO] Pushed {metric_name}={value} to Pushgateway {pushgateway}")
    except Exception as e:
        print("[WARN] Pushgateway push failed:", e)

def append_report_to_metadata(report, models_dir="models"):
    path = os.path.join(models_dir, "model_metadata.json")
    existing = {}
    if os.path.exists(path):
        try:
            existing = json.load(open(path))
        except Exception:
            existing = {}
    existing.setdefault("drift_reports", [])
    existing["drift_reports"].append(report)
    # keep last 100 entries
    existing["drift_reports"] = existing["drift_reports"][-100:]
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
    print("[INFO] Appended drift report to", path)

def normalize_psi(psi):
    # rough normalization to [0,1] where higher indicates worse drift
    if psi <= 0.05:
        return 0.0
    if psi < 0.1:
        return 0.25
    if psi < 0.2:
        return 0.6
    return 1.0

def normalize_ks(stat, pval):
    # pval small => significant difference. stat in [0,1]. return [0,1] where higher => worse
    return float(min(1.0, stat))

# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV path for new (recent) labeled data with Outcome column")
    parser.add_argument("--model-path", default=None, help="optional explicit model path")
    parser.add_argument("--pushgateway", default=os.getenv("PUSHGATEWAY_URL", None), help="Pushgateway URL")
    parser.add_argument("--mlflow", default=os.getenv("MLFLOW_TRACKING_URI", None), help="MLflow tracking URI")
    parser.add_argument("--data-dir", default=os.getenv("DATA_DIR", "data"), help="DATA_DIR for scaler / unscaled df")
    parser.add_argument("--n-perturb", type=int, default=30, help="perturbation rounds per sample")
    parser.add_argument("--psi-buckets", type=int, default=10)
    parser.add_argument("--agg-weights", default=None, help="json string for weights e.g. '{\"acc\":0.5,\"psi\":0.2,\"ks\":0.15,\"entropy\":0.15}'")
    parser.add_argument("--alert-threshold", type=float, default=0.5, help="aggregate drift score threshold to trigger non-zero exit")
    args = parser.parse_args()

    if args.mlflow:
        mlflow.set_tracking_uri(args.mlflow)
        print("[INFO] MLflow tracking URI:", mlflow.get_tracking_uri())

    # load model & artifacts
    model, model_file = load_model(args.model_path)
    scaler, baseline, X_train_unscaled_df = load_scaler_and_baseline(args.data_dir)

    # baseline feature names and distributions
    baseline_feature_names = None
    baseline_feature_distributions = {}
    baseline_feature_stds = {}
    if baseline:
        baseline_feature_names = baseline.get("feature_names", None)
        baseline_feature_distributions = baseline.get("feature_distributions", {})
        baseline_feature_stds = baseline.get("feature_stds", {})
    else:
        if X_train_unscaled_df is not None:
            baseline_feature_names = list(X_train_unscaled_df.columns)
            for c in baseline_feature_names:
                arr = X_train_unscaled_df[c].dropna().values
                baseline_feature_distributions[c] = arr.tolist()
                baseline_feature_stds[c] = float(np.nanstd(arr))
            print("[INFO] Built baseline distributions from X_train_unscaled_df")
        elif scaler is not None:
            # fallback: if scaler exists but unscaled df not present, use scaler mean/stdev approx (not ideal)
            try:
                # scaler may be StandardScaler with mean_ and scale_
                means = getattr(scaler, "mean_", None)
                scales = getattr(scaler, "scale_", None)
                if means is not None and scales is not None:
                    baseline_feature_names = [f"feat_{i}" for i in range(len(means))]
                    for i, nm in enumerate(baseline_feature_names):
                        baseline_feature_stds[nm] = float(scales[i])
                    print("[WARN] Only scaler stats available as baseline (no feature names).")
            except Exception as e:
                print("[WARN] Could not extract baseline from scaler:", e)

    # load new data
    X_df, y, df_new = load_data(args.data)
    feature_names = list(X_df.columns)
    X_vals = X_df.values

    # optionally apply scaler (if scaler present)
    if scaler is not None:
        try:
            X_scaled_vals = scaler.transform(X_vals)
            X_for_model = X_scaled_vals
            print("[INFO] Applied scaler transform to new data.")
        except Exception as e:
            print("[WARN] Scaler transform failed:", e)
            X_for_model = X_vals
    else:
        X_for_model = X_vals

    # compute accuracy
    try:
        y_pred = model.predict(X_for_model)
        acc = float(accuracy_score(y, y_pred))
        print(f"[INFO] Accuracy on new data: {acc:.4f}")
    except Exception as e:
        print("[ERROR] Prediction failed:", e)
        sys.exit(2)

    # compute PSI & KS per feature (use baseline distributions if available; else compute from X_train_unscaled_df if names align)
    psi_results = {}
    ks_results = {}
    for i, fname in enumerate(feature_names):
        new_vals = X_df[fname].dropna().values.astype(float)
        # choose baseline array
        if baseline_feature_distributions and str(fname) in baseline_feature_distributions:
            base_arr = np.array(baseline_feature_distributions[str(fname)])
        else:
            # best-effort: if X_train_unscaled_df present and columns match
            if X_train_unscaled_df is not None and fname in X_train_unscaled_df.columns:
                base_arr = np.array(X_train_unscaled_df[fname].dropna().values)
            else:
                # fallback: cannot compute PSI/KS reliably
                psi_results[fname] = {"psi": float("nan")}
                ks_results[fname] = {"ks_stat": float("nan"), "pval": float("nan")}
                continue
        psi_val = calculate_psi(base_arr, new_vals, buckets=args.psi_buckets)
        psi_results[fname] = {"psi": psi_val}
        try:
            ks_stat, pval = ks_2samp(base_arr, new_vals)
            ks_results[fname] = {"ks_stat": float(ks_stat), "pval": float(pval)}
        except Exception:
            ks_results[fname] = {"ks_stat": float("nan"), "pval": float("nan")}

    # approximate uncertainty via perturbation
    # For perturbation we need baseline stds keyed by feature names
    baseline_stds_map = {}
    if baseline_feature_stds:
        for k, v in baseline_feature_stds.items():
            baseline_stds_map[k] = float(v)
    else:
        # fallback: compute std from X_train_unscaled_df if available
        if X_train_unscaled_df is not None:
            for c in X_train_unscaled_df.columns:
                baseline_stds_map[c] = float(np.nanstd(X_train_unscaled_df[c].dropna().values))
    # prepare baseline_feature_names variable for the function scope
    baseline_feature_names = list(baseline_feature_distributions.keys()) if baseline_feature_distributions else (list(X_train_unscaled_df.columns) if X_train_unscaled_df is not None else None)

    uq_res = approx_uncertainty_via_perturbation(model, X_for_model, baseline_stds_map, n_rounds=args.n_perturb)
    mean_entropy = float(uq_res.get("mean_entropy", 0.0))

    # Build normalized/aggregated drift signals
    # Normalize psi and ks per feature to [0,1], then average across features
    psi_norms = []
    ks_norms = []
    for fname in feature_names:
        p = psi_results.get(fname, {}).get("psi", float("nan"))
        kstat = ks_results.get(fname, {}).get("ks_stat", float("nan"))
        if not math.isnan(p):
            psi_norms.append(normalize_psi(p))
        if not math.isnan(kstat):
            ks_norms.append(normalize_ks(kstat, None))
    mean_psi_norm = float(np.mean(psi_norms)) if psi_norms else 0.0
    mean_ks_norm = float(np.mean(ks_norms)) if ks_norms else 0.0

    # Prepare aggregation weights (defaults)
    if args.agg_weights:
        try:
            weights = json.loads(args.agg_weights)
        except Exception:
            weights = None
    else:
        weights = None
    if not weights:
        weights = {"acc":0.4, "psi":0.2, "ks":0.2, "entropy":0.2}

    # We want higher aggregate => worse. accuracy is inverse (higher accuracy is better) so we use (1-acc)
    score_acc = 1.0 - acc
    score_psi = mean_psi_norm
    score_ks = mean_ks_norm
    # normalize entropy roughly to [0,1] by mapping typical entropy scale (0..~0.7 for binary)
    # we'll cap at 0.7
    score_entropy = min(1.0, mean_entropy / 0.7)

    aggregate_score = (weights["acc"]*score_acc + weights["psi"]*score_psi + weights["ks"]*score_ks + weights["entropy"]*score_entropy) / (sum(weights.values()) or 1.0)
    aggregate_score = float(aggregate_score)

    # Build a report
    ts = int(time.time())
    report = {
        "ts": ts,
        "data": os.path.basename(args.data),
        "model_file": os.path.basename(model_file),
        "accuracy": acc,
        "mean_entropy": mean_entropy,
        "mean_psi_norm": mean_psi_norm,
        "mean_ks_norm": mean_ks_norm,
        "aggregate_score": aggregate_score,
        "psi_per_feature": psi_results,
        "ks_per_feature": ks_results,
        "uq_summary": {"mean_entropy": mean_entropy, "details": {"n_samples": len(X_for_model)}},
        "weights": weights
    }

    print("[INFO] Drift report summary:", json.dumps(report, indent=2))

    # Log to MLflow
    if args.mlflow:
        try:
            with mlflow.start_run(run_name="drift_full_eval") as run:
                mlflow.log_metric(METRIC_ACC, acc)
                mlflow.log_metric(METRIC_MEAN_ENTROPY, mean_entropy)
                mlflow.log_metric(METRIC_AGG_DRIFT, aggregate_score)
                # also log mean psi/ks
                mlflow.log_metric("mean_psi_norm", mean_psi_norm)
                mlflow.log_metric("mean_ks_norm", mean_ks_norm)
                mlflow.set_tag("drift_eval_source", os.path.basename(args.data))
                mlflow.log_param("model_file", os.path.basename(model_file))
                # log per-feature small summary as artifact
                mlflow.log_text(json.dumps(report, indent=2), "drift_report.json")
                print("[INFO] Logged drift metrics to MLflow")
        except Exception as e:
            print("[WARN] MLflow logging failed:", e)

    # Push to Pushgateway key metrics
    if args.pushgateway:
        push_metric(args.pushgateway, METRIC_ACC, acc, job="drift-eval")
        push_metric(args.pushgateway, METRIC_MEAN_ENTROPY, mean_entropy, job="drift-eval")
        push_metric(args.pushgateway, METRIC_AGG_DRIFT, aggregate_score, job="drift-eval")
    else:
        # set local gauges
        _g_acc.set(acc)
        _g_entropy.set(mean_entropy)
        _g_agg.set(aggregate_score)

    # append to models/model_metadata.json drift_reports
    append_report_to_metadata(report)

    # optionally alert by non-zero exit if aggregate exceeds threshold
    if aggregate_score >= args.alert_threshold:
        print(f"[ALERT] Aggregate drift score {aggregate_score:.4f} >= threshold {args.alert_threshold}")
        # exit non-zero to let CI/pipeline detect failure/need-to-retrain
        sys.exit(3)
    else:
        print(f"[OK] Aggregate drift score {aggregate_score:.4f} below threshold {args.alert_threshold}")

    sys.exit(0)
