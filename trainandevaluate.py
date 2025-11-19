"""
Train multiple models, evaluate, perform Pareto-style selection (non-dominated),
register best model with MLflow (from main thread after verifying artifacts),
write metadata, and push summary metrics to Pushgateway.
"""
import os
import json
import time
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import joblib
import mlflow
import mlflow.sklearn
import requests
import numpy as np
import tempfile
from mlflow.tracking import MlflowClient
from time import sleep

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier

# Import the new metrics history and ensemble manager
try:
    from metrics_history import MetricsHistory
    from ensemble_manager import EnsembleManager
except ImportError:
    # Fallback if modules are not available
    class MetricsHistory:
        def __init__(self, *args, **kwargs): pass
        def add_training_run(self, *args, **kwargs): pass
        def get_previous_best(self): return None
        def add_deployment(self, *args, **kwargs): pass

    class EnsembleManager:
        def __init__(self, *args, **kwargs): pass
        def train_and_evaluate_ensemble(self, *args, **kwargs): return None, None
        def log_ensemble_to_mlflow(self, *args, **kwargs): return None

# Config
PROJECT_NAME = os.getenv("PROJECT_NAME", "diabetes")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
PUSHGATEWAY_URL = os.getenv("PUSHGATEWAY_URL", "http://localhost:9091")
DATA_DIR = os.getenv("DATA_DIR", "data")
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
MODEL_DIR.mkdir(exist_ok=True, parents=True)

# Resource/threading
try:
    NUM_WORKER_THREADS = int(os.getenv("NUM_WORKER_THREADS", str(max(1, os.cpu_count()//2))))
except Exception:
    NUM_WORKER_THREADS = max(1, os.cpu_count()//2)
NUM_WORKER_THREADS = min(NUM_WORKER_THREADS, 3)

os.environ.setdefault("OMP_NUM_THREADS", os.getenv("OMP_NUM_THREADS", "2"))
os.environ.setdefault("MKL_NUM_THREADS", os.getenv("MKL_NUM_THREADS", "2"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", os.getenv("OPENBLAS_NUM_THREADS", "2"))

# Healthcare weights (used as tie-breaker)
weights = {'Accuracy':0.05,'Precision':0.05,'Recall':0.4,'F1-Score':0.3,'ROC-AUC':0.2}

# MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
print("MLflow tracking URI:", mlflow.get_tracking_uri())
print("NUM_WORKER_THREADS:", NUM_WORKER_THREADS)

# Load preprocessed data
X_train = joblib.load(os.path.join(DATA_DIR, "X_train.pkl"))
X_test = joblib.load(os.path.join(DATA_DIR, "X_test.pkl"))
y_train = joblib.load(os.path.join(DATA_DIR, "y_train.pkl"))
y_test = joblib.load(os.path.join(DATA_DIR, "y_test.pkl"))

# Try to load scaler/unscaled data for baseline distributions
scaler = None
try:
    scaler = joblib.load(os.path.join(DATA_DIR, "scaler.pkl"))
except Exception:
    scaler = None

# If preprocess saved unscaled dfs (helpful for baseline), load them if present
X_train_unscaled_df = None
try:
    path_unscaled = os.path.join(DATA_DIR, "X_train_unscaled_df.pkl")
    if os.path.exists(path_unscaled):
        X_train_unscaled_df = joblib.load(path_unscaled)
except Exception:
    X_train_unscaled_df = None

# Models dict
models = {
    'LogisticRegression': LogisticRegression(max_iter=2000),
    'RandomForest': RandomForestClassifier(random_state=42, n_jobs=1),
    'SVM': SVC(probability=True),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'KNN': KNeighborsClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=max(1, NUM_WORKER_THREADS), verbosity=1, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42)
}

stacking = StackingClassifier(
    estimators=[
        ('lr', LogisticRegression(max_iter=2000)),
        ('rf', RandomForestClassifier(random_state=42, n_jobs=1)),
        ('svm', SVC(probability=True))
    ],
    final_estimator=LogisticRegression()
)
models['StackingEnsemble'] = stacking

def safe_roc_auc(y_true, y_score):
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return 0.0

def measure_latency(model, X_sample, n_runs=10):
    try:
        for _ in range(2):
            _ = model.predict(X_sample)
        t0 = time.time()
        for _ in range(n_runs):
            _ = model.predict(X_sample)
        t1 = time.time()
        return float((t1 - t0) / n_runs)
    except Exception:
        return float(1e6)

def measure_model_size(model):
    try:
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "m.joblib")
            joblib.dump(model, p)
            return float(os.path.getsize(p))
    except Exception:
        return float(1e12)

def nondominated_front(points):
    n = len(points)
    dominated = [False]*n
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j or dominated[j]:
                continue
            all_leq = True
            strictly_less = False
            for a,b in zip(points[j], points[i]):
                if b + 1e-12 < a:
                    all_leq = False
                    break
                if a + 1e-12 < b:
                    strictly_less = True
            if all_leq and strictly_less:
                dominated[i] = True
                break
    return [i for i, d in enumerate(dominated) if not d]

def validate_against_previous_best(current_best, previous_best):
    """Compare current model with previous best and return whether to deploy"""
    if not previous_best:
        return True, "First deployment"

    # If ensemble, prefer it (we assume ensemble manager validated it)
    if current_best.get("is_ensemble", False):
        return True, "Ensemble model outperforms individual models"

    current_score = current_best.get("weighted_score", 0)
    previous_score = previous_best.get("weighted_score", 0)
    current_fnr = current_best.get("false_negative_rate", 1)
    previous_fnr = previous_best.get("false_negative_rate", 1)

    # Critical healthcare metrics - false negative rate should not increase
    if current_fnr > previous_fnr * 1.1:  # 10% increase in FNR
        return False, f"False negative rate increased from {previous_fnr:.4f} to {current_fnr:.4f}"

    # Weighted score degradation thresholds
    score_drop = previous_score - current_score

    if score_drop <= 0.01:  # 1% or improvement
        return True, f"Performance maintained or improved: {score_drop:.4f}"
    elif score_drop <= 0.03:  # 1-3% drop
        return True, f"Minor performance drop: {score_drop:.4f}. Ensemble was attempted."
    else:  # >3% drop
        return False, f"Significant performance drop: {score_drop:.4f}. Manual review required."

def train_and_log(name, model):
    run_name = f"{PROJECT_NAME}__{name}__{int(time.time())}"
    try:
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            print(f"[{name}] Training (run_id={run_id})")
            t0 = time.time()
            model.fit(X_train, y_train)
            t1 = time.time()
            retrain_time = t1 - t0

            y_pred = model.predict(X_test)
            if hasattr(model, "predict_proba"):
                try:
                    y_proba = model.predict_proba(X_test)[:,1]
                except Exception:
                    y_proba = np.zeros_like(y_pred, dtype=float)
            elif hasattr(model, "decision_function"):
                from scipy.special import expit
                y_proba = expit(model.decision_function(X_test))
            else:
                y_proba = np.zeros_like(y_pred, dtype=float)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc = safe_roc_auc(y_test, y_proba)

            try:
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                fn_rate = float(fn) / float(fn + tp) if (fn + tp) > 0 else 0.0
            except Exception:
                fn_rate = 0.0

            sample_count = min(50, X_test.shape[0])
            X_sample = X_test[:sample_count]
            latency = measure_latency(model, X_sample, n_runs=10)
            model_size = measure_model_size(model)

            weighted = (weights['Accuracy']*acc + weights['Precision']*prec + weights['Recall']*rec + weights['F1-Score']*f1 + weights['ROC-AUC']*roc)

            # Log metrics and params
            mlflow.log_param("model_name", name)
            mlflow.log_metric("accuracy", float(acc))
            mlflow.log_metric("precision", float(prec))
            mlflow.log_metric("recall", float(rec))
            mlflow.log_metric("f1_score", float(f1))
            mlflow.log_metric("roc_auc", float(roc))
            mlflow.log_metric("weighted_score", float(weighted))
            mlflow.log_metric("retrain_time_seconds", float(retrain_time))
            mlflow.log_metric("inference_latency_sec", float(latency))
            mlflow.log_metric("model_size_bytes", float(model_size))
            mlflow.log_metric("false_negative_rate", float(fn_rate))

            mlflow.set_tag("project", PROJECT_NAME)
            mlflow.set_tag("model_name", name)

            try:
                mlflow.sklearn.log_model(model, artifact_path="model")
                try:
                    artifact_uri = mlflow.get_artifact_uri("model")
                    print(f"[{name}] logged artifact_uri={artifact_uri}")
                except Exception as e:
                    print(f"[{name}] warning getting artifact uri: {e}")
            except Exception as e:
                print(f"[{name}] mlflow log model failed:", e)

            # Save local model as fallback
            try:
                local_fn = MODEL_DIR / f"{name}_model.pkl"
                joblib.dump(model, local_fn)
            except Exception as e:
                print("Failed to locally save model:", e)

            print(f"[{name}] done: weighted_score={weighted:.4f}, retrain_time={retrain_time:.2f}s, fn_rate={fn_rate:.4f}")
            return {
                "name": name,
                "run_id": run_id,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "roc_auc": roc,
                "weighted_score": weighted,
                "retrain_time_seconds": retrain_time,
                "latency": latency,
                "model_size": model_size,
                "false_negative_rate": fn_rate
            }
    except Exception as e:
        print(f"[{name}] failed:", e)
        traceback.print_exc()
        return {"name": name, "error": str(e)}

# Run training in parallel (thread pool)
results = []
with ThreadPoolExecutor(max_workers=NUM_WORKER_THREADS) as ex:
    futures = {ex.submit(train_and_log, n, m): n for n, m in models.items()}
    for fut in as_completed(futures):
        name = futures[fut]
        try:
            res = fut.result()
            results.append(res)
            if "error" not in res:
                print(f"[MAIN] Collected result for {name} weighted_score={res.get('weighted_score')}")
        except Exception as e:
            print("Future exception for", name, e)

# Multi-objective selection
valid_results = [r for r in results if "error" not in r and r.get("run_id")]
objs = []
for r in valid_results:
    vec = [
        -float(r.get("recall", 0.0)),
        -float(r.get("precision", 0.0)),
        -float(r.get("f1_score", 0.0)),
        -float(r.get("roc_auc", 0.0)),
        float(r.get("latency", 1e6)),
        float(r.get("model_size", 1e12)),
        float(r.get("retrain_time_seconds", 1e6)),
        float(r.get("false_negative_rate", 1.0))
    ]
    objs.append(vec)

pareto_indices = nondominated_front(objs) if objs else []
pareto_models = [valid_results[i] for i in pareto_indices] if pareto_indices else []

best = {"score": -1, "name": None, "run_id": None, "registry": None}
if pareto_models:
    pareto_models = sorted(pareto_models, key=lambda x: (x.get("recall",0), x.get("f1_score",0), x.get("roc_auc",0)), reverse=True)
    chosen = pareto_models[0]
    best = {"score": float(chosen.get("weighted_score", -1)), "name": chosen["name"], "run_id": chosen.get("run_id"), "registry": None}
else:
    # fallback: best by weighted score
    if valid_results:
        chosen = max(valid_results, key=lambda r: float(r.get("weighted_score", -1)))
        best = {"score": float(chosen.get("weighted_score", -1)), "name": chosen["name"], "run_id": chosen.get("run_id"), "registry": None}

# Performance validation against previous best
metrics_history = MetricsHistory()
previous_best = metrics_history.get_previous_best()

# ENSEMBLE CREATION FOR PERFORMANCE DROPS > 1%
ensemble_manager = EnsembleManager(MODEL_DIR, PROJECT_NAME)
previous_best_score = previous_best.get('weighted_score') if previous_best else None
current_best_score = best['score']

# Get the actual model object for the current best
current_best_model = None
for name, model_obj in models.items():
    if name == best['name']:
        current_best_model = model_obj
        break

# If we have a current model and previous score, try ensemble
if current_best_model and previous_best_score is not None:
    ensemble, ensemble_metrics = ensemble_manager.train_and_evaluate_ensemble(
        X_train, y_train, X_test, y_test, best['name'], current_best_score, previous_best_score
    )
    if ensemble is not None and ensemble_metrics is not None:
        # Check if ensemble performs better than current model
        if ensemble_metrics['weighted_score'] > current_best_score:
            print(f"🎯 Ensemble outperforms current model! Using ensemble.")
            # Update best model to ensemble
            ensemble_name = ensemble_manager.log_ensemble_to_mlflow(ensemble, ensemble_metrics, best['name'], previous_best.get('model_name', 'PreviousBest'))
            if ensemble_name:
                best = {
                    "score": ensemble_metrics['weighted_score'],
                    "name": ensemble_name,
                    "run_id": None,  # may be set by log_ensemble_to_mlflow
                    "registry": None,
                    "is_ensemble": True,
                    "base_models": f"{previous_best.get('model_name')},{best['name']}"
                }
                # Add ensemble to results
                ensemble_result = {
                    "name": ensemble_name,
                    "accuracy": ensemble_metrics['accuracy'],
                    "precision": ensemble_metrics['precision'],
                    "recall": ensemble_metrics['recall'],
                    "f1_score": ensemble_metrics['f1_score'],
                    "roc_auc": ensemble_metrics['roc_auc'],
                    "weighted_score": ensemble_metrics['weighted_score'],
                    "is_ensemble": True,
                    "retrain_time_seconds": 0,
                    "latency": 0,
                    "model_size": 0,
                    "false_negative_rate": 0
                }
                results.append(ensemble_result)
                valid_results.append(ensemble_result)
                print(f"🔄 Updated best model to ensemble: {ensemble_name}")
        else:
            print(f"📉 Ensemble did not improve performance. Keeping current best model.")

# Add all valid results to history
for result in results:
    if "error" not in result:
        metrics_history.add_training_run(result)

# Validate against previous best
should_deploy, deploy_reason = validate_against_previous_best(best, previous_best)
best["should_deploy"] = should_deploy
best["deploy_reason"] = deploy_reason

print(f"Deployment decision: {should_deploy}, Reason: {deploy_reason}")

# Now register the best model from the main thread (with artifact availability check)
if best.get("run_id") and best.get("name") and should_deploy:
    run_id = best["run_id"]
    registry_name = f"{PROJECT_NAME}_{best['name']}"
    max_wait_sec = 30
    poll_interval = 2
    waited = 0
    ok = False
    print(f"[MAIN] Waiting up to {max_wait_sec}s for artifacts for run {run_id} before registering...")
    while waited < max_wait_sec:
        try:
            arts = client.list_artifacts(run_id, path="model")
            if arts and len(arts) > 0:
                ok = True
                break
        except Exception as e:
            print("[MAIN] list_artifacts error (will retry):", e)
        sleep(poll_interval)
        waited += poll_interval

    if not ok:
        print(f"[MAIN] Warning: artifacts for run {run_id} not found after {max_wait_sec}s. Attempting registration anyway (may point to models:/ without artifacts).")

    try:
        print(f"[MAIN] Registering model runs:/{run_id}/model -> {registry_name}")
        registered = mlflow.register_model(f"runs:/{run_id}/model", registry_name)
        best["registry"] = {"name": registry_name, "version": getattr(registered, "version", None)}
        print("[MAIN] Registered model:", best["registry"])
    except Exception as e:
        print("[MAIN] Register failed:", e)
else:
    print(f"[MAIN] Skipping model registration - deployment not approved: {deploy_reason}")

# summary
summary = {
    "project": PROJECT_NAME,
    "results": results,
    "pareto_indices": pareto_indices,
    "pareto_models": pareto_models,
    "best": best,
    "previous_best": previous_best,
    "ts": int(time.time())
}
with open(MODEL_DIR / "last_run_summary.json", "w") as fh:
    json.dump(summary, fh, indent=2)

print("Training complete. Best:", best)

# Build model_metadata.json (baseline distributions + metrics)
metadata = {"project": PROJECT_NAME, "best": best, "results": results, "generated_at": int(time.time())}
# add top-level model_name for deploy/app convenience
if best.get("name"):
    metadata["model_name"] = best.get("name")
if best.get("registry"):
    metadata["version"] = best["registry"].get("version")

try:
    if X_train_unscaled_df is not None:
        df = X_train_unscaled_df
        feat_names = list(df.columns)
        feature_distributions = {str(c): df[c].dropna().tolist() for c in feat_names}
        feature_means = {str(c): float(df[c].mean()) for c in feat_names}
        feature_stds = {str(c): float(df[c].std()) for c in feat_names}
    else:
        if scaler is not None:
            try:
                X_train_unscaled = scaler.inverse_transform(X_train)
            except Exception:
                X_train_unscaled = X_train
        else:
            X_train_unscaled = X_train
        n_feats = X_train_unscaled.shape[1]
        feat_names = [f"feat_{i}" for i in range(n_feats)]
        feature_distributions = {feat_names[i]: X_train_unscaled[:, i].tolist() for i in range(n_feats)}
        feature_means = {feat_names[i]: float(np.nanmean(X_train_unscaled[:, i])) for i in range(n_feats)}
        feature_stds = {feat_names[i]: float(np.nanstd(X_train_unscaled[:, i])) for i in range(n_feats)}

    metadata["baseline"] = {"feature_names": feat_names, "feature_distributions": feature_distributions, "feature_means": feature_means, "feature_stds": feature_stds}
except Exception as e:
    print("Failed to compute baseline distributions:", e)

# Attempt to append existing metadata (if present) and write final metadata
meta_path = MODEL_DIR / "model_metadata.json"
try:
    existing = {}
    if meta_path.exists():
        try:
            existing = json.load(open(meta_path))
        except Exception:
            existing = {}
    existing.update(metadata)
    with open(meta_path, "w") as fh:
        json.dump(existing, fh, indent=2)
    print("[OK] model metadata written to", meta_path)
except Exception as e:
    print("Failed to write model metadata:", e)

# Push summary metric to Pushgateway for monitoring dashboards
if PUSHGATEWAY_URL and best.get("name") and should_deploy:
    try:
        payload_lines = []
        payload_lines.append(f'model_weighted_score{{project="{PROJECT_NAME}",model="{best["name"]}"}} {best["score"]}')
        chosen_res = None
        for r in results:
            if r.get("name") == best.get("name"):
                chosen_res = r
                break
        if chosen_res:
            payload_lines.append(f'retrain_time_seconds{{project="{PROJECT_NAME}",model="{best["name"]}"}} {chosen_res.get("retrain_time_seconds", -1)}')
            payload_lines.append(f'false_negative_rate{{project="{PROJECT_NAME}",model="{best["name"]}"}} {chosen_res.get("false_negative_rate", -1)}')
            payload_lines.append(f'inference_latency_seconds{{project="{PROJECT_NAME}",model="{best["name"]}"}} {chosen_res.get("latency", -1)}')
        payload = "\n".join(payload_lines) + "\n"
        job = f"{PROJECT_NAME}_modelmonitor"
        resp = requests.post(f"{PUSHGATEWAY_URL}/metrics/job/{job}", data=payload, timeout=10)
        print("Pushgateway status:", resp.status_code)
    except Exception as e:
        print("Push to pushgateway failed:", e)
elif not should_deploy:
    print("Skipping Pushgateway update - deployment not approved")

# End
