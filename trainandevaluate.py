# trainandevaluate.py
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
import shutil
import time as pytime

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

# Config from env
PROJECT_NAME = os.getenv("PROJECT_NAME", "diabetes")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
PUSHGATEWAY_URL = os.getenv("PUSHGATEWAY_URL", "http://localhost:9091")
DATA_DIR = os.getenv("DATA_DIR", "data")
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
MODEL_DIR.mkdir(exist_ok=True)

# Resource controls
try:
    NUM_WORKER_THREADS = int(os.getenv("NUM_WORKER_THREADS", str(max(1, os.cpu_count()//2))))
except Exception:
    NUM_WORKER_THREADS = max(1, os.cpu_count()//2)
NUM_WORKER_THREADS = min(NUM_WORKER_THREADS, 3)  # conservative cap for 8GB machine

os.environ.setdefault("OMP_NUM_THREADS", os.getenv("OMP_NUM_THREADS", "2"))
os.environ.setdefault("MKL_NUM_THREADS", os.getenv("MKL_NUM_THREADS", "2"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", os.getenv("OPENBLAS_NUM_THREADS", "2"))

# Metric weights (healthcare priority) -- used as tie-breaker
weights = {'Accuracy':0.05,'Precision':0.05,'Recall':0.4,'F1-Score':0.3,'ROC-AUC':0.2}

# MLflow setup
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
print("MLflow tracking URI:", mlflow.get_tracking_uri())
print("NUM_WORKER_THREADS:", NUM_WORKER_THREADS)

# Load data
X_train = joblib.load(os.path.join(DATA_DIR, "X_train.pkl"))
X_test = joblib.load(os.path.join(DATA_DIR, "X_test.pkl"))
y_train = joblib.load(os.path.join(DATA_DIR, "y_train.pkl"))
y_test = joblib.load(os.path.join(DATA_DIR, "y_test.pkl"))

# For baseline unscaled distributions we will reconstruct using scaler if available
scaler_path = os.path.join(DATA_DIR, "scaler.pkl")
scaler = None
if os.path.exists(scaler_path):
    try:
        scaler = joblib.load(scaler_path)
    except Exception as e:
        print("Failed to load scaler:", e)

# Define models (set n_jobs=1 to avoid nested threading)
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

def measure_latency(model, X_sample, n_runs=20):
    # measure average time per predict call
    try:
        # warmup
        for _ in range(3):
            _ = model.predict(X_sample)
        t0 = pytime.time()
        for _ in range(n_runs):
            _ = model.predict(X_sample)
        t1 = pytime.time()
        avg = (t1 - t0) / n_runs
        return float(avg)
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

def is_better_by_healthcare_priority(a, b):
    # a and b are dicts with metrics; prefer higher recall, then f1, then roc_auc
    for k in ("recall", "f1", "roc_auc", "precision", "accuracy"):
        av = a.get(k, 0)
        bv = b.get(k, 0)
        if av > bv + 1e-9:
            return True
        if bv > av + 1e-9:
            return False
    return False

def nondominated_front(points, objectives_directions):
    """
    Simple non-dominated sorting for Pareto front.
    points: list of numeric vectors
    objectives_directions: list with 1 for minimize, -1 for maximize per objective
    Returns indices of non-dominated points.
    """
    n = len(points)
    dominated = [False]*n
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j or dominated[j]:
                continue
            better_or_equal = True
            strictly_better = False
            for k, dirc in enumerate(objectives_directions):
                vi = points[i][k]*dirc
                vj = points[j][k]*dirc
                if vj < vi - 1e-12:
                    better_or_equal = False
                    break
                if vj > vi + 1e-12:
                    strictly_better = True
            # if j is at least as good as i on all objectives and strictly better on >=1, then i is dominated
            if better_or_equal and strictly_better:
                dominated[i] = True
                break
    return [i for i, d in enumerate(dominated) if not d]

def train_and_log(name, model):
    run_name = f"{PROJECT_NAME}__{name}__{int(time.time())}"
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        try:
            print(f"[{name}] Training (run_id={run_id})")
            model.fit(X_train, y_train)
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

            # measure latency on a small sample
            sample_count = min(50, X_test.shape[0])
            X_sample = X_test[:sample_count]
            latency = measure_latency(model, X_sample, n_runs=10)
            model_size = measure_model_size(model)

            weighted = (weights['Accuracy']*acc + weights['Precision']*prec + weights['Recall']*rec + weights['F1-Score']*f1 + weights['ROC-AUC']*roc)

            mlflow.log_param("model_name", name)
            mlflow.log_metric("accuracy", float(acc))
            mlflow.log_metric("precision", float(prec))
            mlflow.log_metric("recall", float(rec))
            mlflow.log_metric("f1_score", float(f1))
            mlflow.log_metric("roc_auc", float(roc))
            mlflow.log_metric("weighted_score", float(weighted))
            mlflow.log_metric("inference_latency_sec", float(latency))
            mlflow.log_metric("model_size_bytes", float(model_size))
            mlflow.set_tag("project", PROJECT_NAME)
            mlflow.set_tag("model_name", name)

            try:
                mlflow.sklearn.log_model(model, artifact_path="model")
            except Exception as e:
                print("mlflow log model failed:", e)

            # Save a local copy of this model as <name>_model.pkl for fallback
            try:
                local_fn = MODEL_DIR / f"{name}_model.pkl"
                joblib.dump(model, local_fn)
            except Exception as e:
                print("Failed to locally save model:", e)

            print(f"[{name}] done: weighted_score={weighted:.4f}")
            return {
                "name": name,
                "run_id": run_id,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "roc_auc": roc,
                "weighted_score": weighted,
                "latency": latency,
                "model_size": model_size
            }
        except Exception as e:
            print(f"[{name}] failed:", e)
            traceback.print_exc()
            mlflow.set_tag("training_status", "failed")
            return {"name": name, "error": str(e)}

# Parallel training using threads (threads share memory)
best = {"score": -1, "name": None, "run_id": None, "registry": None}
results = []

with ThreadPoolExecutor(max_workers=NUM_WORKER_THREADS) as ex:
    futures = {ex.submit(train_and_log, n, m): n for n, m in models.items()}
    for fut in as_completed(futures):
        name = futures[fut]
        try:
            res = fut.result()
            results.append(res)
            if "error" not in res:
                score = float(res.get("weighted_score", -1))
                if score > best["score"]:
                    # attempt register
                    run_id = res["run_id"]
                    registry_name = f"{PROJECT_NAME}_{res['name']}"
                    try:
                        registered = mlflow.register_model(f"runs:/{run_id}/model", registry_name)
                        best = {"score": score, "name": res['name'], "run_id": run_id, "registry": {"name": registry_name, "version": getattr(registered, "version", None)}}
                    except Exception as e:
                        print("Register failed:", e)
                        best = {"score": score, "name": res['name'], "run_id": run_id, "registry": None}
        except Exception as e:
            print("Future exception for", name, e)

# --- Multi-objective Pareto selection (non-dominated sorting) ---
# Objectives: maximize recall, precision, f1_score, roc_auc; minimize latency and model_size
objs = []
for r in results:
    if "error" in r:
        continue
    # We'll create a vector ordering: [ -recall, -precision, -f1, -roc, latency, model_size ]
    # For nondominated routine, we provide numeric vectors directly and directions list
    obj = [
        -float(r.get("recall", 0.0)),
        -float(r.get("precision", 0.0)),
        -float(r.get("f1_score", 0.0)),
        -float(r.get("roc_auc", 0.0)),
        float(r.get("latency", 1e6)),
        float(r.get("model_size", 1e12))
    ]
    objs.append(obj)

selected_idx = []
if objs:
    # directions: -1 means maximize (we converted), 1 means minimize (latency, size)
    # Our nondominated function expects points and directions where we treat larger*dirc as worse.
    # We'll convert to "points" where higher is worse for each objective; supply directions accordingly.
    # Simpler: call nondominated_front with objectives_directions all = 1 (minimize),
    # because we've already flipped maximize objectives by negation.
    indices = nondominated_front(objs, objectives_directions=[1,1,1,1,1,1])
    selected_idx = indices
    pareto_models = [results[i] for i in selected_idx]
else:
    pareto_models = []

# Tie-breaker using healthcare priorities (recall then f1 then roc_auc)
if pareto_models:
    pareto_models = sorted(pareto_models, key=lambda x: (x.get("recall",0), x.get("f1_score",0), x.get("roc_auc",0)), reverse=True)
    chosen = pareto_models[0]
    # set best accordingly
    best = {"score": float(chosen.get("weighted_score", -1)), "name": chosen["name"], "run_id": chosen.get("run_id"), "registry": None}
    # attempt register chosen model again (idempotent)
    try:
        reg_name = f"{PROJECT_NAME}_{chosen['name']}"
        registered = mlflow.register_model(f"runs:/{chosen.get('run_id')}/model", reg_name)
        best["registry"] = {"name": reg_name, "version": getattr(registered, "version", None)}
    except Exception as e:
        print("Registering chosen model failed:", e)

summary = {
    "project": PROJECT_NAME,
    "results": results,
    "pareto_indices": selected_idx,
    "pareto_models": pareto_models,
    "best": best
}
with open(MODEL_DIR / "last_run_summary.json", "w") as fh:
    json.dump(summary, fh, indent=2)

print("Training complete. Best:", best)

# Save model metadata (include baseline distributions derived from X_train using scaler if available)
metadata = {
    "project": PROJECT_NAME,
    "best": best,
    "results": results,
    "generated_at": int(time.time())
}

# attempt to create baseline unscaled distributions
try:
    if scaler is not None:
        X_train_unscaled = scaler.inverse_transform(X_train)
    else:
        # assume X_train is already unscaled
        X_train_unscaled = X_train
    # column names are unknown because preprocess stored numpy; we will name them feat0..featN
    n_feats = X_train_unscaled.shape[1]
    feat_names = [f"feat_{i}" for i in range(n_feats)]
    feature_distributions = {}
    feature_means = {}
    feature_stds = {}
    for i, name in enumerate(feat_names):
        arr = X_train_unscaled[:, i].tolist()
        feature_distributions[name] = arr
        feature_means[name] = float(np.nanmean(arr))
        feature_stds[name] = float(np.nanstd(arr))

    metadata["baseline"] = {
        "feature_names": feat_names,
        "feature_distributions": feature_distributions,
        "feature_means": feature_means,
        "feature_stds": feature_stds
    }
except Exception as e:
    print("Failed to write baseline distributions:", e)

# write into models/model_metadata.json (augment existing metadata if present)
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

# Push metric to pushgateway
if PUSHGATEWAY_URL and best.get("name"):
    try:
        payload = f'model_weighted_score{{project="{PROJECT_NAME}",model="{best["name"]}"}} {best["score"]}\n'
        job = f"{PROJECT_NAME}_modelmonitor"
        resp = requests.post(f"{PUSHGATEWAY_URL}/metrics/job/{job}", data=payload, timeout=10)
        print("Pushgateway status:", resp.status_code)
    except Exception as e:
        print("Push to pushgateway failed:", e)

