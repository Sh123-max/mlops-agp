# ==========================================================
# IMPORTS
# ==========================================================
import os
import json
import time
import traceback
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import joblib
import mlflow
import mlflow.sklearn

from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    StackingClassifier
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from xgboost import XGBClassifier

from prometheus_client import Gauge, CollectorRegistry, push_to_gateway

# ==========================================================
# OPTIONAL HELPERS (SAFE FALLBACKS)
# ==========================================================
try:
    from metrics_history import MetricsHistory
except Exception:
    class MetricsHistory:
        def add_training_run(self, *a, **k): pass
        def get_previous_best(self): return None
        def add_deployment(self, *a, **k): pass

try:
    from ensemble_manager import EnsembleManager
except Exception:
    class EnsembleManager:
        def __init__(self, *a, **k): pass
        def train_and_evaluate_ensemble(self, *a, **k): return None, None
        def log_ensemble_to_mlflow(self, *a, **k): return None

# ==========================================================
# CONFIG
# ==========================================================
PROJECT_NAME = os.getenv("PROJECT_NAME", "diabetes")
DATA_DIR = os.getenv("DATA_DIR", "data")
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
PUSHGATEWAY_URL = os.getenv("PUSHGATEWAY_URL", "http://localhost:9091")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

# ==========================================================
# PROMETHEUS REGISTRY + METRICS
# ==========================================================
PROM_REGISTRY = CollectorRegistry()

ML_ACCURACY = Gauge("ml_model_accuracy", "Model accuracy", ["project","model"], registry=PROM_REGISTRY)
ML_PRECISION = Gauge("ml_model_precision", "Model precision", ["project","model"], registry=PROM_REGISTRY)
ML_RECALL = Gauge("ml_model_recall", "Model recall", ["project","model"], registry=PROM_REGISTRY)
ML_F1 = Gauge("ml_model_f1_score", "Model F1 score", ["project","model"], registry=PROM_REGISTRY)
ML_ROC = Gauge("ml_model_roc_auc", "Model ROC AUC", ["project","model"], registry=PROM_REGISTRY)
ML_WEIGHTED = Gauge("ml_model_weighted_score", "Weighted score", ["project","model"], registry=PROM_REGISTRY)
ML_FNR = Gauge("ml_model_false_negative_rate", "False negative rate", ["project","model"], registry=PROM_REGISTRY)
ML_RETRAIN = Gauge("ml_model_retrain_time_seconds", "Retrain time", ["project","model"], registry=PROM_REGISTRY)
ML_LATENCY = Gauge("ml_model_inference_latency_seconds", "Latency", ["project","model"], registry=PROM_REGISTRY)
ML_MODEL_SIZE = Gauge("ml_model_size_bytes", "Model size", ["project","model"], registry=PROM_REGISTRY)
ML_BEST = Gauge("ml_best_model_indicator", "Best model flag", ["project","model"], registry=PROM_REGISTRY)

# ==========================================================
# LOAD DATA
# ==========================================================
X_train = joblib.load(f"{DATA_DIR}/X_train.pkl")
X_test = joblib.load(f"{DATA_DIR}/X_test.pkl")
y_train = joblib.load(f"{DATA_DIR}/y_train.pkl")
y_test = joblib.load(f"{DATA_DIR}/y_test.pkl")

# ==========================================================
# MODELS
# ==========================================================
models = {
    "LogisticRegression": LogisticRegression(max_iter=2000),
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42),
    "SVM": SVC(probability=True),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
}

models["StackingEnsemble"] = StackingClassifier(
    estimators=[
        ("lr", LogisticRegression(max_iter=2000)),
        ("rf", RandomForestClassifier(random_state=42)),
        ("svm", SVC(probability=True)),
    ],
    final_estimator=LogisticRegression()
)

weights = {
    "Accuracy": 0.05,
    "Precision": 0.05,
    "Recall": 0.4,
    "F1": 0.3,
    "ROC": 0.2
}

# ==========================================================
# HELPERS
# ==========================================================
def measure_latency(model, X):
    t0 = time.time()
    model.predict(X)
    return (time.time() - t0) / max(1, len(X))

def measure_model_size(model):
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "m.pkl")
        joblib.dump(model, p)
        return os.path.getsize(p)

def safe_roc(y, p):
    try:
        return roc_auc_score(y, p)
    except Exception:
        return 0.0

# ==========================================================
# TRAIN FUNCTION
# ==========================================================
def train_and_log(name, model):
    with mlflow.start_run(run_name=f"{PROJECT_NAME}_{name}") as run:
        t0 = time.time()
        model.fit(X_train, y_train)
        retrain_time = time.time() - t0

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1] if hasattr(model,"predict_proba") else np.zeros_like(y_pred)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = safe_roc(y_test, y_prob)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        fnr = fn / (fn + tp) if (fn+tp)>0 else 0

        latency = measure_latency(model, X_test[:20])
        model_size = measure_model_size(model)

        weighted = (
            weights["Accuracy"]*acc +
            weights["Precision"]*prec +
            weights["Recall"]*rec +
            weights["F1"]*f1 +
            weights["ROC"]*roc
        )

        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc,
            "weighted_score": weighted
        })

        mlflow.sklearn.log_model(model, "model")

        ML_ACCURACY.labels(PROJECT_NAME,name).set(acc)
        ML_PRECISION.labels(PROJECT_NAME,name).set(prec)
        ML_RECALL.labels(PROJECT_NAME,name).set(rec)
        ML_F1.labels(PROJECT_NAME,name).set(f1)
        ML_ROC.labels(PROJECT_NAME,name).set(roc)
        ML_WEIGHTED.labels(PROJECT_NAME,name).set(weighted)
        ML_FNR.labels(PROJECT_NAME,name).set(fnr)
        ML_RETRAIN.labels(PROJECT_NAME,name).set(retrain_time)
        ML_LATENCY.labels(PROJECT_NAME,name).set(latency)
        ML_MODEL_SIZE.labels(PROJECT_NAME,name).set(model_size)

        return {
            "name": name,
            "run_id": run.info.run_id,
            "weighted": weighted
        }

# ==========================================================
# MAIN
# ==========================================================
results = []
with ThreadPoolExecutor(max_workers=3) as ex:
    futures = [ex.submit(train_and_log, n, m) for n,m in models.items()]
    for f in as_completed(futures):
        results.append(f.result())

best = max(results, key=lambda x: x["weighted"])

for r in results:
    ML_BEST.labels(PROJECT_NAME, r["name"]).set(1 if r["name"] == best["name"] else 0)

# ==========================================================
# REGISTER ONLY BEST MODEL (MLFLOW REGISTRY SAFE)
# ==========================================================
def register_best_model_once(best_run_id, project_name):
    registry_name = f"{project_name}_model"

    try:
        versions = client.search_model_versions(f"name='{registry_name}'")
        for v in versions:
            if v.run_id == best_run_id:
                print(f"[MLFLOW] Run {best_run_id} already registered as version {v.version}")
                return
    except Exception as e:
        print("[MLFLOW] Registry lookup failed (safe):", e)

    model_uri = f"runs:/{best_run_id}/model"
    mv = mlflow.register_model(model_uri, registry_name)
    print(f"[MLFLOW] Registered model {registry_name}, version {mv.version}")

register_best_model_once(best["run_id"], PROJECT_NAME)

# ==========================================================
# PUSH PROMETHEUS METRICS
# ==========================================================
if PUSHGATEWAY_URL:
    push_to_gateway(
        PUSHGATEWAY_URL.replace("http://",""),
        job=f"{PROJECT_NAME}_training",
        registry=PROM_REGISTRY
    )

print("PIPELINE COMPLETED. BEST MODEL:", best)
