# ========================= ORIGINAL CODE (RETAINED AS-IS) =========================
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

# ================= PROMETHEUS (ADDED) =================
from prometheus_client import Gauge, CollectorRegistry, push_to_gateway
# =====================================================

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier

# Attempt to import helper modules; fallback stubs if missing
try:
    from metrics_history import MetricsHistory
    from ensemble_manager import EnsembleManager
except ImportError:
    class MetricsHistory:
        def __init__(self, *args, **kwargs): pass
        def add_training_run(self, *args, **kwargs): pass
        def get_previous_best(self): return None
        def add_deployment(self, *args, **kwargs): pass

    class EnsembleManager:
        def __init__(self, *args, **kwargs): pass
        def train_and_evaluate_ensemble(self, *args, **kwargs): return None, None
        def log_ensemble_to_mlflow(self, *args, **kwargs): return None

# Optional: export_retrain_time helper
try:
    from monitoring.metrics_exporter import export_retrain_time
except Exception:
    def export_retrain_time(*args, **kwargs):
        return

PROJECT_NAME = os.getenv("PROJECT_NAME", "diabetes")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
PUSHGATEWAY_URL = os.getenv("PUSHGATEWAY_URL", "http://localhost:9091")
DATA_DIR = os.getenv("DATA_DIR", "data")
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
MODEL_DIR.mkdir(exist_ok=True, parents=True)

try:
    NUM_WORKER_THREADS = int(os.getenv("NUM_WORKER_THREADS", str(max(1, os.cpu_count()//2))))
except Exception:
    NUM_WORKER_THREADS = max(1, os.cpu_count()//2)
NUM_WORKER_THREADS = min(NUM_WORKER_THREADS, 3)

os.environ.setdefault("OMP_NUM_THREADS", os.getenv("OMP_NUM_THREADS", "2"))
os.environ.setdefault("MKL_NUM_THREADS", os.getenv("MKL_NUM_THREADS", "2"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", os.getenv("OPENBLAS_NUM_THREADS", "2"))

weights = {'Accuracy':0.05,'Precision':0.05,'Recall':0.4,'F1-Score':0.3,'ROC-AUC':0.2}

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

# ================= PROMETHEUS REGISTRY + METRICS =================
PROM_REGISTRY = CollectorRegistry()

ML_ACCURACY = Gauge("ml_model_accuracy", "Model accuracy", ["project","model"], registry=PROM_REGISTRY)
ML_PRECISION = Gauge("ml_model_precision", "Model precision", ["project","model"], registry=PROM_REGISTRY)
ML_RECALL = Gauge("ml_model_recall", "Model recall", ["project","model"], registry=PROM_REGISTRY)
ML_F1 = Gauge("ml_model_f1_score", "Model F1 score", ["project","model"], registry=PROM_REGISTRY)
ML_ROC = Gauge("ml_model_roc_auc", "Model ROC AUC", ["project","model"], registry=PROM_REGISTRY)
ML_WEIGHTED = Gauge("ml_model_weighted_score", "Weighted evaluation score", ["project","model"], registry=PROM_REGISTRY)
ML_FNR = Gauge("ml_model_false_negative_rate", "False negative rate", ["project","model"], registry=PROM_REGISTRY)
ML_RETRAIN = Gauge("ml_model_retrain_time_seconds", "Model retrain time", ["project","model"], registry=PROM_REGISTRY)
ML_LATENCY = Gauge("ml_model_inference_latency_seconds", "Inference latency", ["project","model"], registry=PROM_REGISTRY)
ML_MODEL_SIZE = Gauge("ml_model_size_bytes", "Serialized model size", ["project","model"], registry=PROM_REGISTRY)
ML_BEST = Gauge("ml_best_model_indicator", "1 if model is selected as best", ["project","model"], registry=PROM_REGISTRY)
# ================================================================

print("MLflow tracking URI:", mlflow.get_tracking_uri())
print("NUM_WORKER_THREADS:", NUM_WORKER_THREADS)

X_train = joblib.load(os.path.join(DATA_DIR, "X_train.pkl"))
X_test = joblib.load(os.path.join(DATA_DIR, "X_test.pkl"))
y_train = joblib.load(os.path.join(DATA_DIR, "y_train.pkl"))
y_test = joblib.load(os.path.join(DATA_DIR, "y_test.pkl"))

scaler = None
try:
    scaler = joblib.load(os.path.join(DATA_DIR, "scaler.pkl"))
except Exception:
    scaler = None

models = {
    'LogisticRegression': LogisticRegression(max_iter=2000),
    'RandomForest': RandomForestClassifier(random_state=42, n_jobs=1),
    'SVM': SVC(probability=True),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'KNN': KNeighborsClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=1, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42)
}

def safe_roc_auc(y_true, y_score):
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return 0.0

def measure_latency(model, X_sample, n_runs=10):
    try:
        for _ in range(2):
            model.predict(X_sample)
        t0 = time.time()
        for _ in range(n_runs):
            model.predict(X_sample)
        return (time.time() - t0) / n_runs
    except Exception:
        return 1e6

def measure_model_size(model):
    try:
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "m.joblib")
            joblib.dump(model, p)
            return os.path.getsize(p)
    except Exception:
        return 1e12

def train_and_log(name, model):
    with mlflow.start_run(run_name=f"{PROJECT_NAME}_{name}_{int(time.time())}") as run:
        t0 = time.time()
        model.fit(X_train, y_train)
        retrain_time = time.time() - t0

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:,1] if hasattr(model,"predict_proba") else np.zeros_like(y_pred)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = safe_roc_auc(y_test, y_proba)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        fn_rate = fn / (fn + tp) if (fn + tp) else 0

        latency = measure_latency(model, X_test[:50])
        model_size = measure_model_size(model)

        weighted = (
            weights['Accuracy']*acc +
            weights['Precision']*prec +
            weights['Recall']*rec +
            weights['F1-Score']*f1 +
            weights['ROC-AUC']*roc
        )

        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": roc,
            "weighted_score": weighted,
            "false_negative_rate": fn_rate,
            "retrain_time_seconds": retrain_time,
            "inference_latency_sec": latency,
            "model_size_bytes": model_size
        })

        mlflow.sklearn.log_model(model, "model")

        # ===== REGISTER MODEL (MERGED FROM SECOND CODE) =====
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri, f"{PROJECT_NAME}_model")
        # ===================================================

        # ===== PROMETHEUS METRICS PUSH (PER MODEL) =====
        ML_ACCURACY.labels(PROJECT_NAME, name).set(acc)
        ML_PRECISION.labels(PROJECT_NAME, name).set(prec)
        ML_RECALL.labels(PROJECT_NAME, name).set(rec)
        ML_F1.labels(PROJECT_NAME, name).set(f1)
        ML_ROC.labels(PROJECT_NAME, name).set(roc)
        ML_WEIGHTED.labels(PROJECT_NAME, name).set(weighted)
        ML_FNR.labels(PROJECT_NAME, name).set(fn_rate)
        ML_RETRAIN.labels(PROJECT_NAME, name).set(retrain_time)
        ML_LATENCY.labels(PROJECT_NAME, name).set(latency)
        ML_MODEL_SIZE.labels(PROJECT_NAME, name).set(model_size)
        # =================================================

        return {
            "name": name,
            "run_id": run.info.run_id,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": roc,
            "weighted_score": weighted,
            "false_negative_rate": fn_rate,
            "retrain_time_seconds": retrain_time,
            "latency": latency,
            "model_size": model_size
        }

results = []
with ThreadPoolExecutor(max_workers=NUM_WORKER_THREADS) as ex:
    futures = [ex.submit(train_and_log, n, m) for n,m in models.items()]
    for f in as_completed(futures):
        results.append(f.result())

best = max(results, key=lambda x: x["weighted_score"])

# ===== MARK BEST MODEL =====
for r in results:
    ML_BEST.labels(PROJECT_NAME, r["name"]).set(1 if r["name"] == best["name"] else 0)
# ===========================

# ================= SAVE METADATA FOR UI (MERGED) =================
PROJECT_MODEL_DIR = MODEL_DIR / PROJECT_NAME
PROJECT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

metadata = {
    "project": PROJECT_NAME,
    "model_name": best["name"],
    "run_id": best["run_id"],
    "metrics": {
        "accuracy": best["accuracy"],
        "precision": best["precision"],
        "recall": best["recall"],
        "f1_score": best["f1_score"],
        "roc_auc": best["roc_auc"],
        "weighted_score": best["weighted_score"]
    },
    "feature_order": [
        "Pregnancies","Glucose","BloodPressure","SkinThickness",
        "Insulin","BMI","DiabetesPedigreeFunction","Age"
    ],
    "feature_count": 8,
    "timestamp": time.time()
}

with open(PROJECT_MODEL_DIR / "model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)
# ================================================================

# ===== FINAL PUSH TO PUSHGATEWAY =====
if PUSHGATEWAY_URL:
    try:
        push_to_gateway(
            PUSHGATEWAY_URL.replace("http://",""),
            job=f"{PROJECT_NAME}_training",
            registry=PROM_REGISTRY
        )
        print("[PROM] Metrics pushed to Pushgateway")
    except Exception as e:
        print("[PROM] Pushgateway push failed:", e)
# ====================================

print("Training complete. Best model:", best["name"])
