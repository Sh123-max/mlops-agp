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

# ================= PROMETHEUS =================
from prometheus_client import Gauge, CollectorRegistry, push_to_gateway
# ==============================================

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier

PROJECT_NAME = os.getenv("PROJECT_NAME", "diabetes")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
PUSHGATEWAY_URL = os.getenv("PUSHGATEWAY_URL", "http://localhost:9091")
DATA_DIR = os.getenv("DATA_DIR", "data")
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
MODEL_DIR.mkdir(exist_ok=True, parents=True)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

# ================= PROMETHEUS REGISTRY =================
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
# =======================================================

X_train = joblib.load(os.path.join(DATA_DIR, "X_train.pkl"))
X_test = joblib.load(os.path.join(DATA_DIR, "X_test.pkl"))
y_train = joblib.load(os.path.join(DATA_DIR, "y_train.pkl"))
y_test = joblib.load(os.path.join(DATA_DIR, "y_test.pkl"))

weights = {'Accuracy':0.05,'Precision':0.05,'Recall':0.4,'F1-Score':0.3,'ROC-AUC':0.2}

models = {
    'LogisticRegression': LogisticRegression(max_iter=2000),
    'RandomForest': RandomForestClassifier(random_state=42, n_jobs=1),
    'SVM': SVC(probability=True),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'KNN': KNeighborsClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=1, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42)
}

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
        roc = roc_auc_score(y_test, y_proba)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        fn_rate = fn / (fn + tp) if (fn + tp) else 0

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
            "weighted_score": weighted
        })

        mlflow.sklearn.log_model(model, "model")  # 🔴 NO REGISTRATION HERE

        ML_ACCURACY.labels(PROJECT_NAME, name).set(acc)
        ML_PRECISION.labels(PROJECT_NAME, name).set(prec)
        ML_RECALL.labels(PROJECT_NAME, name).set(rec)
        ML_F1.labels(PROJECT_NAME, name).set(f1)
        ML_ROC.labels(PROJECT_NAME, name).set(roc)
        ML_WEIGHTED.labels(PROJECT_NAME, name).set(weighted)

        return {
            "name": name,
            "run_id": run.info.run_id,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": roc,
            "weighted_score": weighted
        }

results = []
with ThreadPoolExecutor(max_workers=3) as ex:
    futures = [ex.submit(train_and_log, n, m) for n,m in models.items()]
    for f in as_completed(futures):
        results.append(f.result())

best = max(results, key=lambda x: x["weighted_score"])

# ================= REGISTER ONLY BEST MODEL =================
from mlflow import register_model

model_uri = f"runs:/{best['run_id']}/model"
registry_name = f"{PROJECT_NAME}_model"

mv = register_model(model_uri=model_uri, name=registry_name)
print(f"[MLFLOW] Registered BEST model as {registry_name}, version {mv.version}")
# ===========================================================

# ================= PROMETHEUS BEST INDICATOR =================
for r in results:
    ML_BEST.labels(PROJECT_NAME, r["name"]).set(1 if r["name"] == best["name"] else 0)
# ============================================================

# ================= SAVE METADATA FOR UI =================
PROJECT_MODEL_DIR = MODEL_DIR / PROJECT_NAME
PROJECT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

with open(PROJECT_MODEL_DIR / "model_metadata.json", "w") as f:
    json.dump({
        "project": PROJECT_NAME,
        "model_name": best["name"],
        "run_id": best["run_id"],
        "metrics": best
    }, f, indent=4)
# =======================================================

# ================= PUSH PROMETHEUS =================
if PUSHGATEWAY_URL:
    push_to_gateway(
        PUSHGATEWAY_URL.replace("http://",""),
        job=f"{PROJECT_NAME}_training",
        registry=PROM_REGISTRY
    )
# ================================================

print("✅ Training complete. BEST model:", best["name"])
