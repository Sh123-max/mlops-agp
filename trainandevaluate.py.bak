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

# Metric weights (healthcare priority)
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

# Define models (set n_jobs=1 to avoid nested threading)
models = {
    'LogisticRegression': LogisticRegression(max_iter=2000) if not hasattr(LogisticRegression, 'n_jobs') else LogisticRegression(max_iter=2000),
    'RandomForest': RandomForestClassifier(random_state=42, n_jobs=1),
    'SVM': SVC(probability=True),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'KNN': KNeighborsClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', nthread=NUM_WORKER_THREADS, verbosity=1, random_state=42),
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

def train_and_log(name, model):
    run_name = f"{PROJECT_NAME}__{name}__{int(time.time())}"
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        try:
            print(f"[{name}] Training (run_id={run_id})")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:,1]
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

            weighted = (weights['Accuracy']*acc + weights['Precision']*prec + weights['Recall']*rec + weights['F1-Score']*f1 + weights['ROC-AUC']*roc)

            mlflow.log_param("model_name", name)
            mlflow.log_metric("accuracy", float(acc))
            mlflow.log_metric("precision", float(prec))
            mlflow.log_metric("recall", float(rec))
            mlflow.log_metric("f1_score", float(f1))
            mlflow.log_metric("roc_auc", float(roc))
            mlflow.log_metric("weighted_score", float(weighted))
            mlflow.set_tag("project", PROJECT_NAME)
            mlflow.set_tag("model_name", name)

            try:
                mlflow.sklearn.log_model(model, artifact_path="model")
            except Exception as e:
                print("mlflow log model failed:", e)

            print(f"[{name}] done: weighted_score={weighted:.4f}")
            return {
                "name": name,
                "run_id": run_id,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "roc_auc": roc,
                "weighted_score": weighted
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
                        best = {"score": score, "name": res['name'], "run_id": run_id, "registry": {"name": registry_name, "version": registered.version}}
                    except Exception as e:
                        print("Register failed:", e)
                        best = {"score": score, "name": res['name'], "run_id": run_id, "registry": None}
        except Exception as e:
            print("Future exception for", name, e)

summary = {
    "project": PROJECT_NAME,
    "results": results,
    "best": best
}
with open(MODEL_DIR / "last_run_summary.json", "w") as fh:
    json.dump(summary, fh, indent=2)

print("Training complete. Best:", best)

# Push metric to pushgateway
if PUSHGATEWAY_URL and best.get("name"):
    try:
        payload = f'model_weighted_score{{project="{PROJECT_NAME}",model="{best["name"]}"}} {best["score"]}\n'
        job = f"{PROJECT_NAME}_modelmonitor"
        resp = requests.post(f"{PUSHGATEWAY_URL}/metrics/job/{job}", data=payload, timeout=10)
        print("Pushgateway status:", resp.status_code)
    except Exception as e:
        print("Push to pushgateway failed:", e)
