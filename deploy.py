# trainandevaluate.py  (REPLACE your old file with this)
import os, json, pickle, traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np

# sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# -------------------------
# Config & resource control
# -------------------------
PROJECT_NAME = os.getenv("PROJECT_NAME", "diabetes")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

# Tune these to match your machine (4 cores, 8GB)
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))   # thread pool
TREE_N_JOBS = int(os.getenv("TREE_N_JOBS", "2"))   # RF/XGB parallelism

# Weights prioritizing recall (medical)
weights = {
    'Accuracy': 0.05,
    'Precision': 0.05,
    'Recall': 0.4,
    'F1-Score': 0.3,
    'ROC-AUC': 0.2
}

# -------------------------
# Load data (expects preprocess.py output in data/)
# -------------------------
def load_data():
    try:
        with open('data/X_train.pkl', 'rb') as f: X_train = pickle.load(f)
        with open('data/X_test.pkl', 'rb') as f: X_test = pickle.load(f)
        with open('data/y_train.pkl', 'rb') as f: y_train = pickle.load(f)
        with open('data/y_test.pkl', 'rb') as f: y_test = pickle.load(f)
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print("Error loading data:", e)
        raise

# -------------------------
# Models
# -------------------------
def get_models():
    return {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'RandomForest': RandomForestClassifier(random_state=42, n_jobs=TREE_N_JOBS),
        'SVM': SVC(probability=True),
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'KNN': KNeighborsClassifier(),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=TREE_N_JOBS, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42)
    }

# -------------------------
# Train + log one model
# -------------------------
def train_and_log(name, model, X_train, y_train, X_test, y_test):
    info = {"name": name, "run_id": None}
    try:
        # Use a named run for clarity
        with mlflow.start_run(run_name=f"{PROJECT_NAME}_{name}") as run:
            run_id = run.info.run_id
            info["run_id"] = run_id
            mlflow.set_tag("project", PROJECT_NAME)
            mlflow.log_param("model_name", name)

            # Fit (quiet)
            model.fit(X_train, y_train)

            # Predictions
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

            # Metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0

            score = (weights['Accuracy'] * acc +
                     weights['Precision'] * prec +
                     weights['Recall'] * rec +
                     weights['F1-Score'] * f1 +
                     weights['ROC-AUC'] * roc)

            # Log metrics
            mlflow.log_metrics({
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "roc_auc": roc,
                "weighted_score": score
            })

            # Artifact path "model" is important so deploy.py can use runs:/.../model
            registered_name = f"{PROJECT_NAME}_{name}"
            try:
                # log and register (preferred)
                mlflow.sklearn.log_model(sk_model=model,
                                         artifact_path="model",
                                         registered_model_name=registered_name)
                print(f"[INFO] Logged & registered model '{registered_name}' (run {run_id})")
            except Exception as e:
                # fallback: log only artifact
                mlflow.sklearn.log_model(sk_model=model, artifact_path="model")
                print(f"[WARN] Registering model failed, logged artifact only. Error: {e}")

            # Return summary
            info.update({"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": roc, "score": score})
            return info
    except Exception as e:
        print(f"[ERROR] Training/logging failed for {name}: {e}")
        traceback.print_exc()
        info["error"] = str(e)
        return info

# -------------------------
# Main
# -------------------------
def main():
    X_train, X_test, y_train, y_test = load_data()
    models = get_models()
    results = []

    # Train models in parallel (threads). Keep max_workers tuned to your machine.
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(train_and_log, name, mdl, X_train, y_train, X_test, y_test): name
                   for name, mdl in models.items()}
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            print("[DONE]", res.get("name"), "run:", res.get("run_id"), "score:", res.get("score"))

    # Determine best model
    valid_results = [r for r in results if r.get("run_id") and r.get("score") is not None]
    if not valid_results:
        raise RuntimeError("No successful model runs to choose best from.")

    best = max(valid_results, key=lambda r: r["score"])
    print("[BEST] Selected:", best)

    # Save best model (local fallback): download artifacts from run to models/best_local_deploy
    best_run = best["run_id"]
    dst = MODEL_DIR / "deployed_model"
    dst.mkdir(parents=True, exist_ok=True)

    try:
        # Use mlflow client to download artifacts (artifact path "model")
        client.download_artifacts(best_run, "model", dst_path=str(dst))
        print(f"[INFO] Downloaded best model artifacts to {dst}")
    except Exception as e:
        print("[WARN] Could not download artifacts after training:", e)

    # Save a pickled best model locally as fallback (attempt)
    try:
        # load pickled model from artifact folder (MLflow saved sklearn format)
        # Fallback: re-fit best model on full data and pickle (cheap on small dataset)
        from sklearn.base import clone
        chosen_model_name = best["name"]
        chosen_model = models[chosen_model_name]
        chosen_model.fit(np.vstack((X_train, X_test)), np.hstack((y_train, y_test)))
        with open(MODEL_DIR / "best_model.pkl", "wb") as f:
            pickle.dump(chosen_model, f)
        print("[INFO] Saved local fallback models/best_model.pkl")
    except Exception as e:
        print("[WARN] Could not save fallback pickle:", e)

    # Write last_run_summary.json
    summary = {"project": PROJECT_NAME, "results": results, "best": best}
    with open(MODEL_DIR / "last_run_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print("[INFO] Wrote", MODEL_DIR / "last_run_summary.json")

    # Write model_metadata.json
    metadata = {
        "model_name": best["name"],
        "run_id": best["run_id"],
        "registry_name": f"{PROJECT_NAME}_{best['name']}",
        "deployed_local_path": str(dst.resolve())
    }
    with open(MODEL_DIR / "model_metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)
    print("[INFO] Wrote", MODEL_DIR / "model_metadata.json")

if __name__ == "__main__":
    main()
