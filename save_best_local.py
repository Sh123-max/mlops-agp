# save_best_local.py
"""
Save best model locally, retrain on full dataset and write models/model_metadata.json.
Usage: python3 save_best_local.py [--project diabetes]
"""
import os, json, sys, argparse, pickle, joblib
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument("--project", default=os.getenv("PROJECT_NAME", "diabetes"))
args = parser.parse_args()
PROJECT = args.project

root = Path(".")
summary_path = root / "models" / "last_run_summary.json"
if not summary_path.exists():
    print("ERROR: models/last_run_summary.json not found. Run trainandevaluate first.")
    sys.exit(1)

summary = json.load(open(summary_path))
best = summary.get("best")
if not best:
    print("ERROR: 'best' not found in last_run_summary.json")
    sys.exit(1)
best_name = best.get("name")
print("[save_best_local] Best model from summary:", best_name)

data_dir = root / "data"
for f in ("X_train.pkl","X_test.pkl","y_train.pkl","y_test.pkl"):
    if not (data_dir / f).exists():
        print(f"ERROR: Missing data file {data_dir/f}. Run preprocess.py first.")
        sys.exit(1)

def try_load(path):
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)

X_train = try_load(data_dir / "X_train.pkl")
X_test  = try_load(data_dir / "X_test.pkl")
y_train = try_load(data_dir / "y_train.pkl")
y_test  = try_load(data_dir / "y_test.pkl")

# Convert to numpy
X_train = X_train if isinstance(X_train, (np.ndarray,)) else np.array(X_train)
X_test = X_test if isinstance(X_test, (np.ndarray,)) else np.array(X_test)
y_train = np.array(y_train).ravel()
y_test = np.array(y_test).ravel()

# combine full dataset
X_full = np.vstack((X_train, X_test))
y_full = np.concatenate((y_train, y_test))

# model mapping — keep in sync with trainandevaluate.py
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

models = {
    "LogisticRegression": LogisticRegression(max_iter=2000),
    "RandomForest": RandomForestClassifier(random_state=42, n_jobs=1),
    "SVM": SVC(probability=True),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "XGBoost": (XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42) if XGBClassifier else None),
    "GradientBoosting": GradientBoostingClassifier(random_state=42)
}

if best_name not in models:
    print("ERROR: best model name not in supported list:", list(models.keys()))
    sys.exit(1)
model = models[best_name]
if model is None:
    print("ERROR: required package missing for model:", best_name)
    sys.exit(1)

print("[save_best_local] Fitting", best_name, "on full dataset ...")
model.fit(X_full, y_full)

outdir = root / "models"
outdir.mkdir(parents=True, exist_ok=True)
pkl_path = outdir / "best_model.pkl"
joblib.dump(model, pkl_path)
print("[save_best_local] Saved model:", pkl_path)

dep = outdir / "deployed_model"
dep.mkdir(parents=True, exist_ok=True)
import shutil
shutil.copy2(pkl_path, dep / pkl_path.name)
print("[save_best_local] Copied to deployed_model:", dep / pkl_path.name)

# quick evaluation on holdout (original X_test)
try:
    preds = model.predict(X_test)
    acc = float(accuracy_score(y_test, preds))
except Exception:
    acc = None

meta = {
    "project": PROJECT,
    "model_name": best_name,
    "registry_name": f"{PROJECT}_{best_name}",
    "version": "local-retrained",
    "source": f"local_file:{(dep / pkl_path.name).name}",
    "deployed_path": str(dep.resolve()),
    "input_shape": int(X_full.shape[1]),
    "accuracy": acc
}
with open(outdir / "model_metadata.json","w") as f:
    json.dump(meta, f, indent=2)
print("[save_best_local] Wrote models/model_metadata.json")
