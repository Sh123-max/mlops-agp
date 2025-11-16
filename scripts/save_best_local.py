"""
Robust retrain & save best model locally.
Prefers joblib to load 'data/*.pkl' (works for sklearn joblib dumps).
Saves models/best_model.pkl and copies to models/deployed_model/.
Also writes models/model_metadata.json for the Flask app.
"""
import json, sys, os, pickle
from pathlib import Path

def try_load(path):
    # 1) try joblib (preferred for sklearn objects)
    try:
        import joblib
        return joblib.load(path)
    except Exception as e_job:
        err_job = e_job
    # 2) try pickle
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e_pk:
        err_pk = e_pk
    # 3) try numpy
    try:
        import numpy as np
        return np.load(path, allow_pickle=True)
    except Exception as e_np:
        err_np = e_np
    raise RuntimeError(f"Failed to load {path!r} by joblib/pickle/numpy.\njoblib err: {err_job}\npickle err: {err_pk}\nnumpy err: {err_np}")

def to_numpy(x):
    import numpy as np
    # if pandas Series -> convert to numpy
    try:
        import pandas as pd
        if isinstance(x, pd.Series) or isinstance(x, pd.DataFrame):
            return x.values
    except Exception:
        pass
    if isinstance(x, np.ndarray):
        return x
    try:
        return np.array(x)
    except Exception:
        return x

# --- main ---
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
print("[INFO] Best model from summary:", best_name)

data_dir = root / "data"
for f in ("X_train.pkl","X_test.pkl","y_train.pkl","y_test.pkl"):
    if not (data_dir / f).exists():
        print(f"ERROR: Missing data file {data_dir/f}. Run preprocess.py first.")
        sys.exit(1)

# load datasets (joblib preferred)
X_train = try_load(data_dir / "X_train.pkl")
X_test  = try_load(data_dir / "X_test.pkl")
y_train = try_load(data_dir / "y_train.pkl")
y_test  = try_load(data_dir / "y_test.pkl")

# normalize / convert to numpy if needed
X_train = to_numpy(X_train)
X_test  = to_numpy(X_test)
y_train = to_numpy(y_train)
y_test  = to_numpy(y_test)

# combine full dataset
import numpy as np
X_full = np.vstack((X_train, X_test))
y_full = np.concatenate((y_train, y_test))

# model mapping — same as trainandevaluate
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
    "LogisticRegression": LogisticRegression(max_iter=1000),
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
    print("ERROR: required package missing for model:", best_name, " (likely xgboost).")
    sys.exit(1)

print("[INFO] Fitting", best_name, "on full dataset ...")
model.fit(X_full, y_full)

outdir = root / "models"
outdir.mkdir(parents=True, exist_ok=True)
pkl_path = outdir / "best_model.pkl"
with open(pkl_path, "wb") as f:
    pickle.dump(model, f)
print("[OK] Saved model:", pkl_path)

# copy into deployed_model
dep = outdir / "deployed_model"
dep.mkdir(parents=True, exist_ok=True)
import shutil
shutil.copy2(pkl_path, dep / pkl_path.name)
print("[OK] Copied to deployed_model:", dep / pkl_path.name)

# metadata for app
meta = {
    "model_name": best_name,
    "registry_name": f"diabetes_{best_name}",
    "version": "local-retrained",
    "source": f"local_file:{str((dep / pkl_path.name).name)}",
    "deployed_path": str(dep.resolve())
}
with open(outdir / "model_metadata.json","w") as f:
    json.dump(meta, f, indent=2)
print("[OK] Wrote models/model_metadata.json")
