# deploy.py
"""
Robust per-project deploy script.
Updated to support Feature Engineering by ensuring baseline and feature metadata
are correctly synced for the Flask API.
"""

import os
import json
import shutil
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime

# ================= CONFIG =================
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

# ================= HELPERS =================
def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def write_metadata(project_dir, metadata):
    _ensure_dir(project_dir)
    meta_path = os.path.join(project_dir, "model_metadata.json")
    # Ensure we don't lose existing drift reports if re-deploying to same dir
    if os.path.exists(meta_path):
        try:
            existing = json.load(open(meta_path))
            if "drift_reports" in existing:
                metadata["drift_reports"] = existing["drift_reports"]
        except: pass

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    return meta_path

# ================= MLflow HELPERS =================
def find_run_for_project(project):
    """Find latest run tagged project=<project>"""
    try:
        df = mlflow.search_runs(
            filter_string=f"tags.project = '{project}'",
            order_by=["attributes.start_time DESC"],
            max_results=1
        )
        if df is not None and len(df) > 0:
            return df.iloc[0].run_id
    except Exception as e:
        print("[WARN] search_runs failed:", e)
    return None

def download_artifacts_from_run(run_id, dst, artifact_path="model"):
    _ensure_dir(dst)
    try:
        uri = f"runs:/{run_id}/{artifact_path}"
        print(f"[INFO] Downloading artifacts from {uri}")
        # Use dst_path to ensure it downloads INTO the deployed_model folder
        return mlflow.artifacts.download_artifacts(uri, dst_path=dst)
    except Exception as e:
        print("[WARN] Artifact download failed:", e)
    return None

def resolve_model_registry_and_download(registry_name, dst):
    try:
        versions = client.get_latest_versions(registry_name)
        if versions:
            mv = versions[0]
            p = download_artifacts_from_run(mv.run_id, dst, "model")
            return p, mv.version
    except Exception as e:
        print("[WARN] Registry resolution failed:", e)
    return None, None

# ================= DEPLOY =================
def deploy_best(project, stage="Staging"):
    # Target directories
    # models/<project>/
    project_dir = _ensure_dir(os.path.join("models"))
    # models/deployed_model/
    deployed_dir = _ensure_dir(os.path.join(project_dir, "deployed_model"))

    # ===== 1️⃣ LOAD TRAINING METADATA =====
    # This file was created by the training script in the root models/ folder
    train_meta_path = os.path.join("models", "model_metadata.json")
    training_meta = {}
    if os.path.exists(train_meta_path):
        training_meta = json.load(open(train_meta_path))
        print("[INFO] Found training metadata for synchronization")

    best_name = training_meta.get("model_name")
    run_id = training_meta.get("run_id") or training_meta.get("best", {}).get("run_id")
   
    # Extract feature info for app.py consistency
    baseline = training_meta.get("baseline", {})
    feat_order = training_meta.get("feature_order") or baseline.get("feature_names")

    metadata = {
        "project": project,
        "model_name": best_name,
        "run_id": run_id,
        "metrics": training_meta.get("results"),
        "best": training_meta.get("best"),
        "feature_order": feat_order,
        "baseline": baseline, # Crucial for drift detection in app.py
        "registry_name": None,
        "version": None,
        "source": None,
        "deployed_path": None,
        "deployed_at": datetime.now().isoformat()
    }

    # Clean the deployed_model folder before new deployment
    if os.path.exists(deployed_dir):
        shutil.rmtree(deployed_dir)
    _ensure_dir(deployed_dir)

    # ===== 2️⃣ DEPLOY FROM RUN_ID (PRIMARY) =====
    if run_id:
        print(f"[INFO] Attempting deployment from run_id: {run_id}")
        p = download_artifacts_from_run(run_id, deployed_dir, "model")
        if p:
            metadata.update({"deployed_path": p, "source": f"runs:/{run_id}/model"})
            write_metadata(project_dir, metadata)
            print(f"✅ Deployed successfully from run_id to {p}")
            return True

    # ===== 3️⃣ FALLBACK: TAG SEARCH =====
    run_id = find_run_for_project(project)
    if run_id:
        p = download_artifacts_from_run(run_id, deployed_dir, "model")
        if p:
            metadata.update({"run_id": run_id, "deployed_path": p, "source": f"runs:/{run_id}/model"})
            write_metadata(project_dir, metadata)
            return True

    # ===== 4️⃣ FALLBACK: REGISTRY =====
    registry_name = f"{project}_{best_name}" if best_name else f"{project}_model"
    p, ver = resolve_model_registry_and_download(registry_name, deployed_dir)
    if p:
        metadata.update({"registry_name": registry_name, "version": ver, "deployed_path": p})
        write_metadata(project_dir, metadata)
        return True

    # ===== 5️⃣ LAST FALLBACK: LOCAL FILES =====
    local_candidates = [
        f"models/{best_name}_model.pkl" if best_name else None,
        f"models/{project}_model.pkl",
        "models/best_model.pkl"
    ]
    for c in filter(None, local_candidates):
        if os.path.exists(c):
            dest = os.path.join(deployed_dir, os.path.basename(c))
            shutil.copy2(c, dest)
            metadata.update({"deployed_path": dest, "source": c, "version": "local-fallback"})
            write_metadata(project_dir, metadata)
            print(f"✅ Deployed successfully from local file: {c}")
            return True

    print(f"❌ Deployment failed for project: {project}")
    return False

# ================= ENTRY =================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=os.getenv("PROJECT_NAME", "diabetes"))
    parser.add_argument("--stage", default="Staging")
    args = parser.parse_args()

    if not deploy_best(args.project, args.stage):
        exit(1)
