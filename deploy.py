# deploy.py
"""
Robust per-project deploy script.

Behavior:
- Prefer models/<project>/model_metadata.json (from training)
- Use run_id directly if available
- Fallback to MLflow tag search
- Fallback to model registry
- Fallback to local artifacts
- Writes UI-compatible metadata
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
        return mlflow.artifacts.download_artifacts(uri, dst_path=dst)
    except Exception as e:
        print("[WARN] Artifact download failed:", e)
    return None

def resolve_model_registry_and_download(registry_name, dst):
    try:
        versions = client.get_latest_versions(registry_name)
        if versions:
            mv = versions[0]
            if mv.run_id:
                p = download_artifacts_from_run(mv.run_id, dst, "model")
                if p:
                    return p, mv.version
            uri = f"models:/{registry_name}/{mv.version}"
            p = mlflow.artifacts.download_artifacts(uri, dst_path=dst)
            return p, mv.version
    except Exception as e:
        print("[WARN] Registry resolution failed:", e)
    return None, None

# ================= DEPLOY =================
def deploy_best(project, stage="Staging"):
    project_dir = _ensure_dir(os.path.join("models", project))
    deployed_dir = _ensure_dir(os.path.join(project_dir, "deployed_model"))

    # ===== 1️⃣ LOAD TRAINING METADATA (NEW PRIMARY SOURCE) =====
    meta_path = os.path.join(project_dir, "model_metadata.json")
    training_meta = {}
    if os.path.exists(meta_path):
        training_meta = json.load(open(meta_path))
        print("[INFO] Using training metadata")

    best_name = training_meta.get("model_name")
    run_id = training_meta.get("run_id")

    metadata = {
        "project": project,
        "model_name": best_name,
        "run_id": run_id,
        "metrics": training_meta.get("metrics"),
        "feature_order": training_meta.get("feature_order"),
        "feature_count": training_meta.get("feature_count"),
        "registry_name": None,
        "version": None,
        "source": None,
        "deployed_path": None,
        "deployed_at": datetime.now().isoformat()
    }

    # ===== 2️⃣ DEPLOY FROM RUN_ID (BEST PATH) =====
    if run_id:
        print(f"[INFO] Deploying from run_id={run_id}")
        p = download_artifacts_from_run(run_id, deployed_dir, "model")
        if p:
            metadata.update({
                "deployed_path": p,
                "source": f"runs:/{run_id}/model"
            })
            write_metadata(project_dir, metadata)
            print("[OK] Deployed from MLflow run")
            return True

    # ===== 3️⃣ FALLBACK: TAG SEARCH =====
    run_id = find_run_for_project(project)
    if run_id:
        print(f"[INFO] Fallback run search found run_id={run_id}")
        p = download_artifacts_from_run(run_id, deployed_dir, "model")
        if p:
            metadata.update({
                "run_id": run_id,
                "deployed_path": p,
                "source": f"runs:/{run_id}/model"
            })
            write_metadata(project_dir, metadata)
            return True

    # ===== 4️⃣ FALLBACK: REGISTRY =====
    if best_name:
        registry_name = f"{project}_model"
        print(f"[INFO] Trying registry {registry_name}")
        p, ver = resolve_model_registry_and_download(registry_name, deployed_dir)
        if p:
            metadata.update({
                "registry_name": registry_name,
                "version": ver,
                "deployed_path": p,
                "source": f"models:/{registry_name}/{ver}"
            })
            write_metadata(project_dir, metadata)
            return True

    # ===== 5️⃣ LAST FALLBACK: LOCAL FILE =====
    local_candidates = [
        f"models/{project}_model.pkl",
        f"models/{project}_model.joblib",
        "models/best_model.pkl",
        "models/best_model.joblib"
    ]

    for c in local_candidates:
        if os.path.exists(c):
            dest = os.path.join(deployed_dir, os.path.basename(c))
            shutil.copy2(c, dest)
            metadata.update({
                "deployed_path": dest,
                "source": c,
                "version": "local-fallback"
            })
            write_metadata(project_dir, metadata)
            return True

    print("[ERROR] Deployment failed for project:", project)
    return False

# ================= ENTRY =================
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--project", default=os.getenv("PROJECT_NAME", "diabetes"))
    p.add_argument("--stage", default="Staging")
    args = p.parse_args()

    if not deploy_best(args.project, args.stage):
        exit(1)

    print("✅ Deployment completed successfully")
