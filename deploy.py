# deploy.py
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
    # This writes to models/<project>/model_metadata.json
    _ensure_dir(project_dir)
    meta_path = os.path.join(project_dir, "model_metadata.json")
    
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
def download_artifacts_from_run(run_id, dst):
    _ensure_dir(dst)
    try:
        # We download the 'model' artifact from MLflow
        uri = f"runs:/{run_id}/model"
        print(f"[INFO] Downloading artifacts from {uri}")
        return mlflow.artifacts.download_artifacts(uri, dst_path=dst)
    except Exception as e:
        print(f"[WARN] Artifact download failed: {e}")
    return None

# ================= DEPLOY =================
def deploy_best(project, stage="Staging"):
    # FIX: Project-specific directories
    project_dir = _ensure_dir(os.path.join("models", project)) 
    deployed_dir = _ensure_dir(os.path.join(project_dir, "deployed_model"))

    # Load the metadata written by the training script (usually in models/model_metadata.json)
    root_meta_path = os.path.join("models", "model_metadata.json")
    
    if not os.path.exists(root_meta_path):
        print(f"❌ No training metadata found at {root_meta_path}")
        return False

    training_meta = json.load(open(root_meta_path))
    
    # Ensure we are deploying the correct project
    if training_meta.get("project") != project:
        print(f"❌ Metadata project ({training_meta.get('project')}) does not match requested ({project})")
        return False

    best_info = training_meta.get("best", {})
    run_id = best_info.get("run_id")
    best_name = best_info.get("name")

    if not run_id:
        print("❌ No run_id found in metadata.")
        return False

    print(f"[INFO] Deploying {project} model: {best_name} (Run: {run_id})")

    # Clean old deployment
    if os.path.exists(deployed_dir):
        shutil.rmtree(deployed_dir)
    _ensure_dir(deployed_dir)

    # 1. Download from MLflow
    p = download_artifacts_from_run(run_id, deployed_dir)
    
    if p:
        # Prepare metadata for the APP to read
        metadata = {
            "project": project,
            "model_name": best_name,
            "run_id": run_id,
            "metrics": training_meta.get("results"),
            "baseline": training_meta.get("baseline"),
            "deployed_at": datetime.now().isoformat()
        }
        write_metadata(project_dir, metadata)
        print(f"✅ Deployed successfully to {deployed_dir}")
        return True

    return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=os.getenv("PROJECT_NAME", "diabetes"))
    args = parser.parse_args()
    if not deploy_best(args.project):
        exit(1)
