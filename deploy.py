# deploy.py
import os
import json
import mlflow
from mlflow.tracking import MlflowClient

MODEL_DIR = os.getenv("MODEL_DIR", "models")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

def deploy_best(project, stage="Staging"):
    summary_path = os.path.join(MODEL_DIR, "last_run_summary.json")
    if not os.path.exists(summary_path):
        raise RuntimeError("No last_run_summary.json found. Run training first.")
    summary = json.load(open(summary_path))
    best = summary.get("best", {})
    best_name = best.get("name")
    if not best_name:
        raise RuntimeError("No best model found in summary.")
    registry_name = f"{project}_{best_name}"
    # try to find version in stage
    try:
        versions = client.get_latest_versions(registry_name, stages=[stage])
        if not versions:
            versions = client.get_latest_versions(registry_name)
    except Exception as e:
        print("Registry listing error:", e)
        versions = []
    if not versions:
        raise RuntimeError(f"No versions found for {registry_name}")
    selected = versions[0]
    dst = os.path.join(MODEL_DIR, "deployed_model")
    os.makedirs(dst, exist_ok=True)
    local_path = mlflow.artifacts.download_artifacts(f"models:/{registry_name}/{selected.version}", dst)
    meta = {"model_name": best_name, "registry_name": registry_name, "version": selected.version}
    with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Downloaded artifacts to", local_path)
    print("Deployment metadata written to models/model_metadata.json")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=os.getenv("PROJECT_NAME", "diabetes"))
    parser.add_argument("--stage", default="Staging")
    args = parser.parse_args()
    deploy_best(args.project, args.stage)
