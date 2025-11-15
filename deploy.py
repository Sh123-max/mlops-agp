import os
import json
import shutil
import mlflow
from mlflow.tracking import MlflowClient

MODEL_DIR = os.getenv("MODEL_DIR", "models")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
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

    # try to find version in stage first, else any versions
    versions = []
    try:
        versions = client.get_latest_versions(registry_name, stages=[stage]) or []
        if not versions:
            versions = client.get_latest_versions(registry_name) or []
    except Exception as e:
        print("Registry listing error:", e)
        versions = []

    dst = os.path.join(MODEL_DIR, "deployed_model")
    os.makedirs(dst, exist_ok=True)

    metadata = {
        "model_name": best_name,
        "registry_name": registry_name,
        "version": None,
        "source": None,
        "deployed_path": None,
    }

    if versions:
        selected = versions[0]
        print(f"[INFO] Selected model version: name={selected.name} version={selected.version} stage={selected.current_stage}")
        # selected.source gives an artifact_uri like "runs:/<run_id>/artifacts/model"
        artifact_uri = getattr(selected, "source", None)
        metadata["version"] = selected.version
        metadata["source"] = artifact_uri

        if artifact_uri:
            # download_artifacts accepts an artifact_uri
            try:
                local_path = mlflow.artifacts.download_artifacts(artifact_uri, dst)
                metadata["deployed_path"] = local_path
                with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
                    json.dump(metadata, f, indent=2)
                print("[OK] Downloaded artifacts to", local_path)
                print("[OK] Deployment metadata written to models/model_metadata.json")
                return
            except Exception as e:
                print("[WARN] Failed to download artifacts from registry source:", e)
        else:
            print("[WARN] Model version has no source attribute; falling back to local model if available.")

    # Fallback: if registry empty or download failed, try to copy local best_model.pkl
    local_best = os.path.join(MODEL_DIR, "best_model.pkl")
    if os.path.exists(local_best):
        deployed = os.path.join(dst, "best_model.pkl")
        shutil.copy2(local_best, deployed)
        metadata["deployed_path"] = deployed
        metadata["version"] = metadata.get("version") or "local-fallback"
        with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        print("[OK] Fallback: copied local best_model.pkl to", deployed)
        print("[OK] Deployment metadata written to models/model_metadata.json")
        return

    raise RuntimeError(f"No model could be downloaded or found locally for registry_name={registry_name}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=os.getenv("PROJECT_NAME", "diabetes"))
    parser.add_argument("--stage", default="Staging")
    args = parser.parse_args()
    deploy_best(args.project, args.stage)
