# deploy.py (robust: resolves models:/ -> runs:/ and downloads artifacts)
import os
import json
import shutil
import mlflow
from mlflow.tracking import MlflowClient
import sys

MODEL_DIR = os.getenv("MODEL_DIR", "models")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

def resolve_artifact_uri_from_model_version(registry_name, version):
    """
    Try methods to obtain a downloadable artifact_uri (runs:/...) for a registered model version.
    Uses MlflowClient.get_model_version_download_uri when available; otherwise uses get_model_version
    and composes runs:/<run_id>/<artifact_path> or uses client.download_artifacts.
    Returns artifact_uri (string) or (run_id, path) tuple to use client.download_artifacts fallback.
    """
    # Preferred helper (returns runs:/ URI)
    try:
        uri = client.get_model_version_download_uri(registry_name, version)
        if uri:
            return uri
    except Exception as e:
        print("[DEBUG] get_model_version_download_uri not usable:", e)

    # Fallback: get full ModelVersion object
    try:
        mv = client.get_model_version(registry_name, version)
        # mv.source may be something like "runs:/<run_id>/artifacts/model" or a path to artifact store
        print("[DEBUG] model version object:", {"name": mv.name, "version": mv.version, "source": getattr(mv, 'source', None), "run_id": getattr(mv, 'run_id', None)})
        # If mv.source is already runs:/... use it
        src = getattr(mv, "source", None)
        if src and src.startswith("runs:/"):
            return src
        # If run_id and source path available, attempt to compute artifact path
        run_id = getattr(mv, "run_id", None)
        if run_id:
            # Attempt to determine artifact path from source if possible
            if src and "runs:/" in src:
                return src
            # If source contains '/artifacts/' we can attempt to derive path after 'artifacts/'
            if src and "artifacts/" in src:
                path = src.split("artifacts/", 1)[1]
                return (run_id, path)
            # Last resort: download entire run's top-level artifacts
            return (run_id, None)
    except Exception as e:
        print("[DEBUG] get_model_version fallback failed:", e)

    return None

def deploy_best(project, stage="Staging"):
    summary_path = os.path.join(MODEL_DIR, "last_run_summary.json")
    if not os.path.exists(summary_path):
        print(f"[deploy] Summary file not found at {summary_path}. Cannot find best model.")
        print("Please ensure training produced a valid models/last_run_summary.json (check MLflow connection).")
        sys.exit(1)

    try:
        summary = json.load(open(summary_path))
    except Exception as e:
        print(f"[deploy] Failed to read summary file {summary_path}: {e}")
        sys.exit(1)

    best = summary.get("best", {})
    best_name = best.get("name")
    if not best_name:
        print("[deploy] Summary exists but no best model present.")
        print("Summary contents:")
        print(json.dumps(summary, indent=2))
        # If you want to fallback to a local model, keep the fallback logic below.
        # For now, exit with clear diagnostic so CI logs show the summary contents.
        sys.exit(1)

    registry_name = f"{project}_{best_name}"

    # try to find version in stage first, else any versions
    versions = []
    try:
        versions = client.get_latest_versions(registry_name, stages=[stage]) or []
        if not versions:
            versions = client.get_latest_versions(registry_name) or []
    except Exception as e:
        print("[WARN] Registry listing error:", e)
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
        metadata["version"] = selected.version
        print(f"[INFO] Selected model version: name={selected.name} version={selected.version} stage={getattr(selected, 'current_stage', None)}")
        # Try to use selected.source if available
        artifact_uri = getattr(selected, "source", None)
        metadata["source"] = artifact_uri
        print("[DEBUG] selected.source:", artifact_uri)

        # If source is models:/... we must resolve it to a runs:/ or to (run_id, path)
        resolved = None
        if artifact_uri and str(artifact_uri).startswith("models:/"):
            print("[DEBUG] artifact_uri is models:/... resolving to runs:/ or run_id/path")
            resolved = resolve_artifact_uri_from_model_version(registry_name, selected.version)
            print("[DEBUG] resolved:", resolved)
        elif artifact_uri:
            # If the source is already runs:/... we can use it directly
            resolved = artifact_uri

        if resolved:
            try:
                # If resolved is a tuple (run_id, path) use client.download_artifacts
                if isinstance(resolved, tuple):
                    run_id, path = resolved
                    print(f"[INFO] Using client.download_artifacts run_id={run_id} path={path or '<top>'} dst={dst}")
                    client.download_artifacts(run_id, path or '', dst)
                    metadata["deployed_path"] = dst
                    with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
                        json.dump(metadata, f, indent=2)
                    print("[OK] Downloaded artifacts via client.download_artifacts")
                    return
                else:
                    # resolved is artifact_uri (string), call mlflow.artifacts.download_artifacts(artifact_uri, dst)
                    print(f"[INFO] Downloading artifact_uri={resolved} to dst={dst}")
                    local_path = mlflow.artifacts.download_artifacts(artifact_uri=resolved, dst_path=dst)
                    metadata["deployed_path"] = local_path
                    with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
                        json.dump(metadata, f, indent=2)
                    print("[OK] Downloaded artifacts to", local_path)
                    print("[OK] Deployment metadata written to models/model_metadata.json")
                    return
            except Exception as e:
                print("[WARN] Failed to download artifacts from resolved source:", e)

        else:
            print("[WARN] Could not resolve artifact_uri for model version; falling back to local model if available.")

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

    print(f"[ERROR] No model could be downloaded or found locally for registry_name={registry_name}")
    print("Summary contents for debugging:")
    print(json.dumps(summary, indent=2))
    sys.exit(1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=os.getenv("PROJECT_NAME", "diabetes"))
    parser.add_argument("--stage", default="Staging")
    args = parser.parse_args()
    deploy_best(args.project, args.stage)

