# deploy.py (robust: resolves models:/ -> runs:/ and downloads artifacts)
import os
import json
import shutil
import mlflow
from mlflow.tracking import MlflowClient

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
        src = getattr(mv, "source", None)
        if src and src.startswith("runs:/"):
            return src
        run_id = getattr(mv, "run_id", None)
        if run_id:
            if src and "artifacts/" in src:
                path = src.split("artifacts/", 1)[1]
                return (run_id, path)
            return (run_id, None)
    except Exception as e:
        print("[DEBUG] get_model_version fallback failed:", e)

    return None

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
        print(f"[INFO] Selected model version: name={selected.name} version={selected.version} stage={selected.current_stage}")
        artifact_uri = getattr(selected, "source", None)
        metadata["source"] = artifact_uri
        print("[DEBUG] selected.source:", artifact_uri)

        resolved = None
        if artifact_uri and str(artifact_uri).startswith("models:/"):
            print("[DEBUG] artifact_uri is models:/... resolving to runs:/ or run_id/path")
            resolved = resolve_artifact_uri_from_model_version(registry_name, selected.version)
            print("[DEBUG] resolved:", resolved)
        elif artifact_uri:
            resolved = artifact_uri

        if resolved:
            try:
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
    local_best = os.path.join(MODEL_DIR, f"{best_name}_model.pkl")
    if os.path.exists(local_best):
        deployed = os.path.join(dst, f"{best_name}_model.pkl")
        shutil.copy2(local_best, deployed)
        metadata["deployed_path"] = deployed
        metadata["version"] = metadata.get("version") or "local-fallback"
        # Attempt to enrich metadata with metrics if available in last_run_summary
        try:
            meta_json = json.load(open(os.path.join(MODEL_DIR, "model_metadata.json"))) if os.path.exists(os.path.join(MODEL_DIR, "model_metadata.json")) else {}
            meta_json.update(metadata)
            with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
                json.dump(meta_json, f, indent=2)
        except Exception:
            with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)
        print("[OK] Fallback: copied local model to", deployed)
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

