# deploy.py
"""
Robust per-project deploy script.

Behavior:
- Look for the latest MLflow run tagged `project=<project>` and use artifacts from that run.
- If not found, try model registry (models:/...) if available.
- Fallback to local artifact `models/<project>_<modelname>_model.pkl` style files.
- Writes per-project metadata to models/<project>/model_metadata.json
- Downloads artifacts to models/<project>/deployed_model/
- Returns non-zero on failure (exit code 1)
"""
import os
import json
import shutil
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime

# Config from env
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def write_metadata(project_dir, metadata):
    _ensure_dir(project_dir)
    meta_path = os.path.join(project_dir, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    return meta_path

def find_run_for_project(project):
    """
    Find the latest run tagged project=<project>.
    Uses mlflow.search_runs (DataFrame) because it is the most stable API across versions.
    Returns run_id (str) or None.
    """
    try:
        # search_runs returns a pandas DataFrame with columns including 'run_id' and 'tags...'
        # filter on tag "project"
        filter_str = f"tags.project = '{project}'"
        # order by start_time desc
        df = mlflow.search_runs(filter_string=filter_str, order_by=["attributes.start_time DESC"], max_results=5)
        if df is not None and len(df) > 0:
            run_id = df.iloc[0].run_id
            return run_id
    except Exception as e:
        print("[WARN] mlflow.search_runs by tag failed:", e)
    # fallback: search by tag using client.search_runs (if available)
    try:
        # try client.search_runs across all experiments (some mlflow versions require experiment ids)
        # try default experiment ids None (implementation dependent)
        runs = client.search_runs(experiment_ids=None, filter_string=f"tags.project = '{project}'", run_view_type=1, max_results=1)
        if runs:
            return runs[0].info.run_id
    except Exception:
        pass
    return None

def download_artifacts_from_run(run_id, dst, artifact_path="model"):
    """Try to download artifacts using mlflow.artifacts.download_artifacts or client.download_artifacts"""
    _ensure_dir(dst)
    # Try mlflow.artifacts.download_artifacts
    try:
        uri = f"runs:/{run_id}/{artifact_path}"
        print(f"[INFO] Using mlflow.artifacts.download_artifacts for {uri} -> {dst}")
        local_path = mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path=dst)
        return local_path
    except Exception as e:
        print("[WARN] mlflow.artifacts.download_artifacts failed:", e)
    # Try client.download_artifacts
    try:
        print(f"[INFO] Trying MlflowClient.download_artifacts run_id={run_id} path={artifact_path} dst={dst}")
        client.download_artifacts(run_id, artifact_path, dst)
        return dst
    except Exception as e:
        print("[WARN] MlflowClient.download_artifacts failed:", e)
    return None

def resolve_model_registry_and_download(registry_name, dst):
    """
    If a model registry entry exists (models:/...), try to resolve and download artifacts.
    """
    try:
        # mlflow.artifacts.download_artifacts accepts models:/ URIs as well
        model_uri = f"models:/{registry_name}/staging"  # try stage name variant - may fail harmlessly
    except Exception:
        model_uri = None

    # Try primary mlflow API: mlflow.artifacts.download_artifacts with models:/
    if registry_name:
        # First try latest versions via client.get_latest_versions
        try:
            vers = []
            try:
                vers = client.get_latest_versions(registry_name) or []
            except Exception:
                # some mlflow server setups or versions might not permit this
                vers = []
            if vers:
                # pick first
                mv = vers[0]
                source = getattr(mv, "source", None)
                run_id = getattr(mv, "run_id", None)
                print("[INFO] mlflow registry version source:", source, "run_id:", run_id)
                if run_id:
                    p = download_artifacts_from_run(run_id, dst, artifact_path="model")
                    if p:
                        return p, mv.version
                # fallback to mlflow.artifacts.download_artifacts on models:/ URI
                try:
                    uri = f"models:/{registry_name}/{mv.version}"
                    print(f"[INFO] Trying mlflow.artifacts.download_artifacts for {uri}")
                    p = mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path=dst)
                    return p, mv.version
                except Exception as e:
                    print("[WARN] mlflow.artifacts.download_artifacts for models:/ failed:", e)
        except Exception as e:
            print("[WARN] registry resolution failed:", e)
    return None, None

def deploy_best(project, stage="Staging"):
    project_dir = os.path.join("models", project)
    _ensure_dir(project_dir)
    deployed_dir = os.path.join(project_dir, "deployed_model")
    _ensure_dir(deployed_dir)

    # Wish: prefer last_run_summary.json local file per-project, otherwise use global models/last_run_summary.json
    local_summary = os.path.join(project_dir, "last_run_summary.json")
    global_summary = os.path.join("models", "last_run_summary.json")
    summary_path = local_summary if os.path.exists(local_summary) else global_summary
    if not os.path.exists(summary_path):
        print("[WARN] No last_run_summary.json for project; will try to find runs in MLflow")
        summary = {}
    else:
        summary = json.load(open(summary_path))

    # find best model name (fallback to "best" object from summary)
    best = summary.get("best", {})
    best_name = best.get("name", None)
    deploy_reason = best.get("deploy_reason", "No reason provided")
    should_deploy = best.get("should_deploy", True)

    # If summary exists and deployment is blocked, abort and write an alerts file
    if summary and not should_deploy:
        print(f"[BLOCKED] Deployment blocked for project {project}: {deploy_reason}")
        alert_file = os.path.join(project_dir, "deployment_blocked_alerts.json")
        alerts = []
        if os.path.exists(alert_file):
            try:
                alerts = json.load(open(alert_file, "r"))
            except Exception:
                alerts = []
        alerts.append({
            "timestamp": datetime.now().isoformat(),
            "project": project,
            "best": best,
            "reason": deploy_reason
        })
        with open(alert_file, "w") as f:
            json.dump(alerts, f, indent=2)
        return False

    # Try to discover run by tag
    run_id = find_run_for_project(project)
    metadata = {
        "model_name": best_name or "unknown",
        "project": project,
        "registry_name": None,
        "version": None,
        "source": None,
        "deployed_path": None,
        "feature_count": best.get("feature_count") if isinstance(best, dict) else None,
        "feature_order": best.get("feature_order") if isinstance(best, dict) else None,
        "metrics": best,
        "deployed_at": datetime.now().isoformat()
    }

    # If run found, try to download artifacts from run
    if run_id:
        print(f"[INFO] Found run for project {project}: run_id={run_id}. Attempting to download artifacts...")
        p = download_artifacts_from_run(run_id, deployed_dir, artifact_path="model")
        if p:
            metadata.update({"deployed_path": p, "source": f"runs:/{run_id}/model"})
            write_metadata(project_dir, metadata)
            print("[OK] Downloaded artifacts from run and wrote metadata")
            return True
        else:
            print("[WARN] Failed to download artifacts from run; continuing to registry fallback")

    # If summary best_name exists, try registry name pattern <project>_<modelname>
    if best_name:
        registry_name = f"{project}_{best_name}"
        print(f"[INFO] Trying registry name: {registry_name}")
        p, ver = resolve_model_registry_and_download(registry_name, deployed_dir)
        if p:
            metadata.update({"deployed_path": p, "registry_name": registry_name, "version": ver, "source": p})
            write_metadata(project_dir, metadata)
            print("[OK] Deployed from registry or models:/ URI")
            return True
        else:
            print("[WARN] Could not obtain registry artifact for", registry_name)

    # Last fallback: try local artifact with common names
    local_candidates = [
        os.path.join("models", f"{best_name}_model.pkl") if best_name else None,
        os.path.join("models", f"{best_name}_model.joblib") if best_name else None,
        os.path.join("models", "best_model.pkl"),
        os.path.join("models", "best_model.joblib"),
        os.path.join("models", f"{project}_model.pkl"),
        os.path.join("models", f"{project}_model.joblib")
    ]
    for c in local_candidates:
        if not c:
            continue
        if os.path.exists(c):
            print(f"[INFO] Fallback: copying local model {c} -> {deployed_dir}")
            dest = os.path.join(deployed_dir, os.path.basename(c))
            shutil.copy2(c, dest)
            metadata.update({"deployed_path": dest, "version": "local-fallback", "source": c})
            write_metadata(project_dir, metadata)
            print("[OK] Copied local fallback model and wrote metadata")
            return True

    print("[ERROR] No model could be downloaded or located for project:", project)
    return False

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--project", default=os.getenv("PROJECT_NAME", "diabetes"), help="Project name (diabetes/heart)")
    p.add_argument("--stage", default="Staging", help="Model registry stage (unused if not registered)")
    args = p.parse_args()

    ok = deploy_best(args.project, stage=args.stage)
    if ok:
        print("Deployment completed successfully")
    else:
        print("Deployment FAILED")
        exit(1)
