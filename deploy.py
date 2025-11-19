import os
import json
import shutil
import joblib
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime

MODEL_DIR = os.getenv("MODEL_DIR", "models")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

try:
    from metrics_history import MetricsHistory
except ImportError:
    class MetricsHistory:
        def __init__(self, *args, **kwargs): pass
        def add_deployment(self, *args, **kwargs): pass

def resolve_artifact_uri_from_model_version(registry_name, version):
    try:
        uri = client.get_model_version_download_uri(registry_name, version)
        if uri:
            return uri
    except Exception as e:
        print("[DEBUG] get_model_version_download_uri not usable:", e)
    try:
        mv = client.get_model_version(registry_name, version)
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

def is_ensemble_model(model_name):
    return str(model_name).startswith("Ensemble_")

def deploy_best(project, stage="Staging"):
    summary_path = os.path.join(MODEL_DIR, "last_run_summary.json")
    if not os.path.exists(summary_path):
        raise RuntimeError("No last_run_summary.json found. Run training first.")
    summary = json.load(open(summary_path))
    best = summary.get("best", {})
    best_name = best.get("name")
    if not best_name:
        raise RuntimeError("No best model found in summary.")
    should_deploy = best.get("should_deploy", True)
    deploy_reason = best.get("deploy_reason", "No deployment reason provided")
    if not should_deploy:
        print(f"Deployment blocked: {deploy_reason}")
        trigger_manual_intervention_alert(best, deploy_reason)
        return False
    registry_name = f"{project}_{best_name}"
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
    metadata = {"model_name": best_name, "registry_name": registry_name, "version": None, "source": None, "deployed_path": None}
    if is_ensemble_model(best_name):
        print(f"Loading ensemble model: {best_name}")
        ensemble_path = os.path.join(MODEL_DIR, f"{best_name}_model.pkl")
        if os.path.exists(ensemble_path):
            try:
                deployed_ensemble_path = os.path.join(dst, f"{best_name}_model.pkl")
                shutil.copy2(ensemble_path, deployed_ensemble_path)
                metadata["deployed_path"] = deployed_ensemble_path
                metadata["model_type"] = "ensemble"
                metadata["version"] = "ensemble-latest"
                print(f"[DEPLOYED] Ensemble model deployed: {best_name} -> {deployed_ensemble_path}")
                metrics_history = MetricsHistory()
                metrics_history.add_deployment({
                    "model_name": best_name,
                    "registry_name": registry_name,
                    "metrics": best,
                    "stage": stage,
                    "deployment_reason": deploy_reason,
                    "is_ensemble": True
                })
                with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
                    json.dump(metadata, f, indent=2)
                return True
            except Exception as e:
                print(f"Failed to load ensemble model: {e}")
        else:
            print(f"Ensemble model file not found: {ensemble_path}")
    if versions:
        selected = versions[0]
        metadata["version"] = selected.version
        print(f"[INFO] Selected model version: name={selected.name} version={selected.version} stage={getattr(selected,'current_stage',None)}")
        artifact_uri = getattr(selected, "source", None)
        metadata["source"] = artifact_uri
        print("[DEBUG] selected.source:", artifact_uri)
        run_id = getattr(selected, "run_id", None)
        if run_id:
            try:
                print(f"[INFO] Trying client.download_artifacts run_id={run_id} path='model' dst={dst}")
                client.download_artifacts(run_id, "model", dst)
                metadata["deployed_path"] = dst
                metadata["version"] = metadata.get("version") or "from-run"
                with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
                    json.dump(metadata, f, indent=2)
                print("[OK] Downloaded artifacts via client.download_artifacts")
                print(f"[DEPLOYED] model_name={best_name} registry_name={registry_name} version={metadata.get('version')} deployed_path={metadata.get('deployed_path')}")
                metrics_history = MetricsHistory()
                metrics_history.add_deployment({
                    "model_name": best_name,
                    "registry_name": registry_name,
                    "metrics": best,
                    "stage": stage,
                    "deployment_reason": deploy_reason
                })
                return True
            except Exception as e:
                print("[WARN] client.download_artifacts failed:", e)
        resolved = None
        if artifact_uri and str(artifact_uri).startswith("models:/"):
            resolved = resolve_artifact_uri_from_model_version(registry_name, selected.version)
            print("[DEBUG] resolved:", resolved)
        elif artifact_uri:
            resolved = artifact_uri
        if resolved:
            try:
                if isinstance(resolved, tuple):
                    run_id2, path = resolved
                    print(f"[INFO] Using client.download_artifacts run_id={run_id2} path={path or '<top>'} dst={dst}")
                    client.download_artifacts(run_id2, path or '', dst)
                    metadata["deployed_path"] = dst
                    with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
                        json.dump(metadata, f, indent=2)
                    print("[OK] Downloaded artifacts via client.download_artifacts (tuple path)")
                    print(f"[DEPLOYED] model_name={best_name} registry_name={registry_name} version={metadata.get('version')} deployed_path={metadata.get('deployed_path')}")
                    metrics_history = MetricsHistory()
                    metrics_history.add_deployment({
                        "model_name": best_name,
                        "registry_name": registry_name,
                        "metrics": best,
                        "stage": stage,
                        "deployment_reason": deploy_reason
                    })
                    return True
                else:
                    print(f"[INFO] Downloading artifact_uri={resolved} to dst={dst}")
                    local_path = mlflow.artifacts.download_artifacts(artifact_uri=resolved, dst_path=dst)
                    metadata["deployed_path"] = local_path
                    with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
                        json.dump(metadata, f, indent=2)
                    print("[OK] Downloaded artifacts to", local_path)
                    print("[OK] Deployment metadata written to models/model_metadata.json")
                    print(f"[DEPLOYED] model_name={best_name} registry_name={registry_name} version={metadata.get('version')} deployed_path={local_path}")
                    metrics_history = MetricsHistory()
                    metrics_history.add_deployment({
                        "model_name": best_name,
                        "registry_name": registry_name,
                        "metrics": best,
                        "stage": stage,
                        "deployment_reason": deploy_reason
                    })
                    return True
            except Exception as e:
                print("[WARN] Failed to download artifacts from resolved source:", e)
        else:
            print("[WARN] Could not resolve artifact_uri for model version; falling back to local model if available.")
    local_best = os.path.join(MODEL_DIR, f"{best_name}_model.pkl")
    if os.path.exists(local_best):
        deployed = os.path.join(dst, f"{best_name}_model.pkl")
        shutil.copy2(local_best, deployed)
        metadata["deployed_path"] = deployed
        metadata["version"] = metadata.get("version") or "local-fallback"
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
        print(f"[DEPLOYED] model_name={best_name} registry_name={registry_name} version={metadata.get('version')} deployed_path={deployed}")
        metrics_history = MetricsHistory()
        metrics_history.add_deployment({
            "model_name": best_name,
            "registry_name": registry_name,
            "metrics": best,
            "stage": stage,
            "deployment_reason": deploy_reason
        })
        return True
    raise RuntimeError(f"No model could be downloaded or found locally for registry_name={registry_name}")

def trigger_manual_intervention_alert(current_best, reason):
    alert_data = {
        "timestamp": datetime.now().isoformat(),
        "current_model": current_best,
        "block_reason": reason,
        "requires_manual_review": True
    }
    alert_file = os.path.join(MODEL_DIR, "deployment_blocked_alerts.json")
    alerts = []
    if os.path.exists(alert_file):
        with open(alert_file, 'r') as f:
            alerts = json.load(f)
    alerts.append(alert_data)
    with open(alert_file, 'w') as f:
        json.dump(alerts, f, indent=2)
    print(f"MANUAL INTERVENTION REQUIRED: {reason}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=os.getenv("PROJECT_NAME", "diabetes"))
    parser.add_argument("--stage", default="Staging")
    args = parser.parse_args()
    success = deploy_best(args.project, args.stage)
    if success:
        print("Deployment completed successfully")
        try:
            mm = json.load(open(os.path.join(MODEL_DIR, "model_metadata.json")))
            name = mm.get("model_name") or mm.get("best", {}).get("name")
            version = mm.get("version") or (mm.get("best", {}).get("registry") or {}).get("version")
            print(f"[DEPLOYED-METADATA] model_name={name} version={version}")
        except Exception:
            pass
    else:
        print("Deployment was blocked or failed")
        exit(1)
