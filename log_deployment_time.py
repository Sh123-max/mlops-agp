from prometheus_client import Gauge
import mlflow

DEPLOY_TIME = Gauge("deployment_cycle_seconds", "Model deployment cycle time")

with open("deployment_time.txt") as f:
    t = float(f.read())

DEPLOY_TIME.set(t)
mlflow.log_metric("deployment_cycle_seconds", t)
