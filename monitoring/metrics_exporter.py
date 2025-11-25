# monitoring/metrics_exporter.py
import os
from prometheus_client import Gauge, CollectorRegistry, push_to_gateway

PUSHGATEWAY_URL = os.getenv("PUSHGATEWAY_URL", "http://localhost:9091")

def export_retrain_time(retrain_time: float, job: str="model_retraining"):
    registry = CollectorRegistry()
    g = Gauge("retrain_time_seconds", "Time taken for model retraining", registry=registry)
    g.set(float(retrain_time))
    try:
        push_to_gateway(PUSHGATEWAY_URL, job=job, registry=registry)
        print(f"[metrics_exporter] Pushed retrain_time={retrain_time} to {PUSHGATEWAY_URL} job={job}")
    except Exception as e:
        print("[metrics_exporter] Failed to push to Pushgateway:", e)
