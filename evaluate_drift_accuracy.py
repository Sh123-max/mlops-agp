import numpy as np
from prometheus_client import Gauge

ACCURACY_DRIFT = Gauge("accuracy_after_drift", "Model accuracy after drift window")

def evaluate_7_day_accuracy(model, data_7_days):
    X, y = data_7_days
    acc = model.score(X, y)
    ACCURACY_DRIFT.set(acc)
    mlflow.log_metric("accuracy_after_drift", acc)

