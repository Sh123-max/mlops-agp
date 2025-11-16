from prometheus_client import Gauge

RETRAIN_TIME = Gauge("retrain_time_seconds", "Time taken for model retraining")

# After training completes
RETRAIN_TIME.set(retrain_time)

