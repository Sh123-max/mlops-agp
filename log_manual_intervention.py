from prometheus_client import Counter
import mlflow

MANUAL_INTERVENTION = Counter(
    "manual_intervention_count",
    "Number of manual interventions"
)

MANUAL_INTERVENTION.inc()
mlflow.log_metric("manual_intervention_trigger", 1)
