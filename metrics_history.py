import json
import os
from datetime import datetime

class MetricsHistory:
    def __init__(self, history_file="models/metrics_history.json"):
        self.history_file = history_file
        self.history = self.load_history()
    
    def load_history(self):
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return {"training_runs": [], "deployments": []}
    
    def add_training_run(self, run_data):
        run_entry = {
            "timestamp": datetime.now().isoformat(),
            "run_id": run_data.get("run_id"),
            "model_name": run_data.get("name"),
            "metrics": {
                "weighted_score": run_data.get("weighted_score"),
                "accuracy": run_data.get("accuracy"),
                "recall": run_data.get("recall"),
                "precision": run_data.get("precision"),
                "f1_score": run_data.get("f1_score"),
                "roc_auc": run_data.get("roc_auc"),
                "false_negative_rate": run_data.get("false_negative_rate")
            },
            "pareto_front": run_data.get("pareto_models", []) != []
        }
        self.history["training_runs"].append(run_entry)
        self.save()
    
    def get_previous_best(self):
        deployments = self.history.get("deployments", [])
        if deployments:
            return deployments[-1]
        return None
    
    def add_deployment(self, deployment_data):
        self.history["deployments"].append({
            "timestamp": datetime.now().isoformat(),
            **deployment_data
        })
        self.save()
    
    def save(self):
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
