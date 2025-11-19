import os
import json
import joblib
import mlflow
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from mlflow.tracking import MlflowClient

class EnsembleManager:
    def __init__(self, model_dir="models", project_name="diabetes"):
        self.model_dir = model_dir
        self.project_name = project_name
        self.client = MlflowClient()
    
    def load_previous_best_model(self):
        try:
            history_path = os.path.join(self.model_dir, "metrics_history.json")
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    history = json.load(f)
                
                deployments = history.get("deployments", [])
                if deployments:
                    last_deployment = deployments[-1]
                    model_name = last_deployment.get("model_name")
                    registry_name = f"{self.project_name}_{model_name}"
                    versions = self.client.get_latest_versions(registry_name)
                    if versions:
                        model_uri = f"models:/{registry_name}/{versions[0].version}"
                        return mlflow.sklearn.load_model(model_uri)
        except Exception as e:
            print(f"Error loading previous best model: {e}")
        return None
    
    def load_current_best_model(self, current_model_name):
        try:
            model_path = os.path.join(self.model_dir, f"{current_model_name}_model.pkl")
            if os.path.exists(model_path):
                return joblib.load(model_path)
        except Exception as e:
            print(f"Error loading current best model: {e}")
        return None
    
    def create_ensemble(self, previous_model, current_model, ensemble_method='soft'):
        if previous_model is None or current_model is None:
            return None
            
        estimators = [
            ('previous_best', previous_model),
            ('current_best', current_model)
        ]
        
        if ensemble_method == 'soft':
            ensemble = VotingClassifier(estimators=estimators, voting='soft')
        else:
            ensemble = VotingClassifier(estimators=estimators, voting='hard')
            
        return ensemble
    
    def evaluate_ensemble(self, ensemble, X_test, y_test):
        y_pred = ensemble.predict(X_test)
        y_proba = ensemble.predict_proba(X_test)[:, 1] if hasattr(ensemble, 'predict_proba') else None
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0
        }
        
        weights = {'Accuracy':0.05, 'Precision':0.05, 'Recall':0.4, 'F1-Score':0.3, 'ROC-AUC':0.2}
        weighted_score = (
            weights['Accuracy'] * metrics['accuracy'] +
            weights['Precision'] * metrics['precision'] +
            weights['Recall'] * metrics['recall'] +
            weights['F1-Score'] * metrics['f1_score'] +
            weights['ROC-AUC'] * metrics['roc_auc']
        )
        
        metrics['weighted_score'] = weighted_score
        return metrics
    
    def should_create_ensemble(self, current_score, previous_score, threshold=0.01):
        if previous_score is None:
            return False
        performance_drop = previous_score - current_score
        return performance_drop > threshold

    def train_and_evaluate_ensemble(self, X_train, y_train, X_test, y_test, 
                                  current_model_name, current_score, previous_score):
        if not self.should_create_ensemble(current_score, previous_score):
            return None, None
            
        print("Performance drop >1% detected. Creating ensemble...")
        
        previous_model = self.load_previous_best_model()
        current_model = self.load_current_best_model(current_model_name)
        
        if previous_model is None or current_model is None:
            print("Could not load models for ensemble")
            return None, None
        
        ensemble = self.create_ensemble(previous_model, current_model, ensemble_method='soft')
        
        if ensemble is None:
            print("Failed to create ensemble")
            return None, None
        
        print("Training ensemble on combined knowledge...")
        ensemble.fit(X_train, y_train)
        
        ensemble_metrics = self.evaluate_ensemble(ensemble, X_test, y_test)
        
        print(f"Ensemble Performance:")
        print(f"   Weighted Score: {ensemble_metrics['weighted_score']:.4f}")
        print(f"   Current Model:  {current_score:.4f}")
        print(f"   Previous Model: {previous_score:.4f}")
        
        return ensemble, ensemble_metrics

    def log_ensemble_to_mlflow(self, ensemble, ensemble_metrics, current_model_name, previous_model_name):
        try:
            ensemble_name = f"Ensemble_{previous_model_name}_{current_model_name}"
            
            with mlflow.start_run(run_name=f"{self.project_name}__{ensemble_name}__ensemble"):
                mlflow.log_param("model_name", ensemble_name)
                mlflow.log_param("ensemble_type", "VotingClassifier")
                mlflow.log_param("voting", "soft")
                mlflow.log_param("base_models", f"{previous_model_name},{current_model_name}")
                
                mlflow.log_metric("accuracy", ensemble_metrics['accuracy'])
                mlflow.log_metric("precision", ensemble_metrics['precision'])
                mlflow.log_metric("recall", ensemble_metrics['recall'])
                mlflow.log_metric("f1_score", ensemble_metrics['f1_score'])
                mlflow.log_metric("roc_auc", ensemble_metrics['roc_auc'])
                mlflow.log_metric("weighted_score", ensemble_metrics['weighted_score'])
                mlflow.log_metric("is_ensemble", 1.0)
                
                mlflow.set_tag("project", self.project_name)
                mlflow.set_tag("model_name", ensemble_name)
                mlflow.set_tag("ensemble", "true")
                
                mlflow.sklearn.log_model(ensemble, artifact_path="model")
                
                local_path = os.path.join(self.model_dir, f"{ensemble_name}_model.pkl")
                joblib.dump(ensemble, local_path)
                
                print(f"Ensemble logged to MLflow: {ensemble_name}")
                
                return ensemble_name
                
        except Exception as e:
            print(f"Failed to log ensemble to MLflow: {e}")
            return None
