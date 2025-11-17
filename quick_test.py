import os, joblib, time, mlflow, mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
X_train = joblib.load("data/X_train.pkl")
y_train = joblib.load("data/y_train.pkl")
X_test  = joblib.load("data/X_test.pkl")
y_test  = joblib.load("data/y_test.pkl")

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI","http://localhost:5001"))
print("Using MLflow URI:", mlflow.get_tracking_uri())

with mlflow.start_run(run_name="quick_test_lr"):
    m = LogisticRegression(max_iter=2000)
    m.fit(X_train[:200], y_train[:200])
    preds = m.predict(X_test[:200])
    acc = accuracy_score(y_test[:200], preds)
    mlflow.log_metric("acc_sample", float(acc))
    mlflow.sklearn.log_model(m, artifact_path="model")
    print("Logged quick model, acc:", acc)
