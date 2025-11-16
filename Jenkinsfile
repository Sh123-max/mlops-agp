// Jenkinsfile (declarative pipeline) for the MLOps flow:
// Checkout -> Preprocess -> Train -> Deploy -> Evaluate drift accuracy -> (Manual intervention logging optional)
// Assumes Jenkins agent has Python, docker, and environment configured. Adjust paths as needed.

pipeline {
    agent any

    environment {
        PROJECT_NAME = "${env.PROJECT_NAME ?: 'diabetes'}"
        DATA_DIR = "${env.DATA_DIR ?: 'data'}"
        MODEL_DIR = "${env.MODEL_DIR ?: 'models'}"
        MLFLOW_TRACKING_URI = "${env.MLFLOW_TRACKING_URI ?: 'http://localhost:5001'}"
    }

    options {
        timestamps()
        buildDiscarder(logRotator(numToKeepStr: '30'))
        timeout(time: 2, unit: 'HOURS')
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Preprocess') {
            steps {
                sh """
                    python3 preprocess.py
                """
            }
        }

        stage('Train') {
            steps {
                script {
                    // record training start time
                    env.TRAIN_START = sh(script: "python3 - <<'PY'\nimport time\nprint(int(time.time()))\nPY", returnStdout: true).trim()
                    sh "python3 trainandevaluate.py 2>&1 | tee train_log.txt"
                    env.TRAIN_END = sh(script: "python3 - <<'PY'\nimport time\nprint(int(time.time()))\nPY", returnStdout: true).trim()
                }
            }
        }

        stage('Record retrain time metric') {
            steps {
                script {
                    // compute and save retrain_time_seconds to file for later use/archival
                    sh """
                        python3 - <<'PY'
import os, json, time
start = int(os.environ.get('TRAIN_START', '0'))
end = int(os.environ.get('TRAIN_END', '0'))
t = end - start if (start and end) else 0
with open('retrain_time.txt','w') as f:
    f.write(str(t))
print("retrain_time_seconds:", t)
PY
                    """
                    archiveArtifacts artifacts: 'train_log.txt,retrain_time.txt', fingerprint: true
                }
            }
        }

        stage('Deploy') {
            steps {
                script {
                    def deployStart = System.currentTimeMillis()
                    // call deploy.py to fetch model artifacts and copy into models/deployed_model
                    sh "python3 deploy.py --project ${env.PROJECT_NAME} --stage Staging"
                    def deployEnd = System.currentTimeMillis()
                    def deploySeconds = (deployEnd - deployStart) / 1000
                    writeFile file: 'deployment_time.txt', text: deploySeconds.toString()
                    archiveArtifacts artifacts: 'deployment_time.txt', fingerprint: true
                }
            }
        }

        stage('Evaluate Drift Accuracy (7-day)') {
            steps {
                // run a small evaluation that computes accuracy on test set (or a 7-day window if available)
                sh '''
python3 - <<'PY'
import joblib, os, json
from sklearn.metrics import accuracy_score
# load deployed model (fallback local)
model = None
deploy_dir = os.path.join("models","deployed_model")
if os.path.exists(deploy_dir):
    for root,_,files in os.walk(deploy_dir):
        for f in files:
            if f.endswith(".pkl") or f.endswith(".joblib"):
                try:
                    model = joblib.load(os.path.join(root,f))
                    break
                except Exception:
                    pass
        if model:
            break
if model is None:
    # fallback to any local model
    for f in os.listdir("models"):
        if f.endswith("_model.pkl"):
            try:
                model = joblib.load(os.path.join("models", f))
                break
            except Exception:
                pass

# load test set
if os.path.exists(os.path.join("data","X_test.pkl")) and os.path.exists(os.path.join("data","y_test.pkl")):
    X_test = joblib.load(os.path.join("data","X_test.pkl"))
    y_test = joblib.load(os.path.join("data","y_test.pkl"))
    try:
        y_pred = model.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
    except Exception:
        acc = 0.0
else:
    acc = 0.0

# write metric and append to model metadata
meta_path = os.path.join("models","model_metadata.json")
meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
meta.setdefault("evaluations", [])
entry = {"ts": int(time.time()), "accuracy_last_test": acc}
meta["evaluations"].append(entry)
meta["evaluations"] = meta["evaluations"][-20:]
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print("accuracy_after_drift:", acc)
PY
'''
                archiveArtifacts artifacts: 'models/model_metadata.json', fingerprint: true
            }
        }

        stage('Manual Intervention Logging (optional)') {
            when {
                anyOf {
                    // trigger when build was user-triggered
                    triggeredBy 'UserIdCause'
                    expression { return params.get('FORCE_MANUAL_LOG', false) == true }
                }
            }
            steps {
                sh '''
python3 - <<'PY'
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
import os
# simple manual intervention recorder file
with open('manual_intervention.txt','a') as fh:
    fh.write("manual_intervention\\n")
# Optionally push a metric to pushgateway if configured
pg = os.environ.get('PUSHGATEWAY_URL', None)
if pg:
    try:
        from prometheus_client import Gauge, CollectorRegistry, push_to_gateway
        registry = CollectorRegistry()
        g = Gauge('manual_intervention_count', 'Manual interventions count', registry=registry)
        g.set(1)
        push_to_gateway(pg + '/metrics/job/${PROJECT_NAME}_manual', registry=registry)
    except Exception as e:
        print("Pushgateway manual log failed:", e)
print("Manual intervention logged.")
PY
'''
                archiveArtifacts artifacts: 'manual_intervention.txt', fingerprint: true
            }
        }
    }

    post {
        success {
            script {
                // optionally push deployment_time.txt and retrain_time.txt content to an external metrics collector
                def deployTime = readFile('deployment_time.txt').trim()
                def retrainTime = readFile('retrain_time.txt').trim()
                echo "Deployment time (s): ${deployTime}"
                echo "Retrain time (s): ${retrainTime}"
            }
        }
        failure {
            mail to: 'you@example.com',
                 subject: "Pipeline failed: ${env.JOB_NAME} #${env.BUILD_NUMBER}",
                 body: "See Jenkins console output for details."
        }
    }
}

