pipeline {
    agent any

    environment {
        PROJECT_NAME = "${env.PROJECT_NAME ?: 'diabetes'}"
        DATA_DIR = "${env.DATA_DIR ?: 'data'}"
        MODEL_DIR = "${env.MODEL_DIR ?: 'models'}"
        MLFLOW_TRACKING_URI = "${env.MLFLOW_TRACKING_URI ?: 'http://localhost:5001'}"
        VENV_DIR = "venv"
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
    stage('Clean Virtual Environment') {
    steps {
        sh 'rm -rf venv'
    }
}

        stage('Setup Environment') {
            steps {
                script {
                    // Create virtual environment if not exists
                    if (!fileExists(env.VENV_DIR)) {
                        sh "python3 -m venv ${env.VENV_DIR}"
                    }
                    // Upgrade pip and install requirements in venv
                    sh """
                        . ${env.VENV_DIR}/bin/activate
                        pip install --upgrade pip
                        pip install -r requirements.txt
                    """
                }
            }
        }

        stage('Check Python Modules') {
            steps {
                script {
                    sh """
                        . ${env.VENV_DIR}/bin/activate
                        python -c "import pandas" || pip install pandas
                    """
                }
            }
        }

        stage('Preprocess') {
            steps {
                sh """
                    . ${env.VENV_DIR}/bin/activate
                    python preprocess.py
                """
            }
        }

        stage('Train') {
            steps {
                script {
                    env.TRAIN_START = sh(script: """
                        . ${env.VENV_DIR}/bin/activate
                        python -c "import time; print(int(time.time()))"
                        """, returnStdout: true).trim()
                    sh """
                        . ${env.VENV_DIR}/bin/activate
                        python trainandevaluate.py 2>&1 | tee train_log.txt
                    """
                    env.TRAIN_END = sh(script: """
                        . ${env.VENV_DIR}/bin/activate
                        python -c "import time; print(int(time.time()))"
                        """, returnStdout: true).trim()
                }
            }
        }

        stage('Record retrain time metric') {
            steps {
                script {
                    sh """
                        . ${env.VENV_DIR}/bin/activate
                        python - <<'PY'
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
                    sh """
                        . ${env.VENV_DIR}/bin/activate
                        python deploy.py --project ${env.PROJECT_NAME} --stage Staging
                    """
                    def deployEnd = System.currentTimeMillis()
                    def deploySeconds = (deployEnd - deployStart) / 1000
                    writeFile file: 'deployment_time.txt', text: deploySeconds.toString()
                    archiveArtifacts artifacts: 'deployment_time.txt', fingerprint: true
                }
            }
        }

        stage('Evaluate Drift Accuracy (7-day)') {
            steps {
                sh """
                    . ${env.VENV_DIR}/bin/activate
                    python - <<'PY'
import joblib, os, json, time
from sklearn.metrics import accuracy_score

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
    for f in os.listdir("models"):
        if f.endswith("_model.pkl"):
            try:
                model = joblib.load(os.path.join("models", f))
                break
            except Exception:
                pass

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
                """
                archiveArtifacts artifacts: 'models/model_metadata.json', fingerprint: true
            }
        }

        stage('Manual Intervention Logging (optional)') {
            when {
                anyOf {
                    triggeredBy 'UserIdCause'
                    expression { return params.get('FORCE_MANUAL_LOG', false) == true }
                }
            }
            steps {
                sh """
                    . ${env.VENV_DIR}/bin/activate
                    python - <<'PY'
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
import os

with open('manual_intervention.txt','a') as fh:
    fh.write("manual_intervention\\n")

pg = os.environ.get('PUSHGATEWAY_URL', None)
if pg:
    try:
        registry = CollectorRegistry()
        g = Gauge('manual_intervention_count', 'Manual interventions count', registry=registry)
        g.set(1)
        push_to_gateway(pg + '/metrics/job/${PROJECT_NAME}_manual', registry=registry)
    except Exception as e:
        print("Pushgateway manual log failed:", e)

print("Manual intervention logged.")
PY
                """
                archiveArtifacts artifacts: 'manual_intervention.txt', fingerprint: true
            }
        }
    }

    post {
        success {
            script {
                def deployTime = readFile('deployment_time.txt').trim()
                def retrainTime = readFile('retrain_time.txt').trim()
                echo "Deployment time (s): ${deployTime}"
                echo "Retrain time (s): ${retrainTime}"
            }
        }
    }
}
