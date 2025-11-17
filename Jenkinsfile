// Jenkinsfile (Declarative) - agent any + venv-based, lightweight, robust pip/scipy install
pipeline {
    agent any

    environment {
        PROJECT_NAME = "${env.PROJECT_NAME ?: 'diabetes'}"
        DATA_DIR = "${env.DATA_DIR ?: 'data'}"
        MODEL_DIR = "${env.MODEL_DIR ?: 'models'}"
        MLFLOW_TRACKING_URI = "${env.MLFLOW_TRACKING_URI ?: 'http://localhost:5001'}"
        REQUIREMENTS = "${env.REQUIREMENTS ?: 'requirements.txt'}"
        ENABLE_MAIL = "${env.ENABLE_MAIL ?: 'false'}" // set to 'true' if Jenkins mail configured
        INSTALL_BUILD_DEPS = "${env.INSTALL_BUILD_DEPS ?: 'false'}" // opt-in if wheel install fails
        VENV_DIR = "${env.VENV_DIR ?: '.venv'}"
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

        stage('Prepare Python env & deps (venv)') {
            steps {
                // create a per-build venv and install dependencies reliably
                sh '''
set -euo pipefail
echo "Using python from: $(which python3 || true)"
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found on this node. Please install python3 (and pip) or use a node that has it." >&2
  exit 3
fi

# create venv
python3 -m venv "${VENV_DIR}"
. "${VENV_DIR}/bin/activate"

# upgrade packaging tools
python -m pip install --upgrade pip setuptools wheel

# install dependencies preferring binary wheels to avoid heavy compilation (scipy)
if [ -f "${REQUIREMENTS}" ]; then
  echo "Installing from ${REQUIREMENTS} with --prefer-binary"
  pip install --no-cache-dir --prefer-binary -r "${REQUIREMENTS}" || INSTALL_RC=$?
else
  echo "No requirements.txt found; installing minimal runtime deps"
  pip install --no-cache-dir --prefer-binary pandas numpy scikit-learn joblib prometheus_client || INSTALL_RC=$?
fi

# fallback: if pip failed and INSTALL_BUILD_DEPS=true, install apt build deps and retry
if [ -n "${INSTALL_RC:-}" ]; then
  echo "pip install returned rc=${INSTALL_RC}"
  if [ "${INSTALL_BUILD_DEPS}" = "true" ]; then
    echo "INSTALL_BUILD_DEPS=true -> attempting apt-get build deps and retry"
    if command -v apt-get >/dev/null 2>&1; then
      apt-get update -y && apt-get install -y build-essential gfortran libatlas-base-dev
      if [ -f "${REQUIREMENTS}" ]; then
        pip install --no-cache-dir -r "${REQUIREMENTS}"
      else
        pip install --no-cache-dir pandas numpy scikit-learn joblib prometheus_client
      fi
    else
      echo "apt-get not available on this node; cannot install system build deps." >&2
      exit ${INSTALL_RC}
    fi
  else
    echo "INSTALL_BUILD_DEPS is false -> not installing system build deps. Failing the build to keep pipeline light." >&2
    exit ${INSTALL_RC}
  fi
fi

# verify essential packages
python - <<'PY'
import sys, importlib
reqs = ['pandas','numpy','sklearn','joblib']
missing = [r for r in reqs if importlib.util.find_spec(r) is None]
if missing:
    print("Missing python packages:", missing, file=sys.stderr)
    sys.exit(4)
print("Python deps OK")
PY
'''
            }
        }

        stage('Preprocess') {
            steps {
                sh '''
set -euo pipefail
. "${VENV_DIR}/bin/activate"
python preprocess.py
'''
            }
        }

        stage('Train') {
            steps {
                script {
                    env.TRAIN_START = sh(script: "bash -lc '. ${VENV_DIR}/bin/activate; python - <<'PY'\nimport time\nprint(int(time.time()))\nPY'", returnStdout: true).trim()
                    sh "bash -lc '. ${VENV_DIR}/bin/activate; python trainandevaluate.py 2>&1 | tee train_log.txt'"
                    env.TRAIN_END = sh(script: "bash -lc '. ${VENV_DIR}/bin/activate; python - <<'PY'\nimport time\nprint(int(time.time()))\nPY'", returnStdout: true).trim()
                }
            }
        }

        stage('Record retrain time metric') {
            steps {
                sh '''
set -euo pipefail
. "${VENV_DIR}/bin/activate"
python - <<'PY'
import os, json
start = int(os.environ.get('TRAIN_START', '0'))
end = int(os.environ.get('TRAIN_END', '0'))
t = end - start if (start and end) else 0
with open('retrain_time.txt','w') as f:
    f.write(str(t))
print("retrain_time_seconds:", t)
PY
'''
                archiveArtifacts artifacts: 'train_log.txt,retrain_time.txt', fingerprint: true
            }
        }

        stage('Deploy') {
            steps {
                sh '''
set -euo pipefail
. "${VENV_DIR}/bin/activate"
python deploy.py --project "${PROJECT_NAME}" --stage Staging || true
# create a placeholder deployment_time.txt if deploy.py does not write it
if [ ! -f deployment_time.txt ]; then
  echo "0" > deployment_time.txt
fi
'''
                archiveArtifacts artifacts: 'deployment_time.txt', fingerprint: true
            }
        }

        stage('Evaluate Drift Accuracy (7-day)') {
            steps {
                sh '''
set -euo pipefail
. "${VENV_DIR}/bin/activate"
python - <<'PY'
import joblib, os, json, time
from sklearn.metrics import accuracy_score
model = None
deploy_dir = os.path.join("models","deployed_model")
if os.path.exists(deploy_dir):
    for root,_,files in os.walk(deploy_dir):
        for f in files:
            if f.endswith((".pkl", ".joblib")):
                try:
                    model = joblib.load(os.path.join(root,f))
                    break
                except Exception:
                    pass
        if model:
            break
if model is None and os.path.exists("models"):
    for f in os.listdir("models"):
        if f.endswith(("_model.pkl", ".pkl", ".joblib")):
            try:
                model = joblib.load(os.path.join("models", f))
                break
            except Exception:
                pass
acc = 0.0
if model is not None and os.path.exists(os.path.join("data","X_test.pkl")) and os.path.exists(os.path.join("data","y_test.pkl")):
    X_test = joblib.load(os.path.join("data","X_test.pkl"))
    y_test = joblib.load(os.path.join("data","y_test.pkl"))
    try:
        y_pred = model.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
    except Exception:
        acc = 0.0
meta_path = os.path.join("models","model_metadata.json")
meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
meta.setdefault("evaluations", [])
entry = {"ts": int(time.time()), "accuracy_last_test": acc}
meta["evaluations"].append(entry)
meta["evaluations"] = meta["evaluations"][-20:]
os.makedirs("models", exist_ok=True)
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
                    triggeredBy 'UserIdCause'
                    expression { return (params != null && params.containsKey('FORCE_MANUAL_LOG') && params.FORCE_MANUAL_LOG.toString().toBoolean()) }
                }
            }
            steps {
                sh '''
set -euo pipefail
. "${VENV_DIR}/bin/activate"
python - <<'PY'
import os
with open('manual_intervention.txt','a') as fh:
    fh.write("manual_intervention\\n")
pg = os.environ.get('PUSHGATEWAY_URL')
if pg:
    try:
        from prometheus_client import Gauge, CollectorRegistry, push_to_gateway
        registry = CollectorRegistry()
        g = Gauge('manual_intervention_count', 'Manual interventions count', registry=registry)
        g.set(1)
        push_to_gateway(pg + '/metrics/job/${PROJECT_NAME}_manual', registry=registry)
    except Exception as e:
        print("Pushgateway push failed:", e)
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
                echo "SUCCESS: reading archived times if present"
                if (fileExists('deployment_time.txt')) {
                    echo "Deployment time (s): " + readFile('deployment_time.txt').trim()
                }
                if (fileExists('retrain_time.txt')) {
                    echo "Retrain time (s): " + readFile('retrain_time.txt').trim()
                }
            }
        }
        failure {
            script {
                if (env.ENABLE_MAIL == 'true') {
                    mail to: 'you@example.com',
                         subject: "Pipeline failed: ${env.JOB_NAME} #${env.BUILD_NUMBER}",
                         body: "See Jenkins console output for details."
                } else {
                    echo "Build failed but mail disabled (ENABLE_MAIL!=true)."
                }
            }
        }
    }
}
