pipeline {
    agent any

    environment {
        PROJECT_NAME = "${env.PROJECT_NAME ?: 'diabetes'}"
        DATA_DIR     = "${env.DATA_DIR ?: 'data'}"
        MODEL_DIR    = "${env.MODEL_DIR ?: 'models'}"
        MLFLOW_TRACKING_URI = "${env.MLFLOW_TRACKING_URI ?: 'http://localhost:5001'}"
        REQUIREMENTS = "${env.REQUIREMENTS ?: 'requirements.txt'}"
        ENABLE_MAIL  = "${env.ENABLE_MAIL ?: 'false'}"
        INSTALL_BUILD_DEPS = "true"
        VENV_DIR     = "${env.VENV_DIR ?: '.venv'}"
        PYTHONUNBUFFERED = "1"
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
                sh('''bash -s <<'BASH'
#!/usr/bin/env bash
set -euo pipefail

echo "NODE PYTHON PATH: $(which python3 || true)"
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found on this node. Please install python3 (and pip) or use a node that has it." >&2
  exit 3
fi

python3 -m venv "${VENV_DIR}"
. "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel

echo "Installing numpy first (prefer binary wheel)"
pip install --no-cache-dir --prefer-binary numpy || true

INSTALLED_OK=0

if [ -f "${REQUIREMENTS}" ]; then
  echo "Installing from ${REQUIREMENTS} (prefer binary wheels)"
  if pip install --no-cache-dir --prefer-binary -r "${REQUIREMENTS}"; then
    echo "pip install succeeded"
    INSTALLED_OK=1
  else
    echo "pip install failed"
    INSTALLED_OK=0
  fi
else
  echo "No requirements.txt found; installing minimal runtime deps"
  if pip install --no-cache-dir --prefer-binary pandas numpy scikit-learn joblib prometheus_client prometheus-flask-exporter; then
    echo "pip install succeeded (minimal set)"
    INSTALLED_OK=1
  else
    echo "pip install failed (minimal set)"
    INSTALLED_OK=0
  fi
fi

if [ "${INSTALLED_OK}" -eq 0 ]; then
  echo "Initial pip install failed."
  if [ "${INSTALL_BUILD_DEPS}" = "true" ]; then
    echo "INSTALL_BUILD_DEPS=true -> installing system build deps and retrying pip install"
    if command -v apt-get >/dev/null 2>&1; then
      apt-get update -y
      apt-get install -y build-essential gfortran libatlas-base-dev libopenblas-dev liblapack-dev python3-dev pkg-config
      if [ -f "${REQUIREMENTS}" ]; then
        pip install --no-cache-dir -r "${REQUIREMENTS}"
      else
        pip install --no-cache-dir pandas numpy scikit-learn joblib prometheus_client prometheus-flask-exporter
      fi
      INSTALLED_OK=1
    else
      echo "apt-get not available on this node; cannot install system build deps." >&2
      exit 5
    fi
  else
    echo "INSTALL_BUILD_DEPS is false -> not installing system build deps. Exiting to keep node light." >&2
    exit 4
  fi
fi

python - <<'PY'
import importlib, sys
reqs = ['pandas','numpy','sklearn','joblib']
missing = [r for r in reqs if importlib.util.find_spec(r) is None]
if missing:
    sys.stderr.write('Missing python packages: %s\n' % missing)
    sys.exit(6)
else:
    print('Python deps OK')
PY

echo "Prepared Python environment in ${VENV_DIR}"
BASH
''')
            }
        }

        stage('Preprocess') {
            steps {
                sh('''bash -s <<'BASH'
#!/usr/bin/env bash
set -euo pipefail
. "${VENV_DIR}/bin/activate"
if [ -f "preprocess.py" ]; then
  python preprocess.py
else
  echo "preprocess.py not found - skipping"
fi
BASH
''')
            }
        }

        stage('Train') {
            steps {
                script {
                    env.TRAIN_START = sh(script: '''bash -s <<'BASH'
#!/usr/bin/env bash
. "${VENV_DIR}/bin/activate"
python - <<'PY'
import time
print(int(time.time()))
PY
BASH
''', returnStdout: true).trim()
                }

                sh('''bash -s <<'BASH'
#!/usr/bin/env bash
set -euo pipefail
. "${VENV_DIR}/bin/activate"
if [ -f "trainandevaluate.py" ]; then
  python trainandevaluate.py 2>&1 | tee train_log.txt
else
  echo "trainandevaluate.py not found - skipping" | tee train_log.txt
fi
BASH
''')

                script {
                    env.TRAIN_END = sh(script: '''bash -s <<'BASH'
#!/usr/bin/env bash
. "${VENV_DIR}/bin/activate"
python - <<'PY'
import time
print(int(time.time()))
PY
BASH
''', returnStdout: true).trim()
                }
            }
        }

        stage('Record retrain time metric') {
            steps {
                sh('''bash -s <<'BASH'
#!/usr/bin/env bash
set -euo pipefail
. "${VENV_DIR}/bin/activate"
python - <<'PY'
import os
start = int(os.environ.get('TRAIN_START', '0'))
end = int(os.environ.get('TRAIN_END', '0'))
t = end - start if (start and end) else 0
with open('retrain_time.txt', 'w') as f:
    f.write(str(t))
print('retrain_time_seconds:', t)
PY
BASH
''')
                archiveArtifacts artifacts: 'train_log.txt,retrain_time.txt', fingerprint: true
            }
        }

        stage('Deploy') {
            steps {
                sh('''bash -s <<'BASH'
#!/usr/bin/env bash
set -euo pipefail
. "${VENV_DIR}/bin/activate"
if [ -f "deploy.py" ]; then
  python deploy.py --project "${PROJECT_NAME}" --stage Staging || true
else
  echo "deploy.py not found - skipping"
fi
if [ ! -f deployment_time.txt ]; then
  echo 0 > deployment_time.txt
fi
BASH
''')
                archiveArtifacts artifacts: 'deployment_time.txt', fingerprint: true
            }
        }

        stage('Evaluate Drift Accuracy (7-day)') {
            steps {
                sh('''bash -s <<'BASH'
#!/usr/bin/env bash
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
if model is not None and os.path.exists(os.path.join("data","X_test.pkl")) and os.path.exists(os.path.join("data","y
