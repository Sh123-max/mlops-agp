// Jenkinsfile (Declarative) - quote & dollar-safe version
pipeline {
    agent any

    environment {
        PROJECT_NAME = "${env.PROJECT_NAME ?: 'diabetes'}"
        DATA_DIR = "${env.DATA_DIR ?: 'data'}"
        MODEL_DIR = "${env.MODEL_DIR ?: 'models'}"
        MLFLOW_TRACKING_URI = "${env.MLFLOW_TRACKING_URI ?: 'http://localhost:5001'}"
        REQUIREMENTS = "${env.REQUIREMENTS ?: 'requirements.txt'}"
        ENABLE_MAIL = "${env.ENABLE_MAIL ?: 'false'}"
        INSTALL_BUILD_DEPS = "${env.INSTALL_BUILD_DEPS ?: 'false'}"
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
                sh('''bash -lc <<'BASH'
set -euo pipefail

echo "NODE PYTHON PATH: $(which python3 || true)"
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found on this node. Please install python3 (and pip) or use a node that has it." >&2
  exit 3
fi

python3 -m venv "${VENV_DIR}"
. "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel

if [ -f "${REQUIREMENTS}" ]; then
  echo "Installing from ${REQUIREMENTS} (preferring binary wheels)"
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
    echo "INSTALL_BUILD_DEPS=true -> attempting to install system build deps and retry"
    if command -v apt-get >/dev/null 2>&1; then
      apt-get update -y && apt-get install -y build-essential gfortran libatlas-base-dev
      if [ -f "${REQUIREMENTS}" ]; then
        pip install --no-cache-dir -r "${REQUIREMENTS}"
      else
        pip install --no-cache-dir pandas numpy scikit-learn joblib prometheus_client prometheus-flask-exporter
      fi
    else
      echo "apt-get not available on this node; cannot install system build deps." >&2
      exit 5
    fi
  else
    echo "INSTALL_BUILD_DEPS is false -> not installing system build deps. Failing to keep pipeline light." >&2
    exit 4
  fi
fi

# quick python check for essential packages
python -c "import importlib,sys; reqs=['pandas','numpy','sklearn','joblib']; missing=[r for r in reqs if importlib.util.find_spec(r) is None]; \
if missing: \
    sys.stderr.write('Missing python packages: %s\\n' % missing); \
    sys.exit(6); \
else: \
    print('Python deps OK')"
BASH
''')
            }
        }

        stage('Preprocess') {
            steps {
                sh('''bash -lc <<'BASH'
set -euo pipefail
. "${VENV_DIR}/bin/activate"
python preprocess.py
BASH
''')
            }
        }

        stage('Train') {
            steps {
                script {
                    env.TRAIN_START = sh(returnStdout: true, script: '''bash -lc <<'BASH'
. "${VENV_DIR}/bin/activate"
python - <<'PY'
import time
print(int(time.time()))
PY
BASH
''').trim()

                    sh('''bash -lc <<'BASH'
set -euo pipefail
. "${VENV_DIR}/bin/activate"
python trainandevaluate.py 2>&1 | tee train_log.txt
BASH
''')

                    env.TRAIN_END = sh(returnStdout: true, script: '''bash -lc <<'BASH'
. "${VENV_DIR}/bin/activate"
python - <<'PY'
import time
print(int(time.time()))
PY
BASH
''').trim()
                }
            }
        }

        stage('Record retrain time metric') {
            steps {
                sh('''bash -lc <<'BASH'
set -euo pipefail
. "${VENV_DIR}/bin/activate"
python - <<'PY'
import os
start = int(os.environ.get("TRAIN_START", "0"))
end = int(os.environ.get("TRAIN_END", "0"))
t = end - start if (start and end) else 0
with open("retrain_time.txt","w") as f:
    f.write(str(t))
print("retrain_time_seconds:", t)
PY
BASH
''')
                archiveArtifacts artifacts: 'train_log.txt,retrain_time.txt', fingerprint: true
            }
        }

        stage('Deploy') {
            steps {
                sh('''bash -lc <<'BASH'
set -euo pipefail
. "${VENV_DIR}/bin/activate"
python deploy.py --project "${PROJECT_NAME}" --stage Staging || true
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
                sh('''bash -lc <<'BASH'
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
BASH
''')
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
                sh('''bash -lc <<'BASH'
set -euo pipefail
. "${VENV_DIR}/bin/activate"
python - <<'PY'
import os
with open("manual_intervention.txt","a") as fh:
    fh.write("manual_intervention\\n")
pg = os.environ.get("PUSHGATEWAY_URL")
if pg:
    try:
        from prometheus_client import Gauge, CollectorRegistry, push_to_gateway
        registry = CollectorRegistry()
        g = Gauge("manual_intervention_count", "Manual interventions count", registry=registry)
        g.set(1)
        push_to_gateway(pg + "/metrics/job/${PROJECT_NAME}_manual", registry=registry)
    except Exception as e:
        print("Pushgateway push failed:", e)
print("Manual intervention logged.")
PY
BASH
''')
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
