pipeline {
    agent any

    environment {
        PROJECT_NAME = "${env.PROJECT_NAME ?: 'diabetes'}"
        DATA_DIR    = "${env.DATA_DIR ?: 'data'}"
        MODEL_DIR   = "${env.MODEL_DIR ?: 'models'}"
        MLFLOW_TRACKING_URI = "${env.MLFLOW_TRACKING_URI ?: 'http://localhost:5001'}"
        CONDA_PATH  = "/home/shreekar/miniconda3"
        CONDA_ENV   = "mlops-agp"
        FLASK_SERVICE = "mlops-flask.service"
    }

    options {
        timeout(time: 1, unit: 'HOURS')
        timestamps()
        ansiColor('xterm')
    }

    stages {
        stage('Check changes (data)') {
            steps {
                script {
                    sh "git fetch --all --quiet || true"
                    def changedFiles = sh(script: "git diff --name-only HEAD~1 HEAD || true", returnStdout: true).trim()
                    echo "Changed files (HEAD~1..HEAD):\\n${changedFiles}"
                    // You can gate the pipeline using changedFiles if desired.
                }
            }
        }

        stage('Preprocess') {
            steps {
                echo "Running preprocessing for project ${env.PROJECT_NAME}..."
                sh """
                    set -e
                    . ${CONDA_PATH}/etc/profile.d/conda.sh
                    conda activate ${CONDA_ENV}
                    export PROJECT_NAME=${PROJECT_NAME}
                    python3 preprocess.py
                """
            }
        }

        stage('Train & Evaluate') {
            steps {
                echo "Running training and evaluation..."
                script {
                    env.TRAIN_START = sh(script: "python3 -c 'import time; print(int(time.time()))'", returnStdout: true).trim()
                    sh """
                        set -e
                        . ${CONDA_PATH}/etc/profile.d/conda.sh
                        conda activate ${CONDA_ENV}
                        python3 trainandevaluate.py 2>&1 | tee train_log.txt
                    """
                    env.TRAIN_END = sh(script: "python3 -c 'import time; print(int(time.time()))'", returnStdout: true).trim()
                }
            }
        }

        stage('Model Validation') {
            steps {
                echo "Validating model performance against previous best..."
                sh """
                    set -e
                    . ${CONDA_PATH}/etc/profile.d/conda.sh
                    conda activate ${CONDA_ENV}

                    python3 - <<'PY'
import json, os, sys

summary_path = os.path.join('${MODEL_DIR}', 'last_run_summary.json')
if not os.path.exists(summary_path):
    print('No summary file found at:', summary_path)
    # create a file for downstream debugging/artifacts
    with open('validation_failure.txt', 'w') as f:
        f.write('No last_run_summary.json found. Check train_log.txt for errors.')
    sys.exit(1)

summary = json.load(open(summary_path))
best = summary.get('best', {})
should_deploy = best.get('should_deploy', True)
deploy_reason = best.get('deploy_reason', 'No reason provided')

if not should_deploy:
    print(f'Model validation blocked: {deploy_reason}')
    with open('validation_failure.txt', 'w') as f:
        f.write(deploy_reason)
    sys.exit(1)

print(f'Model validation passed: {deploy_reason}')
PY
                """
            }
            post {
                failure {
                    echo "Model validation failed - archiving train_log.txt and validation_failure.txt for debugging"
                    archiveArtifacts artifacts: 'validation_failure.txt,train_log.txt', allowEmptyArchive: true, fingerprint: true
                    sh '''
                        echo "Manual intervention required due to model performance degradation"
                    '''
                }
            }
        }

        stage('Ensemble Creation Check') {
            steps {
                echo "Checking if ensemble was created..."
                sh """
                    python3 - <<'PY'
import json, os
summary_path = os.path.join('${MODEL_DIR}', 'last_run_summary.json')
if os.path.exists(summary_path):
    summary = json.load(open(summary_path))
    best = summary.get('best', {})
    if best.get('is_ensemble', False):
        print('ENSEMBLE MODEL CREATED: ' + best.get('name', ''))
        with open('ensemble_created.txt', 'w') as f:
            f.write('Ensemble created: ' + best.get('name', ''))
    else:
        print('No ensemble created - performance maintained')
else:
    print('No last_run_summary.json found - skipping ensemble check')
PY
                """
            }
            post {
                always {
                    archiveArtifacts artifacts: 'ensemble_created.txt', allowEmptyArchive: true, fingerprint: true
                }
            }
        }

        stage('Record Retrain Time Metric') {
            steps {
                echo "Recording retrain time metric..."
                sh """
                    python3 - <<'PY'
import os
start = int(os.environ.get('TRAIN_START', '0'))
end = int(os.environ.get('TRAIN_END', '0'))
t = end - start if (start and end) else 0
with open('retrain_time.txt','w') as f:
    f.write(str(t))
print("retrain_time_seconds:", t)
PY
                """
                archiveArtifacts artifacts: 'train_log.txt,retrain_time.txt', allowEmptyArchive: true, fingerprint: true
            }
        }

        stage('Deploy Model') {
            steps {
                echo "Deploying model..."
                sh """
                    set -e
                    . ${CONDA_PATH}/etc/profile.d/conda.sh
                    conda activate ${CONDA_ENV}
                    python3 deploy.py

                    if [ -f "${MODEL_DIR}/model_metadata.json" ]; then
                        python3 - <<'PY'
import json, os, sys
p = os.path.join("${MODEL_DIR}", "model_metadata.json")
try:
    m = json.load(open(p))
    name = m.get('model_name') or m.get('best', {}).get('name') or 'unknown'
    version = m.get('version') or (m.get('best', {}).get('registry') or {}).get('version') or 'N/A'
    print(f"JENKINS: Deployed model -> name={name} version={version}")
except Exception as e:
    print("JENKINS: Failed to parse model_metadata.json:", e)
    sys.exit(0)
PY
                    else
                        echo "JENKINS: model_metadata.json not present after deployment"
                    fi
                """
            }
        }

        stage('Run Flask App & Health Check') {
            steps {
                echo "Checking Flask service health..."
                sh """
                    # optional: ensure your flask is running and listening on port 5000
                    if curl --max-time 5 --silent --fail http://localhost:5000/health; then
                        echo 'Flask service is healthy!'
                    else
                        echo 'Flask service did not respond correctly!'
                        exit 1
                    fi
                """
            }
        }
    }

    post {
        always {
            echo "Cleaning up workspace..."
            cleanWs()
        }
    }
}
