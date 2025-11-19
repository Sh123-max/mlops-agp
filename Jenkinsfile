pipeline {
    agent any

    environment {
        PROJECT_NAME = "${env.PROJECT_NAME ?: 'diabetes'}"
        DATA_DIR = "${env.DATA_DIR ?: 'data'}"
        MODEL_DIR = "${env.MODEL_DIR ?: 'models'}"
        MLFLOW_TRACKING_URI = "${env.MLFLOW_TRACKING_URI ?: 'http://localhost:5001'}"
        CONDA_PATH = "/home/shreekar/miniconda3"
        CONDA_ENV = "mlops-agp"
        FLASK_SERVICE = "mlops-flask.service"
    }

    options {
        timeout(time: 1, unit: 'HOURS')
        timestamps()
        ansiColor('xterm')
    }

    stages {

        stage('Check changes (diabetes.csv)') {
            steps {
                script {
                    // Ensure we have recent refs to diff against
                    sh "git fetch --all --quiet || true"

                    // Get changed files between last two commits
                    // NOTE: this uses HEAD~1 HEAD (works for typical push events where HEAD~1 exists)
                    def changedFiles = sh(script: "git diff --name-only HEAD~1 HEAD || true", returnStdout: true).trim()
                    echo "Changed files (HEAD~1..HEAD):\\n${changedFiles}"

                    if (!changedFiles?.contains('diabetes.csv')) {
                        echo "No changes to diabetes.csv detected — skipping pipeline."
                        // Mark build as not built and exit the pipeline early
                        currentBuild.result = 'NOT_BUILT'
                        return
                    } else {
                        echo "diabetes.csv changed — continuing pipeline."
                    }
                }
            }
        }

        stage('Preprocess') {
            steps {
                echo "Running preprocessing..."
                sh """
                    set -e
                    . ${CONDA_PATH}/etc/profile.d/conda.sh
                    conda activate ${CONDA_ENV}
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

            python3 - <<-'PY'
import json, os, sys

summary_path = os.path.join('${MODEL_DIR}', 'last_run_summary.json')
if not os.path.exists(summary_path):
    print('No summary file found')
    sys.exit(1)

summary = json.load(open(summary_path))
best = summary.get('best', {})
should_deploy = best.get('should_deploy', True)
deploy_reason = best.get('deploy_reason', 'No reason provided')

if not should_deploy:
    print(f'Model validation failed: {deploy_reason}')
    # Write to file for artifact collection
    with open('validation_failure.txt', 'w') as f:
        f.write(deploy_reason)
    sys.exit(1)

print(f'Model validation passed: {deploy_reason}')
PY
        """
    }
    post {
        failure {
            echo "Model validation failed - performance degradation detected"
            archiveArtifacts artifacts: 'validation_failure.txt', fingerprint: true
            // Optionally trigger manual review workflow
            sh """
                echo "Manual intervention required due to model performance degradation"
            """
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
                archiveArtifacts artifacts: 'train_log.txt,retrain_time.txt', fingerprint: true
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
                """
            }
        }

        stage('Run Flask App & Health Check') {
            steps {
                echo "Checking Flask service health..."
                sh """
                    # Optional: just check service health via curl (assumes systemd-managed Flask or an already-running Flask)
                    if curl --max-time 5 --silent --fail http://localhost:5000/health; then
                        echo '[✔] Flask service is healthy!'
                    else
                        echo '[✘] Flask service did not respond correctly!'
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
