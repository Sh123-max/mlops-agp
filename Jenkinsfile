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
                    # Optional: just check service health via curl
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
